#!/usr/bin/env python3
"""Efficiency measurements for the paper's section 4.1 / table: parameter
counts, single-image inference time and peak inference VRAM.

Requires a quiet GPU (run it when no other job occupies the device).
Parameter counts are read from the real checkpoint; the trainable/unfrozen
split uses the exact predicate of train_scripts/train.py.
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from diffusers.models import AutoencoderKL

from diffusion import DPMS
from diffusion.model.nets import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from tools.prepare_semantic_masks import build_semantic_masks

OUT = ROOT / 'results/efficiency'
OUT.mkdir(parents=True, exist_ok=True)

TRAINABLE_PREFIXES = ('semantic_adapters.', 'semantic_layer_scale',
                      'semantic_time_gate.', 'semantic_token_cross_attention.',
                      'semantic_token_gate', 'semantic_token_layer_gate')

MODELS = {
    'frozen': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0),
    'tpscda': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0),
}


def gpu_name():
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            text=True).strip()
    except Exception:
        return torch.cuda.get_device_name(0)


def measure(tag, args, tokenizer, nlp, t5):
    cfg = MODELS[tag]
    print(f'--- {tag} ---', flush=True)
    latent = args.image_size // 8
    model = PixArt_XL_2(input_size=latent, lewei_scale=1,
                        semantic_conditioning=cfg['semantic_conditioning'],
                        semantic_adapter_dim=64, semantic_dropout=0,
                        semantic_residual_scale=cfg['semantic_residual_scale'],
                        semantic_token_attention=cfg['semantic_token_attention'],
                        semantic_token_gate_max=cfg['semantic_token_gate_max'])
    state = torch.load(cfg['checkpoint'], map_location='cpu')
    state = state.get('state_dict', state); state.pop('pos_embed', None)
    model.load_state_dict(state, strict=False)
    del state

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith(TRAINABLE_PREFIXES))

    model = model.half().cuda().eval()
    vae = AutoencoderKL.from_pretrained(
        str(ROOT / 'output/pretrained_models/sd-vae-ft-ema')).half().cuda().eval()

    prompts = ["a green bench and a blue bowl"] * args.batch_size
    with torch.inference_mode():
        emb, text_mask = t5.get_text_embeddings(prompts)
        emb = emb[:, None].cuda().half()
        text_mask = text_mask.cuda()
        null = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None].half()
        hw = torch.full((len(prompts), 2), args.image_size, device='cuda', dtype=torch.float16)
        ar = torch.ones((len(prompts), 1), device='cuda', dtype=torch.float16)
        di = {'img_hw': hw, 'aspect_ratio': ar}
        if cfg['semantic_conditioning']:
            di['semantic_token_masks'] = torch.stack([
                torch.from_numpy(build_semantic_masks(p, nlp, tokenizer, emb.shape[2]))
                for p in prompts]).cuda().half()

        # warm-up
        for _ in range(args.warmup):
            n = torch.randn(len(prompts), 4, latent, latent, device='cuda', dtype=torch.float16)
            s = DPMS(model.forward_with_dpmsolver, condition=emb, uncondition=null,
                     cfg_scale=args.cfg_scale,
                     model_kwargs=dict(data_info=di, mask=text_mask))
            lat = s.sample(n, steps=args.steps, order=2, skip_type='time_uniform',
                           method='multistep')
            _ = vae.decode((lat / 0.18215).half()).sample
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(args.repeats):
            n = torch.randn(len(prompts), 4, latent, latent, device='cuda',
                            dtype=torch.float16)
            torch.cuda.synchronize(); t0 = time.time()
            s = DPMS(model.forward_with_dpmsolver, condition=emb, uncondition=null,
                     cfg_scale=args.cfg_scale,
                     model_kwargs=dict(data_info=di, mask=text_mask))
            lat = s.sample(n, steps=args.steps, order=2, skip_type='time_uniform',
                           method='multistep')
            _ = vae.decode((lat / 0.18215).half()).sample
            torch.cuda.synchronize(); t1 = time.time()
            times.append((t1 - t0) / len(prompts))
        peak = torch.cuda.max_memory_allocated() / 1e9
    del model, vae
    torch.cuda.empty_cache()
    return dict(
        method=tag, checkpoint=cfg['checkpoint'],
        total_params=total, trainable_params=trainable,
        trainable_fraction=trainable / total,
        batch_size=args.batch_size, steps=args.steps, cfg=args.cfg_scale,
        image_size=args.image_size,
        seconds_per_image_mean=float(np.mean(times)),
        seconds_per_image_std=float(np.std(times)),
        seconds_per_image_min=float(np.min(times)),
        seconds_per_image_max=float(np.max(times)),
        peak_vram_gb_decode_inclusive=peak,
        n_timed_runs=len(times),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--cfg-scale', type=float, default=4.0)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--t5-cache', default=str(ROOT / 'output/pretrained_models/t5_ckpts'))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.t5_cache + '/t5-v1_1-xxl',
                                              use_fast=True, local_files_only=True)
    import spacy
    nlp = spacy.load('en_core_web_sm')
    t5 = T5Embedder(device='cuda', local_cache=True, cache_dir=args.t5_cache,
                    torch_dtype=torch.float, model_max_length=120)

    rows = [measure(t, args, tokenizer, nlp, t5) for t in ('frozen', 'tpscda')]

    with open(OUT / 'efficiency.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (OUT / 'efficiency.json').write_text(json.dumps(dict(
        protocol=dict(batch_size=args.batch_size, steps=args.steps,
                      cfg_scale=args.cfg_scale, image_size=args.image_size,
                      sampler='DPM-Solver order=2 multistep'),
        gpu=gpu_name(), cuda=torch.version.cuda, pytorch=torch.__version__,
        python=platform.python_version(), host=platform.node(),
        note=('seconds_per_image is wall-clock per image for the full '
              'sampling loop INCLUDING VAE decode; peak_vram is the peak '
              'allocated torch memory during that loop.'),
        results=rows), indent=2, ensure_ascii=False))
    for r in rows:
        print(f"{r['method']:8s} {r['seconds_per_image_mean']:.3f} s/img "
              f"peak {r['peak_vram_gb_decode_inclusive']:.1f} GB "
              f"trainable {r['trainable_params']:,}")


if __name__ == '__main__':
    main()
