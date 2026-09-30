#!/usr/bin/env python3
"""Generate GenEval images for one method, in the layout its evaluator expects.

    <out>/<prompt_index:05d>/
        metadata.jsonl        <- the N-th line of GenEval evaluation_metadata.jsonl
        samples/0000.png ...  <- n_samples images

Only the `color_attr` subset is produced (100 prompts), which is what the
paper's Table 6 needs.  Protocol matches the rest of this work: 512x512,
DPM-Solver 20 steps, CFG 4.0.  Every image has a deterministic per-image seed so
the run is resume-safe.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from diffusion import DPMS
from diffusion.model.nets import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from tools.prepare_semantic_masks import build_semantic_masks

METHODS = {
    'tpscda': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0),
    'frozen': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0),
    # the checkpoint this paper reports; run alongside frozen so the GenEval
    # number has a same-protocol control (the published PixArt-α value comes
    # from a different evaluation stack)
    'tpscda_nodistill': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers_nodistill/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=sorted(METHODS))
    ap.add_argument('--geneval-dir', default='/tmp/ge/geneval')
    ap.add_argument('--out-root', default='/root/compbench_work/geneval/images')
    ap.add_argument('--n-samples', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--tag-filter', default='color_attr')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--cfg-scale', type=float, default=4.0)
    ap.add_argument('--t5-cache', default=str(ROOT / 'output/pretrained_models/t5_ckpts'))
    ap.add_argument('--vae-path', default=str(ROOT / 'output/pretrained_models/sd-vae-ft-ema'))
    args = ap.parse_args()

    cfg = METHODS[args.method]
    out_root = Path(args.out_root) / args.method
    out_root.mkdir(parents=True, exist_ok=True)
    log = open(out_root / 'generation.log', 'a', buffering=1)

    def say(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
        log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}\n")

    meta_path = Path(args.geneval_dir) / 'prompts/evaluation_metadata.jsonl'
    metas = []
    for i, line in enumerate(meta_path.read_text().splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        if args.tag_filter in rec.get('tag', ''):
            metas.append((i, rec))
    say(f'method={args.method} tag={args.tag_filter} prompts={len(metas)} '
        f'x {args.n_samples} images')

    device, dtype = 'cuda', torch.float16
    torch.manual_seed(args.seed)
    latent = args.image_size // 8
    model = PixArt_XL_2(input_size=latent, lewei_scale=1,
                        semantic_conditioning=cfg['semantic_conditioning'],
                        semantic_adapter_dim=64, semantic_dropout=0,
                        semantic_residual_scale=cfg['semantic_residual_scale'],
                        semantic_token_attention=cfg['semantic_token_attention'],
                        semantic_token_gate_max=cfg['semantic_token_gate_max']
                        ).to(device, dtype=dtype).eval()
    st = torch.load(cfg['checkpoint'], map_location='cpu')
    st = st.get('state_dict', st); st.pop('pos_embed', None)
    miss, unexp = model.load_state_dict(st, strict=False)
    say(f'checkpoint loaded; missing={len(miss)} unexpected={len(unexp)}')
    del st; torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, dtype=dtype).eval()
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_cache,
                    torch_dtype=torch.float, model_max_length=120)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.t5_cache + '/t5-v1_1-xxl',
                                              use_fast=True, local_files_only=True)
    import spacy
    nlp = spacy.load('en_core_web_sm')

    t0 = time.time()
    done = 0
    with torch.inference_mode():
        for bi in range(0, len(metas), args.batch_size):
            chunk = metas[bi:bi + args.batch_size]
            # expand each prompt to n_samples
            jobs = [(idx, rec, s) for idx, rec in chunk for s in range(args.n_samples)]
            jobs = [j for j in jobs
                    if not (out_root / f'{j[0]:05d}' / 'samples' / f'{j[2]:04d}.png').exists()]
            if not jobs:
                continue
            prompts = [j[1]['prompt'] for j in jobs]
            emb, text_mask = t5.get_text_embeddings(prompts)
            emb = emb[:, None].to(device=device, dtype=dtype)
            null = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None].to(dtype)
            hw = torch.full((len(prompts), 2), args.image_size, device=device, dtype=dtype)
            ar = torch.ones((len(prompts), 1), device=device, dtype=dtype)
            di = {'img_hw': hw, 'aspect_ratio': ar}
            if cfg['semantic_conditioning']:
                di['semantic_token_masks'] = torch.stack([
                    torch.from_numpy(build_semantic_masks(p, nlp, tokenizer, emb.shape[2]))
                    for p in prompts]).to(device=device, dtype=dtype)
            noise = torch.stack([
                torch.randn(4, latent, latent, device=device, dtype=dtype,
                            generator=torch.Generator(device=device).manual_seed(
                                (args.seed * 7_919 + j[0] * 131 + j[2]) % (2 ** 31 - 1)))
                for j in jobs])
            solver = DPMS(model.forward_with_dpmsolver, condition=emb, uncondition=null,
                          cfg_scale=args.cfg_scale,
                          model_kwargs=dict(data_info=di, mask=text_mask))
            lat = solver.sample(noise, steps=args.steps, order=2,
                                skip_type='time_uniform', method='multistep')
            imgs = vae.decode((lat / 0.18215).to(dtype)).sample
            for (idx, rec, s), im in zip(jobs, imgs):
                d = out_root / f'{idx:05d}'
                (d / 'samples').mkdir(parents=True, exist_ok=True)
                if not (d / 'metadata.jsonl').exists():
                    (d / 'metadata.jsonl').write_text(json.dumps(rec))
                save_image(im.float(), d / 'samples' / f'{s:04d}.png',
                           normalize=True, value_range=(-1, 1))
                done += 1
            el = time.time() - t0
            say(f'  {done} images, {el/60:.1f} min, {el/max(done,1):.2f} s/img')

    (out_root / 'config.json').write_text(json.dumps(dict(
        experiment_name=f'geneval_{args.method}',
        model_name='PixArt-XL-2-512x512',
        checkpoint=cfg['checkpoint'], method=args.method,
        semantic_conditioning=cfg['semantic_conditioning'],
        semantic_token_attention=cfg['semantic_token_attention'],
        semantic_token_gate_max=cfg['semantic_token_gate_max'],
        semantic_residual_scale=cfg['semantic_residual_scale'],
        prompt_file=str(meta_path), prompt_version='GenEval official prompts',
        tag_filter=args.tag_filter, n_prompts=len(metas), n_samples=args.n_samples,
        seed=args.seed, resolution=args.image_size,
        sampler='DPM-Solver (order=2, multistep, time_uniform)',
        num_steps=args.steps, CFG=args.cfg_scale,
        GPU=torch.cuda.get_device_name(0), CUDA_version=torch.version.cuda,
        PyTorch_version=torch.__version__, Python_version=sys.version.split()[0],
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        total_seconds=round(time.time() - t0, 1)), indent=2, ensure_ascii=False))
    say(f'DONE {args.method} {done} images in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
