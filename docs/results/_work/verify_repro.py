#!/usr/bin/env python3
"""Re-generate asset/samples.txt group_01 for a method and compare bit-exactly
with the images already on disk.  T5 is kept on CPU so this can run while a
generation job occupies the GPU.

Usage: verify_repro.py --method tpscda|frozen --out <dir>
"""
import argparse
import sys
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
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
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0,
        reference='output/coco2017_eval_learnable_layers/learnable_layers/seed_43'),
    'frozen': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        reference='output/coco2017_eval/baseline/seed_43'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=sorted(METHODS))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = METHODS[args.method]

    prompts = [l.strip() for l in
               (ROOT / 'output/coco2017_eval_learnable_layers/prompt_groups_64/group_01.txt'
                ).read_text().splitlines() if l.strip()]
    out = Path(args.out); (out / 'group_01').mkdir(parents=True, exist_ok=True)

    device, dtype = 'cuda', torch.float16
    torch.manual_seed(43)
    model = PixArt_XL_2(input_size=64, lewei_scale=1,
                        semantic_conditioning=cfg['semantic_conditioning'],
                        semantic_adapter_dim=64, semantic_dropout=0,
                        semantic_residual_scale=cfg['semantic_residual_scale'],
                        semantic_token_attention=cfg['semantic_token_attention'],
                        semantic_token_gate_max=cfg['semantic_token_gate_max']
                        ).to(device, dtype=dtype).eval()
    st = torch.load(cfg['checkpoint'], map_location='cpu')
    st = st.get('state_dict', st); st.pop('pos_embed', None)
    model.load_state_dict(st, strict=False)
    del st; torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(
        str(ROOT / 'output/pretrained_models/sd-vae-ft-ema')).to(device, dtype=dtype).eval()

    from transformers import AutoTokenizer
    t5 = T5Embedder(device='cpu', local_cache=True,
                    cache_dir=str(ROOT / 'output/pretrained_models/t5_ckpts'),
                    torch_dtype=torch.float, model_max_length=120)
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / 'output/pretrained_models/t5_ckpts/t5-v1_1-xxl'),
        use_fast=True, local_files_only=True)
    import spacy
    nlp = spacy.load('en_core_web_sm')

    with torch.inference_mode():
        emb, text_mask = t5.get_text_embeddings(prompts)
        emb = emb[:, None].to(device=device, dtype=dtype)
        text_mask = text_mask.to(device)
        null = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None].to(dtype)
        hw = torch.full((len(prompts), 2), 512, device=device, dtype=dtype)
        ar = torch.ones((len(prompts), 1), device=device, dtype=dtype)
        di = {'img_hw': hw, 'aspect_ratio': ar}
        if cfg['semantic_conditioning']:
            di['semantic_token_masks'] = torch.stack([
                torch.from_numpy(build_semantic_masks(p, nlp, tokenizer, emb.shape[2]))
                for p in prompts]).to(device=device, dtype=dtype)
        noise = torch.randn(len(prompts), 4, 64, 64, device=device, dtype=dtype)
        solver = DPMS(model.forward_with_dpmsolver, condition=emb, uncondition=null,
                      cfg_scale=4.0, model_kwargs=dict(data_info=di, mask=text_mask))
        lat = solver.sample(noise, steps=20, order=2, skip_type='time_uniform',
                            method='multistep')
        imgs = vae.decode((lat / 0.18215).to(dtype)).sample
        for p, im in zip(prompts, imgs):
            save_image(im.float(), out / 'group_01' / (p[:100].replace('/', '_') + '.png'),
                       normalize=True, value_range=(-1, 1))

    new = sorted((out / 'group_01').glob('*.png'))
    ref_dir = ROOT / cfg['reference'] / 'group_01'
    old = sorted(ref_dir.glob('*.png')) if ref_dir.is_dir() else []
    if not old:
        print(f'{args.method}: no reference at {ref_dir}; wrote {len(new)} images')
        return
    ok = 0
    for a, b in zip(new, old):
        A = np.asarray(Image.open(a).convert('RGB')); B = np.asarray(Image.open(b).convert('RGB'))
        if A.shape == B.shape and np.array_equal(A, B):
            ok += 1
    print(f'{args.method}: {ok}/{len(new)} bit-identical vs {cfg["reference"]}')


if __name__ == '__main__':
    main()
