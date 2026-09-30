#!/usr/bin/env python3
"""Generate fixed prompt groups with a trained SCDA PixArt checkpoint."""
import argparse
import sys
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from diffusion import DPMS
from diffusion.model.nets import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from tools.prepare_semantic_masks import build_semantic_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--groups-dir', default='prompt_groups')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--cfg-scale', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--semantic-conditioning', action='store_true',
                   help='Enable SCDA semantic residual injection (default: baseline).')
    p.add_argument('--semantic-adapter-dim', type=int, default=64,
                   help='SCDA adapter bottleneck dimension used by the checkpoint.')
    p.add_argument('--semantic-residual-scale', type=float, default=1.0)
    p.add_argument('--semantic-token-attention', action='store_true',
                   help='Use token-level role/pair-aware SCDA attention instead of pooled residuals.')
    p.add_argument('--semantic-token-gate-max', type=float, default=1.0,
                   help='Maximum multiplier for the learned token-attention gate.')
    p.add_argument('--t5-cache', default='output/pretrained_models/t5_ckpts')
    p.add_argument('--vae-path', default='output/pretrained_models/sd-vae-ft-ema')
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    torch.manual_seed(args.seed)

    latent_size = args.image_size // 8
    model = PixArt_XL_2(input_size=latent_size, lewei_scale=0.5 if args.image_size == 256 else 1,
                        semantic_conditioning=args.semantic_conditioning,
                        semantic_adapter_dim=args.semantic_adapter_dim,
                        semantic_dropout=0,
                        semantic_residual_scale=args.semantic_residual_scale,
                        semantic_token_attention=args.semantic_token_attention,
                        semantic_token_gate_max=args.semantic_token_gate_max).to(device, dtype=dtype).eval()
    state = torch.load(args.checkpoint, map_location='cpu')
    state = state.get('state_dict', state)
    state.pop('pos_embed', None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'checkpoint loaded; missing={len(missing)} unexpected={len(unexpected)}')

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, dtype=dtype).eval()
    t5 = T5Embedder(device='cpu', local_cache=True, cache_dir=args.t5_cache,
                    torch_dtype=torch.float, model_max_length=120)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.t5_cache + '/t5-v1_1-xxl', use_fast=True, local_files_only=True)
    import spacy
    nlp = spacy.load('en_core_web_sm')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for group_file in sorted(Path(args.groups_dir).glob('group_*.txt')):
        group_out = output / group_file.stem
        group_out.mkdir(exist_ok=True)
        prompts = [line.strip() for line in group_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        with torch.inference_mode():
            emb, text_mask = t5.get_text_embeddings(prompts)
            emb = emb[:, None].to(device=device, dtype=dtype)
            null = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None].to(dtype)
            semantic_masks = torch.stack([
                torch.from_numpy(build_semantic_masks(prompt, nlp, tokenizer, emb.shape[2]))
                for prompt in prompts
            ]).to(device=device, dtype=dtype) if args.semantic_conditioning else None
            hw = torch.full((len(prompts), 2), args.image_size, device=device, dtype=dtype)
            ar = torch.ones((len(prompts), 1), device=device, dtype=dtype)
            data_info = {'img_hw': hw, 'aspect_ratio': ar}
            if semantic_masks is not None:
                data_info['semantic_token_masks'] = semantic_masks
            text_mask = text_mask.to(device)
            kwargs = dict(data_info=data_info, mask=text_mask)
            noise = torch.randn(len(prompts), 4, latent_size, latent_size, device=device, dtype=dtype)
            solver = DPMS(model.forward_with_dpmsolver, condition=emb, uncondition=null,
                          cfg_scale=args.cfg_scale, model_kwargs=kwargs)
            latents = solver.sample(noise, steps=args.steps, order=2,
                                    skip_type='time_uniform', method='multistep')
            images = vae.decode((latents / 0.18215).to(dtype)).sample
            for prompt, image in zip(prompts, images):
                filename = prompt[:100].replace('/', '_') + '.png'
                save_image(image.float(), group_out / filename, normalize=True, value_range=(-1, 1))
        print(f'{group_file.stem}: {len(prompts)} images -> {group_out}')


if __name__ == '__main__':
    main()
