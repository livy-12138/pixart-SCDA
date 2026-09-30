#!/usr/bin/env python3
"""Is the degraded generation caused by fp16 inference?

Runs the SAME prompt/seed through this codebase's sampling path twice, once in
fp16 and once in fp32, so the only variable is the dtype.  The official
scripts/inference.py runs the model in fp32 and casts caption embeddings with
.float(); generate_scda_samples.py runs everything in fp16.
"""
import sys
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

PROMPTS = ['a green bench and a blue bowl', 'A bathroom with beige tile and a white toilet.']
CKPT = ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'


def build(dtype):
    m = PixArt_XL_2(input_size=64, lewei_scale=1, semantic_conditioning=False,
                    semantic_dropout=0)
    st = torch.load(CKPT, map_location='cpu')
    st = st.get('state_dict', st); st.pop('pos_embed', None)
    m.load_state_dict(st, strict=False)
    return m.to('cuda', dtype=dtype).eval()


def run(dtype, tag, out):
    m = build(dtype)
    vae = AutoencoderKL.from_pretrained(
        str(ROOT / 'output/pretrained_models/sd-vae-ft-ema')).to('cuda', dtype=dtype).eval()
    from transformers import AutoTokenizer
    t5 = T5Embedder(device='cpu', local_cache=True,
                    cache_dir=str(ROOT / 'output/pretrained_models/t5_ckpts'),
                    torch_dtype=torch.float, model_max_length=120)
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'output/pretrained_models/t5_ckpts/t5-v1_1-xxl'),
        use_fast=True, local_files_only=True)
    import spacy
    nlp = spacy.load('en_core_web_sm')

    torch.manual_seed(43)
    with torch.inference_mode():
        emb, tmask = t5.get_text_embeddings(PROMPTS)
        # official inference.py: caption_embs.float()[:, None]
        cond = emb.float()[:, None].to('cuda', dtype=dtype)
        tmask = tmask.to('cuda')
        null = m.y_embedder.y_embedding[None].repeat(len(PROMPTS), 1, 1)[:, None].to(dtype)
        hw = torch.full((len(PROMPTS), 2), 512, device='cuda', dtype=dtype)
        ar = torch.ones((len(PROMPTS), 1), device='cuda', dtype=dtype)
        di = {'img_hw': hw, 'aspect_ratio': ar}
        noise = torch.randn(len(PROMPTS), 4, 64, 64, device='cuda', dtype=dtype)
        s = DPMS(m.forward_with_dpmsolver, condition=cond, uncondition=null,
                 cfg_scale=4.0, model_kwargs=dict(data_info=di, mask=tmask))
        lat = s.sample(noise, steps=20, order=2, skip_type='time_uniform', method='multistep')
        imgs = vae.decode((lat / 0.18215).to(dtype)).sample
        for p, im in zip(PROMPTS, imgs):
            fn = out / f"{tag}_{p.replace(' ','_').replace('.','')[:36]}.png"
            save_image(im.float(), fn, normalize=True, value_range=(-1, 1))
            print('saved', fn, flush=True)
    del m, vae
    torch.cuda.empty_cache()


if __name__ == '__main__':
    out = Path('/tmp/dtypeprobe'); out.mkdir(exist_ok=True)
    run(torch.float16, 'fp16', out)
    run(torch.float32, 'fp32', out)
