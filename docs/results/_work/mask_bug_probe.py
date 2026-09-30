#!/usr/bin/env python3
"""Verify the cross-attention masking bug hypothesis.

PixArt.forward passes `y_lens` (a python LIST of valid token counts) down to
PixArtBlock.forward -> cross_attn(x, y, mask).  In the modified
MultiHeadCrossAttention the SDPA fallback branch does

    if isinstance(mask, (list, tuple)):
        key_padding = None          # <- silently disables masking

so every image patch attends to the T5 *padding* tokens.  Short prompts are
almost entirely padding (max length 120), long prompts are not -- which matches
the observed pattern (CompBench short prompts melt, samples.txt long prompts
look fine).

This script monkey-patches the mask handling at runtime ONLY (no repo file is
modified) and regenerates the same prompt/seed so the two outputs can be
compared directly.
"""
import sys
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from diffusion.model.nets import PixArt_blocks
from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention

PROMPTS = ['a green bench and a blue bowl', 'A bathroom with beige tile and a white toilet.']
CKPT = ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'

_ORIG = MultiHeadCrossAttention.forward


def fixed_forward(self, x, cond, mask=None):
    B, N, C = x.shape
    q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
    kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
    k, v = kv.unbind(2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # B,H,*,D
    key_padding = None
    if mask is not None:
        if isinstance(mask, (list, tuple)):
            L = k.shape[-2]
            lens = torch.as_tensor([int(m) for m in mask], device=k.device)[:, None]
            idx = torch.arange(L, device=k.device)[None, :]
            key_padding = (idx < lens)[:, None, None, :]      # True = keep
        else:
            key_padding = mask.bool()[:, None, None, :]
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=key_padding)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def build():
    from diffusion.model.nets import PixArt_XL_2
    m = PixArt_XL_2(input_size=64, lewei_scale=1, semantic_conditioning=False,
                    semantic_dropout=0)
    st = torch.load(CKPT, map_location='cpu')
    st = st.get('state_dict', st); st.pop('pos_embed', None)
    m.load_state_dict(st, strict=False)
    return m.half().cuda().eval()


def run(tag, out):
    from diffusers.models import AutoencoderKL
    from torchvision.utils import save_image
    from diffusion import DPMS
    from diffusion.model.t5 import T5Embedder

    m = build()
    vae = AutoencoderKL.from_pretrained(
        str(ROOT / 'output/pretrained_models/sd-vae-ft-ema')).half().cuda().eval()
    t5 = T5Embedder(device='cpu', local_cache=True,
                    cache_dir=str(ROOT / 'output/pretrained_models/t5_ckpts'),
                    torch_dtype=torch.float, model_max_length=120)
    torch.manual_seed(43)
    with torch.inference_mode():
        emb, tmask = t5.get_text_embeddings(PROMPTS)
        cond = emb[:, None].cuda().half()
        tmask = tmask.cuda()
        null = m.y_embedder.y_embedding[None].repeat(len(PROMPTS), 1, 1)[:, None].half()
        hw = torch.full((len(PROMPTS), 2), 512, device='cuda', dtype=torch.float16)
        ar = torch.ones((len(PROMPTS), 1), device='cuda', dtype=torch.float16)
        noise = torch.randn(len(PROMPTS), 4, 64, 64, device='cuda', dtype=torch.float16)
        s = DPMS(m.forward_with_dpmsolver, condition=cond, uncondition=null, cfg_scale=4.0,
                 model_kwargs=dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=tmask))
        lat = s.sample(noise, steps=20, order=2, skip_type='time_uniform', method='multistep')
        imgs = vae.decode((lat / 0.18215).half()).sample
        for p, im in zip(PROMPTS, imgs):
            save_image(im.float(), out / f"{tag}_{p.replace(' ','_').replace('.','')[:34]}.png",
                       normalize=True, value_range=(-1, 1))
    del m, vae
    torch.cuda.empty_cache()
    print('done', tag, flush=True)


if __name__ == '__main__':
    out = Path('/tmp/maskprobe'); out.mkdir(exist_ok=True)
    MultiHeadCrossAttention.forward = _ORIG
    run('buggy', out)
    MultiHeadCrossAttention.forward = fixed_forward
    run('fixed', out)
