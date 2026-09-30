#!/usr/bin/env python3
"""Runtime restore of the cross-attention key-padding mask (ISSUE-011).

`diffusion/model/nets/PixArt_blocks.py` in this repo rewrote
`MultiHeadCrossAttention.forward` to add an SDPA fallback for the case where
xformers is unavailable.  In that fallback:

    if isinstance(mask, (list, tuple)):
        key_padding = None          # <- mask silently dropped

But `PixArt.forward` does NOT pass a tensor mask down to the blocks; it passes

    y_lens = mask.sum(dim=1).tolist()
    x = auto_grad_checkpoint(block, x, y, t0, y_lens)

i.e. a plain python list of per-sample valid-token counts.  Because
`_HAS_XFORMERS and (mask is None or B == 1)` is False for B > 1, the SDPA
fallback always runs in batched generation, so **every image patch attends to
all 120 T5 tokens including padding**.  Short prompts are almost entirely
padding, which is why T2I-CompBench prompts degrade while the long
asset/samples.txt prompts look acceptable.

This module monkey-patches the method at runtime ONLY.  No repository file is
modified.  Upstream PixArt-alpha expresses the same semantics with
`xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)`; the SDPA
equivalent is a boolean key-padding mask where True means "keep".

Usage:
    import mask_fix; mask_fix.apply()
"""
import torch
import torch.nn.functional as F


def corrected_forward(self, x, cond, mask=None):
    B, N, C = x.shape
    q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
    kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
    k, v = kv.unbind(2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B,H,*,D)
    key_padding = None
    if mask is not None:
        if isinstance(mask, (list, tuple)):
            L = k.shape[-2]
            lens = torch.as_tensor([int(m) for m in mask], device=k.device)[:, None]
            idx = torch.arange(L, device=k.device)[None, :]
            key_padding = (idx < lens)[:, None, None, :]        # True = keep
        else:
            key_padding = mask.bool()[:, None, None, :]
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=key_padding)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def apply():
    from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention
    MultiHeadCrossAttention.forward = corrected_forward
    return True
