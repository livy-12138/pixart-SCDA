#!/usr/bin/env python3
"""Mechanism visualisation for the paper's section 4.3.6.

For a fixed probe latent and a set of diffusion timesteps we read, at selected
DiT layers, the image-patch -> text-token cross-attention produced by the
FROZEN PixArt cross-attention (present in BOTH models, so the comparison is
like-for-like) and, for TP-SCDA, additionally the structured object->attribute
pair bias B computed by SemanticTokenCrossAttention.

The attention weights are recomputed from q/k inside a wrapper, because the
model itself uses fused SDPA / xformers kernels that do not return maps.  The
model file is NOT modified.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

from diffusion.model.nets import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from tools.prepare_semantic_masks import build_semantic_masks

FONT = ROOT / 'results/_work/fonts/NotoSansCJKsc-Regular.otf'
if FONT.exists():
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(str(FONT))
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

OUT = ROOT / 'results/visualization/mechanism'
OUT.mkdir(parents=True, exist_ok=True)

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

CASES = [
    dict(prompt='a green bench and a blue bowl',
         object_token='bench', attribute_token='green'),
    dict(prompt='a red car and a yellow clock',
         object_token='car', attribute_token='red'),
]


def build(tag):
    cfg = MODELS[tag]
    m = PixArt_XL_2(input_size=64, lewei_scale=1,
                    semantic_conditioning=cfg['semantic_conditioning'],
                    semantic_adapter_dim=64, semantic_dropout=0,
                    semantic_residual_scale=cfg['semantic_residual_scale'],
                    semantic_token_attention=cfg['semantic_token_attention'],
                    semantic_token_gate_max=cfg['semantic_token_gate_max'])
    st = torch.load(cfg['checkpoint'], map_location='cpu')
    st = st.get('state_dict', st); st.pop('pos_embed', None)
    m.load_state_dict(st, strict=False)
    return m.half().cuda().eval()


def find_token_index(tokenizer, prompt, word):
    ids = tokenizer(prompt, add_special_tokens=True)['input_ids']
    # locate the sub-token(s) covering `word` via offsets
    enc = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    off = enc['offset_mapping']
    lo = prompt.lower().find(word.lower())
    hits = [i for i, (a, b) in enumerate(off) if b > a and a < lo + len(word) and b > lo]
    return hits, ids


def run_probe(model, emb_cond, mask_cond, sem_mask, timesteps, layers, capture_bias):
    null_cond = model.y_embedder.y_embedding[None][:, None].to(emb_cond.dtype)
    """Single forward pass per timestep; capture cross-attention maps."""
    num_blocks = len(model.blocks)
    captured = {}
    counter = {'i': 0}
    handles = []

    for li in layers:
        ca = model.blocks[li].cross_attn

        def make_hook(li_):
            def hook(module, args):
                # args = (x, cond, mask)
                x, cond = args[0], args[1]
                mk = args[2] if len(args) > 2 else None
                B, N, C = x.shape
                q = module.q_linear(x).view(B, N, module.num_heads, module.head_dim).transpose(1, 2)
                kv = module.kv_linear(cond).view(B, -1, 2, module.num_heads, module.head_dim)
                k, _ = kv.unbind(2)
                k = k.transpose(1, 2)
                logits = torch.matmul(q, k.transpose(-2, -1)) * (module.head_dim ** -0.5)
                if mk is not None and not isinstance(mk, (list, tuple)):
                    key_padding = mk.bool()[:, None, None, :]
                    logits = logits.masked_fill(~key_padding, torch.finfo(logits.dtype).min)
                w = torch.softmax(logits.float(), dim=-1)
                captured.setdefault('cross', {})[li_] = w.detach()
            return hook
        handles.append(ca.register_forward_pre_hook(make_hook(li)))

    if capture_bias:
        sem = model.semantic_token_cross_attention
        state = {}

        def sem_hook(module, args, kwargs):
            state['call'] = counter['i']
            counter['i'] += 1
        handles.append(sem.register_forward_pre_hook(sem_hook, with_kwargs=True))

        orig_forward = sem.forward

        def patched(image_tokens, text_tokens, role_masks, text_valid_mask=None,
                    return_role_outputs=False):
            li = state.get('call', -1)
            with torch.no_grad():
                B, patches, hidden = image_tokens.shape
                length = text_tokens.shape[1]
                q = sem.q_proj(image_tokens).view(B, patches, sem.num_heads, sem.head_dim).transpose(1, 2)
                k = sem.k_proj(text_tokens).view(B, length, sem.num_heads, sem.head_dim).transpose(1, 2)
                logits = torch.matmul(q, k.transpose(-2, -1)) * sem.scale
                roles = role_masks.to(logits.dtype)
                logits = logits + torch.einsum('brl,r->bl', roles, sem.role_bias)[:, None, None, :]
                om, am = roles[:, 0], roles[:, 1]
                orep = sem.object_pair_proj(text_tokens)
                arep = sem.attribute_pair_proj(text_tokens)
                pair = torch.sigmoid(torch.matmul(orep, arep.transpose(1, 2)) * sem.scale)
                pair = pair * om[:, :, None] * am[:, None, :]
                obj_logits = logits.masked_fill(om[:, None, None, :] <= 0,
                                                torch.finfo(logits.dtype).min)
                obj_attn = torch.softmax(obj_logits, dim=-1).mean(dim=1)
                bias = torch.bmm(obj_attn, pair)
                captured.setdefault('bias', {})[li] = (
                    F.softplus(sem.pair_strength) * bias[:, None, :, :]).detach()
                captured.setdefault('objattn', {})[li] = obj_attn.detach()
                captured.setdefault('pair', {})[li] = pair.detach()
            return orig_forward(image_tokens, text_tokens, role_masks,
                                text_valid_mask, return_role_outputs)

        sem.forward = patched

    try:
        for t in timesteps:
            captured.setdefault('meta', {})[t] = None
            x = torch.randn(1, 4, 64, 64, device='cuda', dtype=torch.float16,
                            generator=torch.Generator(device='cuda').manual_seed(43))
            x2 = torch.cat([x, x], dim=0)
            # Solver batches as [unconditional, conditional] (see
            # diffusion/model/dpm_solver.py: c_in = cat([unconditional, condition])).
            y2 = torch.cat([null_cond, emb_cond], dim=0)
            hw = torch.full((2, 2), 512, device='cuda', dtype=torch.float16)
            ar = torch.ones((2, 1), device='cuda', dtype=torch.float16)
            di = {'img_hw': hw, 'aspect_ratio': ar}
            if sem_mask is not None:
                # batch-1 masks -> the model expands to [zeros, masks], which is
                # exactly the real conditional/unconditional split.
                di['semantic_token_masks'] = sem_mask
            ts = torch.full((2,), float(t), device='cuda', dtype=torch.float16)
            with torch.no_grad():
                model.forward(x2, ts, y2, mask=mask_cond, data_info=di)
            captured['meta'][t] = True
            counter['i'] = 0
    finally:
        for h in handles:
            h.remove()
        if capture_bias:
            sem.forward = orig_forward
    return captured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timesteps', default='100,250,400,550,700,850')
    ap.add_argument('--layers', default='0,6,13,20,27')
    ap.add_argument('--t5-cache', default=str(ROOT / 'output/pretrained_models/t5_ckpts'))
    args = ap.parse_args()
    timesteps = [int(x) for x in args.timesteps.split(',')]
    layers = [int(x) for x in args.layers.split(',')]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.t5_cache + '/t5-v1_1-xxl',
                                              use_fast=True, local_files_only=True)
    import spacy
    nlp = spacy.load('en_core_web_sm')
    # T5-XXL is huge; keep it on CPU so this analysis can run while a
    # generation job occupies the GPU.
    t5 = T5Embedder(device='cpu', local_cache=True, cache_dir=args.t5_cache,
                    torch_dtype=torch.float, model_max_length=120)

    index = {}
    for case in CASES:
        p = case['prompt']
        emb, mask = t5.get_text_embeddings([p])
        emb = emb[:, None].cuda().half()
        mask = mask.cuda()
        null_cond = None  # filled after model build (needs y_embedder)
        sem = torch.stack([torch.from_numpy(build_semantic_masks(p, nlp, tokenizer, emb.shape[2]))]
                          ).cuda().half()
        oi, _ = find_token_index(tokenizer, p, case['object_token'])
        ai, _ = find_token_index(tokenizer, p, case['attribute_token'])
        case['obj_idx'], case['attr_idx'] = oi, ai
        print(f"{p!r}: object {case['object_token']!r} -> {oi}, "
              f"attribute {case['attribute_token']!r} -> {ai}", flush=True)

        for tag in ('frozen', 'tpscda'):
            model = build(tag)
            cap = run_probe(model, emb, mask, sem, timesteps, layers,
                            capture_bias=(tag == 'tpscda'))
            index[f'{tag}|{p}'] = dict(
                object_token_index=oi, attribute_token_index=ai,
                layers=layers, timesteps=timesteps)
            draw(cap, case, tag, timesteps, layers)
            del model
            torch.cuda.empty_cache()

    (OUT / 'mechanism_index.json').write_text(json.dumps(
        dict(cases=[{k: v for k, v in c.items()} for c in CASES],
             note=('Patch response heatmaps. "cross" is the frozen PixArt '
                   'cross-attention present in both models; "bias" is the '
                   'structured object->attribute pair bias B that only TP-SCDA '
                   'adds. All maps are for a fixed probe latent (seed 43).'),
             index=index), indent=2, ensure_ascii=False))
    print('wrote', OUT)


def draw(cap, case, tag, timesteps, layers):
    prompt = case['prompt']
    aidx = case['attr_idx'][0] if case['attr_idx'] else 0
    safe = prompt[:60].replace(' ', '_').replace('/', '_')
    cross = cap.get('cross', {})
    if not cross:
        return
    fig, axes = plt.subplots(len(layers), len(timesteps),
                             figsize=(2.0 * len(timesteps), 2.0 * len(layers)))
    for r, li in enumerate(layers):
        for c, t in enumerate(timesteps):
            ax = axes[r, c]
            w = cross.get(li)
            if w is None:
                ax.axis('off'); continue
            v = w[1].mean(0)[:, aidx].float().cpu().numpy()
            g = int(round(v.shape[0] ** 0.5))
            m = v.reshape(g, g)
            im = ax.imshow(m, cmap='inferno')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f't={t}', fontsize=9)
            if c == 0:
                ax.set_ylabel(f'层{li}', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
    name = 'TP-SCDA' if tag == 'tpscda' else '冻结 PixArt'
    fig.suptitle(f'{name}：图像 Patch 对属性 token「{case["attribute_token"]}」的交叉注意力响应\n'
                 f'提示词：{prompt}（固定探针 latent，seed 43）', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f'cross_attn_{tag}_{safe}.png', dpi=170)
    plt.close(fig)

    if tag == 'tpscda' and cap.get('bias'):
        fig, axes = plt.subplots(len(layers), len(timesteps),
                                 figsize=(2.0 * len(timesteps), 2.0 * len(layers)))
        for r, li in enumerate(layers):
            for c, t in enumerate(timesteps):
                ax = axes[r, c]
                b = cap['bias'].get(li)
                if b is None:
                    ax.axis('off'); continue
                v = b[1, 0][:, aidx].float().cpu().numpy()
                g = int(round(v.shape[0] ** 0.5))
                m = v.reshape(g, g)
                im = ax.imshow(m, cmap='viridis')
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f't={t}', fontsize=9)
                if c == 0:
                    ax.set_ylabel(f'层{li}', fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f'TP-SCDA：结构化对象→属性偏置 B 对属性 token「{case["attribute_token"]}」的分量\n'
                     f'提示词：{prompt}（固定探针 latent，seed 43）', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / f'pair_bias_tpscda_{safe}.png', dpi=170)
        plt.close(fig)


if __name__ == '__main__':
    main()
