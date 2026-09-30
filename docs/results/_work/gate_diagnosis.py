#!/usr/bin/env python3
"""Why did the learnable layer gate not move?  Measure its gradient directly.

Runs a small number of realistic training steps through the TP-SCDA model
(diffusion MSE + the same semantic L2 regulariser used in training) and reports
the gradient norm of every trainable parameter group, plus how the gate
gradient scales with the gate's own value.

Nothing is trained or written; this is a read-only diagnostic.
"""
import sys
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from diffusion.model.nets import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from tools.prepare_semantic_masks import build_semantic_masks

CKPT = ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth'
TRAINABLE = ('semantic_adapters.', 'semantic_layer_scale', 'semantic_time_gate.',
             'semantic_token_cross_attention.', 'semantic_token_gate',
             'semantic_token_layer_gate')

PROMPTS = [
    'a green bench and a blue bowl',
    'a red car and a gold clock',
    'rubber gloves and a fluffy teddy bear',
    'A dog is walking on a leash with its owner.',
    'a diamond tiara and a round bracelet',
    'three cats sitting on a wooden table',
    'a plastic toy and a glass bottle',
    'a blue bench and a green bowl',
]


def main():
    dev, dt = 'cuda', torch.float32          # fp32 so the gradient numbers are clean
    torch.manual_seed(43)

    model = PixArt_XL_2(input_size=64, lewei_scale=1, semantic_conditioning=True,
                        semantic_adapter_dim=64, semantic_dropout=0,
                        semantic_residual_scale=0.0, semantic_token_attention=True,
                        semantic_token_gate_max=0.08).to(dev, dtype=dt)
    st = torch.load(CKPT, map_location='cpu')
    st = st.get('state_dict', st); st.pop('pos_embed', None)
    missing, unexpected = model.load_state_dict(st, strict=False)
    print(f'checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}')

    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(TRAINABLE)
    model.train()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'output/pretrained_models/t5_ckpts/t5-v1_1-xxl'),
        use_fast=True, local_files_only=True)
    t5 = T5Embedder(device='cpu', local_cache=True,
                    cache_dir=str(ROOT / 'output/pretrained_models/t5_ckpts'),
                    torch_dtype=torch.float, model_max_length=120)
    import spacy
    nlp = spacy.load('en_core_web_sm')

    with torch.no_grad():
        emb, tmask = t5.get_text_embeddings(PROMPTS)
        emb = emb[:, None].to(dev, dtype=dt)
        tmask = tmask.to(dev)
        sem = torch.stack([torch.from_numpy(build_semantic_masks(p, nlp, tok, emb.shape[2]))
                           for p in PROMPTS]).to(dev, dtype=dt)

    B = len(PROMPTS)
    x0 = torch.randn(B, 4, 64, 64, device=dev, dtype=dt)
    eps = torch.randn(B, 4, 64, 64, device=dev, dtype=dt)
    t = torch.randint(0, 1000, (B,), device=dev)
    ab = torch.cos((t.float() / 1000.0 + 0.008) / 1.008 * torch.pi / 2) ** 2   # cosine schedule proxy
    xt = ab.sqrt().view(B, 1, 1, 1) * x0 + (1 - ab).sqrt().view(B, 1, 1, 1) * eps
    hw = torch.full((B, 2), 512, device=dev, dtype=dt)
    ar = torch.ones((B, 1), device=dev, dtype=dt)

    out = model(xt, t, emb, mask=tmask,
                data_info={'img_hw': hw, 'aspect_ratio': ar, 'semantic_token_masks': sem})
    pred = out.chunk(2, dim=1)[0]
    diff_loss = F.mse_loss(pred.float(), eps.float())
    reg = model._last_semantic_regularization
    total = diff_loss + 0.02 * reg
    model.zero_grad(set_to_none=True)
    total.backward()

    print(f'\ndiffusion loss = {diff_loss.item():.4f}   '
          f'semantic L2 reg = {float(reg):.4e}   (coef 0.02)')

    groups = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        key = n.split('.')[0] if not n.startswith('semantic_token_cross_attention') \
            else 'semantic_token_cross_attention'
        if n.startswith('semantic_token_layer_gate'):
            key = 'semantic_token_layer_gate'
        elif n.startswith('semantic_token_gate'):
            key = 'semantic_token_gate'
        groups.setdefault(key, []).append((n, p.grad.norm().item(), p.numel()))

    print(f"\n{'parameter group':38s} {'grad norm':>12s} {'params':>10s} {'grad/param':>12s}")
    for k in sorted(groups, key=lambda k: -sum(g[1] for g in groups[k])):
        gn = sum(g[1] for g in groups[k]); np_ = sum(g[2] for g in groups[k])
        print(f'{k:38s} {gn:12.4e} {np_:10,d} {gn/np_:12.4e}')

    # how far did the gate actually move during the whole run?
    lg = st['semantic_token_layer_gate'].float()
    init = torch.full_like(lg, -4.0); init[:, 0] = 4.0
    g, gi = torch.sigmoid(lg), torch.sigmoid(init)
    print(f'\nlayer gate: |w - w_init| max = {(lg - init).abs().max():.3e}, '
          f'gate |sigmoid(w) - sigmoid(w_init)| max = {(g - gi).abs().max():.3e}')
    print(f'scalar token gate effective value = '
          f'{0.08 * float(torch.tanh(st["semantic_token_gate"])):.5f}')

    # gradient vs gate value: d/dw sigmoid(w) = sigmoid(w)(1-sigmoid(w))
    import math
    for w in (-4.0, -2.0, 0.0, 2.0, 4.0):
        s = 1 / (1 + math.exp(-w))
        print(f'  w={w:+5.1f}  lambda={s:.4f}  dlambda/dw={s*(1-s):.4f}')


if __name__ == '__main__':
    main()
