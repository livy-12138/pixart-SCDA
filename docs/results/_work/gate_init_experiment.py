#!/usr/bin/env python3
"""Does the gate's *initialisation* explain why it never learns?

Holds everything else fixed (same checkpoint, same synthetic batch, same
Adam/lr/steps) and only changes where ``semantic_token_layer_gate`` starts:

  A. current code      w = -4.0 / +4.0   -> lambda = 0.018 / 0.982  (sigmoid saturated)
  B. warmed-up init    w = -1.0 / +1.0   -> lambda = 0.269 / 0.731
  C. current init, semantic L2 regulariser switched off

Reports how far lambda actually travels.  This is a controlled probe, not a
substitute for retraining -- it isolates one variable.
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
STEPS = 150
LR = 3.536e-6          # the effective LR of the real run
REG_COEF = 0.02
PROMPTS = ['a green bench and a blue bowl', 'a red car and a gold clock',
           'rubber gloves and a fluffy teddy bear', 'a plastic toy and a glass bottle']


def run(init_w, reg_coef, label):
    dev, dt = 'cuda', torch.float32
    torch.manual_seed(43)
    model = PixArt_XL_2(input_size=64, lewei_scale=1, semantic_conditioning=True,
                        semantic_adapter_dim=64, semantic_dropout=0,
                        semantic_residual_scale=0.0, semantic_token_attention=True,
                        semantic_token_gate_max=0.08).to(dev, dtype=dt)
    st = torch.load(CKPT, map_location='cpu')
    st = st.get('state_dict', st); st.pop('pos_embed', None)
    model.load_state_dict(st, strict=False)

    for n, p in model.named_parameters():
        p.requires_grad = n.startswith('semantic_token_layer_gate')
    model.train()

    init = torch.full((28, 4), float(init_w), device=dev)
    init[:, 0] = -float(init_w)
    model.semantic_token_layer_gate.data.copy_(init)
    lam0 = torch.sigmoid(init).detach().clone()

    opt = torch.optim.AdamW([model.semantic_token_layer_gate], lr=LR,
                            weight_decay=0.01, eps=1e-10)

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
        emb = emb[:, None].to(dev, dtype=dt); tmask = tmask.to(dev)
        sem = torch.stack([torch.from_numpy(build_semantic_masks(p, nlp, tok, emb.shape[2]))
                           for p in PROMPTS]).to(dev, dtype=dt)

    B = len(PROMPTS)
    g = torch.Generator(device=dev).manual_seed(7)
    x0 = torch.randn(B, 4, 64, 64, device=dev, dtype=dt, generator=g)
    eps = torch.randn(B, 4, 64, 64, device=dev, dtype=dt, generator=g)
    t = torch.randint(0, 1000, (B,), device=dev, generator=g)
    ab = torch.cos((t.float() / 1000.0 + 0.008) / 1.008 * torch.pi / 2) ** 2
    xt = ab.sqrt().view(B, 1, 1, 1) * x0 + (1 - ab).sqrt().view(B, 1, 1, 1) * eps
    hw = torch.full((B, 2), 512, device=dev, dtype=dt)
    ar = torch.ones((B, 1), device=dev, dtype=dt)

    for step in range(STEPS):
        out = model(xt, t, emb, mask=tmask,
                    data_info={'img_hw': hw, 'aspect_ratio': ar, 'semantic_token_masks': sem})
        pred = out.chunk(2, dim=1)[0]
        loss = F.mse_loss(pred.float(), eps.float()) + reg_coef * model._last_semantic_regularization
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    lam1 = torch.sigmoid(model.semantic_token_layer_gate.detach()).clone()
    d = (lam1 - lam0)
    print(f'{label:34s} |dlambda| max={d.abs().max():.3e}  mean={d.abs().mean():.3e}  '
          f'|dw| max={(model.semantic_token_layer_gate.detach()-init).abs().max():.3e}')
    del model
    torch.cuda.empty_cache()
    return d.abs().max().item()


if __name__ == '__main__':
    print(f'{STEPS} AdamW steps, lr={LR}, batch={len(PROMPTS)}, fp32\n')
    a = run(-4.0, REG_COEF, 'A current init (w=+-4)')
    b = run(-1.0, REG_COEF, 'B warmed init (w=+-1)')
    c = run(-4.0, 0.0, 'C current init, reg=0')
    print(f'\ngate travel: B/A = {b/a:.1f}x   C/A = {c/a:.1f}x')
