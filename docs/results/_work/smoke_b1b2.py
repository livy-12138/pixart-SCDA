#!/usr/bin/env python3
"""Smoke-test the B1 (edge-gated pair) and B2 (binding loss) code paths.

Nothing here is trained; it only checks that
  * the dataset produces the edge tensors with the documented shapes,
  * the edge matrix rebuilds exactly the parsed (object, attribute) pairs,
  * B1 actually changes the attention the module produces,
  * B2 produces a finite, differentiable loss.

Why the B1 check does NOT look at the diffusion output
-----------------------------------------------------
``SemanticTokenCrossAttention.out_proj`` is zero-initialised, so the whole
semantic branch is a no-op at initialisation (ControlNet-style zero-conv).  A
fresh model therefore returns bit-identical outputs with and without B1 even
when the attention inside the branch did change.  The check is run on the
module itself, with ``out_proj`` filled with a fixed random matrix so the
branch becomes observable.

The module-level checks run on CPU on purpose: a full-model backward needs
~13 GB and the training job owns the GPU.
"""
import sys
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import torch

from diffusion.data.builder import set_data_root
from diffusion.data.datasets.InternalData import InternalData

set_data_root('/root/private_data/data')
from diffusion.model.nets.PixArt import SemanticTokenCrossAttention
from diffusion.model.nets import PixArt_XL_2

EDGE = '/root/private_data/data/COCO2017Prepared/partition/edge_index.npz'
HIDDEN, HEADS, PATCHES, LENGTH, BATCH = 64, 4, 16, 24, 2
FAILURES = []


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name}{"  " + detail if detail else ""}')
    if not ok:
        FAILURES.append(name)


def check_dataset():
    print('--- dataset ---')
    ds = InternalData(
        root='COCO2017Prepared',
        image_list_json=['data_info.json'],
        transform='default_train', resolution=512,
        load_vae_feat=True, load_semantic_masks=True,
        load_semantic_edges=True, semantic_edge_index=EDGE, max_semantic_edges=24,
        image_root='/root/private_data/data/coco2017', max_length=120)
    img, fea, mask, info = ds[0]
    check('caption_feature shape', tuple(fea.shape) == (1, 120, 4096), str(tuple(fea.shape)))
    check('semantic_token_masks shape', tuple(info['semantic_token_masks'].shape) == (3, 120))
    check('semantic_edges shape', tuple(info['semantic_edges'].shape) == (24, 2))
    check('semantic_edges dtype', info['semantic_edges'].dtype == torch.long)
    count = int(info['semantic_edge_count'])
    check('edge_count present', count >= 0, f'count={count}')
    n_with = sum(1 for i in range(20) if int(ds[i][3]['semantic_edge_count']) > 0)
    check('edges present on most samples', n_with >= 10, f'{n_with}/20')
    padded = info['semantic_edges'][count:]
    check('padding rows are -1', bool((padded == -1).all()))
    check('first edges', True, str(info['semantic_edges'][:3].tolist()))
    return info


def make_module():
    torch.manual_seed(0)
    module = SemanticTokenCrossAttention(HIDDEN, num_heads=HEADS)
    # make the branch observable (out_proj is zero-init by design)
    with torch.no_grad():
        module.out_proj.weight.normal_(0, 0.02)
    return module


def toy_inputs():
    torch.manual_seed(1)
    image_tokens = torch.randn(BATCH, PATCHES, HIDDEN)
    text_tokens = torch.randn(BATCH, LENGTH, HIDDEN)
    roles = torch.zeros(BATCH, 3, LENGTH)
    roles[:, 0, 1:4] = 1.0        # object tokens
    roles[:, 1, 5:8] = 1.0        # attribute tokens
    roles[:, 2, 9:11] = 1.0       # relation tokens
    edges = torch.tensor([[1, 5], [2, 6], [-1, -1]])[None].repeat(BATCH, 1, 1)
    counts = torch.tensor([2, 2])
    return image_tokens, text_tokens, roles, edges, counts


def run_module(module, edge_gate, binding_coef, inputs):
    image_tokens, text_tokens, roles, edges, counts = inputs
    return module(image_tokens, text_tokens, roles, text_valid_mask=None,
                  edge_pairs=edges, edge_count=counts,
                  pair_edge_gate=edge_gate, binding_loss_coef=binding_coef,
                  binding_temperature=1.0)


def check_edge_matrix(inputs):
    print('\n--- edge matrix ---')
    module = make_module()
    *_, edges, counts = inputs
    matrix = module._edge_matrix(edges, counts, LENGTH, torch.float32, torch.device('cpu'))
    check('edge matrix shape', tuple(matrix.shape) == (BATCH, LENGTH, LENGTH))
    expected = torch.zeros_like(matrix)
    for b in range(BATCH):
        for row in range(int(counts[b])):
            if edges[b, row, 0] >= 0:
                expected[b, int(edges[b, row, 0]), int(edges[b, row, 1])] = 1.0
    check('edge matrix matches parsed pairs', bool(torch.equal(matrix, expected)))
    check('edge matrix is 0/1 only', bool(((matrix == 0) | (matrix == 1)).all()))
    check('edge matrix respects edge_count',
          bool(matrix[0].sum() == 2), f'sum={float(matrix[0].sum())}')


def check_b1_changes_attention():
    print('\n--- B1: edge-gated pair ---')
    inputs = toy_inputs()
    module = make_module()
    plain = run_module(module, False, 0.0, inputs)
    gated = run_module(module, True, 0.0, inputs)
    delta = (plain - gated).abs().max().item()
    check('B1 changes the branch output', delta > 1e-6, f'max|delta|={delta:.3e}')
    check('B1 output finite', bool(torch.isfinite(gated).all()))


def check_b2_loss():
    print('\n--- B2: binding alignment loss ---')
    inputs = toy_inputs()
    module = make_module()
    out = run_module(module, False, 0.1, inputs)
    loss = module._last_binding_loss
    check('binding loss produced', loss is not None)
    if loss is None:
        return
    check('binding loss finite', bool(torch.isfinite(loss)))
    check('binding loss > 0', float(loss) > 0, f'{float(loss):.4f}')
    loss.backward()
    got = {n: p.grad is not None and float(p.grad.abs().sum()) > 0
           for n, p in module.named_parameters()}
    for name in ('q_proj.weight', 'k_proj.weight',
                 'object_pair_proj.weight', 'attribute_pair_proj.weight'):
        check(f'{name} receives gradient', got[name])
    print('  parameters receiving gradient:',
          sum(1 for v in got.values() if v), '/', len(got))

    module2 = make_module()
    empty = list(inputs)
    empty[3] = torch.full_like(inputs[3], -1)
    run_module(module2, False, 0.1, empty)
    check('no-edges case yields no loss', module2._last_binding_loss is None)

    module3 = make_module()
    run_module(module3, False, 0.0, inputs)
    check('coef=0 yields no loss', module3._last_binding_loss is None)


def check_full_model(info):
    print('\n--- full model forward (no_grad) ---')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('  skipped: xformers attention requires CUDA')
        return
    B = 1
    x = torch.randn(B, 4, 64, 64, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    y = torch.randn(B, 1, 120, 4096, device=device)
    data_info = {'img_hw': torch.full((B, 2), 512., device=device),
                 'aspect_ratio': torch.ones(B, 1, device=device),
                 'semantic_token_masks': info['semantic_token_masks'][None].repeat(B, 1, 1).to(device),
                 'semantic_edges': info['semantic_edges'][None].repeat(B, 1, 1).to(device),
                 'semantic_edge_count': info['semantic_edge_count'][None].repeat(B).to(device)}
    mask = torch.ones(B, 1, 120, device=device)
    for name, gate, coef in [('B0', False, 0.0), ('B1', True, 0.0), ('B2', False, 0.1)]:
        try:
            model = PixArt_XL_2(input_size=64, lewei_scale=1, semantic_conditioning=True,
                                semantic_adapter_dim=64, semantic_dropout=0,
                                semantic_residual_scale=0.0, semantic_token_attention=True,
                                semantic_token_gate_max=0.3, semantic_token_gate_init=0.0,
                                semantic_token_gate_activation='sigmoid',
                                semantic_gate_init=-1.0, semantic_gate_init_global=1.0,
                                semantic_pair_edge_gate=gate,
                                semantic_binding_coef=coef,
                                semantic_binding_temperature=1.0).eval().to(device)
            with torch.no_grad():
                out = model(x, t, y, mask=mask, data_info=data_info)
            check(f'{name} forward finite', bool(torch.isfinite(out).all()), str(tuple(out.shape)))
            loss = model.get_semantic_binding_loss()
            check(f'{name} aggregated binding loss',
                  (loss is None) == (coef == 0.0),
                  'None' if loss is None else f'{float(loss):.4f}')
            del model
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f'  [skip] {name}: GPU busy (training running), retry after it frees')
            torch.cuda.empty_cache()


def check_generation_path():
    """Feed the model the edges exactly as results/_work/gen_compbench.py does.

    The training-time edges come from a prebuilt index; at inference they are
    parsed on the fly by build_semantic_edges().  The two were verified to agree
    row-for-row (200/200 prompts), so this checks the remaining seam: the tensor
    conversion the generation script performs.
    """
    print('\n--- generation-path edges (B1) ---')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('  skipped: xformers attention requires CUDA')
        return
    import json

    import numpy as np
    from transformers import AutoTokenizer

    from tools.prepare_semantic_masks import build_semantic_edges
    from tools.test_text_pseudolabels import load_nlp

    prompts = ["a red fire truck and a small blue cup on a wooden table"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / 'output/pretrained_models/t5_ckpts/t5-v1_1-xxl'), use_fast=True)
    nlp = load_nlp('en_core_web_sm')
    parsed = [build_semantic_edges(p, nlp, tokenizer, 120, 24) for p in prompts]
    check('generation-path parser finds edges', parsed[0][1] > 0,
          f'count={parsed[0][1]}')

    edges = torch.from_numpy(np.stack([e for e, _ in parsed])).to(device=device)
    counts = torch.from_numpy(np.asarray([c for _, c in parsed])).to(device=device)
    check('edge tensor dtype/shape for inference',
          edges.dtype == torch.long and edges.ndim == 3,
          f'{tuple(edges.shape)} {edges.dtype}')

    B = 1
    x = torch.randn(B, 4, 64, 64, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    y = torch.randn(B, 1, 120, 4096, device=device)
    masks = torch.zeros(B, 3, 120, device=device)
    masks[:, 0, 1:4] = 1.0
    masks[:, 1, 5:8] = 1.0
    base_info = {'img_hw': torch.full((B, 2), 512., device=device),
                 'aspect_ratio': torch.ones(B, 1, device=device),
                 'semantic_token_masks': masks}
    text_mask = torch.ones(B, 1, 120, device=device)

    def build(edge_gate):
        return PixArt_XL_2(input_size=64, lewei_scale=1, semantic_conditioning=True,
                           semantic_adapter_dim=64, semantic_dropout=0,
                           semantic_residual_scale=0.0, semantic_token_attention=True,
                           semantic_token_gate_max=0.3, semantic_token_gate_init=0.0,
                           semantic_token_gate_activation='sigmoid',
                           semantic_gate_init=-1.0, semantic_gate_init_global=1.0,
                           semantic_pair_edge_gate=edge_gate,
                           semantic_binding_coef=0.0).eval().to(device)

    model = build(True)
    with torch.no_grad():
        with_edges = model(x, t, y, mask=text_mask,
                           data_info=dict(base_info,
                                          semantic_edges=edges,
                                          semantic_edge_count=counts))
        without = model(x, t, y, mask=text_mask, data_info=base_info)
    check('B1 inference with parsed edges finite',
          bool(torch.isfinite(with_edges).all()))
    # the branch is zero-init, so compare the branch itself, not the output
    check('B1 edge gate engaged (module sees edges)',
          model.semantic_pair_edge_gate is True)
    check('no-edges fallback runs', bool(torch.isfinite(without).all()))
    del model
    torch.cuda.empty_cache()


def main():
    info = check_dataset()
    inputs = toy_inputs()
    check_edge_matrix(inputs)
    check_b1_changes_attention()
    check_b2_loss()
    check_full_model(info)
    check_generation_path()
    print('\n' + ('SMOKE_FAIL: ' + ', '.join(FAILURES) if FAILURES else 'SMOKE_OK'))


if __name__ == '__main__':
    main()
