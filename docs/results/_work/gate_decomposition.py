#!/usr/bin/env python3
"""Decompose each checkpoint's semantic injection into global vs role parts.

The forward pass computes

    x = x + semantic_gate_scale() * ( g_global * delta_all
                                    + sum_r g_role_r * delta_role_r )

so the quantity that actually reaches the residual stream is the PRODUCT of the
global scale and each layer gate -- not the layer gate on its own.  Comparing
raw gate values between two checkpoints is misleading: it hid the fact that the
role gates are the ones that moved most.
"""
import sys
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import torch

# gate_max and the activation are CONFIG, not weights -- a checkpoint carries no
# record of them, so they must be supplied per run or the effective strength is
# computed with the wrong formula (this bug made an earlier version report 0.166
# for the paper checkpoint whose audited value is 0.0164).
CKPTS = {
    'tpscda (paper)': (
        ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth',
        0.08, 'tanh'),
    'gateopt': (
        ROOT / 'output/coco2017_token_pair_gate_opt/checkpoints/epoch_1_step_14786.pth',
        0.30, 'sigmoid'),
    'B1 (running)': (
        ROOT / 'output/coco2017_token_pair_b1_edgegate/checkpoints/epoch_1_step_14786.pth',
        0.30, 'sigmoid'),
    'B2 (queued)': (
        ROOT / 'output/coco2017_token_pair_b2_bindingloss/checkpoints/epoch_1_step_14786.pth',
        0.30, 'sigmoid'),
}
ROLES = ('global', 'object', 'attribute', 'relation')


def main():
    print(f"{'checkpoint':<16} {'scale':>9} " +
          ' '.join(f'g_{r:>9}' for r in ROLES) + '   ' +
          ' '.join(f'inj_{r:>7}' for r in ROLES))
    results = {}
    for name, (path, gate_max, activation) in CKPTS.items():
        if not path.exists():
            print(f'{name:<16} (not found yet)')
            continue
        state = torch.load(path, map_location='cpu')
        state = state.get('state_dict', state)
        raw = state.get('semantic_token_gate')
        lay = state.get('semantic_token_layer_gate')
        if raw is None or lay is None:
            print(f'{name:<16} (no token gate)')
            continue
        shaped = torch.tanh(raw) if activation == 'tanh' else torch.sigmoid(raw)
        scale = float(gate_max * shaped)
        gates = torch.sigmoid(lay).mean(dim=0)          # (4,)
        inj = gates * scale
        results[name] = (scale, gates, inj)
        print(f'{name:<16} {scale:>9.5f} ' +
              ' '.join(f'{float(g):>10.5f}' for g in gates) + '   ' +
              ' '.join(f'{float(v):>10.5f}' for v in inj))

    if 'tpscda (paper)' in results and 'gateopt' in results:
        _, _, a = results['tpscda (paper)']
        _, _, b = results['gateopt']
        print('\nrelative change gateopt / paper checkpoint:')
        for i, role in enumerate(ROLES):
            r = float(b[i] / a[i].clamp_min(1e-12))
            print(f'  {role:<10} injection x{r:.2f}')

    out = ROOT / 'results/improvement_report/gate_decomposition.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as fh:
        fh.write('checkpoint,scale,' + ','.join(f'gate_{r}' for r in ROLES) + ','
                 + ','.join(f'injection_{r}' for r in ROLES) + '\n')
        for name, (scale, gates, inj) in results.items():
            fh.write(f'{name},{scale:.6f},'
                     + ','.join(f'{float(g):.6f}' for g in gates) + ','
                     + ','.join(f'{float(v):.6f}' for v in inj) + '\n')
    print('\nwrote', out)


if __name__ == '__main__':
    main()
