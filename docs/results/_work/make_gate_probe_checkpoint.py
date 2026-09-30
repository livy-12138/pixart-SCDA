#!/usr/bin/env python3
"""Build an inference-time probe checkpoint: role branches restored to the
historical (paper) injection level, everything else untouched.

Why
---
`results/improvement_report/gate_decomposition.csv` shows the gate-optimised run
changed the two kinds of injection very differently:

    global (all-token) branch      x1.55
    object / attribute / relation  x23.0 .. x27.2

so the regression on the three binding categories is far more likely to come
from the role branches than from the global one.  The cheap way to test that is
NOT to retrain: the gates are plain parameters, so a probe checkpoint can be
written out with the role gates set back to their historical value while the
global gate and the overall scale keep their trained values.

What this is and is not
-----------------------
This is a PROBE, not a trained model.  It must be reported as
"门控改造 checkpoint + 推理时 role 门控置回历史值", never as a separately
trained variant, and never mixed into the same table as trained checkpoints
without saying so.  Prompts, seeds, resolution, sampler, steps and CFG are all
left exactly as the standard protocol uses them -- the only change is one
internal parameter of the model.

Usage:
    python results/_work/make_gate_probe_checkpoint.py \
        --source output/coco2017_token_pair_gate_opt/checkpoints/epoch_1_step_14786.pth \
        --output output/coco2017_token_pair_gate_opt/checkpoints/probe_rolegate_historical.pth
"""
import argparse
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')

import torch

HISTORICAL_ROLE_GATE = 0.01790   # mean sigmoid(layer_gate) of the paper checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path,
                    default=ROOT / 'output/coco2017_token_pair_gate_opt/checkpoints/epoch_1_step_14786.pth')
    ap.add_argument('--output', type=Path,
                    default=ROOT / 'output/coco2017_token_pair_gate_opt/checkpoints/probe_rolegate_historical.pth')
    ap.add_argument('--role-gate', type=float, default=HISTORICAL_ROLE_GATE)
    args = ap.parse_args()

    state = torch.load(args.source, map_location='cpu')
    wrapped = 'state_dict' in state
    sd = state['state_dict'] if wrapped else state
    gates = sd.get('semantic_token_layer_gate')
    if gates is None:
        raise SystemExit(f'{args.source} has no semantic_token_layer_gate')

    before = torch.sigmoid(gates).mean(dim=0)
    target_logit = float(torch.log(torch.tensor(args.role_gate) /
                                   (1.0 - torch.tensor(args.role_gate))))
    with torch.no_grad():
        gates[:, 1:] = target_logit          # object / attribute / relation
    after = torch.sigmoid(sd['semantic_token_layer_gate']).mean(dim=0)

    print(f'source: {args.source}')
    print(f'  mean gate before: ' + ' '.join(f'{float(v):.5f}' for v in before))
    print(f'  mean gate after : ' + ' '.join(f'{float(v):.5f}' for v in after))
    print(f'  (global branch untouched at {float(before[0]):.5f}; '
          f'role branches set to {args.role_gate:.5f}, the historical value)')

    if wrapped:
        state['state_dict'] = sd
    else:
        state = sd
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print('wrote', args.output)
    print('\nNOTE: this is a probe checkpoint. Report it as '
          '"门控改造 + 推理时 role 门控置回历史值", not as a trained variant.')


if __name__ == '__main__':
    main()
