#!/usr/bin/env python3
"""Read the gate-drift DIRECTION of every run at matched steps.

The direction of the drift is the early readout for whether the semantic branch
is becoming useful: in every run so far the optimiser drives the gate DOWN
(0.150 -> 0.063 without a binding objective), i.e. it decides the branch is not
worth its cost.  If a binding objective makes the branch useful, the gate should
drift UP instead.  That is visible in training_metrics.csv long before the two
hours of evaluation finish.

Usage:
    python results/_work/gate_drift_compare.py [--steps 2500,4000,6000,8000]
"""
import argparse
import csv
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
RUNS = {
    'gateopt   (no B2)': ROOT / 'output/coco2017_token_pair_gate_opt/experiment_tables/training_metrics.csv',
    'B1        (edge gate)': ROOT / 'output/coco2017_token_pair_b1_edgegate/experiment_tables/training_metrics.csv',
    'B2        (bind, eff0.15)': ROOT / 'output/coco2017_token_pair_b2_bindingloss/experiment_tables/training_metrics.csv',
    'B2neutral (bind, eff0.016)': ROOT / 'output/coco2017_token_pair_b2_neutralgate/experiment_tables/training_metrics.csv',
}
FIELDS = ['semantic_gate_effective', 'semantic_gate_drift_max',
          'semantic_gate_layer_std_max', 'semantic_binding_loss']


def series(path):
    if not path.exists():
        return None
    rows = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                rows[int(float(row['global_step']))] = row
            except (KeyError, ValueError):
                continue
    return rows or None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', default='2500,4000,6000,8000,10000,12000')
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(',') if s.strip()]

    data = {name: series(path) for name, path in RUNS.items()}

    print(f"{'run':<28}{'step':>7}{'eff':>10}{'drift_max':>11}{'layer_std':>11}{'bind_loss':>11}")
    print('-' * 78)
    for step in steps:
        for name, rows in data.items():
            if not rows:
                continue
            row = rows.get(step)
            if row is None:
                continue
            vals = []
            for field in FIELDS:
                try:
                    vals.append(f"{float(row[field]):>10.5f}")
                except (KeyError, ValueError):
                    vals.append(f"{'n/a':>10}")
            print(f"{name:<28}{step:>7}" + ''.join(vals))
        print('-' * 78)

    print('\nInterpretation: with no binding objective the gate drifts DOWN')
    print('(0.150 -> 0.063).  A binding objective that makes the branch worth')
    print('opening would show up as an UPWARD or at least flat trajectory.')


if __name__ == '__main__':
    main()
