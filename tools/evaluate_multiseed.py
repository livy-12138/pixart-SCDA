#!/usr/bin/env python3
"""Generate and score a fixed prompt set across checkpoints and seeds."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prompts', default='asset/samples.txt')
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='label=checkpoint[:adapter_dim], baseline uses dim 0')
    p.add_argument('--seeds', nargs='+', type=int, default=[43, 44, 45])
    p.add_argument('--output-root', default='output/multiseed_eval')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--cfg-scale', type=float, default=4.0)
    p.add_argument('--semantic-residual-scale', type=float, default=1.0,
                   help='Residual scale passed to semantic SCDA inference.')
    p.add_argument('--semantic-token-attention', action='store_true')
    p.add_argument('--semantic-token-gate-max', type=float, default=1.0)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    prompt_lines = [x.strip() for x in (root / args.prompts).read_text().splitlines() if x.strip()]
    group_dir = root / args.output_root / 'prompt_groups_64'
    group_dir.mkdir(parents=True, exist_ok=True)
    # Eight prompts per group keeps inference memory bounded and preserves names.
    for old in group_dir.glob('group_*.txt'):
        old.unlink()
    for idx in range(0, len(prompt_lines), 8):
        (group_dir / f'group_{idx // 8 + 1:02d}.txt').write_text(
            '\n'.join(prompt_lines[idx:idx + 8]) + '\n')

    rows = []
    generator = root / 'tools/generate_scda_samples.py'
    for spec in args.checkpoints:
        label, value = spec.split('=', 1)
        parts = value.rsplit(':', 1)
        checkpoint = parts[0]
        dim = int(parts[1]) if len(parts) == 2 else 64
        semantic = label != 'baseline'
        for seed in args.seeds:
            out = root / args.output_root / label / f'seed_{seed}'
            existing = [x for x in out.rglob('*.png') if '.ipynb' not in str(x)] if out.exists() else []
            if len(existing) >= len(prompt_lines):
                print(f'Skip existing {label} seed={seed}: {len(existing)} images', flush=True)
                rows.append({'model': label, 'seed': seed, 'output_dir': str(out),
                             'prompts': len(prompt_lines), 'steps': args.steps,
                             'cfg_scale': args.cfg_scale})
                continue
            cmd = [sys.executable, str(generator), '--checkpoint', checkpoint,
                   '--groups-dir', str(group_dir), '--output-dir', str(out),
                   '--image-size', '512', '--steps', str(args.steps),
                   '--cfg-scale', str(args.cfg_scale), '--seed', str(seed)]
            if semantic:
                cmd += ['--semantic-conditioning', '--semantic-adapter-dim', str(dim),
                        '--semantic-residual-scale', str(args.semantic_residual_scale)]
                if args.semantic_token_attention:
                    cmd += ['--semantic-token-attention']
                if args.semantic_token_attention:
                    cmd += ['--semantic-token-gate-max', str(args.semantic_token_gate_max)]
            subprocess.run(cmd, cwd=root, check=True)
            rows.append({'model': label, 'seed': seed, 'output_dir': str(out),
                         'prompts': len(prompt_lines), 'steps': args.steps,
                         'cfg_scale': args.cfg_scale})
    manifest = root / args.output_root / 'evaluation_manifest.csv'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {manifest}')


if __name__ == '__main__':
    main()
