#!/usr/bin/env python3
"""Paired significance test for T2I-CompBench++ scores.

The benchmark reports a mean over prompts, and `compare_improvement.py` reports
the difference of two such means.  A difference of means is not by itself
evidence of an improvement: with 300 prompts per category the standard error is
large enough that +-0.02 happens by chance.  This script pairs the methods at the
PROMPT level (same prompt, same seed, same protocol) and bootstraps the paired
difference, which removes the prompt-difficulty variance that dominates the
unpaired comparison.

Output: one row per (category, method, baseline) with the delta and its 95%
interval, plus a markdown table.

Usage:
    python results/_work/compbench_paired_test.py \
        --methods gateopt b1 b2 tpscda \
        --output results/improvement_report/compbench_paired_tests.csv
"""
import argparse
import csv
import random
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
CB = Path('/root/compbench_work/results')

CATEGORIES = ['color', 'shape', 'texture', 'spatial', '3d_spatial', 'numeracy',
              'non_spatial', 'complex']
CN = {'color': '颜色', 'shape': '形状', 'texture': '纹理', 'spatial': '2D空间',
      '3d_spatial': '3D空间', 'numeracy': '计数', 'non_spatial': '非空间',
      'complex': '复杂'}
# The three attribute-binding categories are the ones TP-SCDA targets.
BINDING = {'color', 'shape', 'texture'}


def load(method, category):
    """Mean score per prompt_index (only successfully scored rows).

    A prompt can have several images -- the `complex` category uses the official
    10 images per prompt while every other category here uses 1.  Keying the
    dict by prompt_index without averaging would silently keep only the LAST
    image of each prompt, which is what an earlier version of this script did.
    """
    path = CB / method / category / 'per_prompt.csv'
    if not path.exists():
        return None
    grouped = {}
    for row in csv.DictReader(path.open()):
        if row.get('status') != 'ok':
            continue
        try:
            grouped.setdefault(int(row['prompt_index']), []).append(float(row['prediction']))
        except (KeyError, ValueError):
            continue
    if not grouped:
        return None
    return {index: sum(values) / len(values) for index, values in grouped.items()}


def paired_bootstrap(a, b, draws=10000, seed=12345):
    """Difference b - a over the shared prompts, with a 95% interval."""
    rng = random.Random(seed)
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diffs = [b[k] - a[k] for k in keys]
    n = len(diffs)
    mean = sum(diffs) / n
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(draws))
    lo = boot[int(0.025 * draws)]
    hi = boot[int(0.975 * draws)]
    return dict(n=n, mean_a=sum(a[k] for k in keys) / n,
                mean_b=sum(b[k] for k in keys) / n,
                delta=mean, ci_low=lo, ci_high=hi,
                better=sum(1 for d in diffs if d > 0),
                worse=sum(1 for d in diffs if d < 0),
                significant=(lo > 0) or (hi < 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', default='frozen')
    ap.add_argument('--methods', nargs='+',
                    default=['tpscda', 'gateopt', 'b1', 'b2'])
    ap.add_argument('--draws', type=int, default=10000)
    ap.add_argument('--output', type=Path,
                    default=ROOT / 'results/improvement_report/compbench_paired_tests.csv')
    args = ap.parse_args()

    rows = []
    for method in args.methods:
        for category in CATEGORIES:
            base = load(args.baseline, category)
            other = load(method, category)
            if base is None or other is None:
                rows.append(dict(method=method, category=category, n=0,
                                 mean_baseline='', mean_method='', delta='',
                                 ci_low='', ci_high='', better='', worse='',
                                 significant='', verdict='NOT COMPLETED'))
                continue
            stat = paired_bootstrap(base, other, draws=args.draws)
            if stat is None:
                rows.append(dict(method=method, category=category, n=0,
                                 mean_baseline='', mean_method='', delta='',
                                 ci_low='', ci_high='', better='', worse='',
                                 significant='', verdict='NOT COMPLETED'))
                continue
            if stat['significant'] and stat['delta'] > 0:
                verdict = '提升（显著）'
            elif stat['significant'] and stat['delta'] < 0:
                verdict = '退步（显著）'
            else:
                verdict = '无显著差异'
            rows.append(dict(
                method=method, category=category, n=stat['n'],
                mean_baseline=f"{stat['mean_a']:.4f}",
                mean_method=f"{stat['mean_b']:.4f}",
                delta=f"{stat['delta']:+.4f}",
                ci_low=f"{stat['ci_low']:+.4f}",
                ci_high=f"{stat['ci_high']:+.4f}",
                better=stat['better'], worse=stat['worse'],
                significant=stat['significant'], verdict=verdict))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['method', 'category', 'n', 'mean_baseline', 'mean_method', 'delta',
              'ci_low', 'ci_high', 'better', 'worse', 'significant', 'verdict']
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('wrote', args.output)

    for method in args.methods:
        print(f'\n=== {method} vs {args.baseline} (paired, 95% CI) ===')
        print(f"{'类别':<8}{'基线':>9}{method:>10}{'差值':>9}{'95% 区间':>22}{'判定':>14}")
        for row in rows:
            if row['method'] != method:
                continue
            if row['n'] == 0:
                print(f"{CN[row['category']]:<8}{'N/A':>9}{'N/A':>10}{'—':>9}{'—':>22}{'NOT COMPLETED':>14}")
                continue
            ci = f"[{row['ci_low']}, {row['ci_high']}]"
            print(f"{CN[row['category']]:<8}{row['mean_baseline']:>9}"
                  f"{row['mean_method']:>10}{row['delta']:>9}{ci:>22}{row['verdict']:>14}")


if __name__ == '__main__':
    main()
