#!/usr/bin/env python3
"""Summarize paired CLIP binding-proxy evaluations with prompt bootstrap CIs."""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


METRICS = {
    'object': ('object_correct', lambda row: True),
    'attribute': ('presence_correct', lambda row: True),
    'binding': ('binding_correct', lambda row: row['binding_case'] == '1'),
}


def mean(rows, key, include):
    selected = [float(row[key]) for row in rows if include(row)]
    return sum(selected) / len(selected)


def quantile(values, q):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='output/attribute_binding_metrics_all.csv')
    parser.add_argument('--output', default='output/attribute_binding_metrics_all_bootstrap.csv')
    parser.add_argument('--baseline', default='baseline')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260909)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with (root / args.input).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    by_model_prompt = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_model_prompt[row['model']][row['prompt_index']].append(row)
    if args.baseline not in by_model_prompt:
        raise ValueError(f'Baseline {args.baseline!r} not found')

    prompts = sorted(by_model_prompt[args.baseline])
    models = list(by_model_prompt)
    rng = random.Random(args.seed)
    bootstrap = {model: {name: [] for name in METRICS} for model in models}
    bootstrap_delta = {model: {name: [] for name in METRICS} for model in models}
    for _ in range(args.iterations):
        sample = [rng.choice(prompts) for _ in prompts]
        for model in models:
            sampled = [row for prompt in sample for row in by_model_prompt[model][prompt]]
            baseline_sampled = [row for prompt in sample for row in by_model_prompt[args.baseline][prompt]]
            for name, (key, include) in METRICS.items():
                value = mean(sampled, key, include)
                bootstrap[model][name].append(value)
                bootstrap_delta[model][name].append(value - mean(baseline_sampled, key, include))

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['model']
    for name in METRICS:
        fields.extend([f'{name}_accuracy', f'{name}_correct', f'{name}_n',
                       f'{name}_ci_low', f'{name}_ci_high',
                       f'{name}_delta_vs_baseline', f'{name}_delta_ci_low',
                       f'{name}_delta_ci_high'])
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in models:
            record = {'model': model}
            rows_for_model = [row for prompt_rows in by_model_prompt[model].values() for row in prompt_rows]
            for name, (key, include) in METRICS.items():
                selected = [row for row in rows_for_model if include(row)]
                record[f'{name}_accuracy'] = mean(rows_for_model, key, include)
                record[f'{name}_correct'] = sum(int(row[key]) for row in selected)
                record[f'{name}_n'] = len(selected)
                record[f'{name}_ci_low'] = quantile(bootstrap[model][name], 0.025)
                record[f'{name}_ci_high'] = quantile(bootstrap[model][name], 0.975)
                record[f'{name}_delta_vs_baseline'] = record[f'{name}_accuracy'] - mean(
                    [row for prompt_rows in by_model_prompt[args.baseline].values() for row in prompt_rows], key, include)
                record[f'{name}_delta_ci_low'] = quantile(bootstrap_delta[model][name], 0.025)
                record[f'{name}_delta_ci_high'] = quantile(bootstrap_delta[model][name], 0.975)
            writer.writerow(record)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
