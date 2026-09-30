#!/usr/bin/env python3
"""Parser / token-alignment descriptive statistics.

The paper asks for precision/recall/F1 of object, attribute and edge extraction.
There is NO human ground truth on this server, so this script deliberately
reports only DESCRIPTIVE statistics: how often the rule parser finds each
element, and how often the character-span -> T5-subtoken alignment maps a parsed
element onto zero tokens.  Reporting P/R/F1 without ground truth would be
fabrication, so none is produced here.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import numpy as np
from tools.prepare_semantic_masks import clean_caption, overlapping_indices
from tools.test_text_pseudolabels import build_labels, load_nlp, non_space_tokens

CATEGORIES = ['color', 'shape', 'texture', 'spatial', '3d_spatial',
              'numeracy', 'non_spatial', 'complex']


def analyse(prompt, nlp, tokenizer, sequence_length):
    cleaned = clean_caption(prompt)
    doc = nlp(cleaned)
    labels = build_labels(doc)
    words = non_space_tokens(doc)
    enc = tokenizer(cleaned, max_length=sequence_length, padding='max_length',
                    truncation=True, add_special_tokens=True,
                    return_offsets_mapping=True)
    offsets = enc['offset_mapping']
    if hasattr(offsets, 'tolist'):
        offsets = offsets.tolist()
    # tokenizer truncation: does the tokenizer report tokens beyond the limit?
    full = tokenizer(cleaned, add_special_tokens=True, truncation=False,
                     return_offsets_mapping=True)['input_ids']
    truncated = max(0, len(full) - sequence_length)

    objects = labels['objects']
    attrs = labels['attribute_token_indices']
    relations = labels['relations']

    obj_words = [i for o in objects for i in o['token_indices']]
    rel_words = [i for r in relations for i in r['token_indices']]

    obj_mapped = overlapping_indices(obj_words, words, offsets) if obj_words else set()
    attr_mapped = overlapping_indices(attrs, words, offsets) if attrs else set()
    rel_mapped = overlapping_indices(rel_words, words, offsets) if rel_words else set()

    return dict(
        n_objects=len(objects),
        n_attributes=len(attrs),
        n_relations=len(relations),
        obj_word_count=len(obj_words),
        attr_word_count=len(attrs),
        rel_word_count=len(rel_words),
        obj_aligned=len(obj_mapped),
        attr_aligned=len(attr_mapped),
        rel_aligned=len(rel_mapped),
        truncated_tokens=truncated,
        n_tokens=len(full),
    )


def summarise(rows, label):
    n = len(rows)
    if n == 0:
        return {}
    def frac(pred):
        return sum(1 for r in rows if pred(r)) / n
    # alignment failure = parsed element exists but maps to zero T5 sub-tokens
    def align_fail(rows_, kind):
        wc, ac = f'{kind}_word_count', f'{kind}_aligned'
        failed = [r for r in rows_ if r[wc] > 0 and r[ac] == 0]
        return len(failed), sum(1 for r in rows_ if r[wc] > 0)
    out = dict(
        split=label,
        n_prompts=n,
        prompts_with_object=frac(lambda r: r['n_objects'] > 0),
        prompts_with_attribute=frac(lambda r: r['n_attributes'] > 0),
        prompts_with_relation=frac(lambda r: r['n_relations'] > 0),
        prompts_with_object_and_attribute=frac(lambda r: r['n_objects'] > 0 and r['n_attributes'] > 0),
        mean_objects=float(np.mean([r['n_objects'] for r in rows])),
        mean_attributes=float(np.mean([r['n_attributes'] for r in rows])),
        mean_relations=float(np.mean([r['n_relations'] for r in rows])),
        median_objects=float(np.median([r['n_objects'] for r in rows])),
        median_attributes=float(np.median([r['n_attributes'] for r in rows])),
        median_relations=float(np.median([r['n_relations'] for r in rows])),
        prompts_truncated_by_t5=frac(lambda r: r['truncated_tokens'] > 0),
        mean_truncated_tokens=float(np.mean([r['truncated_tokens'] for r in rows])),
    )
    for kind in ('obj', 'attr', 'rel'):
        n_fail, n_cand = align_fail(rows, kind)
        out[f'{kind}_alignment_failure_rate'] = (n_fail / n_cand) if n_cand else 0.0
        out[f'{kind}_elements_considered'] = n_cand
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'results/parser_alignment'))
    ap.add_argument('--compbench-dir', default='/tmp/cb/T2I-CompBench/examples/dataset')
    ap.add_argument('--sequence-length', type=int, default=120)
    ap.add_argument('--t5-cache', default=str(ROOT / 'output/pretrained_models/t5_ckpts'))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.t5_cache + '/t5-v1_1-xxl',
                                              use_fast=True, local_files_only=True)
    nlp = load_nlp('en_core_web_sm')

    out_dir = Path(args.out_dir)
    (out_dir / 'raw').mkdir(parents=True, exist_ok=True)

    splits = {}
    samples = [l.strip() for l in (ROOT / 'asset/samples.txt').read_text().splitlines() if l.strip()]
    splits['asset_samples_64'] = samples
    for cat in CATEGORIES:
        p = Path(args.compbench_dir) / f'{cat}_val.txt'
        splits[f'compbench_{cat}'] = [l.strip() for l in p.read_text().splitlines() if l.strip()]

    all_rows = []
    summaries = []
    for name, prompts in splits.items():
        rows = []
        for i, p in enumerate(prompts):
            r = analyse(p, nlp, tokenizer, args.sequence_length)
            r['split'] = name
            r['prompt_id'] = i
            r['prompt'] = p
            rows.append(r)
        all_rows.extend(rows)
        s = summarise(rows, name)
        summaries.append(s)
        print(f'{name:24s} n={len(rows):4d}  obj={s["prompts_with_object"]*100:5.1f}%  '
              f'attr={s["prompts_with_attribute"]*100:5.1f}%  '
              f'rel={s["prompts_with_relation"]*100:5.1f}%  '
              f'obj_align_fail={s["obj_alignment_failure_rate"]*100:5.2f}%  '
              f'trunc={s["prompts_truncated_by_t5"]*100:5.1f}%', flush=True)

    fields = list(all_rows[0].keys())
    with open(out_dir / 'raw' / 'per_prompt.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    sfields = list(summaries[0].keys())
    with open(out_dir / 'summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sfields)
        w.writeheader(); w.writerows(summaries)
    (out_dir / 'summary.json').write_text(json.dumps({
        'note': ('Descriptive statistics only. No human ground truth was available, '
                 'so precision/recall/F1 are deliberately NOT reported.'),
        'sequence_length': args.sequence_length,
        'splits': summaries,
    }, indent=2, ensure_ascii=False))
    print('wrote', out_dir)


if __name__ == '__main__':
    main()
