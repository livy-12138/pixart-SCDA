#!/usr/bin/env python3
"""Attribute-swap binding proxy on the CompBench colour prompts.

Why this exists
---------------
The caption-level metrics (CLIPScore, CompBench B-VQA) are dominated by whether
the right objects are present.  They do not isolate the failure mode TP-SCDA
targets: an attribute attached to the WRONG object.

The standard way to isolate it (SynGen-style) is to compare each image against
two descriptions -- the correct one and one where the attributes are exchanged
between the objects -- and count how often the correct one wins.  The paper's
existing binding proxy does exactly this, but with only 13 pair cases x 3 seeds
= 39 judgements, which has almost no power.

This script applies the same idea to the 300 CompBench colour prompts, whose
images ALREADY EXIST for every method, so the sample size goes up ~20x at no
generation cost.

Naming: the output is an `Attribute-presence proxy` / `Binding proxy`
in the sense of the paper.  It is NOT a recognition accuracy and must not be
reported as one.

Usage:
    python tools/score_binding_swap.py --methods frozen tpscda gateopt b2 \
        --output results/binding_swap/binding_swap.csv
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from tools.prepare_semantic_masks import clean_caption
from tools.test_text_pseudolabels import build_labels, load_nlp

IMAGES = Path('/root/compbench_work/images')
PROMPTS = Path('/tmp/cb/T2I-CompBench/examples/dataset/color_val.txt')


def object_attribute_groups(prompt, nlp):
    """[(attribute_words, object_words)] per parsed object, in prompt order.

    Uses the same parser as the training-time masks, so "what counts as an
    attribute of which object" is identical to what the model was trained on.
    """
    doc = nlp(clean_caption(prompt))
    labels = build_labels(doc)
    groups = []
    for obj in labels['objects']:
        attrs = obj.get('attributes') or []
        if attrs:
            groups.append((list(attrs), list(obj['token_indices'])))
    return groups, doc


def swapped_prompt(prompt, nlp):
    """Exchange the attribute spans between the two objects, if there are two.

    Only prompts whose two attribute spans have the SAME number of words are
    accepted.  Otherwise the exchange produces a malformed sentence -- e.g.
    "the soft pillow ... the hard rocking chair" would come out as
    "the rocking pillow ... the hard soft chair", which is a different sentence
    rather than the same sentence with the attributes exchanged, and the
    comparison would no longer isolate binding.
    """
    groups, doc = object_attribute_groups(prompt, nlp)
    if len(groups) != 2:
        return None
    (a0, o0), (a1, o1) = groups[0], groups[1]
    if not a0 or not a1 or len(a0) != len(a1):
        return None
    tokens = [t.text for t in doc]
    out = list(tokens)
    for slot, src in ((a0, a1), (a1, a0)):
        for pos, word_index in enumerate(slot):
            out[word_index] = tokens[src[pos]]
    text = ''
    for i, tok in enumerate(out):
        text += tok if (i == 0 or tok in ',.!?') else ' ' + tok
    # doc.text preserves the original casing of the first word; restoring it
    # keeps the pair differing ONLY in the exchanged attributes
    if prompt[:1].isupper() and text:
        text = text[0].upper() + text[1:]
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--methods', nargs='+',
                    default=['frozen', 'tpscda', 'gateopt', 'b2'])
    ap.add_argument('--category', default='color')
    ap.add_argument('--prompts', type=Path, default=None,
                    help='prompt file; defaults to <category>_val.txt')
    ap.add_argument('--model-path', default='/tmp/clip-vit-base-patch32')
    ap.add_argument('--output', type=Path,
                    default=ROOT / 'results/binding_swap/binding_swap.csv')
    args = ap.parse_args()

    nlp = load_nlp('en_core_web_sm')
    prompt_file = args.prompts or (
        PROMPTS.parent / f'{args.category}_val.txt')
    prompts = [l.strip() for l in prompt_file.read_text().splitlines() if l.strip()]
    print(f'{len(prompts)} prompts')

    pairs = []
    for i, prompt in enumerate(prompts):
        sw = swapped_prompt(prompt, nlp)
        if sw and sw != prompt:
            pairs.append((i, prompt, sw))
    print(f'{len(pairs)} prompts admit a clean 2-object attribute swap')
    if not pairs:
        raise SystemExit('no swappable prompts -- check the parser')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_path, local_files_only=True)

    rows = []
    for method in args.methods:
        img_dir = IMAGES / method / args.category
        if not img_dir.exists():
            print(f'{method}: no image directory {img_dir}')
            continue
        # every repeat of every swappable prompt
        todo = []
        for idx, prompt, swapped in pairs:
            for path in sorted(img_dir.glob(f'{prompt}_*.png')):
                todo.append((idx, prompt, swapped, path))
        if not todo:
            print(f'{method}: no images found under {img_dir}')
            continue
        per_image = []
        with torch.no_grad():
            for start in range(0, len(todo), 16):
                chunk = todo[start:start + 16]
                images = [Image.open(path).convert('RGB') for _, _, _, path in chunk]
                text = [t for _, p, s, _ in chunk for t in (p, s)]
                inp = processor(text=text, images=images, return_tensors='pt',
                                padding=True, truncation=True).to(device)
                out = model(**inp)
                ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
                te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
                sims = ie @ te.T                        # (B, 2B)
                rows_idx = torch.arange(len(images), device=sims.device)
                sims = torch.stack([sims[rows_idx, 2 * rows_idx],
                                    sims[rows_idx, 2 * rows_idx + 1]], dim=1)
                for (idx, prompt, swapped, path), (sc, ss) in zip(chunk, sims.tolist()):
                    per_image.append(dict(prompt_index=idx, prompt=prompt,
                                          swapped=swapped, image=path.name,
                                          sim_correct=round(sc, 6),
                                          sim_swapped=round(ss, 6),
                                          correct_wins=1 if sc > ss else 0))
        # cluster by prompt: the paired test must treat a prompt, not an image,
        # as the independent unit
        by_prompt = {}
        for rec in per_image:
            by_prompt.setdefault(rec['prompt_index'], []).append(rec['correct_wins'])
        per_prompt = []
        for idx, prompt, swapped in pairs:
            wins = by_prompt.get(idx)
            if not wins:
                continue
            per_prompt.append(dict(prompt_index=idx, prompt=prompt, swapped=swapped,
                                   n_images=len(wins),
                                   correct_wins=round(sum(wins) / len(wins), 6)))
        n_prompt = len(per_prompt)
        correct_wins = sum(r['correct_wins'] for r in per_prompt)
        rows.append(dict(method=method, n=n_prompt, n_images=len(per_image),
                         correct_wins=round(correct_wins, 4),
                         binding_proxy=round(correct_wins / n_prompt, 6)))
        print(f'{method:<10} prompts={n_prompt:<4} images={len(per_image):<5} '
              f'binding proxy = {correct_wins / n_prompt:.4f}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with (args.output.parent / f'per_prompt_{method}_{args.category}.csv').open(
                'w', newline='') as handle:
            w = csv.DictWriter(handle, fieldnames=list(per_prompt[0].keys()))
            w.writeheader(); w.writerows(per_prompt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=['method', 'n', 'n_images',
                                               'correct_wins', 'binding_proxy'])
        w.writeheader(); w.writerows(rows)
    print('wrote', args.output)

    # paired bootstrap of the difference vs the frozen baseline
    base = next((r for r in rows if r['method'] == 'frozen'), None)
    if base:
        base_pp = {int(r['prompt_index']): float(r['correct_wins']) for r in
                   csv.DictReader((args.output.parent /
                                   f'per_prompt_frozen_{args.category}.csv').open())}
        rng = random.Random(12345)
        print('\npaired bootstrap vs frozen (10 000 draws, 95% interval):')
        for r in rows:
            if r['method'] == 'frozen':
                continue
            pp = {int(x['prompt_index']): float(x['correct_wins']) for x in
                  csv.DictReader((args.output.parent /
                                  f"per_prompt_{r['method']}_{args.category}.csv").open())}
            keys = sorted(set(base_pp) & set(pp))
            diffs = [pp[k] - base_pp[k] for k in keys]
            n = len(diffs)
            mean = sum(diffs) / n
            boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                          for _ in range(10000))
            lo, hi = boot[250], boot[9750]
            verdict = ('提升(显著)' if lo > 0 else '退步(显著)' if hi < 0 else '无显著差异')
            print(f'  {r["method"]:<10} delta={mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}')


if __name__ == '__main__':
    main()
