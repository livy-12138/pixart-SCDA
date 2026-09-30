#!/usr/bin/env python3
"""Compute CLIP contrastive proxies for object/attribute binding."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def image_path(sample_root, prompt_index, prompts):
    prompt = prompts[prompt_index - 1]
    group = f"group_{(prompt_index - 1) // 8 + 1:02d}"
    filename = prompt[:100].replace('/', '_') + '.png'
    return sample_root / group / filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='asset/attribute_binding_cases.csv')
    parser.add_argument('--prompts', default='asset/samples.txt')
    parser.add_argument('--evaluations', nargs='+', required=True,
                        help='label=eval_root; eval_root contains label/seed_N')
    parser.add_argument('--seeds', nargs='+', type=int, default=[43, 44, 45])
    parser.add_argument('--model-path', default='/tmp/clip-vit-base-patch32')
    parser.add_argument('--output', default='output/attribute_binding_metrics.csv')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Number of binding cases per CLIP forward pass.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prompts = [x.strip() for x in (root / args.prompts).read_text().splitlines() if x.strip()]
    cases = list(csv.DictReader((root / args.cases).open()))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_path, local_files_only=True)
    records = []
    for spec in args.evaluations:
        label, directory = spec.split('=', 1)
        for seed in args.seeds:
            sample_root = root / directory / label / f'seed_{seed}'
            for start in range(0, len(cases), args.batch_size):
                batch = cases[start:start + args.batch_size]
                texts, images = [], []
                for case in batch:
                    path = image_path(sample_root, int(case['prompt_index']), prompts)
                    if not path.exists():
                        raise FileNotFoundError(path)
                    captions = [case['object_caption'], case['object_distractor'],
                                case['positive_caption'], case['neutral_caption'],
                                case['swapped_caption']]
                    image = Image.open(path).convert('RGB')
                    texts.extend(captions)
                    images.extend([image] * len(captions))
                inputs = processor(text=texts, images=images,
                                   return_tensors='pt', padding=True, truncation=True,
                                   max_length=model.config.text_config.max_position_embeddings).to(device)
                with torch.inference_mode():
                    output = model(**inputs)
                    sims = torch.nn.functional.cosine_similarity(
                        output.text_embeds, output.image_embeds).cpu().tolist()
                for offset, case in enumerate(batch):
                    case_sims = sims[offset * 5:(offset + 1) * 5]
                    object_margin = case_sims[0] - case_sims[1]
                    presence_margin = case_sims[2] - case_sims[3]
                    binding_margin = case_sims[2] - case_sims[4]
                    records.append({
                        'model': label, 'seed': seed, 'prompt_index': case['prompt_index'],
                        'binding_case': case['binding'], 'object_margin': object_margin,
                        'object_correct': int(object_margin > 0),
                        'presence_margin': presence_margin,
                        'presence_correct': int(presence_margin > 0),
                        'binding_margin': binding_margin,
                        'binding_correct': int(binding_margin > 0),
                    })
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['model'], int(row['seed']))].append(row)
    summary = out.with_name(out.stem + '_summary.csv')
    fields = ['model', 'seed', 'n', 'binding_n', 'object_accuracy',
              'presence_accuracy', 'binding_accuracy', 'object_margin',
              'presence_margin', 'binding_margin']
    with summary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (model_name, seed), rows in sorted(grouped.items()):
            bind = [r for r in rows if r['binding_case'] == '1']
            mean = lambda key, xs: sum(float(r[key]) for r in xs) / len(xs)
            writer.writerow({
                'model': model_name, 'seed': seed, 'n': len(rows), 'binding_n': len(bind),
                'object_accuracy': mean('object_correct', rows),
                'presence_accuracy': mean('presence_correct', rows),
                'binding_accuracy': mean('binding_correct', bind),
                'object_margin': mean('object_margin', rows),
                'presence_margin': mean('presence_margin', rows),
                'binding_margin': mean('binding_margin', bind),
            })
    print(f'Wrote {out} and {summary}')


if __name__ == '__main__':
    main()
