#!/usr/bin/env python3
"""Score fixed prompt/image directories with a local CLIP model."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def image_path(root, group, prompt):
    return root / group / (prompt[:100].replace('/', '_') + '.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', default='asset/samples.txt')
    parser.add_argument('--evaluations', nargs='+', required=True,
                        help='label=output/eval/root; each root contains label/seed_N directories')
    parser.add_argument('--seeds', type=int, nargs='+', default=[43, 44, 45])
    parser.add_argument('--model-path', default='/tmp/clip-vit-base-patch32')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', default='output/multiseed_clip_scores.csv')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prompts = [line.strip() for line in (root / args.prompts).read_text().splitlines() if line.strip()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_path, local_files_only=True)
    records = []
    for spec in args.evaluations:
        label, directory = spec.split('=', 1)
        for seed in args.seeds:
            sample_root = root / directory / label / f'seed_{seed}'
            pairs = []
            for index, prompt in enumerate(prompts):
                path = image_path(sample_root, f'group_{index // 8 + 1:02d}', prompt)
                if not path.exists():
                    raise FileNotFoundError(path)
                pairs.append((prompt, path))
            scores = []
            for start in range(0, len(pairs), args.batch_size):
                batch = pairs[start:start + args.batch_size]
                inputs = processor(text=[x[0] for x in batch], images=[Image.open(x[1]).convert('RGB') for x in batch],
                                   return_tensors='pt', padding=True, truncation=True,
                                   max_length=model.config.text_config.max_position_embeddings).to(device)
                with torch.inference_mode():
                    output = model(**inputs)
                    sims = torch.nn.functional.cosine_similarity(output.text_embeds, output.image_embeds).cpu().tolist()
                scores.extend(sims)
            for index, score in enumerate(scores):
                records.append({'model': label, 'seed': seed, 'prompt_index': index + 1,
                                'group': index // 8 + 1, 'clip_score': score})
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['model'], row['seed'])].append(row['clip_score'])
    summary = out.with_name(out.stem + '_summary.csv')
    with summary.open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['model', 'seed', 'n', 'clip_mean', 'clip_std'])
        for (label, seed), values in sorted(grouped.items()):
            tensor = torch.tensor(values)
            writer.writerow([label, seed, len(values), float(tensor.mean()), float(tensor.std(unbiased=False))])
    print(f'Wrote {out} and {summary}')


if __name__ == '__main__':
    main()
