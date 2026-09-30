#!/usr/bin/env python3
"""Compute CLIP-space distribution and pixel-diversity metrics for existing T2I samples.

These are exploratory diagnostics, not replacements for standard Inception
FID/KID. They are useful here because all generated images and the local CLIP
checkpoint are already available, while a formal 10k/30k FID run is absent.
"""
import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def images(directory):
    return sorted(path for path in Path(directory).rglob('*') if path.suffix.lower() in {'.png', '.jpg', '.jpeg'})


def read_manifests(paths):
    models = defaultdict(list)
    for path in paths:
        with open(path, newline='') as handle:
            for row in csv.DictReader(handle):
                models[row['model']].append(Path(row['output_dir']))
    return models


def embeddings(model, processor, paths, device, batch_size):
    result = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch_images = []
        for path in batch_paths:
            with Image.open(path) as image:
                batch_images.append(image.convert('RGB'))
        inputs = processor(images=batch_images, return_tensors='pt').to(device)
        with torch.inference_mode():
            feature = model.get_image_features(**inputs)
            feature = torch.nn.functional.normalize(feature, dim=-1)
        result.append(feature.cpu())
    return torch.cat(result)


def cov(features):
    centered = features - features.mean(0, keepdim=True)
    return centered.T @ centered / (len(features) - 1)


def sqrt_psd(matrix):
    values, vectors = torch.linalg.eigh((matrix + matrix.T) * .5)
    return (vectors * values.clamp_min(0).sqrt()) @ vectors.T


def frechet(left, right):
    mean_left, mean_right = left.mean(0), right.mean(0)
    cov_left, cov_right = cov(left), cov(right)
    middle = sqrt_psd(cov_left) @ cov_right @ sqrt_psd(cov_left)
    return float((mean_left - mean_right).square().sum() + torch.trace(cov_left) + torch.trace(cov_right) - 2 * torch.trace(sqrt_psd(middle)))


def kid(left, right, subsets=20, size=192):
    generator = torch.Generator().manual_seed(43)
    count = min(len(left), len(right), size)
    values = []
    for _ in range(subsets):
        x = left[torch.randperm(len(left), generator=generator)[:count]]
        y = right[torch.randperm(len(right), generator=generator)[:count]]
        kernel = lambda a, b: ((a @ b.T) / a.shape[1] + 1).pow(3)
        kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
        values.append(float(((kxx.sum() - kxx.diag().sum()) + (kyy.sum() - kyy.diag().sum())) / (count * (count - 1)) - 2 * kxy.mean()))
    return float(np.mean(values)), float(np.std(values, ddof=1))


def ssim(left, right):
    def gray(image):
        return cv2.cvtColor(np.asarray(image.convert('RGB').resize((256, 256))), cv2.COLOR_RGB2GRAY).astype(np.float64)
    left, right = gray(left), gray(right)
    c1, c2 = 6.5025, 58.5225
    mean_left, mean_right = cv2.GaussianBlur(left, (11, 11), 1.5), cv2.GaussianBlur(right, (11, 11), 1.5)
    var_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mean_left * mean_left
    var_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mean_right * mean_right
    cross = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mean_left * mean_right
    return float((((2 * mean_left * mean_right + c1) * (2 * cross + c2)) /
                  ((mean_left * mean_left + mean_right * mean_right + c1) * (var_left + var_right + c2))).mean())


def diversity(sample_dirs):
    matched = defaultdict(list)
    for directory in sample_dirs:
        for path in images(directory):
            matched[path.relative_to(directory)].append(path)
    values = []
    for paths in matched.values():
        for first, second in combinations(paths, 2):
            with Image.open(first) as left, Image.open(second) as right:
                values.append(ssim(left, right))
    return len(values), 1 - float(np.mean(values)) if values else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', nargs='+', required=True)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--reference-limit', type=int, default=1000)
    parser.add_argument('--model-path', default='/tmp/clip-vit-base-patch32')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', default='output/clip_distribution_metrics.csv')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_path, local_files_only=True)
    ref_paths = images(args.reference_dir)[:args.reference_limit]
    reference = embeddings(model, processor, ref_paths, device, args.batch_size)
    rows = []
    for name, directories in sorted(read_manifests(args.manifests).items()):
        generated_paths = [path for directory in directories for path in images(directory)]
        generated = embeddings(model, processor, generated_paths, device, args.batch_size)
        kid_mean, kid_std = kid(generated, reference)
        pairs, diversity_score = diversity(directories)
        row = {
            'model': name, 'generated_images': len(generated_paths), 'reference_images': len(ref_paths),
            'clip_frechet_distance_exploratory': frechet(generated.to(device), reference.to(device)),
            'clip_kid_mean_exploratory': kid_mean, 'clip_kid_std_exploratory': kid_std,
            'pair_count': pairs, 'one_minus_ms_ssim_proxy_mean': diversity_score,
            'note': 'Exploratory CLIP embedding distribution metric; not standard Inception FID/KID. MS-SSIM proxy uses grayscale 256px pairs.'
        }
        print(row, flush=True)
        rows.append(row)
    output = root / args.output
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
