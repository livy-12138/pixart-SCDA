#!/usr/bin/env python3
"""Compute exploratory image-quality and diversity metrics for T2I samples.

The script deliberately labels FID/KID as exploratory when fewer than 10,000
generated images are available. It uses a shared COCO validation reference and
the evaluation manifests already emitted by the project inference scripts.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3


def list_images(directory):
    return sorted(path for path in directory.rglob('*')
                  if path.suffix.lower() in {'.png', '.jpg', '.jpeg'})


def load_rgb(path, size=299):
    with Image.open(path) as image:
        image = image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def batched_inception(model, paths, device, batch_size):
    features, probabilities = [], []
    captured = []

    def capture(_, __, output):
        captured.append(torch.flatten(output, 1))

    hook = model.avgpool.register_forward_hook(capture)
    try:
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([load_rgb(path) for path in paths[start:start + batch_size]]).to(device)
            captured.clear()
            with torch.inference_mode():
                logits = model(batch)
                probabilities.append(torch.softmax(logits, dim=1).cpu())
                features.append(captured[0].cpu())
    finally:
        hook.remove()
    return torch.cat(features), torch.cat(probabilities)


def covariance(features):
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.T @ centered / (features.shape[0] - 1)


def psd_sqrt(matrix):
    values, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    return (vectors * values.clamp_min(0).sqrt()) @ vectors.T


def frechet_distance(generated, reference):
    gen_mean, ref_mean = generated.mean(0), reference.mean(0)
    gen_cov, ref_cov = covariance(generated), covariance(reference)
    gen_sqrt = psd_sqrt(gen_cov)
    middle = gen_sqrt @ ref_cov @ gen_sqrt
    trace_sqrt = torch.trace(psd_sqrt(middle))
    return float((gen_mean - ref_mean).dot(gen_mean - ref_mean)
                 + torch.trace(gen_cov) + torch.trace(ref_cov) - 2 * trace_sqrt)


def polynomial_kernel(left, right):
    return ((left @ right.T) / left.shape[1] + 1.0).pow(3)


def kid_subsets(generated, reference, subset_size, subsets, generator):
    n = min(subset_size, generated.shape[0], reference.shape[0])
    values = []
    for _ in range(subsets):
        gen_index = torch.randperm(generated.shape[0], generator=generator)[:n]
        ref_index = torch.randperm(reference.shape[0], generator=generator)[:n]
        left, right = generated[gen_index], reference[ref_index]
        kxx, kyy, kxy = polynomial_kernel(left, left), polynomial_kernel(right, right), polynomial_kernel(left, right)
        estimate = ((kxx.sum() - kxx.diag().sum()) + (kyy.sum() - kyy.diag().sum())) / (n * (n - 1))
        estimate -= 2 * kxy.mean()
        values.append(float(estimate))
    return float(np.mean(values)), float(np.std(values, ddof=1))


def inception_score(probabilities, splits=10):
    chunks = torch.tensor_split(probabilities, min(splits, len(probabilities)))
    scores = []
    for chunk in chunks:
        marginal = chunk.mean(0, keepdim=True)
        kl = (chunk * (chunk.clamp_min(1e-12).log() - marginal.clamp_min(1e-12).log())).sum(1)
        scores.append(float(torch.exp(kl.mean())))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def ssim(left, right):
    left = cv2.cvtColor(np.asarray(left.convert('RGB').resize((256, 256))), cv2.COLOR_RGB2GRAY).astype(np.float64)
    right = cv2.cvtColor(np.asarray(right.convert('RGB').resize((256, 256))), cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = 6.5025, 58.5225
    mu_left, mu_right = cv2.GaussianBlur(left, (11, 11), 1.5), cv2.GaussianBlur(right, (11, 11), 1.5)
    sigma_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mu_left * mu_left
    sigma_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mu_right * mu_right
    sigma_cross = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mu_left * mu_right
    value = ((2 * mu_left * mu_right + c1) * (2 * sigma_cross + c2)) / ((mu_left ** 2 + mu_right ** 2 + c1) * (sigma_left + sigma_right + c2))
    return float(value.mean())


def diversity_metrics(sample_dirs, device, lpips_model):
    by_name = defaultdict(list)
    for sample_dir in sample_dirs:
        for path in list_images(sample_dir):
            by_name[path.relative_to(sample_dir)].append(path)
    lpips_values, ssim_values = [], []
    for paths in by_name.values():
        if len(paths) < 2:
            continue
        for left_path, right_path in combinations(paths, 2):
            with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
                ssim_values.append(ssim(left_image, right_image))
                if lpips_model is not None:
                    left = load_rgb(left_path, 256).unsqueeze(0).to(device) * 2 - 1
                    right = load_rgb(right_path, 256).unsqueeze(0).to(device) * 2 - 1
                    with torch.inference_mode():
                        lpips_values.append(float(lpips_model(left, right).item()))
    result = {'pair_count': len(ssim_values), 'ms_ssim_proxy_mean': 1 - float(np.mean(ssim_values)) if ssim_values else math.nan}
    result['lpips_mean'] = float(np.mean(lpips_values)) if lpips_values else math.nan
    return result


def read_models(manifests):
    models = defaultdict(list)
    for manifest in manifests:
        with Path(manifest).open(newline='') as handle:
            for row in csv.DictReader(handle):
                models[row['model']].append(Path(row['output_dir']))
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', nargs='+', required=True)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--reference-limit', type=int, default=5000)
    parser.add_argument('--feature-dim', type=int, default=512,
                        help='Deterministic prefix dimension for tractable exploratory FID/KID.')
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--kid-subsets', type=int, default=10)
    parser.add_argument('--output', default='output/image_distribution_metrics.csv')
    parser.add_argument('--disable-lpips', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(43)

    reference_paths = list_images(Path(args.reference_dir))[:args.reference_limit]
    if len(reference_paths) < 2:
        raise ValueError('Need at least two reference images')
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False).to(device).eval()
    reference_features, _ = batched_inception(model, reference_paths, device, args.batch_size)
    # Full 2048-D covariance eigendecomposition is unnecessarily expensive in
    # this environment. A fixed prefix makes every model comparable while the
    # output is explicitly labelled exploratory rather than standard FID.
    feature_dim = min(args.feature_dim, reference_features.shape[1])
    reference_features = reference_features[:, :feature_dim]
    models = read_models(args.manifests)

    lpips_model = None
    lpips_status = 'not installed or disabled'
    if not args.disable_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').to(device).eval()
            lpips_status = 'AlexNet LPIPS'
        except Exception as error:
            lpips_status = f'unavailable: {type(error).__name__}: {error}'

    rows = []
    kid_generator = torch.Generator().manual_seed(43)
    for name, sample_dirs in sorted(models.items()):
        generated_paths = []
        for sample_dir in sample_dirs:
            generated_paths.extend(list_images(sample_dir))
        generated_features, generated_probabilities = batched_inception(model, generated_paths, device, args.batch_size)
        generated_features = generated_features[:, :feature_dim]
        fid = frechet_distance(generated_features.to(device), reference_features.to(device))
        kid_mean, kid_std = kid_subsets(generated_features, reference_features, 192, args.kid_subsets, kid_generator)
        is_mean, is_std = inception_score(generated_probabilities)
        diversity = diversity_metrics(sample_dirs, device, lpips_model)
        rows.append({
            'model': name, 'generated_images': len(generated_paths), 'reference_images': len(reference_paths),
            'fid_inception_v3_exploratory': fid, 'kid_mean_exploratory': kid_mean,
            'kid_std_exploratory': kid_std, 'inception_score_mean_exploratory': is_mean,
            'inception_score_std_exploratory': is_std, **diversity,
            'lpips_backend': lpips_status,
            'feature_dim': feature_dim,
            'note': 'Exploratory only: reduced Inception feature dimension and fewer than 10k generated images.'
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
