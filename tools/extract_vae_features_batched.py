#!/usr/bin/env python3
"""Batch-extract PixArt VAE posterior mean/std features for an InternalData manifest."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from diffusers.models import AutoencoderKL
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def stem(path):
    return '_'.join(path.rsplit('/', 1)).rsplit('.', 1)[0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json-path', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--vae-path', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=8)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    out = Path(args.output_root) / f'{args.image_size}resolution' / 'noflip'
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(args.json_path).read_text())
    transform = T.Compose([T.Resize(args.image_size), T.CenterCrop(args.image_size), T.ToTensor(), T.Normalize([.5], [.5])])
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device=device, dtype=dtype).eval()
    pending = []
    for r in records:
        target = out / f'{stem(r["path"])}.npy'
        if not target.exists():
            pending.append((r, target))
    print(f'records={len(records)} existing={len(records)-len(pending)} pending={len(pending)} batch={args.batch_size}')
    class ImageSet(Dataset):
        def __len__(self): return len(pending)
        def __getitem__(self, i):
            r, target = pending[i]
            try:
                return transform(Image.open(Path(args.dataset_root) / r['path']).convert('RGB')), str(target), ''
            except Exception as exc:
                return torch.zeros(3, args.image_size, args.image_size), str(target), str(exc)
    loader = DataLoader(ImageSet(), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0, prefetch_factor=4)
    for images, targets, errors in tqdm(loader, desc='VAE features'):
        valid = [(t, e) for t, e in zip(targets, errors) if not e]
        if not valid:
            continue
        images = images[[not bool(e) for e in errors]]
        with torch.inference_mode():
            posterior = vae.encode(images.to(device=device, dtype=dtype)).latent_dist
            values = torch.cat([posterior.mean, posterior.std], dim=1).float().cpu().numpy()
        for value, (target, _) in zip(values, valid): np.save(target, value)
    print('done')

if __name__ == '__main__':
    main()
