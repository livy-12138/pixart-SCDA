"""Prepare MS COCO 2014 captions for the PixArt InternalData format.

This creates ``partition/data_info.json`` and, optionally, the T5 and VAE
features consumed by ``InternalData``.  COCO image paths remain relative to
``--image-root`` (for example ``train2014/COCO_train2014_000000000009.jpg``).
"""
import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True, help='COCO captions_*.json')
    parser.add_argument('--image-root', required=True, help='Directory containing train2014/ or val2014/')
    parser.add_argument('--output-root', required=True, help='Directory used as InternalData root')
    parser.add_argument('--max-items', type=int, default=0, help='0 means all images')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--verify-images', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    with open(args.annotations, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    captions = {}
    for item in annotations['annotations']:
        captions.setdefault(item['image_id'], []).append(item['caption'].strip())
    images = annotations['images']
    random.shuffle(images)
    if args.max_items > 0:
        images = images[:args.max_items]

    records = []
    missing = 0
    for item in tqdm(images, desc='Building COCO metadata'):
        file_name = item['file_name']
        # The annotation file contains only the filename; infer the COCO split
        # (including the 2017 expansion) from its filename.
        annotation_name = Path(args.annotations).name.lower()
        if 'val2017' in annotation_name:
            split = 'val2017'
        elif 'train2017' in annotation_name:
            split = 'train2017'
        elif 'val' in annotation_name:
            split = 'val2014'
        else:
            split = 'train2014'
        rel_path = f'{split}/{file_name}'
        image_path = image_root / rel_path
        if args.verify_images and not image_path.exists():
            missing += 1
            continue
        prompt_list = captions.get(item['id'], [])
        if not prompt_list:
            continue
        height, width = int(item['height']), int(item['width'])
        records.append({
            'path': rel_path,
            'prompt': random.choice(prompt_list),
            'height': height,
            'width': width,
            'ratio': height / max(width, 1),
        })

    (output_root / 'partition').mkdir(parents=True, exist_ok=True)
    with open(output_root / 'partition' / 'data_info.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)
    print(f'Wrote {len(records)} records to {output_root / "partition" / "data_info.json"}')
    if missing:
        print(f'Skipped {missing} missing images')


if __name__ == '__main__':
    main()
