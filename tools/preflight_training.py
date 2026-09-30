#!/usr/bin/env python3
"""Fail-fast resource check for reproducible PixArt runs."""
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
cfg = ROOT / 'configs/PixArt_xl2_coco2014_gate.py'
checks = {
    'dataset_link_target': Path('/data/pixart/coco2014'),
    'dataset_metadata': Path('/data/pixart/coco2014/COCO2014Prepared10K/data_info.json'),
    'vae': ROOT / 'models/sd-vae-ft-ema/config.json',
    't5': ROOT / 'output/pretrained_models/t5_ckpts/t5-v1_1-xxl/config.json',
    'pretrained': ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth',
}
result = {name: {'path': str(path), 'exists': path.exists()} for name, path in checks.items()}
out = ROOT / 'experiments/preflight.json'; out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
if not all(item['exists'] for item in result.values()):
    print('PRECHECK FAILED: restore missing mounted assets before training.', file=sys.stderr)
    sys.exit(2)
