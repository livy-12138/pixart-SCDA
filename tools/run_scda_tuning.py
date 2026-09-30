#!/usr/bin/env python3
"""Run reproducible SCDA tuning trials sequentially and collect metrics/samples."""
import csv, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'configs/PixArt_xl2_coco2014_scda.py'
OUT = ROOT / 'output/scda_tuning_512'
CKPT = ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'
GROUPS = ROOT / 'experiments/run_003_gate_v3/prompt_groups'

TRIALS = [
    dict(lr=5e-5, wd=3e-2, dropout=.10, bottleneck=64, gate=.05),
    dict(lr=1e-4, wd=3e-2, dropout=.10, bottleneck=64, gate=.05),
    dict(lr=2e-4, wd=3e-2, dropout=.10, bottleneck=64, gate=.05),
    dict(lr=1e-4, wd=1e-2, dropout=.10, bottleneck=64, gate=.05),
    dict(lr=1e-4, wd=6e-2, dropout=.10, bottleneck=64, gate=.05),
    dict(lr=1e-4, wd=3e-2, dropout=0., bottleneck=64, gate=.05),
    dict(lr=1e-4, wd=3e-2, dropout=.20, bottleneck=64, gate=.05),
    dict(lr=1e-4, wd=3e-2, dropout=.10, bottleneck=32, gate=.05),
    dict(lr=1e-4, wd=3e-2, dropout=.10, bottleneck=128, gate=.05),
    dict(lr=1e-4, wd=3e-2, dropout=.10, bottleneck=64, gate=.10),
]

def make_config(i, h):
    d = OUT / f'trial_{i:02d}'; d.mkdir(parents=True, exist_ok=True)
    cfg = d / 'config.py'
    text = f"exec(compile(open({str(BASE)!r}, encoding='utf-8').read(), {str(BASE)!r}, 'exec'))\n"
    text += f"optimizer = dict(type='AdamW', lr={h['lr']!r}, weight_decay={h['wd']!r}, eps=1e-10)\n"
    text += f"semantic_dropout = {h['dropout']!r}\nsemantic_adapter_dim = {h['bottleneck']}\ncondition_gate_scale = {h['gate']!r}\n"
    text += f"work_dir = {str(d)!r}\nload_from = {str(CKPT)!r}\nseed = 43\n"
    cfg.write_text(text)
    return d, cfg

def last_epoch_row(path):
    p = path / 'experiment_tables/epoch_summary.csv'
    if not p.exists(): return {}
    with p.open() as f: rows=list(csv.DictReader(f))
    return rows[-1] if rows else {}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = OUT / 'tuning_summary.csv'
    for i, h in enumerate(TRIALS, 1):
        d, cfg = make_config(i, h)
        done = d / 'checkpoints/epoch_10_step_2700.pth'
        if not done.exists():
            with (d/'train_console.log').open('w') as log:
                subprocess.run(['accelerate','launch','--num_processes','1','train_scripts/train.py',str(cfg)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if done.exists():
            sample_dir = d / 'samples'
            subprocess.run([sys.executable, 'tools/generate_scda_samples.py', '--checkpoint', str(done), '--groups-dir', str(GROUPS), '--output-dir', str(sample_dir), '--image-size','512','--steps','20','--cfg-scale','4.0','--seed','43','--semantic-conditioning','--semantic-adapter-dim',str(h['bottleneck'])], cwd=ROOT, check=False)
        metrics = last_epoch_row(d)
        # Keep hyperparameters authoritative; rename colliding logged fields.
        metrics = {k: v for k, v in metrics.items() if k not in h and k != 'trial'}
        row = dict(trial=i, **h, checkpoint=str(done) if done.exists() else '', **metrics)
        fields=list(row)
        write_header=not summary.exists()
        with summary.open('a', newline='') as f:
            w=csv.DictWriter(f, fieldnames=fields); w.writeheader() if write_header else None; w.writerow(row)
        (OUT/'latest_status.json').write_text(json.dumps(row, indent=2))
    print('all trials complete')

if __name__ == '__main__': main()
