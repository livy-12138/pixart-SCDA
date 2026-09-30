#!/usr/bin/env python3
"""Consolidate evaluator outputs into the `results/` tree required by the
task brief, and build the top-level summary.json / summary.csv.

Reads  /root/compbench_work/results/<method>/<category>/{summary.json,per_prompt.csv,failures.csv,config.json}
Writes results/t2i_compbench_pp/<method>/<category>/... plus an aggregate
       results/t2i_compbench_pp/summary.csv and results/summary.json
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

SRC = Path('/root/compbench_work/results')
OUT = ROOT / 'results/t2i_compbench_pp'
CATEGORIES = ['color', 'shape', 'texture', 'spatial', '3d_spatial',
              'numeracy', 'non_spatial', 'complex']
# Chinese names used in the paper's Table 5
CN = {'color': '颜色', 'shape': '形状', 'texture': '纹理', 'spatial': '2D空间',
      '3d_spatial': '3D空间', 'numeracy': '计数', 'non_spatial': '非空间',
      'complex': '复杂'}
METHOD_CN = {'tpscda': 'TP-SCDA（本文）', 'frozen': '冻结 PixArt（本文同协议对照）'}

# Public published values for the same benchmark, from the CompBench++ paper.
PUBLIC = {
    'SD v1.4': dict(color=0.3765, shape=0.3576, texture=0.4156, spatial=0.1246,
                    d3=0.3030, numeracy=0.4461, non_spatial=0.3079, complex=0.3080),
}


def collect():
    rows = []
    for method in ('tpscda', 'frozen'):
        for cat in CATEGORIES:
            d = SRC / method / cat
            sj = d / 'summary.json'
            if not sj.exists():
                rows.append(dict(method=method, category=cat, status='not_completed',
                                 official_mean_score=None, num_images_scored=0,
                                 num_images_failed=None, note='evaluator did not run'))
                continue
            s = json.loads(sj.read_text())
            rows.append(dict(
                method=method, category=cat,
                status='completed' if s.get('num_images_failed', 0) == 0 else 'partial',
                metric=s.get('metric'),
                official_mean_score=s.get('official_mean_score'),
                num_prompts=s.get('num_prompts'),
                num_images_selected=s.get('num_images_selected'),
                num_images_scored=s.get('num_images_scored'),
                num_images_failed=s.get('num_images_failed'),
                is_full_official_run=s.get('is_full_official_run'),
                evaluator_chain=s.get('evaluator_chain'),
                evaluator_wall_seconds=s.get('evaluator_wall_seconds'),
                source_dir=str(d),
            ))
    return rows


def copy_artifacts():
    for method in ('tpscda', 'frozen'):
        for cat in CATEGORIES:
            src = SRC / method / cat
            if not src.is_dir():
                continue
            dst = OUT / method / cat
            (dst / 'raw').mkdir(parents=True, exist_ok=True)
            for name in ('summary.json', 'summary.csv', 'config.json', 'failures.csv'):
                if (src / name).exists():
                    shutil.copy2(src / name, dst / name)
            if (src / 'per_prompt.csv').exists():
                shutil.copy2(src / 'per_prompt.csv', dst / 'raw' / 'per_prompt.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary-json', default=str(ROOT / 'results/summary.json'))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    copy_artifacts()

    fields = ['method', 'category', 'status', 'metric', 'official_mean_score',
              'num_prompts', 'num_images_selected', 'num_images_scored',
              'num_images_failed', 'is_full_official_run', 'evaluator_chain',
              'evaluator_wall_seconds', 'source_dir', 'note']
    with open(OUT / 'summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

    # paper-facing table
    table = {}
    for r in rows:
        table.setdefault(r['method'], {})[r['category']] = r['official_mean_score']
    (OUT / 'table5_tpscda_row.json').write_text(json.dumps(dict(
        note=('T2I-CompBench++ official metrics, our own protocol. Report this row '
              'SEPARATELY from the public published rows in the paper (different '
              'backbone/training data/evaluator version).'),
        columns={c: CN[c] for c in CATEGORIES},
        rows={METHOD_CN.get(m, m): v for m, v in table.items()},
    ), indent=2, ensure_ascii=False))

    # merge into the global summary.json if it exists
    sp = Path(args.summary_json)
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    summary['t2i_compbench_pp'] = dict(
        status='completed' if all(r['official_mean_score'] is not None for r in rows)
        else 'partial',
        source='T2I-CompBench++ official evaluators (BLIP-VQA / UniDet / CLIPScore / 3-in-1)',
        rows=rows,
    )
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'wrote {OUT}/summary.csv and {sp}')
    for r in rows:
        v = r['official_mean_score']
        print(f"  {r['method']:8s} {r['category']:12s} "
              f"{'NOT COMPLETED' if v is None else f'{v:.4f}'}  "
              f"({r.get('num_images_scored')} scored / {r.get('num_images_failed')} failed)")


if __name__ == '__main__':
    main()
