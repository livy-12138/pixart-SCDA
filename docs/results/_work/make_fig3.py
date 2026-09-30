#!/usr/bin/env python3
"""Figure 3: paired Frozen PixArt vs TP-SCDA generations at an identical seed.

Two sources of paired images are used, both of which are genuinely
seed-matched:

* CompBench runs -- the generator derives each image's noise from
  (base seed, category index, prompt index, repeat), so
  `<prompt>_<idx>.png` uses exactly the same noise for both methods.
* the existing 64-prompt evaluation sets (`output/coco2017_eval/*/seed_43`)
  for the complex-prompt row.

Writes results/visualization/... PNGs plus an index JSON that records the
prompt, the source, and the exact file used for each panel.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FONT = ROOT / 'results/_work/fonts/NotoSansCJKsc-Regular.otf'
if FONT.exists():
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(str(FONT))
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

CB = Path('/root/compbench_work/images')
OUT = ROOT / 'results/visualization'

# (group, category, prompt, chinese gloss, binding note)
CASES = [
    ('binding', 'color', 'a green bench and a blue bowl',
     '绿色长凳 + 蓝色碗', '两对象各自带颜色属性'),
    ('binding', 'color', 'a blue bench and a green bowl',
     '蓝色长凳 + 绿色碗', '对象相同、颜色互换（属性交换对照）'),
    ('binding', 'texture', 'a plastic toy and a glass bottle',
     '塑料玩具 + 玻璃瓶', '两种材质属性分别绑定到两个对象'),
    ('binding', 'texture', 'a metallic desk lamp and a fluffy sweater',
     '金属台灯 + 毛绒毛衣', '材质属性绑定'),
    ('binding', 'shape', 'an oblong cucumber and a teardrop plum',
     '长椭圆黄瓜 + 水滴形李子', '形状属性绑定'),
    ('binding', 'shape', 'a cubic ice cube and a spherical ice bucket',
     '立方体冰块 + 球形冰桶', '形状属性绑定'),
    ('complex', 'complex', None, None, None),
]


def find_compbench(method, category, prompt):
    d = CB / method / category
    if not d.is_dir():
        return None
    hits = sorted(d.glob(f'{prompt}_*.png'))
    return hits[0] if hits else None


def find_existing(method, index):
    """Existing 64-prompt eval set; method in {frozen, tpscda}."""
    sub = {'frozen': 'baseline', 'tpscda': 'learnable_layers'}[method]
    base = ROOT / 'output/coco2017_eval_learnable_layers' if method == 'tpscda' \
        else ROOT / 'output/coco2017_eval'
    imgs = sorted(base.glob(f'{sub}/seed_43/*/*.png'))
    return imgs[index] if index < len(imgs) else None


def load(p):
    return np.asarray(Image.open(p).convert('RGB'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--complex-index', type=int, default=0)
    ap.add_argument('--complex-prompt-index', type=int, default=0)
    args = ap.parse_args()

    samples = [l.strip() for l in (ROOT / 'asset/samples.txt').read_text().splitlines() if l.strip()]

    rows = []
    for group, cat, prompt, gloss, note in CASES:
        if group == 'complex':
            f = find_existing('frozen', args.complex_index)
            t = find_existing('tpscda', args.complex_index)
            if not (f and t):
                continue
            rows.append(dict(group='complex', prompt=samples[args.complex_prompt_index],
                             gloss='复杂长提示词（64 条固定提示词集）',
                             note='来自已有 64 提示词评测集，seed 43',
                             frozen=f, tpscda=t))
        else:
            f = find_compbench('frozen', cat, prompt)
            t = find_compbench('tpscda', cat, prompt)
            if not (f and t):
                print(f'skip (images not generated yet): [{cat}] {prompt}', flush=True)
                continue
            rows.append(dict(group=group, prompt=prompt, gloss=gloss, note=note,
                             frozen=f, tpscda=t))
    if not rows:
        print('no paired images available yet')
        return

    fig, axes = plt.subplots(len(rows), 2,
                             figsize=(7.4, 3.7 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])
    for r, row in enumerate(rows):
        a, b = axes[r]
        a.imshow(load(row['frozen']))
        a.set_xticks([]); a.set_yticks([])
        b.imshow(load(row['tpscda']))
        b.set_xticks([]); b.set_yticks([])
        if r == 0:
            a.set_title('冻结 PixArt', fontsize=12, pad=8)
            b.set_title('TP-SCDA（本文）', fontsize=12, pad=8)
        p = row['prompt'] if len(row['prompt']) < 60 else row['prompt'][:57] + '…'
        a.set_ylabel(f'「{row["gloss"]}」\n{p}\n{row["note"]}',
                     fontsize=7.5, rotation=0, ha='right', va='center', labelpad=6)
    fig.tight_layout(rect=(0.16, 0, 1, 0.99))
    fig.savefig(OUT / 'fig3_paired_binding.png', dpi=200)
    plt.close(fig)
    print('wrote fig3_paired_binding.png')

    idx = [dict(group=x['group'], prompt=x['prompt'], gloss=x['gloss'], note=x['note'],
                frozen_image=str(x['frozen']), tpscda_image=str(x['tpscda']))
           for x in rows]
    (OUT / 'fig3_index.json').write_text(json.dumps(dict(
        description='Frozen PixArt vs TP-SCDA, identical seed per image '
                    '(CompBench per-image deterministic seeds; seed 43 for the '
                    '64-prompt set). Binding correctness is to be judged by the '
                    'reader / annotator -- no automatic claim is made here.',
        panels=idx), indent=2, ensure_ascii=False))
    print('wrote fig3_index.json')


if __name__ == '__main__':
    main()
