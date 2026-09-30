#!/usr/bin/env python3
"""Build the human binding-annotation sheet (paper section 4.4).

SynGen's three-way judgement (Proper / Improper / Entity Neglect) needs human
annotators.  This script only produces the worksheet plus a blinding key -- it
does NOT fabricate any annotation, and no Cohen's kappa is computed here.

Layout: one row per (item, blinded method).  200 prompts x 2 methods = 400 rows.
The method is blinded as A/B with a per-row random assignment; the mapping is
written to a separate key file that annotators must not open until done.
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
IMG = Path('/root/compbench_work/images')
OUT = ROOT / 'results/human_binding'
OUT.mkdir(parents=True, exist_ok=True)
N_PROMPTS = 200
CATS = ['color', 'shape', 'texture']


def main():
    rng = random.Random(20260918)
    items = []
    for cat in CATS:
        prompts = [l.strip() for l in
                   (Path('/tmp/cb/T2I-CompBench/examples/dataset') /
                    f'{cat}_val.txt').read_text().splitlines() if l.strip()]
        for p in prompts:
            f = sorted((IMG / 'tpscda' / cat).glob(f'{p}_*.png'))
            b = sorted((IMG / 'frozen' / cat).glob(f'{p}_*.png'))
            if f and b:
                items.append((cat, p, str(b[0].resolve()), str(f[0].resolve())))
            if len(items) >= N_PROMPTS:
                break
        if len(items) >= N_PROMPTS:
            break

    rows, key = [], []
    for i, (cat, prompt, frozen_img, tpscda_img) in enumerate(items):
        flip = rng.random() < 0.5
        a_img, b_img = (tpscda_img, frozen_img) if flip else (frozen_img, tpscda_img)
        a_m, b_m = ('tpscda', 'frozen') if flip else ('frozen', 'tpscda')
        rows.append(dict(item_id=i, category=cat, prompt=prompt,
                         image_A=a_img, image_B=b_img,
                         annotator1='', annotator2='', notes=''))
        key.append(dict(item_id=i, category=cat, prompt=prompt,
                        A=a_m, B=b_m, A_path=a_img, B_path=b_img))

    fields = ['item_id', 'category', 'prompt', 'image_A', 'image_B',
              'annotator1', 'annotator2', 'notes']
    with open(OUT / 'annotation_sheet.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(OUT / 'annotation_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv', 'w',
              newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0].keys()))
        w.writeheader(); w.writerows(key)

    (OUT / 'README.md').write_text(f"""# 人工绑定标注（论文 4.4 节）

**本目录不含任何人工标注结果。** 标注必须由两名标注者完成。

- `annotation_sheet.csv` —— 工作表，{len(rows)} 行（{len(items)} 条提示词 × 2 个方法，方法已盲化为 A/B）
- `annotation_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv` —— 盲化对照表，**标注完成前不要打开**
- 图片路径为服务器绝对路径：`{IMG}`

## 标注说明（SynGen 三分类）

对每一行，给定提示词与图 A、图 B，分别判断：

| 取值 | 含义 |
|---|---|
| `proper` | 提示词中的对象都出现，且属性绑定到正确的对象 |
| `improper` | 对象都出现，但至少一个属性绑定到了错误的对象 |
| `neglect` | 至少一个提示词中的对象没有出现（实体遗漏） |

在 `annotator1` / `annotator2` 两列分别填写；`notes` 可记录疑难案例。

## 统计（标注完成后）

本脚本**不计算** Proper / Improper / Entity Neglect 比例与 Cohen's kappa，
也不生成 95% 置信区间——这些必须在真实标注完成后由作者计算。
""", encoding='utf-8')
    print(f'wrote {len(rows)} rows for {len(items)} prompts -> {OUT}')


if __name__ == '__main__':
    main()
