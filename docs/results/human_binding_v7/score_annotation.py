"""人工标注的统计：各模型三类比例、Cohen's kappa、95% 自助区间。

用法：python3 score_annotation.py <填好的 annotation_sheet.csv> [key.csv]

约定（见工作表 README）：
  · annotator1 / annotator2 两列，每列形如 `A:proper B:improper`；
  · A / B 是盲化后的图 A、图 B；
  · 取值 proper / improper / neglect（SynGen 三分类）。

统计口径：
  · 每条提示词贡献两个判定（图A、图B），分别归到两个模型；
  · kappa 在**两名标注者的逐判定配对**上计算（400 对）；
  · 置信区间按**提示词聚类**自助重采样（同一提示词的两个判定不独立）；
  · 两人不一致的条目单列计数，默认不计入比例，需第三人裁决或讨论定稿。
"""
import csv
import json
import random
import sys
from pathlib import Path

LABELS = ('proper', 'improper', 'neglect')
UNSURE = 'unsure'      # 填表页里的"无法判断"：单独计数，不计入比例与 kappa
DEFAULT_KEY = Path('/root/private_data/PixArt-alpha-attentiongate/results/'
                   'human_binding_v7/annotation_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv')


def parse_cell(cell):
    """'A:proper B:improper' -> {'A': 'proper', 'B': 'improper'}"""
    out = {}
    for tok in (cell or '').replace(',', ' ').split():
        if ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k, v = k.strip().upper(), v.strip().lower()
        if k in ('A', 'B') and v in LABELS + (UNSURE,):
            out[k] = v
    return out


def cohen_kappa(a, b):
    """两名标注者在配对标签上的一致性（标准 Cohen's kappa）。"""
    if not a:
        return None
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for lab in LABELS:
        pa = sum(1 for x in a if x == lab) / n
        pb = sum(1 for y in b if y == lab) / n
        pe += pa * pb
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    sheet = Path(sys.argv[1])
    key_p = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_KEY
    key = {int(r['item_id']): r for r in csv.DictReader(key_p.open())}
    rows = list(csv.DictReader(sheet.open()))

    pairs = []          # (item_id, model, label1, label2)
    skipped = []
    n_unsure = 0
    for r in rows:
        iid = int(r['item_id'])
        c1, c2 = parse_cell(r.get('annotator1')), parse_cell(r.get('annotator2'))
        for side in ('A', 'B'):
            model = key[iid][f'model_for_{side}']
            l1, l2 = c1.get(side), c2.get(side)
            if l1 == UNSURE or l2 == UNSURE:
                n_unsure += 1
            if l1 is None or l2 is None or l1 == UNSURE or l2 == UNSURE:
                skipped.append((iid, side, '缺标注/无法判断'))
                continue
            pairs.append((iid, model, l1, l2))

    if not pairs:
        print('没有可用的标注。工作表里的 annotator1 / annotator2 还是空的？')
        return

    # ---- 各模型的三类比例（只用两人一致的条目） --------------------------
    per_model = {}
    disagree = 0
    for iid, model, l1, l2 in pairs:
        d = per_model.setdefault(model, {k: 0 for k in LABELS})
        if l1 == l2:
            d[l1] += 1
        else:
            disagree += 1
    print(f'可用判定 {len(pairs)} 个；两人一致 {len(pairs)-disagree}、'
          f'不一致 {disagree}（不一致的暂不计入比例，需裁决）')
    if n_unsure:
        print(f'其中标了"无法判断"的 {n_unsure} 处（不计入比例与 kappa）')
    if skipped:
        print(f'未计入的判定 {len(skipped)} 处，例如 {skipped[:3]}')
    print()
    total_agree = sum(sum(v.values()) for v in per_model.values())
    for model, d in sorted(per_model.items()):
        n = sum(d.values())
        if not n:
            continue
        parts = '  '.join(f'{k}={d[k]} ({100*d[k]/n:.1f}%)' for k in LABELS)
        print(f'{model:22s} n={n:4d}  {parts}')

    # ---- Cohen's kappa（配对判定） ---------------------------------------
    a = [p[2] for p in pairs]
    b = [p[3] for p in pairs]
    k = cohen_kappa(a, b)
    # 按提示词聚类的自助区间
    by_item = {}
    for iid, _, l1, l2 in pairs:
        by_item.setdefault(iid, []).append((l1, l2))
    items = list(by_item.values())
    rng = random.Random(43)
    boots = []
    for _ in range(2000):
        sample = [x for it in (rng.choice(items) for _ in items) for x in it]
        kk = cohen_kappa([x for x, _ in sample], [y for _, y in sample])
        if kk is not None:
            boots.append(kk)
    lo, hi = (sorted(boots)[int(0.025*len(boots))],
              sorted(boots)[int(0.975*len(boots))]) if boots else (None, None)
    print()
    print(f"Cohen's kappa = {k:.3f}  95% 自助区间 [{lo:.3f}, {hi:.3f}]"
          f"（按提示词聚类，{len(items)} 条提示词 × 2 判定）")

    out = dict(n_judgements=len(pairs), n_disagree=disagree,
               per_model={m: v for m, v in per_model.items()},
               kappa=k, kappa_ci95=[lo, hi], n_items=len(items))
    p = sheet.with_name('annotation_scores.json')
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print('->', p)
    print()
    print('提醒：不一致的条目要先裁决（第三人判定或两人讨论）再定稿；'
          '定稿后把两人的列改成一致，重跑本脚本即可给出最终比例。')


if __name__ == '__main__':
    main()
