#!/usr/bin/env python3
"""Which metrics did the gate-optimised TP-SCDA actually improve?

Compares three models on both protocols:
  frozen        -- PixArt-alpha baseline
  tpscda        -- the paper's current TP-SCDA checkpoint (gate effectively off)
  gateopt       -- TP-SCDA with the gate controls from GATE_OPTIMIZATION.md

For the CompBench metrics the three are on the SAME protocol (300 prompts per
category, 1 image/prompt, same code state).  For the 64-prompt protocol we also
report the paired-bootstrap 95% interval against frozen, because the binding
proxy uses only 39 case-seed judgements and point estimates alone are not
enough to claim an improvement.
"""
import csv
import json
import os
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
CB = Path('/root/compbench_work/results')
OUT = ROOT / 'results/improvement_report'

CATS = ['color', 'shape', 'texture', 'spatial', '3d_spatial', 'numeracy',
        'non_spatial', 'complex']
CN = {'color': '颜色', 'shape': '形状', 'texture': '纹理', 'spatial': '2D空间',
      '3d_spatial': '3D空间', 'numeracy': '计数', 'non_spatial': '非空间',
      'complex': '复杂'}
PUB = {'color': 0.6690, 'shape': 0.4927, 'texture': 0.6477, 'spatial': 0.2064,
       '3d_spatial': 0.3901, 'numeracy': 0.5058, 'non_spatial': 0.3197, 'complex': 0.3433}
# The three attribute-binding categories (BLIP-VQA) are the ones TP-SCDA targets.
BINDING_CATS = {'color', 'shape', 'texture'}
NON_BINDING = [c for c in CATS if c not in BINDING_CATS]

L = []


def f(v):
    return 'N/A' if v is None else f'{v:.4f}'


def g(method, cat):
    p = CB / method / cat / 'summary.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get('official_mean_score')


L.append('# 门控改造后的 TP-SCDA：哪些指标提升了\n')
L.append('对比三个模型，**同一协议、同一代码状态**：\n')
L.append('- `frozen` —— PixArt-α 基线')
L.append('- `tpscda` —— 论文当前的 TP-SCDA（门控实际处于关闭状态，有效强度 0.0164）')
L.append('- `gateopt` —— 采用 `GATE_OPTIMIZATION.md` 门控控制的 TP-SCDA\n')

L.append('## 1. T2I-CompBench++ 官方八项\n')
hdr = f"| 类别 | 冻结 PixArt | TP-SCDA（现） | **TP-SCDA（门控改造）** | 改造 vs 现 | 改造 vs 冻结 | 公开 PixArt-α |"
L.append(hdr)
L.append('|---|---:|---:|---:|---:|---:|---:|')
tot = {'gateopt_vs_tpscda': [], 'gateopt_vs_frozen': [], 'binding': [], 'nonbinding': []}
for c in CATS:
    fz, tp, go = g('frozen', c), g('tpscda', c), g('gateopt', c)
    if None in (fz, tp, go):
        L.append(f"| {CN[c]} | {f(fz)} | {f(tp)} | {f(go)} | — | — | {PUB[c]:.4f} |")
        continue
    d1, d2 = go - tp, go - fz
    tot['gateopt_vs_tpscda'].append(d1)
    tot['gateopt_vs_frozen'].append(d2)
    tot['binding' if c in BINDING_CATS else 'nonbinding'].append(d2)
    L.append(f"| {CN[c]} | {f(fz)} | {f(tp)} | **{f(go)}** | {d1:+.4f} | {d2:+.4f} | {PUB[c]:.4f} |")
if tot['gateopt_vs_tpscda']:
    b = tot['binding']
    nb = tot['nonbinding']
    L.append('')
    L.append(f"**绑定类（颜色/形状/纹理）相对冻结的平均变化：{sum(b)/len(b):+.4f}**"
             if b else '')
    L.append(f"**非绑定类相对冻结的平均变化：{sum(nb)/len(nb):+.4f}**" if nb else '')
    L.append('')
    L.append('> 单张图的评测有噪声，且这里每个类别只有 300 个样本；'
             '判断"是否提升"应看绑定类是否一致为正、且非绑定类不退化，')
    L.append('> 而不是看单个类别的正负号。')

L.append('\n## 2. 64 提示词协议：CLIPScore 与属性绑定 proxy\n')
sp = ROOT / 'output/tables234_fixed/final_multiseed_clip_scores_summary.csv'
if sp.exists():
    from collections import defaultdict
    agg = defaultdict(list)
    for r in csv.DictReader(sp.open()):
        agg[r['model']].append(float(r['clip_mean']))
    L.append('| 模型 | CLIPScore | 相对冻结 |')
    L.append('|---|---:|---:|')
    base = sum(agg['baseline']) / len(agg['baseline']) if 'baseline' in agg else None
    for m, name in [('baseline', '冻结 PixArt'), ('learnable_layers', 'TP-SCDA（现）'),
                    ('token_pair', 'Token-pair'), ('gateopt', 'TP-SCDA（门控改造）')]:
        if m not in agg:
            L.append(f'| {name} | N/A | — |')
            continue
        v = sum(agg[m]) / len(agg[m])
        L.append(f'| {name} | {v:.6f} | {(100*(v-base)/base):+.2f}% |' if base else f'| {name} | {v:.6f} | — |')

bp = ROOT / 'output/tables234_fixed/attribute_binding_metrics_all_bootstrap.csv'
if bp.exists():
    L.append('\n**属性绑定 proxy 相对冻结的差（百分点，10 000 次 paired bootstrap，95% 区间）**\n')
    L.append('| 模型 | 对象存在 | 属性存在 | 绑定 |')
    L.append('|---|---:|---:|---:|')
    CN2 = {'baseline': '冻结 PixArt', 'learnable_layers': 'TP-SCDA（现）',
           'token_pair': 'Token-pair', 'gateopt': 'TP-SCDA（门控改造）'}
    for r in csv.DictReader(bp.open()):
        if r['model'] not in CN2:
            continue
        def cell(a, lo, hi):
            return f"{100*float(r[a]):+.2f} [{100*float(r[lo]):+.2f}, {100*float(r[hi]):+.2f}]"
        L.append(f"| {CN2[r['model']]} | {cell('object_delta_vs_baseline','object_delta_ci_low','object_delta_ci_high')} "
                 f"| {cell('attribute_delta_vs_baseline','attribute_delta_ci_low','attribute_delta_ci_high')} "
                 f"| {cell('binding_delta_vs_baseline','binding_delta_ci_low','binding_delta_ci_high')} |")
    L.append('\n> **区间下界 > 0 才算稳定提升。** 下界为 0 只能写"呈提升趋势"。')

L.append('\n## 3. 结论口径\n')
L.append('- 表 5 的三列（颜色/形状/纹理）是 TP-SCDA 的**目标指标**；'
         '非空间/计数等不应退化。')
L.append('- 若绑定类一致为正、非绑定类持平，即达到参数高效适配的合理预期。')
L.append('- 不要期望八项全面超过公开 PixArt-α —— 冻结基线就是原始 PixArt-α 权重'
         '（逐张量验证 maxdiff=0），差距来自主干与训练数据，不是适配方法。')

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'IMPROVEMENT_REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('wrote', OUT / 'IMPROVEMENT_REPORT.md')
print('\n'.join(L[:60]))
