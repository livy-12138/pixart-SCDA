#!/usr/bin/env python3
"""Generate results/EXPERIMENT_REPORT.md from the evaluator outputs.

Pulls the official T2I-CompBench++ numbers from both result roots:
  * /root/compbench_work/results        -- mask-restored run (ISSUE-011 fix)
  * /root/compbench_work/results_buggy  -- mask-disabled run (current repo code)
so the report shows both the corrected values and the impact of the bug.
"""
import json
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
FIXED = Path('/root/compbench_work/results')
BUGGY = Path('/root/compbench_work/results_buggy')
ORDER = ['color', 'shape', 'texture', 'spatial', '3d_spatial', 'numeracy',
         'non_spatial', 'complex']
CN = {'color': '颜色', 'shape': '形状', 'texture': '纹理', 'spatial': '2D空间',
      '3d_spatial': '3D空间', 'numeracy': '计数', 'non_spatial': '非空间',
      'complex': '复杂'}
PUBLIC_PIXART = {'color': 0.6690, 'shape': 0.4927, 'texture': 0.6477,
                 'spatial': 0.2064, '3d_spatial': 0.3901, 'numeracy': 0.5058,
                 'non_spatial': 0.3197, 'complex': 0.3433}


def load(root, method, cat):
    p = Path(root) / method / cat / 'summary.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(d):
    if d is None:
        return 'NOT COMPLETED'
    v = d.get('official_mean_score')
    if v is None:
        return 'NOT COMPLETED'
    return f'{v:.4f}'


def main():
    L = []
    L.append('# TP-SCDA 实验补全报告\n')
    L.append('生成时间：2026-09-18　服务器：`/root/private_data/PixArt-alpha-attentiongate`\n')

    L.append('## 1. 完成了什么 / 没完成什么\n')
    L.append('| 项 | 状态 | 说明 |')
    L.append('|---|---|---|')
    L.append('| 表5 T2I-CompBench++（TP-SCDA） | 见 §2 | 官方 val 集 300 prompt/类，1 图/prompt |')
    L.append('| 表5 同协议对照（Frozen PixArt） | 见 §2 | 与 TP-SCDA 完全同协议 |')
    L.append('| 表6 GenEval Color Attribution | **NOT COMPLETED** | mmdet 2.x 不可用，见 ISSUE-010 |')
    L.append('| 图3 成对对比 | 见 §4 | CompBench 同 seed 配对 |')
    L.append('| 4.3.6 机制可视化 | 见 §4 | 层×时间步 patch 响应热力图 |')
    L.append('| 4.1 训练与效率指标 | 见 §3 | 参数量/时长/显存/loss/gate |')
    L.append('| 3.2 parser/token 对齐 | 见 §5 | 仅描述性统计 |')
    L.append('| 4.4 人工绑定标注 | 见 §5 | 仅生成标注表 |')
    L.append('')
    L.append('## 2. T2I-CompBench++（官方评测器）\n')
    L.append('**重要：本表两组数值的条件不同，不可混用。**\n')
    L.append(f'| 类别 | 冻结 PixArt（修复掩码） | TP-SCDA（修复掩码） | TP-SCDA（当前仓库代码，掩码丢失） | 公开 PixArt-α（参考） |')
    L.append('|---|---:|---:|---:|---:|')
    for c in ORDER:
        f_frozen = fmt(load(FIXED, 'frozen', c))
        f_tp = fmt(load(FIXED, 'tpscda', c))
        b_tp = fmt(load(BUGGY, 'tpscda', c))
        pub = f"{PUBLIC_PIXART[c]:.4f}"
        L.append(f'| {CN[c]} | {f_frozen} | **{f_tp}** | {b_tp} | {pub} |')
    L.append('')
    L.append('> 公开 PixArt-α 数值来自 T2I-CompBench++ 论文原始协议（原版 PixArt-α、10 图/prompt），')
    L.append('> 与本文的骨干微调状态、图片数、评测器版本不完全一致，**不构成统一条件下的排名**。\n')
    L.append('评测器：T2I-CompBench++ 官方 BLIP-VQA / UniDet / CLIPScore / 3-in-1，仓库文件未改动。')
    L.append('所有数值均可追溯到 `results/t2i_compbench_pp/<method>/<category>/raw/per_prompt.csv`。\n')

    L.append('## 3. 训练与效率（论文 4.1 节）\n')
    tp = ROOT / 'results/training'
    if (tp / 'trainable_params.txt').exists():
        txt = (tp / 'trainable_params.txt').read_text().splitlines()
        for line in txt[:6]:
            if line.strip():
                L.append(f'- {line}')
    ck = tp / 'checkpoint_info.json'
    if ck.exists():
        d = json.loads(ck.read_text())
        L.append(f"- 训练时长：{d['training_wall_clock_hours']:.2f} 小时（14 786 步）")
        L.append(f"- 训练峰值显存：{d['gpu_peak_memory_during_training_gb']:.2f} GB")
    L.append('- 优化器：AdamW，lr 1e-5（sqrt 自适应后 3.536e-6），weight decay 0.01，grad clip 1.0')
    L.append('- 微批 8 × 梯度累积 4 = 有效 batch 32；fp16；1 epoch')
    L.append('- loss 曲线：`results/training/loss_curve.png`；逐层门控：`layer_gate.png`；'
             '时间步门控：`timestep_gate.png`')
    L.append('- **注意**：该 checkpoint 的逐层门控几乎未学习，见 ISSUE-008\n')

    L.append('## 4. 可视化\n')
    L.append('- `results/visualization/issue011/issue011_mask_before_after.png` —— '
             '掩码开关对照（ISSUE-011）')
    L.append('- `results/visualization/fig3_paired_binding.png` —— 冻结 PixArt vs TP-SCDA 同 seed 配对')
    L.append('- `results/visualization/mechanism/` —— 层×时间步 patch 响应热力图与结构化偏置 B\n')

    L.append('## 5. 未完成项\n')
    L.append('- **GenEval Color Attribution**：`status = not_completed`，原因见 EXPERIMENT_ISSUES.md ISSUE-010')
    L.append('- **人工绑定标注**：仅生成 `results/human_binding/annotation_sheet.csv`，'
             '**未伪造任何人工结果**')
    L.append('- **parser P/R/F1**：无人工真值，仅输出描述性统计（`results/parser_alignment/`）\n')

    L.append('## 6. 关键结论\n')
    L.append('1. **ISSUE-011 是本次最重要的发现**：仓库的交叉注意力掩码被静默关闭，'
             '短提示词出图严重退化。修复后 T2I-CompBench++ 颜色指标由 '
             f'{fmt(load(BUGGY, "tpscda", "color"))} 提升到 '
             f'{fmt(load(FIXED, "tpscda", "color"))}。')
    L.append('2. **该缺陷影响论文已有的全部生成类结果**（表2/表3/表4 与 4.3.1—4.3.3 节），'
             '建议修复后重跑。')
    L.append('3. **可学习层门控在该 checkpoint 中几乎未学习**（ISSUE-008），'
             '4.3.3 节的归因需要修正表述。')

    out = ROOT / 'results/EXPERIMENT_REPORT.md'
    out.write_text('\n'.join(L) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
