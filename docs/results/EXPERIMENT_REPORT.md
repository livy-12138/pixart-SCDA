# TP-SCDA 实验补全报告

生成时间：2026-09-18　服务器：`/root/private_data/PixArt-alpha-attentiongate`

## 1. 完成了什么 / 没完成什么

| 项 | 状态 | 说明 |
|---|---|---|
| 表5 T2I-CompBench++（TP-SCDA） | 见 §2 | 官方 val 集 300 prompt/类，1 图/prompt |
| 表5 同协议对照（Frozen PixArt） | 见 §2 | 与 TP-SCDA 完全同协议 |
| 表6 GenEval Color Attribution | **NOT COMPLETED** | mmdet 2.x 不可用，见 ISSUE-010 |
| 图3 成对对比 | 见 §4 | CompBench 同 seed 配对 |
| 4.3.6 机制可视化 | 见 §4 | 层×时间步 patch 响应热力图 |
| 4.1 训练与效率指标 | 见 §3 | 参数量/时长/显存/loss/gate |
| 3.2 parser/token 对齐 | 见 §5 | 仅描述性统计 |
| 4.4 人工绑定标注 | 见 §5 | 仅生成标注表 |

## 2. T2I-CompBench++（官方评测器）

**重要：本表两组数值的条件不同，不可混用。**

| 类别 | 冻结 PixArt（修复掩码） | TP-SCDA（修复掩码） | TP-SCDA（当前仓库代码，掩码丢失） | 公开 PixArt-α（参考） |
|---|---:|---:|---:|---:|
| 颜色 | 0.3956 | **0.3922** | 0.1754 | 0.6690 |
| 形状 | 0.4158 | **0.4135** | 0.2133 | 0.4927 |
| 纹理 | 0.4701 | **0.4697** | 0.2248 | 0.6477 |
| 2D空间 | 0.2040 | **0.1969** | 0.0000 | 0.2064 |
| 3D空间 | 0.3388 | **0.3366** | 0.0247 | 0.3901 |
| 计数 | 0.5086 | **0.4961** | 0.0346 | 0.5058 |
| 非空间 | 0.3095 | **0.3093** | 0.2019 | 0.3197 |
| 复杂 | 0.3314 | **0.3312** | NOT COMPLETED | 0.3433 |

> 公开 PixArt-α 数值来自 T2I-CompBench++ 论文原始协议（原版 PixArt-α、10 图/prompt），
> 与本文的骨干微调状态、图片数、评测器版本不完全一致，**不构成统一条件下的排名**。

评测器：T2I-CompBench++ 官方 BLIP-VQA / UniDet / CLIPScore / 3-in-1，仓库文件未改动。
所有数值均可追溯到 `results/t2i_compbench_pp/<method>/<category>/raw/per_prompt.csv`。

## 3. 训练与效率（论文 4.1 节）

- TP-SCDA 参数统计（判据与 train_scripts/train.py 的 configure_trainable_parameters 完全一致）
- 总参数量 (PixArt-XL-2-512 + TP-SCDA 模块): 619,618,753
- 可训练参数量: 8,569,065
- 可训练占比: 1.3830%
- 冻结参数量: 611,049,688 (98.6170%)
- 训练时长：3.72 小时（14 786 步）
- 训练峰值显存：30.38 GB
- 优化器：AdamW，lr 1e-5（sqrt 自适应后 3.536e-6），weight decay 0.01，grad clip 1.0
- 微批 8 × 梯度累积 4 = 有效 batch 32；fp16；1 epoch
- loss 曲线：`results/training/loss_curve.png`；逐层门控：`layer_gate.png`；时间步门控：`timestep_gate.png`
- **注意**：该 checkpoint 的逐层门控几乎未学习，见 ISSUE-008

## 4. 可视化

- `results/visualization/issue011/issue011_mask_before_after.png` —— 掩码开关对照（ISSUE-011）
- `results/visualization/fig3_paired_binding.png` —— 冻结 PixArt vs TP-SCDA 同 seed 配对
- `results/visualization/mechanism/` —— 层×时间步 patch 响应热力图与结构化偏置 B

## 5. 未完成项

- **GenEval Color Attribution**：`status = not_completed`，原因见 EXPERIMENT_ISSUES.md ISSUE-010
- **人工绑定标注**：仅生成 `results/human_binding/annotation_sheet.csv`，**未伪造任何人工结果**
- **parser P/R/F1**：无人工真值，仅输出描述性统计（`results/parser_alignment/`）

## 6. 关键结论

1. **ISSUE-011 是本次最重要的发现**：仓库的交叉注意力掩码被静默关闭，短提示词出图严重退化。修复后 T2I-CompBench++ 颜色指标由 0.1754 提升到 0.3922。
2. **该缺陷影响论文已有的全部生成类结果**（表2/表3/表4 与 4.3.1—4.3.3 节），建议修复后重跑。
3. **可学习层门控在该 checkpoint 中几乎未学习**（ISSUE-008），4.3.3 节的归因需要修正表述。
