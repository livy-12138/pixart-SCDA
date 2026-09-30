# SCDA/PixArt 文生图全指标评测报告

**评测日期：** 2026-09-08  
**评测对象：** Frozen PixArt baseline、COCO2017 pooled SCDA、full-span SCDA、token-pair SCDA、token-pair 蒸馏（Distill-2500）和可学习分支层（Learnable Layers）。  
**结论边界：** 本报告严格区分已实测指标和待执行指标。没有 FID/KID、LPIPS、GenEval、T2I-CompBench、VQA/开放词汇检测或人工标注结果，因此不对这些指标给出虚构数值，也不据此宣称 SCDA 的图像质量或组合语义能力获得全面提升。

## 1. 统一评测协议

- 固定提示词：`asset/samples.txt`，64 条复杂英文 prompt。
- 随机种子：43、44、45；每个模型共 192 张图。
- 推理：512x512，DPM-Solver 20 steps，CFG=4.0。
- 文本图像相似度模型：本地 CLIP ViT-B/32，长文本按模型最大长度截断。
- 属性绑定专项：`asset/attribute_binding_cases.csv` 的 17 条案例，其中 13 条提供可交换属性描述；每模型共 51 条对象/属性判断和 39 条绑定判断。
- 统计单位：总体 CLIP 为 image-prompt 对；专项准确率为 case-seed 对。现有实验仅有 3 个推理 seed，且同一 prompt 的不同 seed 并非独立语义样本，因此数值用于候选筛选，不应替代大规模独立测试集上的显著性结论。

## 2. 已完成的实测结果

### 2.1 总体文本-图像一致性：CLIPScore

| 模型 | CLIP 均值 | Seed 标准差 | 相对 Frozen baseline | 解读 |
|---|---:|---:|---:|---|
| Frozen PixArt baseline | 0.273209 | 0.006460 | 0.00% | 参照模型 |
| COCO2017 pooled SCDA | 0.247061 | 0.004233 | -9.57% | 显著退化趋势 |
| Full-span SCDA | 0.247550 | 0.005966 | -9.39% | 相对 root-only 仅边际改善 |
| Token-pair SCDA | 0.266105 | 0.000975 | -2.60% | 优于 pooled 方案，但仍低于 baseline |
| Token-pair + distillation | 0.273675 | 0.004985 | +0.17% | 基本保持 baseline 对齐 |
| Token-pair + learnable layers | 0.273597 | 0.005394 | +0.14% | 与蒸馏初始化近似，未证明额外增益 |

CLIPScore 衡量图像和完整 prompt 的嵌入相似度，适合检测大范围文本对齐退化，但对细粒度颜色、属性归属、数量和空间关系不够敏感。因此，C5/C6 的约 0.1% 差异不能视为总体质量显著提高；它说明保守蒸馏至少避免了前述 pooled SCDA 的大幅退化。

**原始产物：**

- `output/coco2017_eval/clip_scores*.csv`
- `output/coco2017_eval_fullspan_final/clip_scores*.csv`
- `output/coco2017_eval_token_pair/clip_scores*.csv`
- `output/coco2017_eval_distill2500/clip_scores*.csv`
- `output/coco2017_eval_learnable_layers/clip_scores*.csv`

### 2.2 对象、属性与对象-属性绑定：CLIP contrastive proxy（2026-09-09 全变体重评）

设 `s(I, C)` 为 CLIP 图文余弦相似度。每个案例使用正确描述、干扰对象描述、去属性描述和交换属性描述：

\[
\begin{aligned}
m_{obj} &= s(I,C_{object})-s(I,C_{distractor}),\\
m_{attr} &= s(I,C_{positive})-s(I,C_{neutral}),\\
m_{bind} &= s(I,C_{positive})-s(I,C_{swapped}).
\end{aligned}
\]

对象存在率、属性存在率和绑定准确率分别为对应 margin 大于 0 的比例。绑定准确率只统计 13 个 `binding=1` 案例；margin 为平均相似度差。

| 模型 | 对象存在率 | 属性存在率 | 绑定准确率 | 对象 margin | 属性 margin | 绑定 margin |
|---|---:|---:|---:|---:|---:|---:|
| Frozen PixArt baseline | 86.27% (44/51) | 70.59% (36/51) | 66.67% (26/39) | 0.04384 | 0.02447 | 0.01886 |
| Pooled SCDA | 80.39% (41/51) | **76.47% (39/51)** | 69.23% (27/39) | 0.04052 | **0.02517** | 0.01788 |
| Full-span SCDA | 80.39% (41/51) | 68.63% (35/51) | **74.36% (29/39)** | 0.04067 | 0.02332 | **0.02023** |
| Token-pair SCDA | 84.31% (43/51) | 66.67% (34/51) | 66.67% (26/39) | 0.04353 | 0.02217 | 0.01752 |
| Token-pair + distillation | 84.31% (43/51) | 74.51% (38/51) | 66.67% (26/39) | 0.04370 | 0.02347 | 0.01912 |
| Token-pair + learnable layers | **86.27% (44/51)** | 74.51% (38/51) | 69.23% (27/39) | **0.04436** | 0.02440 | 0.01819 |

为避免把同一 prompt 的三个 seed 视为独立语义样本，进一步以 17 个案例为聚类单元、固定随机种子、10,000 次 paired bootstrap 计算 95% CI。下表是相对 Frozen 的百分点差；CI 跨越 0 代表当前案例规模不足以支持确定的优劣结论。

| 模型 | 对象存在率差（95% CI） | 属性存在率差（95% CI） | 绑定准确率差（95% CI） |
|---|---:|---:|---:|
| Pooled SCDA | -5.88 pp [-17.65, +3.92] | +5.88 pp [-11.76, +25.49] | +2.56 pp [-20.51, +27.78] |
| Full-span SCDA | -5.88 pp [-19.61, +5.88] | -1.96 pp [-17.65, +15.69] | +7.69 pp [-16.67, +33.33] |
| Token-pair SCDA | -1.96 pp [-13.73, +7.84] | -3.92 pp [-17.65, +7.84] | +0.00 pp [-10.26, +10.26] |
| Token-pair + distillation | -1.96 pp [-7.84, +3.92] | +3.92 pp [+0.00, +9.80] | +0.00 pp [-7.14, +7.41] |
| Token-pair + learnable layers | +0.00 pp [-5.88, +5.88] | +3.92 pp [+0.00, +9.80] | +2.56 pp [+0.00, +8.33] |

**结果解读：** Learnable Layers 相比 Frozen 多正确 1 个绑定判断（27/39 对 26/39），并提升属性存在率 3.92 个百分点；Distillation 同样提升属性存在率 3.92 个百分点。它们的 binding 或 attribute CI 下界仍为 0，且所有数值都依赖同一个 CLIP 判定器，因此只能作为“值得扩大专项集验证”的正向信号。Full-span 在 binding proxy 上点估计最高，但总体 CLIPScore 低 9.39%，不改变停止 pooled/full-span 路线的结论。

**实现与产物：** [tools/score_attribute_binding.py](../tools/score_attribute_binding.py)、[tools/summarize_binding_comparison.py](../tools/summarize_binding_comparison.py)、`output/attribute_binding_metrics_all.csv`、`output/attribute_binding_metrics_all_summary.csv`、`output/attribute_binding_metrics_all_bootstrap.csv`。

## 3. 文生图指标覆盖状态

| 维度 | 推荐指标 | 当前状态 | 当前是否可作结论 | 补全所需输入 |
|---|---|---|---|---|
| 整体文本一致性 | CLIPScore | 已完成 | 可以，限于 64 prompt 的筛选集 | 扩展测试 prompt |
| 对象存在 | Open-vocabulary Object Recall/F1 | 未执行 | 不可以 | 目标对象清单、Grounding DINO 或 OWL-ViT |
| 属性存在 | Attribute Accuracy | CLIP proxy 已完成 | 仅可作代理结论 | 属性标签或 VQA 问题 |
| 属性绑定 | Binding Accuracy/Margin | CLIP proxy 已完成 | 仅可作代理结论 | 扩大交换属性案例，最好加 VQA/人工标注 |
| 空间/动作关系 | Relation Accuracy | 未执行 | 不可以 | 关系问题、检测框或人工标注 |
| 数量遵循 | Exact Count Accuracy、Count MAE | 未执行 | 不可以 | 带数量 prompt、检测/计数模型或人工标注 |
| 图像真实度/分布质量 | FID、KID | 未执行 | 不可以 | 足量生成图和独立真实 COCO 图像集 |
| 同 prompt 多样性 | LPIPS、MS-SSIM | 未执行 | 不可以 | 每个 prompt 至少 4 个独立 seed 的生成图 |
| 人类偏好 | ImageReward、HPS v2、PickScore | 未执行 | 不可以 | 对应预训练评估模型及其权重 |
| 问答式文本遵循 | TIFA、VQAScore | 未执行 | 不可以 | 可靠 VQA 模型、prompt 问题集 |
| 组合语义基准 | GenEval、T2I-CompBench | 未执行 | 不可以 | 官方 prompt、评测脚本、检测/VQA依赖 |
| 人工语义正确性 | 双人盲评、Cohen's kappa | 未执行 | 不可以 | 标注说明、独立标注员、仲裁规则 |
| 效率与稳定性 | 参数量、时间/图、显存、NaN/OOM | 部分训练记录 | 仅可描述单次训练 | 系统化 profiling 日志 |

## 4. 尚未完成指标的正式评测方案

### 4.1 FID 与 KID：图像真实度和分布质量

FID/KID 不评价文本遵循，而是比较生成图和真实图的视觉特征分布。应使用独立的 COCO2017 验证集真实图像作为 reference，不能用参与训练的图像或固定 64 条 prompt 的 192 张图作为正式 FID 报告。

- 生成规模：至少 10,000 张用于初步比较；30,000 张更适合作为正式 FID 报告。
- 公平性：各方法用同一 prompt 集、相同分辨率、相同采样步数与 CFG。
- 报告：FID-10k/30k（越低越好）、KID mean +/- std（越低越好）、生成数、reference split 和特征提取器版本。
- 当前阻塞：仓库未安装 `cleanfid`、`torchmetrics` 或 `torch-fidelity`，且未生成独立大规模评测样本。

### 4.2 LPIPS/MS-SSIM：同提示词多样性

每条 prompt 生成至少 4 个新 seed，计算所有 seed 两两图像距离；LPIPS 越高通常表示感知多样性更好，MS-SSIM 越低表示重复性更低。必须与 CLIPScore 联合报告，避免“随机噪声更大”被误判为更有多样性。

- 推荐规模：1,000 条 prompt x 4 seed。
- 报告：mean LPIPS、mean MS-SSIM、CLIPScore，并按 prompt bootstrap 置信区间。
- 当前阻塞：项目没有 `lpips` 依赖，也没有每 prompt >=4 seed 的大规模评测集。

### 4.3 对象、属性、关系和计数：GenEval/T2I-CompBench

这两类基准比总体 CLIP 更适合验证 SCDA 的贡献。

- **GenEval**：报告单对象、双对象、计数、颜色、位置和属性绑定子分数，以及宏平均分。
- **T2I-CompBench**：至少报告 color、shape、texture、spatial、non-spatial、complex 子任务；若方法重点为 object-attribute pair，必须单列 color/shape/texture 和 spatial/non-spatial。
- 公平性：每个 benchmark prompt 使用完全一致的 sampler、steps、CFG 和图像分辨率。不得从 benchmark prompt 中挑选少量容易样本。
- 当前阻塞：项目未包含官方 benchmark 数据、脚本、评估模型或生成结果。

### 4.4 检测/VQA 语义指标

建议在内部测试集补充可解释的结构化指标：

| 任务 | 输出指标 | 推荐判定器 |
|---|---|---|
| 对象是否出现 | Object Recall、Precision、F1、all-object success | Grounding DINO / OWL-ViT |
| 颜色/材质/形状 | Attribute Accuracy、macro-F1 | VQA 或区域属性分类器 |
| 对象-属性绑定 | pair-level Binding Accuracy、Binding Margin | 对象检测 + 区域 VQA；或正确/交换描述 VQAScore |
| 空间关系 | Relation Accuracy | 两对象检测框的几何规则 + VQA 复核 |
| 动作关系 | Action Relation Accuracy | VQA/人工盲评 |
| 数量 | Exact Count Accuracy、Count MAE | 检测器计数 + 人工抽检 |

相比纯 CLIP proxy，推荐的对象-属性绑定流程为：先以 Grounding DINO/OWL-ViT 定位 prompt 中的每个对象，再对每个目标区域询问对应属性，最后按对象对计算正确绑定比例。这样能区分“红色和蓝色都出现”与“红色确实属于球、蓝色确实属于立方体”。

### 4.5 人工盲评

自动指标应由人工评测校验。推荐从最终独立测试集分层抽取至少 200 个 prompt，每个方法使用相同 seed 图并打乱模型名；两名独立标注员给出以下二元标签：

1. 所有关键对象是否存在；
2. 每个属性是否属于正确对象；
3. 空间/动作关系是否正确；
4. 数量是否正确；
5. 整体图像质量是否可接受。

报告每项准确率、配对 bootstrap 95% CI、Cohen's kappa；分歧交由第三名标注员仲裁。对“属性绑定提高”的最终结论，应优先使用这项结果而不是 CLIP margin。

## 5. 最小可发表指标组合

在当前 SCDA 的研究目标下，建议最终论文至少包含：

| 目的 | 最小指标组合 |
|---|---|
| 整体质量 | FID/KID + CLIPScore |
| 文本遵循 | GenEval 或 TIFA/VQAScore + CLIPScore |
| 对象-属性绑定 | 扩展 Binding Accuracy/Margin + 区域 VQA/人工绑定准确率 |
| 关系/数量 | GenEval/T2I-CompBench 子分数或人工 Relation/Count Accuracy |
| 多样性 | LPIPS 或 MS-SSIM |
| 可用性 | 参数量、峰值显存、训练吞吐、推理时间/图、失败率 |

结果须报告均值、标准差和配对 bootstrap 95% CI。若 C5/C6 仅在属性绑定指标提升而 FID/CLIP 持平，正确表述应为“改善组合语义控制且保持整体质量”，而非“全面提升文生图质量”。

## 7. 本轮新增探索性计算结果

### 7.1 CLIP-space 分布距离与 KID 代理

由于当前环境中 Inception-v3 特征提取/协方差分解异常缓慢，未把 Inception FID 冒充为已完成结果。本轮使用项目已经加载的本地 CLIP ViT-B/32 图像嵌入，参考集为 COCO2017 `val2017` 前 1,000 张图，每个模型为 192 张生成图，计算 CLIP-space Fréchet 距离和 polynomial-kernel KID 代理。两者只能用于**同一批模型的探索性相对比较**，不是标准 Inception FID/KID，也不能与其他论文的 FID 数值直接比较。

| 模型 | 生成图数 | CLIP-space Fréchet ↓ | CLIP-space KID ↓ | KID std | 1-MS-SSIM proxy ↑ |
|---|---:|---:|---:|---:|---:|
| Frozen baseline | 192 | 0.573556 | 0.001012 | 0.000024 | 0.842314 |
| Pooled SCDA | 192 | 0.456120 | 0.000629 | 0.000023 | 0.791645 |
| Full-span SCDA | 192 | 0.460248 | 0.000641 | 0.000023 | 0.787495 |
| Token-pair SCDA | 192 | 0.491364 | 0.000685 | 0.000022 | 0.821260 |
| Token-pair + distillation | 192 | 0.567767 | 0.000997 | 0.000023 | 0.843215 |
| Learnable Layers | 192 | 0.571318 | 0.001005 | 0.000024 | 0.842481 |

`1-MS-SSIM proxy` 是同一 prompt、不同 seed 图像两两灰度 SSIM 的 `1-mean(SSIM)`，值越高表示跨 seed 感知差异越大；它不是 LPIPS。每个模型有 192 个跨 seed 图像对。完整逐模型结果保存在 `output/clip_distribution_metrics.csv`，计算脚本为 `tools/score_clip_distribution.py`。

**谨慎解读：** pooled/full-span SCDA 的 CLIP-space 距离较低且 `1-MS-SSIM` 较低，说明其生成图在该参考空间中更集中、跨 seed 变化更小；这不能解释为图像质量更好，反而与其较低的总体 CLIPScore 同时出现，可能反映语义多样性或生成退化。Distillation 与 Learnable Layers 的分布距离和多样性都接近 baseline，支持“保持 baseline 行为”的结论。

### 7.2 尚未完成的指标

本轮仍未完成标准 Inception FID/KID、LPIPS、Inception Score、GenEval、T2I-CompBench、TIFA/VQAScore、Grounding DINO/OWL-ViT 对象与关系指标以及人工盲评。原因是当前仓库没有对应评测权重/官方脚本，且标准 FID 需要至少 10,000 张生成图和统一 reference split。上述指标仍按第 3、4 节方案保留为后续正式评测任务。

## 8. 当前结论

1. Pooled SCDA 和 full-span SCDA 的 CLIP 分别比 Frozen baseline 低 9.57% 与 9.39%，不支持其总体文本图像一致性优于 baseline。
2. Token-pair 机制将退化缩小到 -2.60%；加入 teacher distillation 后，CLIP 均值达到 0.273675，数值上与 baseline 持平且略高。
3. Learnable Layers 的 CLIP 为 0.273597，未超过 Distill-2500 初始化；现有训练记录中 object/attribute/relation layer gate 也几乎未偏离初始化，因此没有证据证明层位置学习带来独立贡献。
4. 在 39 个 CLIP 代理绑定判断中，Learnable Layers 的胜率为 69.23%，高于 baseline 的 66.67%，但仅相当于多 1 个正确判断；该趋势需要更大专项集和非 CLIP 判定器复验。
5. 目前尚不能对真实度、多样性、关系、计数、跨基准组合语义或人类偏好作任何模型优劣结论。

## 9. 研发决策

自 2026-09-08 起，**停止 pooled SCDA 路线**：不再对 pooled residual、其分支组合、残差尺度、正则系数或 full-span mask 投入新的训练预算。历史配置、checkpoint 和评测 CSV 作为可复现负结果保留，不删除、不覆盖。后续资源集中在 token-pair + teacher distillation，并以 token-pair 内部的 role mask、pair bias 和层位置机制作为唯一消融主线。
