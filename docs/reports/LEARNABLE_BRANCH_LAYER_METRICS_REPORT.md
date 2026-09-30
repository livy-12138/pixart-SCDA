# 可学习分支层指标报告

## 摘要

本报告汇总可学习分支层模型的总体 CLIP 指标、对象/属性存在代理指标与属性绑定代理指标，并与 Frozen PixArt baseline 和 step-2,500 蒸馏模型比较。

最终模型的总体 CLIP 为 `0.273597`，相对 baseline `0.273209` 提升 `+0.000388`（`+0.14%`）。属性存在率从 baseline 的 `70.59%` 提升至 `74.51%`，属性绑定准确率从 `66.67%` 提升至 `69.23%`。但绑定 margin 从 `0.01886` 降至 `0.01819`，且专项集仅含 13 个可交换属性案例，因此不能把该小幅差异视作统计显著的绑定增益。

## 评测对象

| 模型 | checkpoint | 训练步数 |
|---|---|---:|
| Frozen PixArt baseline | `output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth` | 0 |
| token-pair distill | `output/coco2017_token_pair_distill_gate008_sparse/checkpoints/epoch_1_step_2500.pth` | 2,500 |
| 可学习分支层 | `output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth` | 14,786 |

## 总体 CLIP 指标

统一评测协议：64 条固定 prompt，seed `43/44/45`，512px，DPM-Solver 20 steps，CFG 4.0，本地 CLIP ViT-B/32。每个模型生成 192 张图片。

| 模型 | seed 43 | seed 44 | seed 45 | CLIP 均值 | seed std | image std | 相对 baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen PixArt baseline | 0.273054 | 0.266827 | 0.279745 | 0.273209 | 0.005275 | 0.066340 | 0.00% |
| token-pair distill step 2,500 | 0.272990 | 0.267941 | 0.280095 | **0.273675** | 0.004985 | 0.064778 | **+0.17%** |
| 可学习分支层 step 14,786 | 0.273797 | 0.266893 | 0.280102 | 0.273597 | 0.005394 | 0.066059 | +0.14% |

### 总体 CLIP 解读

- 可学习分支层比 Frozen baseline 高 `0.000388`，但比其初始化蒸馏模型低 `0.000078`。
- 该差值远小于 seed 间波动（约 `0.005`），因此目前只能确认“未破坏 baseline 且保持微弱正向趋势”，不能确认完整训练优于 step-2,500 checkpoint。
- 当前整体最佳 checkpoint 仍是 step-2,500 蒸馏模型。

## 属性与绑定专项指标

### 协议

从固定 64 条 prompt 中人工筛出 17 条具有明确对象和属性的案例；其中 13 条可构造正确绑定文本与交换属性后的错误绑定文本。每个模型使用三个 seed，共有：

- 对象/属性存在：17 x 3 = 51 个图像案例。
- 属性绑定：13 x 3 = 39 个图像案例。

这些均是 CLIP contrastive proxy，不是目标检测或人工判定。具体案例定义在 `asset/attribute_binding_cases.csv`。

### 指标定义

| 指标 | 定义 | 趋势 |
|---|---|---|
| 对象存在率 | `CLIP(图, 对象描述) > CLIP(图, 干扰对象描述)` 的比例 | 越高越好 |
| 属性存在率 | `CLIP(图, 完整属性描述) > CLIP(图, 去属性中性描述)` 的比例 | 越高越好 |
| 绑定准确率 | `CLIP(图, 正确绑定) > CLIP(图, 交换绑定)` 的比例 | 越高越好 |
| 对象 margin | 对象描述分数减去干扰对象描述分数 | 越高越好 |
| 属性 margin | 完整属性描述分数减去中性描述分数 | 越高越好 |
| 绑定 margin | 正确绑定分数减去交换绑定分数 | 越高越好 |

### 汇总结果

| 模型 | 对象存在率 | 属性存在率 | 绑定准确率 | 对象 margin | 属性 margin | 绑定 margin |
|---|---:|---:|---:|---:|---:|---:|
| Frozen PixArt baseline | 86.27% | 70.59% | 66.67% | 0.04384 | 0.02447 | 0.01886 |
| token-pair distill step 2,500 | 84.31% | **74.51%** | 66.67% | 0.04370 | 0.02347 | **0.01912** |
| 可学习分支层 step 14,786 | 86.27% | **74.51%** | **69.23%** | **0.04436** | 0.02440 | 0.01819 |

### 属性绑定解读

可学习分支层模型将正确绑定胜率从 `26/39` 提升到 `27/39`，即 `66.67%` 提升到 `69.23%`。但它在已正确的案例中平均赢得不如 baseline 明显，因此 binding margin 略低。两个现象并不矛盾：

- 绑定准确率只计算正确绑定是否获胜，margin 的正负跨过 0 即计为一次改进。
- binding margin 计算每个案例的胜出幅度，少数原本 margin 较大的案例变小，就会降低整体均值。
- 此处差值仅 `-0.00067`，小于小规模测试集的预期波动，不能据此认定绑定能力下降。

按案例均值看，主要负向变化来自第 53 条 yarn beach（约 `-0.0045`）、第 58 条 shipwreck（约 `-0.0030`）和第 46 条 blue jay/macaron（约 `-0.0022`）；第 54 条 pumpkin chair 则约提升 `+0.0041`。

## 结论与限制

1. 总体 CLIP：可学习分支层略高于 baseline，但没有超过其 step-2,500 蒸馏初始化。
2. 属性存在：相对 baseline 有 `+3.92` 个百分点提升，说明保守的 token/pair 修正没有损害属性表达。
3. 属性绑定：代理准确率有 `+2.56` 个百分点提升，但只有 39 个观测，证据不足以得出强结论。
4. binding margin 略低不能单独解释为退化；它与准确率的提升同时出现，表明胜率与置信差距发生了不同方向的微小变化。
5. 本专项评测依赖 CLIP，可能受风格、背景、文本长度与 CLIP 对细粒度属性敏感性的影响。

## 建议的后续评测

- 构建 128-256 条短、受控的双对象双属性 prompt，覆盖颜色、材质、大小、纹理和形状。
- 每条 prompt 至少使用 3 个 seed；将绑定样本扩大到至少 384 个图像观测。
- 加入 GroundingDINO 或 OWL-ViT 定位对象，再对对象 crop 做属性分类，并人工复核子集。
- 报告 bootstrap 置信区间或配对显著性检验，避免把小于评测噪声的差异当作模型改进。

## 原始产物

- 总体 CLIP：`output/coco2017_eval_learnable_layers/clip_scores.csv`
- 总体 CLIP 按 seed 汇总：`output/coco2017_eval_learnable_layers/clip_scores_summary.csv`
- 属性/绑定逐案例结果：`output/attribute_binding_metrics.csv`
- 属性/绑定按模型和 seed 汇总：`output/attribute_binding_metrics_summary.csv`
- 案例定义：`asset/attribute_binding_cases.csv`
- 计算脚本：`tools/score_attribute_binding.py`
