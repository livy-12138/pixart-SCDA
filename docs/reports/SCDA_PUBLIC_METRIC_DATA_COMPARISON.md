# SCDA 与相似方法的公开评价指标数据对比

**用途：** 汇总本项目已实测数值和相似文献公开报告的数值，服务于论文表格和实验结论撰写。外部方法没有在本地重新部署；所有外部数字均保留原论文的数据集、模型、采样器和额外输入条件。

## 1. 结论先行

- 本项目最可靠的统一数字是 Frozen PixArt、token-pair SCDA、蒸馏和可学习层模型在同一 64-prompt/3-seed 协议下的 CLIP 与 CLIP binding proxy。
- SynGen 提供了与 SCDA 研究问题最接近的公开人工绑定数据：在其三个数据集上 Proper Binding 为 63.68--94.76，且明显优于其表中的 A&E、Structured Diffusion 和 Stable Diffusion；但这些不是 PixArt 协议，不能直接宣称 SCDA 超过或落后于 SynGen。
- BoxDiff 的 AP/AP50/AP75 数值评价的是“给定 box 后的空间满足度”，属于额外空间条件上限，不是纯文本 object-attribute binding 对比。
- A&E 官方仓库确认了 image-CLIP 和 BLIP-caption-CLIP 两类指标，但没有在 README 中给出可直接迁移的统一总分；这里不填猜测数值。

## 2. 本项目统一评测结果

协议：64 条固定 prompt、seed 43/44/45、512x512、DPM-Solver 20 steps、CFG 4.0、本地 CLIP ViT-B/32。每个模型生成 192 张图。

| 方法 | 输入/基座 | CLIP ↑ | 相对 Frozen | 对象存在 proxy ↑ | 属性存在 proxy ↑ | 绑定 proxy ↑ |
|---|---|---:|---:|---:|---:|---:|
| Frozen PixArt | 纯文本 / PixArt-XL-2 | 0.273209 | 0.00% | 86.27% (44/51) | 70.59% (36/51) | 66.67% (26/39) |
| Pooled SCDA | 文本 + pooled residual / PixArt-XL-2 | 0.247061 | -9.57% | 80.39% (41/51) | **76.47% (39/51)** | 69.23% (27/39) |
| Full-span SCDA | 文本 + full-span residual / PixArt-XL-2 | 0.247550 | -9.39% | 80.39% (41/51) | 68.63% (35/51) | **74.36% (29/39)** |
| Token-pair SCDA | 文本 + token-pair bias / PixArt-XL-2 | 0.266105 | -2.60% | 84.31% (43/51) | 66.67% (34/51) | 66.67% (26/39) |
| Token-pair + distillation | 文本 + token-pair + teacher / PixArt-XL-2 | **0.273675** | **+0.17%** | 84.31% (43/51) | **74.51% (38/51)** | 66.67% (26/39) |
| Token-pair + learnable layers | 文本 + token-pair + learnable layer gate / PixArt-XL-2 | 0.273597 | +0.14% | 86.27% (44/51) | **74.51% (38/51)** | **69.23% (27/39)** |

当前绑定数值是 CLIP contrastive proxy：正确绑定描述的 CLIP 分数高于交换属性描述即计为正确；不是检测器、VQA 或人工准确率。2026-09-09 对六个变体按 17 个案例做 paired bootstrap 后，Learnable Layers 相比 baseline 的 binding 差为 +2.56 pp，95% CI 为 [0.00, +8.33] pp；样本仍很小，不能视作统计显著提升。Full-span 虽有 74.36% binding proxy，但总体 CLIP 低 9.39%，故不恢复该路线。

## 3. 纯文本绑定方法的公开数值

### 3.1 SynGen、Attend-and-Excite、Structured Diffusion

SynGen 论文在三个数据集上进行人工 fine-grained evaluation。Proper Binding 是正确属性-对象映射比例，Improper Binding 是错误映射比例，Entity Neglect 是 prompt 中实体未出现的比例。

| 数据集 | 方法 | Proper Binding ↑ | Improper Binding ↓ | Entity Neglect ↓ |
|---|---|---:|---:|---:|
| A&E (177) | SynGen | **94.76** | 23.81 | 2.82 |
| A&E (177) | Attend-and-Excite | 81.90 | 63.81 | **1.41** |
| A&E (177) | Structured Diffusion | 55.71 | 67.62 | 21.13 |
| A&E (177) | Stable Diffusion | 59.05 | 68.57 | 20.56 |
| DVMP (200) | SynGen | **74.90** | **19.49** | 16.26 |
| DVMP (200) | Attend-and-Excite | 52.47 | 31.64 | **10.77** |
| DVMP (200) | Structured Diffusion | 48.73 | 30.57 | 28.46 |
| DVMP (200) | Stable Diffusion | 47.80 | 30.44 | 26.22 |
| ABC-6K (200) | SynGen | **63.68** | **14.37** | 34.41 |
| ABC-6K (200) | Attend-and-Excite | 56.26 | 26.43 | **33.18** |
| ABC-6K (200) | Structured Diffusion | 51.47 | 29.52 | 34.57 |
| ABC-6K (200) | Stable Diffusion | 52.70 | 27.20 | 36.57 |

SynGen 的人工多数投票概念分离率/视觉偏好也公开报告如下：

| 数据集 | 方法 | Concept separation ↑ | Visual appeal ↑ |
|---|---|---:|---:|
| A&E | SynGen | **38.42** | **37.85** |
| A&E | Attend-and-Excite | 18.08 | 18.65 |
| A&E | Structured Diffusion | 4.52 | 4.52 |
| A&E | Stable Diffusion | 1.69 | 2.26 |
| DVMP | SynGen | **24.84** | **16.00** |
| DVMP | Attend-and-Excite | 13.33 | 12.17 |
| DVMP | Structured Diffusion | 4.33 | 7.83 |
| DVMP | Stable Diffusion | 3.83 | 7.17 |
| ABC-6K | SynGen | **28.00** | **18.34** |
| ABC-6K | Attend-and-Excite | 11.17 | 10.00 |
| ABC-6K | Structured Diffusion | 5.83 | 6.33 |
| ABC-6K | Stable Diffusion | 4.83 | 7.83 |

补充数据：SynGen 论文报告 Stable Diffusion、A&E、SynGen 的平均生成时间分别为 4.0、8.8、9.76 s/image；其 phrase-to-image CLIP 自动评测与人工多数选择的一致率仅 43.5%，说明普通 CLIP 不应替代绑定人工评测。

### 3.2 GenEval：六项对象级组合指标（更多基础模型）

GenEval 使用 Mask2Former 对生成图做对象级验证，并针对颜色使用额外的颜色分类器。其 `color attribution` 要求两个对象及其颜色均正确，因此是与 object-attribute binding 最接近的自动列；`two object` 能补充实体遗漏，`position` 和 `counting` 则覆盖 SCDA 当前未测的关系与数量能力。每个分数均在 0--1 范围内，越高越好。

| 模型 | 总分 | 单对象 | 双对象 | 计数 | 单对象颜色 | 位置关系 | 颜色-对象绑定 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP retrieval（检索基线） | 0.35 | 0.89 | 0.22 | 0.37 | 0.62 | 0.03 | 0.00 |
| minDALL-E | 0.23 | 0.73 | 0.11 | 0.12 | 0.37 | 0.02 | 0.01 |
| Stable Diffusion v1.5 | 0.43 | 0.97 | 0.38 | 0.35 | 0.76 | 0.04 | 0.06 |
| Stable Diffusion v2.1 | 0.50 | 0.98 | 0.51 | 0.44 | 0.85 | 0.07 | 0.17 |
| Stable Diffusion XL | 0.55 | 0.98 | 0.74 | 0.39 | 0.85 | **0.15** | 0.23 |
| IF-XL | **0.61** | 0.97 | 0.74 | **0.66** | 0.81 | 0.13 | **0.35** |

这是一张很适合补入论文的“基础模型参照表”：它表明基础模型在单对象（0.97--0.98）上已接近饱和，但颜色-对象绑定仍仅为 0.06--0.35，位置关系仍仅为 0.04--0.15。注意其生成和评测协议与本项目不同，不能将 Frozen PixArt 的 69.23% CLIP binding proxy 换算为 GenEval 的 0.35，也不能据此形成跨表排名。

### 3.3 T2I-CompBench++：11 个模型、12 项组合指标

T2I-CompBench++ 在 8,000 个组合 prompt 上评测 11 个模型。以下是其 Table XIII 的原始数值。颜色、形状、纹理使用 **disentangled BLIP-VQA**（将每个属性-对象短语拆开提问，概率连乘）；二维空间、三维空间和数量使用 **UniDet**；非空间关系使用 **CLIP**；复杂组合使用三类指标。均为 0--1，越高越好。

#### 3.3.1 自动主指标

| 模型 | 颜色绑定 B-VQA | 形状绑定 B-VQA | 纹理绑定 B-VQA | 2D 空间 UniDet | 3D 空间 UniDet | 数量 UniDet | 非空间 CLIP | 复杂组合 3-in-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stable Diffusion v1.4 | 0.3765 | 0.3576 | 0.4156 | 0.1246 | 0.3030 | 0.4456 | 0.3079 | 0.3080 |
| Stable Diffusion v2 | 0.5065 | 0.4221 | 0.4922 | 0.1342 | 0.3230 | 0.4582 | 0.3127 | 0.3386 |
| Composable Diffusion + SD v2 | 0.4063 | 0.3299 | 0.3645 | 0.0800 | 0.2847 | 0.4272 | 0.2980 | 0.2898 |
| Structured Diffusion + SD v2 | 0.4990 | 0.4218 | 0.4900 | 0.1386 | 0.3224 | 0.4557 | 0.3111 | 0.3355 |
| Attend-and-Excite + SD v2 | 0.6400 | 0.4517 | 0.5963 | 0.1455 | 0.3222 | 0.4773 | 0.3109 | 0.3401 |
| GORS-unbiased + SD v2 | 0.6414 | 0.4546 | 0.6025 | 0.1725 | 0.3300 | 0.4849 | 0.3158 | 0.3470 |
| GORS + SD v2 | 0.6603 | 0.4785 | 0.6287 | 0.1815 | 0.3572 | 0.4830 | 0.3193 | 0.3328 |
| Stable Diffusion XL | 0.5879 | 0.4687 | 0.5299 | 0.2133 | 0.3566 | 0.4991 | 0.3119 | 0.3237 |
| PixArt-alpha-ft | 0.6690 | 0.4927 | 0.6477 | 0.2064 | 0.3901 | 0.5032 | 0.3197 | 0.3433 |
| DALL-E 3 | 0.7785 | **0.6205** | 0.7036 | 0.2865 | 0.3744 | 0.5926 | 0.3003 | **0.3773** |
| Stable Diffusion 3 | **0.8132** | 0.5885 | **0.7334** | **0.3200** | **0.4084** | 0.6174 | 0.3140 | 0.3771 |
| FLUX.1 [schnell] | 0.7407 | 0.5718 | 0.6922 | 0.2863 | 0.3866 | **0.6185** | 0.3127 | 0.3703 |

`PixArt-alpha-ft` 是该论文用 GORS 微调后的 PixArt-alpha，不是本项目的 Frozen PixArt，也不是 token-pair SCDA。它可作为“同为 PixArt 架构、但模型和训练方案不同”的外部参考，不能替代本项目消融基线。

#### 3.3.2 多模态评判指标

同一 Table XIII 还以 GPT-4V 和 ShareGPT4V-CoT 判定非空间关系与复杂组合。它们适合验证自动检测器难以覆盖的动作/语义关系，但不是可与 B-VQA 或 UniDet 相互换算的指标。

| 模型 | 非空间 GPT-4V | 非空间 Share-CoT | 复杂组合 GPT-4V | 复杂组合 Share-CoT |
|---|---:|---:|---:|---:|
| Stable Diffusion v1.4 | 0.7717 | 0.7487 | 0.6453 | 0.7727 |
| Stable Diffusion v2 | 0.8153 | 0.7567 | 0.6483 | 0.7783 |
| Composable Diffusion + SD v2 | 0.5030 | 0.6927 | 0.5637 | 0.7487 |
| Structured Diffusion + SD v2 | 0.8127 | 0.7560 | 0.6400 | 0.7777 |
| Attend-and-Excite + SD v2 | 0.8243 | 0.7593 | 0.6817 | 0.7763 |
| GORS-unbiased + SD v2 | 0.8557 | 0.7650 | 0.6753 | 0.7697 |
| GORS + SD v2 | 0.8420 | 0.7637 | 0.6850 | 0.7737 |
| Stable Diffusion XL | 0.8500 | 0.7673 | 0.7170 | 0.7817 |
| PixArt-alpha-ft | 0.8620 | 0.7747 | 0.7223 | 0.7823 |
| DALL-E 3 | 0.9170 | 0.7853 | 0.8653 | **0.7927** |
| Stable Diffusion 3 | 0.9093 | 0.7782 | 0.8717 | 0.7919 |
| FLUX.1 [schnell] | **0.9213** | **0.7809** | **0.8727** | **0.7927** |

这组公开数据已经覆盖更多可比较模型：Stable Diffusion v1.4/v2/XL、Composable Diffusion、Structured Diffusion、Attend-and-Excite、GORS、PixArt-alpha-ft、DALL-E 3、Stable Diffusion 3、FLUX.1。对于 SCDA 的核心主张，最关键的三列是颜色、形状和纹理绑定 B-VQA；其余列用于证明模型没有以牺牲对象关系、数量或复杂组合为代价。

## 4. 额外空间条件方法

BoxDiff 在文本之外接收 bounding box。论文使用 189 个 prompt、4,274 个有效 box，并以 YOLOv4 评估对象框满足度：

| 方法 | T2I-Sim ↑ | YOLO AP ↑ | AP50 ↑ | AP75 ↑ | 条件 |
|---|---:|---:|---:|---:|---|
| Stable Diffusion | 0.3511 | 2.8 | 未报告 | 未报告 | text + box protocol |
| BoxDiff | **0.3513** | **22.3** | **46.8** | **20.2** | text + user box |

这组 AP 数值不能与 SCDA 的 66.67%/69.23% binding proxy 合并排序：前者是空间框约束，后者是 CLIP 对比判断。

GLIGEN 同样接收 box、keypoint、reference image 等 grounding 条件；其公开资料明确以 COCO/LVIS grounded generation 和 grounding 质量为主要评测方向，但当前报告没有提取一组可与 SCDA 同协议复用的总分，因此不填跨协议数字。

## 5. 指标对齐与推荐主表

| 研究问题 | SCDA 当前值 | 文献对应指标 | 论文主表建议 |
|---|---|---|---|
| 整体文本一致性 | CLIP 0.273209--0.273675 | image-CLIP、BLIP-caption-CLIP | 仅在同一基座/协议下比较；外部论文数值不横排 |
| 对象是否出现 | 86.27% CLIP proxy | Entity Neglect / object coverage | 用 GroundingDINO/OWL-ViT 或人工 object recall 复核 |
| 属性是否出现 | 74.51% CLIP proxy | attribute presence / attribute neglect | 加 VQA 或区域属性分类 |
| 属性是否绑定 | 69.23% CLIP proxy | Proper Binding、Improper Binding | 主指标应改为人工或区域 VQA binding accuracy |
| 空间 grounding | 未执行 | BoxDiff AP/AP50/AP75、GLIGEN grounding | 另设额外 box 条件表 |
| 效率 | 当前未统一 profiling | SynGen 4--9.76 s/image | 统一硬件、steps、batch 后再比较 |

若实际将 SCDA 与更多模型重跑，建议使用下面这组最小且完整的同协议数据，而不是从上述外部表里抽数和本地分数混排：

| 指标组 | 必算数据 | 能回答的问题 | 关联公开基准 |
|---|---|---|---|
| 基础对象 | GenEval：单对象、双对象、计数 | 是否漏实体、是否能同时生成多个对象 | GenEval |
| 属性-对象绑定 | GenEval color attribution；T2I-CompBench++ 色/形/纹理 B-VQA；人工 Proper/Improper Binding | 属性有没有绑定到**正确**对象 | GenEval、T2I-CompBench++、SynGen |
| 关系 | GenEval position；T2I 2D/3D UniDet；非空间关系 | token-pair 注入是否损害或改善关系组合 | GenEval、T2I-CompBench++ |
| 全局对齐与质量 | CLIPScore、FID 或 KID、人工 visual appeal | 整体语义和图像分布是否退化 | 当前本地 CLIP、SynGen |
| 成本与可信度 | s/image、显存、参数/训练量、95% bootstrap CI、人工一致性 | 绑定提升是否来自更大推理代价，是否稳健 | SynGen、统一本地 profiling |

## 6. 可引用的限制性结论

1. Token-pair + distillation 保持 PixArt 的总体 CLIP（0.273675 vs 0.273209），并将属性存在 proxy 从 70.59% 提升到 74.51%；这不是跨论文 SOTA 结论。
2. Learnable Layers 的 binding proxy 为 69.23%（27/39），相比 baseline 仅多 1 个正确判断，必须扩大测试集并加入人工/区域判定。
3. SynGen 的 63.68--94.76 Proper Binding 是公开人工结果，但基于 Stable Diffusion 和不同 prompt/采样协议；只能作为问题难度与指标定义参照。
4. BoxDiff 的 22.3 AP 是额外 box 条件下的空间指标，不能证明其纯文本绑定能力，也不能直接压过 SCDA。
5. T2I-CompBench++ 公开表表明 Attend-and-Excite 在 SD v2 上可将颜色/形状/纹理 B-VQA 提升至 0.6400/0.4517/0.5963，但该实现和本项目 PixArt 不同；它应是 SCDA 的优先同协议外部基线之一。
6. pooled SCDA 的负结果（CLIP -9.57%）应保留在消融表中，但后续训练预算应集中在 token-level pair、蒸馏、关系建模和更大绑定测试集。

## 7. 来源

- SynGen / Linguistic Binding，NeurIPS 2023：[论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/0b08d733a5d45a547344c4e9d88bb8bc-Paper-Conference.pdf)，Table 1、2、6 和 Appendix G.2。
- Attend-and-Excite：[论文](https://arxiv.org/abs/2301.13826)，[官方指标代码](https://github.com/yuval-alaluf/Attend-and-Excite)。
- Structured Diffusion Guidance：[论文](https://arxiv.org/abs/2212.05032)，[官方仓库](https://github.com/weixi-feng/Structured-Diffusion-Guidance)。
- BoxDiff：[论文 PDF](https://github.com/showlab/BoxDiff/blob/main/BoxDiff_ICCV_2023.pdf)，Table 1、3。
- GenEval：[论文](https://arxiv.org/abs/2310.11513)，[官方仓库与主结果表](https://github.com/djghosh13/geneval)。
- T2I-CompBench++：TPAMI 2025 [官方仓库](https://github.com/Karine-Huang/T2I-CompBench)，Table XIII（8,000 prompts，11 个模型）。
- SCDA 本地实验：[COCO2017_TOKEN_PAIR_RECORD.md](../experiments/COCO2017_TOKEN_PAIR_RECORD.md)、[LEARNABLE_BRANCH_LAYER_METRICS_REPORT.md](LEARNABLE_BRANCH_LAYER_METRICS_REPORT.md)。
