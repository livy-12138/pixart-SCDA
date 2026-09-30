# Token-Pair SCDA 的属性绑定专项对比

**目的：** Token-pair SCDA 的机制直接作用于属性 token 与对象 token 的对应关系。本文只将其与属性绑定或组合提示词绑定方法作主比较；通用文生图基础模型、box/layout 条件方法和一般图像质量指标均不构成主排名。

## 1. 主比较对象

| 方法 | 是否直接解决属性-对象绑定 | 输入条件 | 应否进入主表 | 理由 |
|---|---|---|---|---|
| Frozen PixArt | 否 | 纯文本 | 是 | 本方法的同基座基线 |
| Token-pair SCDA（本方法） | 是 | 纯文本 | 是 | 将属性和对象 token 对显式注入 cross-attention |
| Attend-and-Excite | 是，减少主体/修饰词遗漏 | 纯文本 | 是 | 通过测试时 attention 优化强化指定 token |
| Structured Diffusion Guidance | 是，结构化属性/对象关系 | 纯文本 | 是 | 结构化 cross-attention guidance 的直接基线 |
| SynGen | 是，modifier-entity linguistic binding | 纯文本 | 是 | 使用句法依存和 attention map alignment，问题定义最接近 |
| Composable Diffusion | 间接，概念 conjunction | 纯文本 | 辅助表 | 主要面向概念组合/否定，不是 pair-level binding |
| BoxDiff / GLIGEN | 部分 | text + box/layout | 不进入主表 | 获得额外空间条件，适合作为 grounded upper bound |

## 2. 本项目同协议结果：Token-Pair 消融主表

协议：64 条固定 prompt、seed 43/44/45、512x512、DPM-Solver 20 steps、CFG 4.0、CLIP ViT-B/32。对象/属性/绑定均为正确描述与干扰描述的 **CLIP contrastive proxy**，不是人工 Binding Accuracy。

| 方法 | CLIPScore | 对象存在 proxy | 属性存在 proxy | Binding proxy | 与 Frozen 的 binding 差 |
|---|---:|---:|---:|---:|---:|
| Frozen PixArt | 0.273209 | **86.27%** (44/51) | 70.59% (36/51) | 66.67% (26/39) | - |
| Token-pair SCDA | 0.266105 | 84.31% (43/51) | 66.67% (34/51) | 66.67% (26/39) | +0.00 pp [-10.26, +10.26] |
| Token-pair + distillation | **0.273675** | 84.31% (43/51) | **74.51% (38/51)** | 66.67% (26/39) | +0.00 pp [-7.14, +7.41] |
| Token-pair + learnable layers | 0.273597 | **86.27%** (44/51) | **74.51% (38/51)** | **69.23% (27/39)** | +2.56 pp [0.00, +8.33] |

方括号是按 17 个案例聚类、10,000 次 paired bootstrap 的 95% CI。Learnable Layers 比 Frozen 多 1 个正确绑定判断，当前只能表述为绑定 proxy 的正向趋势，不能表述为统计显著提升。

Pooled SCDA 和 Full-span SCDA 不再放入 token-pair 主表：二者不是 token-pair 机制，且 CLIPScore 分别为 0.247061 和 0.247550（相对 Frozen 为 -9.57% 和 -9.39%）。其负结果保留在消融附录。

## 3. 与属性绑定方法的公开人工结果

SynGen 在同一篇论文中以人工细粒度标注评测 SynGen、Attend-and-Excite、Structured Diffusion 和 Stable Diffusion。`Proper Binding` 是正确属性-对象映射比例，`Improper Binding` 是错误映射比例，`Entity Neglect` 是目标实体未出现比例。该表是外部方法之间可直接比较的绑定证据；它与本项目 CLIP proxy 的判定器和基座不同，因此本方法不填入该表。

### 3.1 ABC-6K：最接近双对象-双属性压力测试的公开表

| 方法 | Proper Binding ↑ | Improper Binding ↓ | Entity Neglect ↓ | 判定 |
|---|---:|---:|---:|---|
| SynGen | **63.68** | **14.37** | 34.41 | 人工 |
| Attend-and-Excite | 56.26 | 26.43 | **33.18** | 人工 |
| Structured Diffusion | 51.47 | 29.52 | 34.57 | 人工 |
| Stable Diffusion | 52.70 | 27.20 | 36.57 | 人工 |
| Token-pair + learnable layers（本项目） | 未在该协议评测 | 未在该协议评测 | 未在该协议评测 | - |

SynGen 同时报告 A&E (177 prompts) 和 DVMP (200 prompts) 的人工结果：

| 数据集 | SynGen Proper / Improper / Neglect | Attend-and-Excite Proper / Improper / Neglect | Structured Diffusion Proper / Improper / Neglect | Stable Diffusion Proper / Improper / Neglect |
|---|---|---|---|---|
| A&E | 94.76 / 23.81 / 2.82 | 81.90 / 63.81 / 1.41 | 55.71 / 67.62 / 21.13 | 59.05 / 68.57 / 20.56 |
| DVMP | 74.90 / 19.49 / 16.26 | 52.47 / 31.64 / 10.77 | 48.73 / 30.57 / 28.46 | 47.80 / 30.44 / 26.22 |

## 4. 属性绑定基准的自动指标：T2I-CompBench++

T2I-CompBench++ 以 disentangled BLIP-VQA 计算颜色、形状和纹理绑定。以下只保留属性绑定相关方法和辅助组合基线，不将通用大模型混入 token-pair 主比较。分数在 0--1，越高越好。

| 方法 | 颜色绑定 B-VQA ↑ | 形状绑定 B-VQA ↑ | 纹理绑定 B-VQA ↑ | 方法定位 |
|---|---:|---:|---:|---|
| Stable Diffusion v2 | 0.5065 | 0.4221 | 0.4922 | 基础生成模型 |
| Composable Diffusion + SD v2 | 0.4063 | 0.3299 | 0.3645 | 概念组合辅助基线 |
| Structured Diffusion + SD v2 | 0.4990 | 0.4218 | 0.4900 | 属性/对象结构化 guidance |
| Attend-and-Excite + SD v2 | **0.6400** | **0.4517** | **0.5963** | 测试时 attention 控制 |
| Token-pair + learnable layers（本项目） | 未在该协议评测 | 未在该协议评测 | 未在该协议评测 | PixArt token-pair conditioning |

## 5. 量化总览表：所有可核验的绑定数值

这是当前可直接放入补充材料的完整数值表。所有百分比已换算到 0--1；`-` 表示原论文未在该协议报告该项，**不是 0 分**。同一方法在不同数据集/判定器下分行报告，防止将不同定义强行合并。

| 数据集与判定器 | 方法 | Proper Binding ↑ | Improper Binding ↓ | Entity Neglect ↓ | Color B-VQA ↑ | Shape B-VQA ↑ | Texture B-VQA ↑ | 本地 Binding proxy ↑ | 本地 CLIPScore ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 本地 64 prompts / CLIP ViT-B/32 | Frozen PixArt | - | - | - | - | - | - | 0.6667 | 0.273209 |
| 本地 64 prompts / CLIP ViT-B/32 | Token-pair SCDA | - | - | - | - | - | - | 0.6667 | 0.266105 |
| 本地 64 prompts / CLIP ViT-B/32 | Token-pair + distillation | - | - | - | - | - | - | 0.6667 | **0.273675** |
| 本地 64 prompts / CLIP ViT-B/32 | Token-pair + learnable layers | - | - | - | - | - | - | **0.6923** | 0.273597 |
| ABC-6K / 人工 | SynGen | **0.6368** | **0.1437** | 0.3441 | - | - | - | - | - |
| ABC-6K / 人工 | Attend-and-Excite | 0.5626 | 0.2643 | **0.3318** | - | - | - | - | - |
| ABC-6K / 人工 | Structured Diffusion | 0.5147 | 0.2952 | 0.3457 | - | - | - | - | - |
| ABC-6K / 人工 | Stable Diffusion | 0.5270 | 0.2720 | 0.3657 | - | - | - | - | - |
| T2I-CompBench++ / disentangled BLIP-VQA | Stable Diffusion v2 | - | - | - | 0.5065 | 0.4221 | 0.4922 | - | - |
| T2I-CompBench++ / disentangled BLIP-VQA | Structured Diffusion + SD v2 | - | - | - | 0.4990 | 0.4218 | 0.4900 | - | - |
| T2I-CompBench++ / disentangled BLIP-VQA | Attend-and-Excite + SD v2 | - | - | - | **0.6400** | **0.4517** | **0.5963** | - | - |

**表的正确读法：**

1. 在每一个“数据集与判定器”块内可以做数值排序。例如 ABC-6K 人工标注中 SynGen 的 Proper Binding 为 0.6368，高于 Attend-and-Excite 的 0.5626；T2I-CompBench++ 中 Attend-and-Excite 的三种 B-VQA 分数最高。
2. 本地 Token-pair 行只能与 Frozen PixArt 行直接比较。目前 Learnable Layers 在 binding proxy 上为 0.6923，对 Frozen 的 0.6667 是 +0.0256，但 95% CI 为 [0.0000, 0.0833]，不应写为显著优于基线。
3. 本地行与 ABC-6K/T2I-CompBench++ 行不能纵向比较或计算平均排名。若要将 Token-pair 放入后两块，必须用原始官方 prompt、相同每 prompt 图数和相同判定器实际重跑。

## 6. 最终论文应使用的对比表

要使 Token-pair 与上述方法形成严格的数字对比，必须在同一生成和评测协议下获得以下列。建议主表只保留 Frozen PixArt、Token-pair、Token-pair + distillation、Token-pair + learnable layers、Attend-and-Excite、Structured Diffusion、SynGen。

| 指标 | 优先级 | 原因 |
|---|---|---|
| Proper Binding / Improper Binding / Entity Neglect（人工） | 必须 | 与 SynGen 的核心指标一致，直接区分“颜色都出现”与“颜色归属正确” |
| Color / Shape / Texture B-VQA | 必须 | 分解三种属性，避免只在颜色上有效 |
| GenEval color attribution | 必须 | 检测器驱动的第二个自动绑定判定器 |
| Object Recall 与 all-object success | 必须 | 排除通过遗漏一个对象而获得表面绑定正确的情况 |
| CLIPScore 与 Binding Margin | 辅助 | 监控整体对齐及与现有本地结果的连续性，不能替代人工绑定 |
| s/image、steps、峰值显存 | 必须 | Attend-and-Excite、SynGen 均有额外测试时优化代价 |
| FID/KID、空间关系、计数 | 附录 | 评价一般生成能力，不是 token-pair 属性绑定的主张 |

## 7. 可作出的结论与不可作出的结论

1. 当前可成立的结论是：在 PixArt 同协议下，token-pair + distillation 保持总体 CLIPScore，并提高属性存在 proxy；learnable layers 的 binding proxy 有 +2.56 pp 的正向趋势。
2. 不可声称本方法已超过 SynGen、Attend-and-Excite 或 Structured Diffusion：它们的公开数字来自 Stable Diffusion、不同 prompt 集和人工/B-VQA 判定器。
3. 在未重跑外部方法时，应该把第 2 节写为“本项目消融”，把第 3--4 节写为“属性绑定方法的公开基准结果”，不得合并为单一性能排行。
4. 若只选两个优先外部对比，选 **Attend-and-Excite**（测试时 attention 控制）与 **SynGen**（句法绑定）。Structured Diffusion 作为第三个结构化 attention 基线；Composable Diffusion 只作辅助组合基线。

## 8. 来源

- SynGen / Linguistic Binding，NeurIPS 2023：[论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/0b08d733a5d45a547344c4e9d88bb8bc-Paper-Conference.pdf)，Table 1、2、6。
- Attend-and-Excite：[论文](https://arxiv.org/abs/2301.13826)，[官方代码](https://github.com/yuval-alaluf/Attend-and-Excite)。
- Structured Diffusion Guidance：[论文](https://arxiv.org/abs/2212.05032)，[官方代码](https://github.com/weixi-feng/Structured-Diffusion-Guidance)。
- T2I-CompBench++：[官方仓库](https://github.com/Karine-Huang/T2I-CompBench)，Table XIII。
- 本项目实测：[T2I_ALL_METRICS_EVALUATION_REPORT.md](T2I_ALL_METRICS_EVALUATION_REPORT.md)、`output/attribute_binding_metrics_all_bootstrap.csv`。
