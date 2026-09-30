# SCDA 与相关对象-文本绑定方法的公开指标对比

**用途：** 汇总公开论文/官方项目中实际使用的评价指标，并与本项目已有的 token-pair SCDA 结果对齐。本文不部署外部模型，也不把不同论文的分数直接合并排名。

## 1. 先看结论

公开方法的指标并不统一：Attend-and-Excite 和 SynGen 主要验证文本组合语义、对象覆盖和属性绑定；Structured Diffusion 关注组合提示词的属性/对象关系；GLIGEN 和 BoxDiff 重点验证给定框或布局后的空间 grounding。你的 SCDA 当前使用的是固定 64 条 prompt、3 个 seed 和本地 CLIP ViT-B/32，因此只能与文献中的 **指标类别和评测目标** 对齐，不能把 CLIP 数值当作跨论文的公平排行榜。

就研究问题而言，最有参考价值的外部指标顺序是：

1. 对象存在/对象覆盖率；
2. object-attribute binding accuracy；
3. 关系或空间 grounding accuracy；
4. 整体 CLIPScore；
5. FID/KID、人工偏好和推理成本。

## 2. 公开论文使用的指标

| 方法 | 公开论文/官方项目 | 基座与输入条件 | 论文/官方代码中可确认的指标 | 对 SCDA 的可比性 |
|---|---|---|---|---|
| Attend-and-Excite | [论文](https://arxiv.org/abs/2301.13826)，[官方代码](https://github.com/yuval-alaluf/Attend-and-Excite) | Stable Diffusion；纯文本；推理时优化 latent 和 cross-attention | image-based CLIP similarity；先用 BLIP 生成 caption 后的 text-based CLIP similarity；对象遗漏/组合语义的定性和人工分析 | 高。纯文本、无训练数据；但基座、采样过程和 CLIP 实现不同 |
| SynGen | [Linguistic Binding in Diffusion Models](https://arxiv.org/abs/2306.08877) | Stable Diffusion；纯文本；句法解析后推理时对 attention map 做对齐 | 三个数据集上的人工绑定/组合评估；论文核心目标是 modifier-entity attribute correspondence，另有图文语义对比 | 高。问题定义最接近 object-attribute binding，但论文不是 PixArt/token-pair 训练设定 |
| Structured Diffusion Guidance | [论文](https://arxiv.org/abs/2212.05032) | Stable Diffusion；纯文本；训练免费结构化 attention guidance | 组合文本生成的对象/属性绑定和组合正确性；论文以定性比较及组合语义评测为主 | 中-高。输入条件相同，但推理时 guidance 开销更高 |
| GLIGEN | [论文](https://arxiv.org/abs/2301.07093)，[项目页](https://gligen.github.io/)，[代码](https://github.com/gligen/GLIGEN) | Stable Diffusion；文本 + box/keypoint/image 等 grounding 条件 | COCO/LVIS grounded generation；公开实验同时比较生成质量与 grounding/定位表现，项目页明确报告 zero-shot grounded generation 对布局基线的比较 | 低-中。对象 grounding 很相关，但需要额外空间输入，不能作为纯文本主表直接排名 |
| BoxDiff | [论文](https://arxiv.org/abs/2307.10816)，[代码](https://github.com/showlab/BoxDiff) | Stable Diffusion；文本 + 用户框/草图；训练免费 | 公开实验围绕 box/空间约束是否满足、对象布局正确性、图文一致性和定性结果展开 | 低-中。可作为“额外框输入时的能力上限”，不是同条件绑定对比 |
| DAAM | [论文](https://arxiv.org/abs/2210.04885) | Stable Diffusion；原始文本条件 | token-to-image cross-attention heatmap、词级归因和区域对应分析 | 不是生成增强主基线，更适合做 SCDA 的诊断指标 |
| Prompt-to-Prompt | [论文](https://arxiv.org/abs/2208.01626) | Stable Diffusion；源 prompt/编辑 prompt | 编辑成功率、文本编辑保持度和人工/定性比较 | 适合 attention 控制参照，不是专门的对象属性绑定基准 |

### 2.1 指标含义

- **CLIP image-text similarity：** 图像和原 prompt 的嵌入相似度。适合总体文本一致性，不足以区分“颜色都出现了”与“颜色绑定到正确对象”。
- **BLIP-caption CLIP：** 先对生成图生成 caption，再比较 caption 与目标 prompt。可以补充图像内容覆盖，但仍可能漏掉细粒度绑定错误。
- **Object/subject coverage：** prompt 中每个目标对象是否被检测器、VQA 或人工确认。Attend-and-Excite 的核心问题就是 catastrophic neglect，因此这是最直接的指标。
- **Binding accuracy：** 对 `red ball and blue cube`，比较正确绑定描述与交换描述；应按对象对统计，而不是只统计颜色是否出现。
- **Grounding/box adherence：** 预测对象区域与给定 box 的 IoU、中心偏差或检测成功率。GLIGEN/BoxDiff 的指标依赖额外布局输入，不能用于证明纯文本 SCDA 更强。
- **FID/KID：** 只衡量生成分布与真实图像分布的距离，不直接评价属性归属或关系。
- **人工评测：** 对象存在、属性绑定、关系正确性和总体质量的 pairwise preference/accuracy。SynGen 等绑定方法使用这类评测来补充自动指标。

## 3. 与本项目结果的对齐

本项目统一协议为 64 条 prompt、seed 43/44/45、512x512、DPM-Solver 20 steps、CFG=4.0、本地 CLIP ViT-B/32。

| 方法 | 本项目 CLIP | 对象存在 proxy | 属性存在 proxy | 绑定 proxy | 说明 |
|---|---:|---:|---:|---:|---|
| Frozen PixArt | 0.273209 | 86.27% | 70.59% | 66.67% | 纯文本参照 |
| Pooled SCDA | 0.247061 | 未统一计算 | 未统一计算 | 未统一计算 | 已停止；总体 CLIP 比 baseline 低 9.57% |
| Full-span SCDA | 0.247550 | 未统一计算 | 未统一计算 | 未统一计算 | pooled 路线负结果 |
| Token-pair SCDA | 0.266105 | 未统一计算 | 未统一计算 | 未统一计算 | 保留 token 级 pair bias，但未加蒸馏 |
| Token-pair + distillation | 0.273675 | 84.31% | 74.51% | 66.67% | 总体 CLIP 基本保持 baseline |
| Token-pair + learnable layers | 0.273597 | 86.27% | 74.51% | 69.23% | 比 baseline 多 1/39 个绑定判断，样本太小不能宣称显著提升 |

这里的 object/attribute/binding 是 CLIP contrastive proxy，不是检测器、VQA 或人工标注结果。它们可以与文献的目标对齐，但不能与 SynGen 的人工准确率、GLIGEN 的 grounding 指标或 BoxDiff 的 box adherence 数值直接比较。

## 3.0 文献中可核验的公开数值

下表补充论文正文中可以直接核验的数字。它们不是与 SCDA 的统一重评结果：SynGen 使用 Stable Diffusion v1.4、50 个采样步、CFG 7.5 和人工标注；BoxDiff 使用 Stable Diffusion、额外 bounding-box 条件和 YOLOv4 检测器。因此，这些数字只能说明指标量级和评测定义，不能用于跨论文排序。

### SynGen / Linguistic Binding：人工绑定指标

SynGen 的 fine-grained evaluation 在 Attend-and-Excite（A&E）数据集、DVMP challenge set 和 ABC-6K 上报告三项百分比：Proper Binding 越高越好，Improper Binding 和 Entity Neglect 越低越好。每个数据集随机抽取 200 个 prompt（A&E 使用全部 177 个 prompt），由人工判断属性-对象映射和实体是否出现。

| 数据集 | 方法 | Proper Binding ↑ | Improper Binding ↓ | Entity Neglect ↓ |
|---|---|---:|---:|---:|
| A&E (177 prompts) | SynGen | **94.76** | 23.81 | 2.82 |
| A&E (177 prompts) | Attend-and-Excite | 81.90 | 63.81 | **1.41** |
| A&E (177 prompts) | Structured Diffusion | 55.71 | 67.62 | 21.13 |
| A&E (177 prompts) | Stable Diffusion | 59.05 | 68.57 | 20.56 |
| DVMP (200 prompts) | SynGen | **74.90** | **19.49** | 16.26 |
| DVMP (200 prompts) | Attend-and-Excite | 52.47 | 31.64 | **10.77** |
| DVMP (200 prompts) | Structured Diffusion | 48.73 | 30.57 | 28.46 |
| DVMP (200 prompts) | Stable Diffusion | 47.80 | 30.44 | 26.22 |
| ABC-6K (200 prompts) | SynGen | **63.68** | **14.37** | 34.41 |
| ABC-6K (200 prompts) | Attend-and-Excite | 56.26 | 26.43 | **33.18** |
| ABC-6K (200 prompts) | Structured Diffusion | 51.47 | 29.52 | 34.57 |
| ABC-6K (200 prompts) | Stable Diffusion | 52.70 | 27.20 | 36.57 |

SynGen 还报告了三模型/四模型人工多数投票的概念分离（prompt 匹配程度）和视觉偏好。这里的数值是每个 prompt 的多数胜者比例，不是单张图准确率：

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

论文同时指出，phrase-to-image CLIP 自动指标与人工多数选择的一致率只有 **43.5%**（随机选择为 25%），这正是不能把普通 CLIP 分数当作绑定准确率的直接证据。SynGen 的平均生成时间也被报告为 Stable Diffusion **4.0 s/image**、Attend-and-Excite **8.8 s/image**、SynGen **9.76 s/image**（50 个 prompt 的计时，硬件和实现为论文设置）。

### BoxDiff：额外 box 条件下的空间指标

BoxDiff 在 189 个文本 prompt、4,274 个有效 bounding boxes 上评测，输入包含用户提供的 box；YOLOv4 用于检测生成图中的对象。其最终配置报告：

| 方法 | 输入条件 | T2I-Sim ↑ | YOLO AP ↑ | AP50 ↑ | AP75 ↑ |
|---|---|---:|---:|---:|---:|
| Stable Diffusion | 文本 + box 评测协议中的基线 | 0.3511 | 2.8 | 未报告 | 未报告 |
| BoxDiff | 文本 + box，training-free | **0.3513** | **22.3** | **46.8** | **20.2** |

BoxDiff 还报告其消融中仅 inner-box constraint 的 AP 为 9.8、inner+outer constraints 为 20.2、再加入 corner constraint 后为 22.3。由于 SCDA 不接收 box，BoxDiff 的 AP/IoU 类结果应放在“额外空间条件上限”表，不得与纯文本 binding 主表合并。

### Attend-and-Excite 官方代码能确认的指标

Attend-and-Excite 官方仓库明确提供 image-based CLIP similarity 和 BLIP caption 后的 text-based CLIP similarity 两个计算脚本，但仓库 README 不给出一组可直接迁移到 PixArt/SCDA 的统一总分。其对象遗漏结果主要通过 subject coverage、定性示例和人工分析呈现。报告中应引用指标定义，而不是填写未经同协议复算的数值。

### 数值来源

- SynGen / Linguistic Binding：NeurIPS 2023 正式论文 [PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/0b08d733a5d45a547344c4e9d88bb8bc-Paper-Conference.pdf)，Table 1、Table 2、Table 6 和 Appendix G.2。
- Structured Diffusion 数据集定义与 GLIP 评测接口：[官方仓库](https://github.com/weixi-feng/Structured-Diffusion-Guidance)，`CC-500.txt`、`ABC-6K.txt` 和 `GLIP_eval/`。
- Attend-and-Excite 指标脚本：[官方仓库](https://github.com/yuval-alaluf/Attend-and-Excite)，`metrics/compute_clip_similarity.py` 与 `metrics/blip_captioning_and_clip_similarity.py`。
- BoxDiff：ICCV 2023 [论文 PDF](https://github.com/showlab/BoxDiff/blob/main/BoxDiff_ICCV_2023.pdf)，Table 1 和 Table 3。

## 3.1 指标覆盖对比表

下表回答“每个方法实际评测了什么”。`有` 表示论文或官方代码明确提供了该类评测，`部分` 表示只有辅助分析/定性结果，`未统一` 表示没有可直接复用的统一数值。它不是方法性能排名。

| 方法 | 对象存在/覆盖 | 属性绑定 | 关系/空间 grounding | CLIP/图文一致性 | FID/KID | 人工评测 | 公开可复用数值 |
|---|---|---|---|---|---|---|---|
| Frozen PixArt（本项目） | 有：86.27% CLIP proxy | 有：66.67% binding proxy | 未执行 | 有：0.273209 | 未执行 | 未执行 | 本项目已测 |
| Token-pair + distillation（本项目） | 有：84.31% proxy | 有：66.67% proxy | 未执行 | 有：0.273675 | 未执行 | 未执行 | 本项目已测 |
| Token-pair + learnable layers（本项目） | 有：86.27% proxy | 有：69.23% proxy | 未执行 | 有：0.273597 | 未执行 | 未执行 | 本项目已测 |
| Attend-and-Excite | 有：对象遗漏/subject coverage 分析 | 部分：不显式计算 pair binding | 部分 | 有：image-CLIP、BLIP-caption-CLIP | 未见官方代码统一 FID | 部分/定性 | 官方代码可复算，协议不同 |
| SynGen | 部分：组合样例覆盖 | 有：三数据集人工 binding 评估 | 弱 | 有：CLIP winner/softmax 统计 | 未见官方代码统一 FID | 有 | 官方代码支持复算，论文数据集不同 |
| Structured Diffusion Guidance | 有：CC-500/组合对象 | 有：ABC-6K 属性绑定 | 部分：GLIP eval 接口 | 论文/代码提供辅助相似度分析 | 未统一 | 部分/定性 | 官方 benchmark 可下载，README 未给统一总分 |
| GLIGEN | 有：COCO/LVIS grounding | 有：区域条件下属性/对象 | 有：box/keypoint/image grounding | 有：论文中的质量对比 | 论文中按任务报告 | 部分 | 需要按论文表格和输入条件解读 |
| BoxDiff | 有：给定 box 后对象出现 | 区域约束下可分析 | 有：box adherence/空间约束 | 有：论文中的相似度与定性对比 | 未作为纯文本绑定指标 | 部分 | 需要 box 条件，不能直接与 SCDA 排名 |

### 3.2 SCDA 当前数值的相对变化

以 Frozen PixArt 为参照，当前已完成的数值变化为：

| 模型 | CLIP 相对变化 | 属性存在率变化 | 绑定准确率变化 | 可作出的结论 |
|---|---:|---:|---:|---|
| Pooled SCDA | -9.57% | 未统一 | 未统一 | 总体一致性退化，路线已停止 |
| Full-span SCDA | -9.39% | 未统一 | 未统一 | pooled mask 粒度没有解决退化 |
| Token-pair SCDA | -2.60% | 未统一 | 未统一 | token 级结构明显优于 pooled，但仍低于 baseline |
| Token-pair + distillation | +0.17% | +3.92 个百分点 | 0 个百分点 | 保持总体 CLIP，属性存在有正向趋势 |
| Token-pair + learnable layers | +0.14% | +3.92 个百分点 | +2.56 个百分点 | 绑定 proxy 多 1/39 个正确判断，需扩大测试集 |

这些百分比只适用于本项目固定协议，不能与外部论文报告的百分比互换。尤其是 A&E、SynGen 和 Structured Diffusion 使用 Stable Diffusion 1.4/2.x、不同 prompt 集和不同采样步数；GLIGEN/BoxDiff 还使用额外布局条件。

## 4. 公平的比较方式

### 4.1 纯文本主比较

主表应只放输入条件相同或近似的方法：

| 方法 | 输入 | 建议共同指标 |
|---|---|---|
| Frozen PixArt | 文本 | CLIP、对象存在、属性存在、binding accuracy |
| Attend-and-Excite | 文本 | 同上，加每张图的推理时间/额外 denoising 次数 |
| SynGen/Structured Diffusion | 文本 | 同上，加关系/组合准确率 |
| Token-pair SCDA | 文本 + 句法伪标签 | 同上，加可训练参数量和训练成本 |

### 4.2 额外布局输入的扩展比较

GLIGEN、BoxDiff、MultiDiffusion/InstanceDiffusion 应单独列为“grounded upper bound”。报告其 box/区域输入、检测器或布局来源，并使用 grounding accuracy、box IoU、空间关系成功率；不要把它们与纯文本方法混成一个总体排名。

## 5. 对你最有价值的指标补齐

若目标是证明 object-text binding，而不是一般图文相似度，建议将文献指标映射为下面的正式主表：

| 研究问题 | 首选指标 | 当前状态 |
|---|---|---|
| 对象是否出现 | object recall、all-object success、F1 | 尚未用开放词汇检测器验证 |
| 属性是否出现 | attribute accuracy、macro-F1 | 当前只有 CLIP proxy |
| 属性是否绑定正确 | pair-level binding accuracy、binding margin、人工准确率 | 当前为 39 个 case-seed 的 CLIP proxy |
| 空间/动作关系 | relation accuracy、人工关系准确率 | 尚未执行 |
| 整体文本遵循 | CLIPScore，GenEval/TIFA/VQAScore | 当前只有 CLIPScore |
| 图像质量 | FID/KID | 当前未完成标准 Inception FID/KID |
| 推理代价 | seconds/image、denoising steps、显存 | 应与 Attend-and-Excite 重点比较 |

最低限度的可发表对比应包含：Frozen PixArt、Attend-and-Excite、token-pair SCDA、token-pair + distillation；每个方法用同一 prompt、seed、分辨率、采样步数和 CFG，并报告 object recall、binding accuracy、CLIPScore 和推理时间。

## 6. 不能作出的结论

1. 不能用本项目的 CLIP `0.273675` 与 Attend-and-Excite、SynGen 或 GLIGEN 论文中的 CLIP 数值直接比较。它们的基座、CLIP 版本、prompt 集和采样配置不同。
2. 不能把 GLIGEN/BoxDiff 的 grounding 数值解释为纯文本 SCDA 的失败，因为它们获得了 box/layout 额外信息。
3. 不能把 pooled SCDA 的较低 CLIP-space FID/KID 代理解释为质量更好；其总体 CLIP 已明显退化。
4. 当前 39 个绑定判断只适合筛选候选模型，不能支撑“显著优于 SynGen/Attend-and-Excite”的论文结论。

## 7. 可核验网络来源

- Attend-and-Excite 官方代码中的 `metrics/compute_clip_similarity.py` 和 `metrics/blip_captioning_and_clip_similarity.py` 明确给出两类 CLIP 评测：[GitHub](https://github.com/yuval-alaluf/Attend-and-Excite)。
- SynGen 论文：[arXiv:2306.08877](https://arxiv.org/abs/2306.08877)。论文摘要明确说明句法分析、attention map alignment 和三个数据集上的人工评估。
- Structured Diffusion Guidance：[arXiv:2212.05032](https://arxiv.org/abs/2212.05032)。
- GLIGEN：[CVPR 2023 论文](https://arxiv.org/abs/2301.07093)、[项目页](https://gligen.github.io/)、[官方代码](https://github.com/gligen/GLIGEN)。
- BoxDiff：[ICCV 2023 论文](https://arxiv.org/abs/2307.10816)、[官方代码](https://github.com/showlab/BoxDiff)。
- DAAM：[arXiv:2210.04885](https://arxiv.org/abs/2210.04885)。
- Prompt-to-Prompt：[arXiv:2208.01626](https://arxiv.org/abs/2208.01626)。

**最终判断：** 文献证据支持把 Attend-and-Excite、SynGen 和 Structured Diffusion 作为你的纯文本绑定对比方向；GLIGEN 和 BoxDiff 应作为额外空间条件的上限参照。当前 SCDA 最合理的论文表述是“token-pair + distillation 在保持 PixArt 总体 CLIP 的同时，显示出属性存在和绑定 proxy 的正向趋势”，而不是宣称已经超过这些方法的公开指标。
