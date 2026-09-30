# 面向对象-属性绑定的 Token-Pair 条件适配：冻结 PixArt 中的弱监督文本到图像生成

> 论文初稿。所有“—”均表示尚未完成本地评测，不以估计值、外部论文数值或代理值填充。

## 中文摘要

复杂文本到图像生成中，扩散模型常能生成提示词中出现的对象和属性，却无法保证属性归属于正确对象。例如，对于“一个红色的球和一个蓝色的立方体”，生成图可能同时包含红、蓝两种颜色，却发生颜色交换。已有纯文本方法通常通过测试时注意力优化或句法约束缓解该问题，但它们大多建立在 Stable Diffusion 上，且额外推理开销较大。本文研究冻结 PixArt-XL-2 主干时的对象-属性绑定问题，提出 Token-Pair Semantic-Conditioned Diffusion Adapter（TP-SCDA）。该方法从英文提示词中提取对象 token、属性 token 及依存关系边，并在原始 cross-attention 不被替换的条件下，利用对象注意力响应向合法属性 token 注入 pair-aware logit bias。该结构避免将多个对象和属性池化为单一全局向量，从而保留“哪个属性属于哪个对象”的 token 级对应。为抑制轻量条件分支对冻结基座的行为漂移，本文进一步采用冻结 PixArt 教师的去噪蒸馏，并探索可学习层门控。

在 118,286 条预处理 COCO2017 文本-图像样本上训练，在 64 条固定英文 prompt、3 个随机种子、512x512、DPM-Solver 20 步和 CFG=4.0 的统一协议下评估。Token-pair + distillation 的 CLIPScore 为 0.273675，略高于 Frozen PixArt 的 0.273209；Token-pair + learnable layers 的 CLIPScore 为 0.273597。针对 17 个对象/属性案例（其中 13 个具有交换属性描述）的 CLIP contrastive proxy 中，learnable-layers 变体的对象存在率、属性存在率和绑定准确率分别为 86.27%、74.51% 和 69.23%，相较 Frozen PixArt 的 86.27%、70.59% 和 66.67%，绑定正确数仅增加 1/39。按 prompt 聚类的 paired bootstrap 表明该绑定差的 95% 区间为 [0.00, 8.33] 个百分点，故本文将其解释为正向趋势而非显著提升。本文同时报告 pooled/full-span SCDA 的总体 CLIP 退化结果，并制定面向 SynGen、Attend-and-Excite 与 Structured Diffusion 的统一属性绑定评测方案。本文的贡献是一个可复现的 token-pair 条件结构、诚实的多变体消融与可执行的后续对比协议，而非跨论文的性能排名。

**关键词：** 文本到图像生成；扩散 Transformer；PixArt；对象-属性绑定；cross-attention；参数高效适配；弱监督

## English Abstract

Text-to-image diffusion models may render all entities and attributes in a prompt while assigning attributes to the wrong entities. For example, an image for “a red ball and a blue cube” can contain both colors but swap their ownership. This work studies object-attribute binding while keeping a PixArt-XL-2 backbone frozen. We propose the Token-Pair Semantic-Conditioned Diffusion Adapter (TP-SCDA), a lightweight conditioning mechanism that extracts object tokens, attribute tokens, and dependency edges from English prompts. Rather than pooling semantic roles into a global residual, TP-SCDA preserves token-level correspondences and adds a pair-aware logit bias to the original cross-attention. The bias is routed from image-patch attention over object tokens to their syntactically valid attribute tokens. A frozen PixArt teacher is used for denoising distillation, and learnable layer gates are explored to limit behavioral drift.

We train on 118,286 prepared COCO2017 examples and evaluate under a fixed protocol of 64 prompts, three seeds, 512x512 resolution, DPM-Solver with 20 steps, and CFG 4.0. Token-pair with distillation achieves a CLIPScore of 0.273675, compared with 0.273209 for frozen PixArt; the learnable-layer variant reaches 0.273597. On 17 object-attribute cases, including 13 swapped-attribute binding cases, the learnable-layer variant obtains 86.27% object-presence proxy, 74.51% attribute-presence proxy, and 69.23% binding proxy, versus 86.27%, 70.59%, and 66.67% for frozen PixArt. This amounts to one additional correct binding decision out of 39. A prompt-clustered paired bootstrap gives a 95% interval of [0.00, 8.33] percentage points for the binding difference; therefore, we interpret the result as a positive trend rather than a statistically significant improvement. We also report negative results for pooled and full-span SCDA, and provide a protocol for fair comparisons with binding-oriented methods such as SynGen, Attend-and-Excite, and Structured Diffusion. The paper contributes a reproducible token-pair design, transparent ablations, and an executable evaluation plan rather than a cross-paper ranking claim.

**Keywords:** text-to-image generation; diffusion transformer; PixArt; object-attribute binding; cross-attention; parameter-efficient adaptation; weak supervision

## 1. 引言

扩散模型已成为开放域文本到图像生成的主流技术路线 [1-3]。PixArt-α 以 Transformer 去噪器和强文本编码器在较低训练成本下获得了有竞争力的视觉质量 [2]。但文本条件的整体可用性并不意味着组合语义正确：当一个 prompt 同时包含多个对象、多个颜色或材质、数量词和关系词时，模型可能遗漏实体、混合概念，或者将属性分配给错误对象。此类 object-attribute binding failure 是长提示词可控生成的核心瓶颈之一 [8-13]。

现有工作提供了三类思路。第一类以 Attend-and-Excite [8]、Prompt-to-Prompt [9] 为代表，在采样期直接调节 token 的 cross-attention；其优点是不必训练，但需要额外的优化迭代。第二类以 Structured Diffusion Guidance [10]、SynGen [11] 为代表，使用文本结构或句法关系约束对象、属性和关系的注意力对应。第三类以 GLIGEN [14]、BoxDiff [15] 为代表，引入边界框、布局或其他 grounding 输入获得更强的控制，但已不属于只输入文本的公平设置。

本研究的目标更受限：在不使用框、掩码、参考图或人工场景图，且冻结 PixArt 主干的前提下，能否以少量可训练模块改善属性-对象的 token 对应？前期 pooled SCDA 将对象、属性和关系 token 池化后注入全局残差。尽管训练稳定，该路线的 CLIPScore 比 Frozen PixArt 低 9% 以上，说明池化会损失绑定结构。本文因此转向 token-pair 路线：不再把“red”和“ball”压缩为独立全局向量，而是在模型仍可见完整文本 token 的情况下，利用其依存边建立定向的对象到属性信息路径。

本文贡献如下：

1. 提出 TP-SCDA：一个保持原 PixArt cross-attention、仅向合法对象-属性 token 对注入 bias 的弱监督条件结构。
2. 将对象 token 的图像 patch 注意力响应路由到相连属性 token，并以冻结教师蒸馏抑制参数高效适配带来的整体语义漂移。
3. 在统一多 seed 协议下报告完整消融，包括 pooled/full-span 的负结果、token-pair、蒸馏和可学习层门控，并给出 prompt 聚类 bootstrap 区间。
4. 构建以 Proper Binding、Improper Binding、Entity Neglect、颜色/形状/纹理绑定和效率为核心的后续公平对比方案；未完成的数值显式留空。

## 2. 相关工作

### 2.1 扩散模型与 DiT 文本条件

Stable Diffusion 在潜空间执行扩散过程，并通过文本 cross-attention 对去噪网络进行条件化 [3]。DiT 以 Transformer 替代 U-Net 骨干，展示了可扩展的生成建模能力 [1]；PixArt-α 进一步将大规模文本编码特征与 Diffusion Transformer 结合 [2]。本文保留 PixArt 的原始文本 cross-attention 和冻结权重，避免将主干能力变化混入方法结论。

### 2.2 文本组合与属性绑定

Composable Diffusion 将概念组合视为扩散条件的代数操作 [12]，但并不直接区分每个属性的对象归属。Structured Diffusion Guidance 使用结构化语言表示引导组合生成 [10]。Attend-and-Excite 通过迭代提升未充分激活 token 的 attention，主要缓解对象遗漏 [8]。SynGen 利用句法结构对齐 modifier 与 entity 的 attention map，是最接近本文“语言绑定”问题定义的纯文本方法 [11]。Prompt-to-Prompt [9] 与 DAAM [13] 则说明 cross-attention 可用于 token 对齐控制、编辑和诊断。

### 2.3 Grounded 与布局控制方法

GLIGEN [14]、BoxDiff [15]、MultiDiffusion [25] 和 InstanceDiffusion [26] 能够通过 box、区域或实例条件提供更强的空间/实例控制。它们适合成为能力上限或扩展实验，但由于获得了 TP-SCDA 不使用的额外控制信息，不能与纯文本结果合并排名。

### 2.4 文本到图像评测

CLIPScore [5] 可评估总体图文对齐，FID [6] 和 KID [7] 可描述生成分布质量，但均不能可靠判定“属性是否绑定到正确对象”。GenEval 使用对象检测和颜色判定分解对象、计数、位置与颜色属性 [16]；T2I-CompBench++ 以 disentangled BLIP-VQA 分别评测颜色、形状与纹理绑定 [17]；TIFA [18] 与 VQAScore [19] 使用问答式判定补充细粒度文本遵循。SynGen 的人工 Proper/Improper Binding 和 Entity Neglect 提供了直接面向绑定错误的评测定义 [11]。

## 3. 方法

### 3.1 问题定义

给定英文 prompt p、VAE latent z_t 和扩散时间步 t，冻结的 PixArt 去噪器预测噪声 epsilon。标准训练目标为：

`L_diff = E[||epsilon - epsilon_theta(z_t, t, p)||_2^2]`。

令 T5 文本编码结果为 `Y = {y_1, ..., y_N}`，其中 N 为有效文本 token 数。提示词解析器产生对象 token 集 O、属性 token 集 A，以及有向对象-属性边集 `E_OA ⊆ O x A`。本文目标不是重新预测完整场景图，而是在 E_OA 的约束下，让属性 token 的图像响应更多地来自对应对象的局部区域。

### 3.2 弱监督文本解析与 token 对齐

对每个英文 prompt，使用依存句法和词性规则提取名词短语中心词作为对象候选，使用 `amod`、颜色/材质/形状形容词、`compound` 等关系提取属性候选。属性与其修饰的名词中心词组成 edge；并列结构保留独立对象节点，不把所有形容词广播给所有名词。解析得到字符区间后，与 T5 fast tokenizer 的 offset mapping 求正长度交叠，以获得对象掩码 `M_O`、属性掩码 `M_A` 和 pair 邻接矩阵 `P ∈ {0,1}^{N x N}`。

解析标签是弱监督信号而非人工真值。对于无法可靠解析的长句、否定、隐喻或跨短语修饰，P 中不强制建立边。后续实验需在人工抽样集上报告对象、属性和 edge 的 precision、recall、F1；这些数值目前为“—”。

### 3.3 Token-pair 条件偏置

第 l 个 DiT block 的图像 patch 表示为 `X_l ∈ R^{M x D}`。原始文本 cross-attention 为：

`A_l = softmax(Q_l(X_l) K_l(Y)^T / sqrt(d))`，

`C_l = A_l V_l(Y)`。

TP-SCDA 不替换 A_l。先由对象 token 产生每个 patch 的对象响应 `h_{p,o}`，再将该响应经合法对象-属性边 P 路由给属性 token。对对象 o 和属性 a 的可学习兼容度定义为：

`g(o,a) = softplus(u_o^T v_a + b_pair)`，

其中 u_o、v_a 由低维投影得到，softplus 保证初始 pair strength 非负。属性 token a 在 patch p 上的 pair-aware bias 为：

`B_{p,a} = sum_{o in O} h_{p,o} P_{o,a} g(o,a)`。

最终注意力为：

`A'_l = softmax(Q_l K_l^T / sqrt(d) + lambda_l(t) B_l)`，

`C'_l = A'_l V_l(Y)`。

B 只写入属性 token 的 logit，且仅由 P 中的合法对象传递，因而不会将“red ball”的响应直接扩散到不相连的“blue cube”。原始文本 token 仍完整参与 K、V，故基座已有的语义能力不会被文本池化替换。

### 3.4 教师蒸馏与可学习层门控

直接训练新增偏置可能破坏冻结基座的整体文本一致性。为此，使用 Frozen PixArt 教师产生噪声预测 `epsilon_T`，并在扩散目标外加入蒸馏项：

`L = L_diff + lambda_dist E[||epsilon_theta(z_t,t,p) - epsilon_T(z_t,t,p)||_2^2]`。

该项约束 TP-SCDA 在一般文本上接近基座，同时允许其在 pair bias 激活处学习偏移。进一步地，对每层设置 role gate `lambda_l(t)`；其由层参数和时间步嵌入产生，用于检验对象/属性信息应注入何处。可学习层实验的 gate 主要保持在初始化附近，因此其独立贡献尚未被证明。

### 3.5 模型架构与推理流程

| 模块 | 输入 | 输出 | 可训练状态 | 作用 |
|---|---|---|---|---|
| T5 文本编码器 | prompt | token 特征 Y | 冻结 | 保留 PixArt 原始文本条件 |
| 依存解析与对齐器 | prompt、offset | M_O、M_A、P | 无参数 | 给出弱监督对象-属性边 |
| Pair projection | 对象/属性 token | g(o,a) | 可训练 | 学习 pair 兼容度 |
| Pair-aware attention bias | patch-object attention、P、g | B | 可训练 scale | 将对象区域响应路由给属性 token |
| DiT cross-attention | X_l、Y、B | C'_l | PixArt 冻结 | 维持原条件通路并叠加局部偏置 |
| Teacher branch | z_t、t、p | epsilon_T | 冻结 | 蒸馏约束，训练时使用 |

推理时输入只有文本，不要求任何空间标注。步骤为：(1) 解析 prompt；(2) 对齐 T5 token 并构造 P；(3) 在 DPM-Solver 去噪的各步中计算 pair bias；(4) 输出图像。与 Attend-and-Excite 相比，TP-SCDA 不进行每张图的 latent 优化；其额外时间与显存开销仍需 profiling，表中留空。

## 4. 实验设置

### 4.1 数据和训练配置

训练使用 118,286 条预处理 COCO2017 样本，输入为缓存 VAE latent、T5 文本特征及规则生成的 token 角色掩码。分辨率为 512x512，微批量为 8、梯度累积为 4（有效 batch size 32），优化器为 AdamW。PixArt 主干保持冻结，token-pair 模块和相关投影可训练。token-pair 主 run 在 epoch 5、step 60,000 的 checkpoint 评估；distillation checkpoint 在 step 2,500 评估；learnable-layers checkpoint 在 step 14,786 评估。

| 项目 | 配置 |
|---|---|
| 基座 | PixArt-XL-2-512x512 |
| 训练文本数据 | COCO2017 prepared，118,286 样本 |
| 图像分辨率 | 512x512 |
| 有效 batch size | 32 |
| 优化器 | AdamW |
| 主干 / T5 | 冻结 |
| 训练精确时间、峰值显存、可训练参数量 | — |

### 4.2 统一推理与本地评测协议

使用 `asset/samples.txt` 的 64 条固定英文 prompt，随机种子为 43、44、45，每个模型 192 张图。采样器为 DPM-Solver，20 step，CFG=4.0，分辨率 512x512。本地 CLIP ViT-B/32 计算图像与完整 prompt 的 cosine similarity。对象/属性/绑定专项由 17 个案例组成：对象存在比较目标对象描述与干扰对象描述；属性存在比较完整描述与去属性描述；绑定比较正确属性描述与交换属性描述。后者只适用于 13 个 pair 案例，共 39 个 case-seed 判断。

该专项是 CLIP contrastive proxy，不能等同人工 Proper Binding 或 B-VQA。为估计不确定性，按 prompt 案例而非按 seed 重采样，采用 10,000 次 paired bootstrap 给出相对 Frozen 的 95% 区间。

## 5. 实验结果

### 5.1 总体图文对齐

| 方法 | CLIPScore ↑ | 相对 Frozen | 说明 |
|---|---:|---:|---|
| Frozen PixArt | 0.273209 | 0.00% | 同基座基线 |
| Pooled SCDA | 0.247061 | -9.57% | 池化残差路线，停止 |
| Full-span SCDA | 0.247550 | -9.39% | 池化掩码变体，停止 |
| Token-pair SCDA | 0.266105 | -2.60% | 保留 token 对，但未蒸馏 |
| Token-pair + distillation | **0.273675** | **+0.17%** | 当前最优总体 CLIP |
| Token-pair + learnable layers | 0.273597 | +0.14% | 与蒸馏 checkpoint 接近 |

Token-pair 显著减小了 pooled 路线的退化，蒸馏后总体 CLIP 恢复到 Frozen 基线附近。然而 0.17% 的差异小于当前 3 seed 协议可支持的显著性范围，不能被表述为整体质量提升。

### 5.2 对象、属性与绑定专项

| 方法 | 对象存在 proxy ↑ | 属性存在 proxy ↑ | Binding proxy ↑ | Binding margin ↑ |
|---|---:|---:|---:|---:|
| Frozen PixArt | **86.27%** (44/51) | 70.59% (36/51) | 66.67% (26/39) | 0.01886 |
| Pooled SCDA | 80.39% (41/51) | **76.47%** (39/51) | 69.23% (27/39) | 0.01788 |
| Full-span SCDA | 80.39% (41/51) | 68.63% (35/51) | **74.36%** (29/39) | **0.02023** |
| Token-pair SCDA | 84.31% (43/51) | 66.67% (34/51) | 66.67% (26/39) | 0.01752 |
| Token-pair + distillation | 84.31% (43/51) | 74.51% (38/51) | 66.67% (26/39) | 0.01912 |
| Token-pair + learnable layers | **86.27%** (44/51) | 74.51% (38/51) | 69.23% (27/39) | 0.01819 |

| 方法 | 对象差 vs Frozen（95% CI） | 属性差 vs Frozen（95% CI） | 绑定差 vs Frozen（95% CI） |
|---|---:|---:|---:|
| Token-pair SCDA | -1.96 pp [-13.73, +7.84] | -3.92 pp [-17.65, +7.84] | +0.00 pp [-10.26, +10.26] |
| Token-pair + distillation | -1.96 pp [-7.84, +3.92] | +3.92 pp [+0.00, +9.80] | +0.00 pp [-7.14, +7.41] |
| Token-pair + learnable layers | +0.00 pp [-5.88, +5.88] | +3.92 pp [+0.00, +9.80] | +2.56 pp [+0.00, +8.33] |

learnable-layers 的 binding proxy 较 Frozen 增加 2.56 个百分点，但仅对应 1 个额外正确案例；区间下界为 0。更高的 full-span 点估计与其严重的总体 CLIP 退化并存，不能作为恢复 pooled 路线的证据。

### 5.3 消融实验

| 消融因素 | 对照 | 观察 | 结论 |
|---|---|---|---|
| 池化 vs token-pair | Pooled 0.247061；Token-pair 0.266105 CLIP | token-pair 将相对退化从 -9.57% 缩小到 -2.60% | 池化会损失绑定结构；保留 token 粒度是必要设计 |
| 教师蒸馏 | Token-pair 0.266105；Distillation 0.273675 CLIP | CLIP 恢复 +0.007570 | 蒸馏有效抑制基座漂移，但不证明绑定提高 |
| 可学习层门控 | Distillation 0.273675；Learnable 0.273597 CLIP | 差异 -0.000078；binding 26/39 到 27/39 | 当前 gate 未显示独立贡献，需更大绑定集 |
| Pooled/full-span | Frozen 0.273209 CLIP | 分别 -9.57%、-9.39% | 保留为负结果，不追加训练预算 |

### 5.4 与属性绑定方法的公开对比

外部方法不在本项目部署。下表保留原论文的评测协议与数字，不能与第 5.2 节的 CLIP proxy 合并成总排名。

| 数据集/判定器 | 方法 | Proper Binding ↑ | Improper Binding ↓ | Entity Neglect ↓ | Color B-VQA ↑ | Shape B-VQA ↑ | Texture B-VQA ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| ABC-6K / 人工 | SynGen [11] | **0.6368** | **0.1437** | 0.3441 | — | — | — |
| ABC-6K / 人工 | Attend-and-Excite [8] | 0.5626 | 0.2643 | **0.3318** | — | — | — |
| ABC-6K / 人工 | Structured Diffusion [10] | 0.5147 | 0.2952 | 0.3457 | — | — | — |
| ABC-6K / 人工 | Stable Diffusion | 0.5270 | 0.2720 | 0.3657 | — | — | — |
| T2I-CompBench++ / B-VQA | Stable Diffusion v2 [17] | — | — | — | 0.5065 | 0.4221 | 0.4922 |
| T2I-CompBench++ / B-VQA | Structured Diffusion + SD v2 [17] | — | — | — | 0.4990 | 0.4218 | 0.4900 |
| T2I-CompBench++ / B-VQA | Attend-and-Excite + SD v2 [17] | — | — | — | **0.6400** | **0.4517** | **0.5963** |
| ABC-6K / 人工；T2I-CompBench++ / B-VQA | TP-SCDA（本方法） | — | — | — | — | — | — |

## 6. 待补做的公平对比实验

### 6.1 统一主对比协议

对比对象为 Frozen PixArt、TP-SCDA、TP-SCDA + distillation、TP-SCDA + learnable layers，以及纯文本绑定方法 Attend-and-Excite、Structured Diffusion、SynGen。若外部方法不能直接适配 PixArt attention 接口，应同时报告“原作者 Stable Diffusion 协议结果”和“同 prompt/同评测器的复现结果”，并禁止混合排名。

| 方法 | Proper Binding ↑ | Improper Binding ↓ | Entity Neglect ↓ | Color B-VQA ↑ | Shape B-VQA ↑ | Texture B-VQA ↑ | GenEval color attribution ↑ | Time/image ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen PixArt | — | — | — | — | — | — | — | — |
| Attend-and-Excite | — | — | — | — | — | — | — | — |
| Structured Diffusion | — | — | — | — | — | — | — | — |
| SynGen | — | — | — | — | — | — | — | — |
| TP-SCDA | — | — | — | — | — | — | — | — |
| TP-SCDA + distillation | — | — | — | — | — | — | — | — |
| TP-SCDA + learnable layers | — | — | — | — | — | — | — | — |

### 6.2 数据、采样与统计方案

1. **人工绑定集：** 使用 ABC-6K、A&E、DVMP 或构造至少 200 条双对象双属性 prompt。每个 prompt 固定生成数、随机种子、分辨率、采样器与 CFG。两名独立标注员分别标记对象存在、属性归属和错误绑定；报告 Proper Binding、Improper Binding、Entity Neglect、Cohen's kappa 和按 prompt 配对 bootstrap CI。
2. **自动绑定集：** 使用 T2I-CompBench++ 的 color、shape、texture 子集，运行官方 disentangled BLIP-VQA；另运行 GenEval 的 two-object、colors、color attribution，以检测器与 VQA 的不同偏差交叉验证。
3. **全局质量：** 在同一生成 prompt 集上报告 CLIPScore。FID-10k、KID、LPIPS、峰值显存和 s/image 当前均为“—”；在准备固定 COCO reference、10,000 张以上样本和统一硬件后补充。
4. **显著性：** 所有二元成功率按 prompt 聚类 bootstrap；模型差异报告 95% CI。绑定主张必须同时在人工指标和至少一个自动绑定指标上成立，不能仅使用 CLIP margin。

### 6.3 计划中的表格与图件

| 编号 | 内容 | 当前状态 |
|---|---|---|
| 表 1 | 方法与架构对比 | 已完成 |
| 表 2 | 本地总体 CLIP 与绑定 proxy | 已完成 |
| 表 3 | Token-pair 消融 | 已完成 |
| 表 4 | 与属性绑定方法的统一主比较 | 数据待补 |
| 表 5 | FID/KID、LPIPS、效率与显存 | 数据待补 |
| 图 1 | TP-SCDA 架构：解析器、pair 图、cross-attention bias、蒸馏分支 | 待绘制 |
| 图 2 | 双对象双属性案例的 attention/生成可视化 | 待生成 |
| 图 3 | 属性类别和 prompt 难度的分组柱状图与 CI | 数据待补 |

## 7. 讨论与局限性

第一，当前绑定指标是本地 CLIP 对比代理。SynGen 已发现 phrase-to-image CLIP 与人工多数选择的一致率有限 [11]，因此本研究不能把 69.23% proxy 解释为同等意义的人工绑定准确率。第二，17 个案例和 39 个 binding case-seed 判断不足以稳定估计小差异；learnable-layers 的一个额外正确案例需要通过至少数百个独立 prompt 验证。第三，规则依存解析会在并列、长修饰、数量、否定和抽象概念上出错，且其标签质量尚未人工审计。第四，TP-SCDA 的模型、训练参数、采样时间和显存尚未形成完整 profiling；参数效率和运行效率主张必须等待补测。第五，外部方法大多基于 Stable Diffusion，PixArt 的文本编码器和注意力实现不同，因此跨论文公开数值只能用于任务定位，不能用于 SOTA 宣称。

## 8. 结论

本文提出 TP-SCDA，在冻结 PixArt 中以 token-pair 而非 pooled semantic residual 表达对象-属性绑定。统一评测显示，token-pair + distillation 保持了 Frozen PixArt 的总体 CLIPScore，learnable-layers 在小规模 CLIP binding proxy 中表现出 +2.56 个百分点的正向趋势。与此同时，pooled 和 full-span SCDA 的显著 CLIP 退化表明，丢失 token 对应关系的全局池化不是可行方向。当前证据不足以证明 TP-SCDA 超过现有属性绑定方法；下一阶段应优先在 SynGen 风格人工标注、T2I-CompBench++ B-VQA、GenEval color attribution 和效率指标上对齐协议，并以人工绑定评测作为主要结论依据。

## 参考文献

[1] Peebles W, Xie S. Scalable Diffusion Models with Transformers. ICCV, 2023.

[2] Chen J, Yu J, Ge C, et al. PixArt-alpha: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis. ICLR, 2024.

[3] Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR, 2022.

[4] Raffel C, Shazeer N, Roberts A, et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR, 2020.

[5] Hessel J, Holtzman A, Forbes M, Bras R, Choi Y. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. EMNLP, 2021.

[6] Heusel M, Ramsauer H, Unterthiner T, Nessler B, Hochreiter S. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. NeurIPS, 2017.

[7] Binkowski M, Sutherland D J, Arbel M, Gretton A. Demystifying MMD GANs. ICLR, 2018.

[8] Chefer H, Alaluf Y, Vinker Y, et al. Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models. ACM TOG, 2023.

[9] Hertz A, Mokady R, Tenenbaum J, Aberman K, Pritch Y, Cohen-Or D. Prompt-to-Prompt Image Editing with Cross Attention Control. ICLR, 2023.

[10] Feng W, He X, Fu T-J, Jampani V, Akula A, Narayana P, Basu S. Training-free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis. ICLR, 2023.

[11] Lee J, Kim K, Kim G. Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment. NeurIPS, 2023.

[12] Liu N, Li S, Du Y, Torralba A, Tenenbaum J B. Compositional Visual Generation with Composable Diffusion Models. ECCV, 2022.

[13] Tang R, Pandey A, Jiang Z, Yang G, Kumar K, Agarwal A, Hays J. What the DAAM: Interpreting Stable Diffusion Using Cross Attention. ACL, 2023.

[14] Li Y, Liu H, Wu Q, et al. GLIGEN: Open-Set Grounded Text-to-Image Generation. CVPR, 2023.

[15] Xie J, et al. BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion. ICCV, 2023.

[16] Ghosh D, Hajishirzi H, Schmidt L. GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. NeurIPS, 2024.

[17] Huang K, Duan C, Sun K, Xie E, Li Z, Liu X. T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation. IEEE TPAMI, 2025.

[18] Hu Y, Liu B, Kasai J, et al. TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering. ICCV, 2023.

[19] Lin C, Wu Z, et al. VQAScore: A Vision-Language Model for Text-to-Image Evaluation. arXiv:2404.01222, 2024.

[20] Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.

[21] Zhang L, Rao A, Agrawala M. Adding Conditional Control to Text-to-Image Diffusion Models. ICCV, 2023.

[22] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.

[23] Mou C, Wang X, Xie L, et al. T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability for Text-to-Image Diffusion Models. AAAI, 2024.

[24] Saharia C, Chan W, Saxena S, et al. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. NeurIPS, 2022.

[25] Bar-Tal O, Ofri-Amar D, Fridman R, et al. MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation. ICML, 2023.

[26] Wang X, et al. InstanceDiffusion: Instance-level Control for Image Generation. CVPR, 2024.
