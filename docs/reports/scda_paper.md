# 面向文本语义分层条件的 PixArt 扩散 Transformer 适配器：设计、训练稳定性与负结果分析

## 摘要

文本到图像扩散模型在复杂提示词下常出现对象遗漏、属性绑定错误和关系不遵循等问题。为在不依赖边界框、分割掩码或人工场景图标注的前提下增强文本条件表达，本文设计了文本语义条件扩散适配器（Semantic-Conditioned DiT Adapter, SCDA）。该方法以英文提示词为输入，利用依存句法和规则伪标注得到对象、属性和关系 token 掩码；随后将全局、对象、属性和关系四类 T5 条件分别池化，经零初始化瓶颈适配器映射为残差，并按照 DiT 层级和扩散时间步注入 PixArt 图像 token。训练时冻结 PixArt 主干与 T5，仅更新新增适配器、层级尺度及时间门控参数。

在 8,617 条 COCO2014 样本上进行的 10 epoch 训练表明，SCDA 可稳定收敛。为避免单样本或单随机种子结论，本文在 64 条固定文本、3 个随机种子、512x512、DPM-Solver 20 步、CFG=4.0 的统一协议下，对 baseline、两组历史 gate 试验、原始改进 SCDA 及加入残差正则的 SCDA 使用 CLIP ViT-B/32 重评分。baseline 的 CLIP 均值为 0.273209，原始改进 SCDA 为 0.264862，加入正则后为 0.268161；正则方案较原始 SCDA 提升 1.25%，但仍较 baseline 低 1.85%。结果表明，残差正则减轻了退化并保持数值稳定，但尚未证实整体文本图像一致性超过 baseline。

**关键词：** 文本到图像生成；扩散 Transformer；PixArt；条件适配器；文本图像一致性；负结果

## 1. 引言

扩散 Transformer（Diffusion Transformer, DiT）将图像潜变量建模为 token 序列，并使用文本编码器输出作为条件进行迭代去噪。PixArt 等模型具备较强的开放域生成能力，但提示词中的全局场景、对象、属性与对象间关系通常以同一文本序列进入 cross-attention。模型需要隐式决定不同词语的作用时机与重要性，在多对象、细粒度属性和空间/动作关系场景中可能产生对象竞争、属性串位或关系失真。

显式可控生成往往依赖边界框、分割图、姿态或场景图，这些控制信号提高了标注和推理成本。本文研究一个更受限的问题：仅利用原始英文文本，能否以轻量、参数高效的方式向已有 PixArt 主干增加语义角色信息？为此，我们构建 SCDA：用句法规则从文本中得到对象、属性、关系伪标签；将语义角色汇聚成四种条件；再以零初始化残差在不同 DiT 层与时间步注入。

本文的贡献在于方法实现、可复现实验协议及负结果分析，而不是宣称已取得超越 baseline 的效果。第一，给出从词级依存分析到 T5 子词的字符区间对齐流程。第二，给出不改变原 cross-attention 且可由零初始化保证初始等价性的语义残差结构。第三，在训练与推理配置一致的前提下完成多 seed 评测，并报告当前方法未超过 baseline 的结果。第四，基于此结果给出明确的实验决策规则，避免通过选择性提示词或单 seed 夸大收益。

## 2. 相关工作

DiT 在潜空间中通过 Transformer 去噪，文本条件通常经 cross-attention 或条件归一化进入网络。PixArt-α 采用 T5-XXL 文本特征和大规模 DiT 结构，是研究条件注入机制的合适基础。参数高效微调中的 adapter、低秩更新与残差调制提供了在冻结主干时加入新能力的思路。

文本图像一致性和组合生成研究关注对象存在、计数、属性绑定、空间关系与动作关系。CLIPScore 可作为整体图文语义相似性的自动代理，但不能充分衡量组合关系。因此，本文将 CLIP 用于统一的初步比较，而不将其视为唯一的结构遵循证据。结构化文本条件常采用场景图或检测标注；SCDA 的差异在于只使用规则伪标注，代价是必须面对标签噪声与对齐误差。

## 3. 方法

### 3.1 问题定义与基线

给定提示词 p、扩散时间步 t、干净潜变量 z0 和噪声 epsilon，去噪网络预测噪声的标准目标为：

L_diff = E[||epsilon - epsilon_theta(z_t, t, p)||^2]。

PixArt 将 T5 token 表示投影到隐藏维度 D=1152，并在每个 DiT block 的文本 cross-attention 中使用完整文本序列。SCDA 保留这一路径，因此任何性能变化可归因于新增的残差条件，而非删除或替换基线文本条件。

### 3.2 弱监督语义角色标注

对英文 prompt 使用 spaCy 依存分析和规则进行标注。名词或专有名词短语中心词标为对象；amod、compound、nummod、poss 等修饰词标为属性；动词及其宾语、介词短语等标为关系。为与预先提取的 T5 特征一致，文本先经过与特征提取相同的清洗流程。对于 spaCy 词区间 (s_i,e_i) 与 T5 fast tokenizer 子词区间 (a_j,b_j)，若两者存在正长度交叠，则该子词属于相应语义组。

每个样本保存三个二值掩码 M_obj、M_attr、M_rel，形状为 [3,120]。全局条件使用 T5 attention mask。该标注不声称是人工真值；它是可扩展的弱监督信号，后续应通过人工抽样估计其 precision、recall 和错误模式。

### 3.3 语义池化与零初始化适配器

设 Y 为投影后的文本 token，v 为有效 token 掩码，m^k 为类别 k 的掩码。第 k 类条件为：

c^k = sum_j(m_j^k v_j Y_j) / max(1, sum_j m_j^k v_j)，k 属于 {global, object, attribute, relation}。

空类别的输出为零。每一类通过一个 bottleneck adapter 变换：A_k(c)=W_up,k SiLU(W_down,k c)。本实验中 bottleneck 维度 r=64；上投影和 bias 被置零。因此在初始状态 A_k(c)=0，加载同一 PixArt checkpoint 时 SCDA 初始前向与 baseline 等价。

### 3.4 层级和时间步注入

对第 l 个 block，残差表示为：

R_l(t) = s * sum_k u_lk beta_lk gamma_k(t) A_k(c^k)，

其中 u_lk 是固定层窗口，beta_lk 是可学习尺度，gamma_k(t)=sigmoid(W_t e_t)_k 是时间门控，s 是全局残差系数。全局分支作用于全部层；对象分支作用于前半层；关系分支作用于中间层；属性分支作用于后半层且避开最后四层。该分配是待验证的归纳偏置，而非已经被实验证实的最佳策略。

### 3.5 训练策略

主干 PixArt 和 T5 均冻结。可训练部分包括四个 adapter、28x4 层级尺度和时间门控，总计约 0.60M 参数。对象、属性与关系分支以 0.1 的概率独立 dropout，全局条件始终保留。训练仍只使用扩散噪声预测损失，没有直接的对象、属性或关系监督项。

## 4. 实验设置

### 4.1 数据、硬件和训练配置

主实验使用 COCO2014Prepared10K 中可用的 8,617 条样本及其 512 分辨率 VAE 特征、T5 caption 特征和语义掩码，在一张 NVIDIA L20（46 GB）上训练 10 epoch，共 2,700 step。batch size 为 32，AdamW 学习率为 5e-5，weight decay 为 0.03，adapter dim 为 64，semantic dropout 为 0.1，残差系数为 0.25，启用 global、object、attribute、relation 四个分支。

为验证链路，还进行过 512 条样本、160 step 的 SCDA pilot。历史上存在 Gate v2 与 Gate v3 两组训练日志，但其完整训练配置未全部留存，本文只将其作为可评分的历史对照，不将其用于严格的训练参数因果分析。

### 4.2 统一生成与评价协议

所有可生成方法使用 `asset/samples.txt` 的 64 条固定文本和种子 43、44、45，共 192 张图像/方法。统一使用 512x512、DPM-Solver、20 step、CFG=4.0。改进 SCDA 推理显式设置 semantic_residual_scale=0.25，与训练一致。使用本地 CLIP ViT-B/32 计算匹配 prompt 与图像 embedding 的余弦相似度。对于超过 CLIP 位置上限的提示词，以 77 token 截断；这一规则对所有方法相同。

目前未准备完整且对齐的真实 COCO 参考图像集合，因此不报告 FID/KID，且不以图像清晰度或熵等代理指标代替保真度结论。

## 5. 结果

### 5.1 训练稳定性

SCDA pilot 的扩散损失从 0.2076 降至 0.1516，语义残差范数从 0 增长至 0.0577，未出现 NaN/Inf。全量 improved SCDA 的最终 diffusion loss 为 0.132721，最终梯度范数为 0.009002，semantic residual norm 为 0.407930，10 epoch 内未出现 NaN/Inf。这证明新增路径被优化器激活并且训练数值稳定；但扩散损失和残差范数本身不是文本一致性提升的证据。

| 方法 | 数据/步数 | 关键配置 | 可核验训练结果 |
|---|---:|---|---|
| Gate v2 / Trial 02 | 历史日志 70 点 | 完整配置缺失 | loss 0.1907 -> 0.1513；历史记录曾出现 grad_norm NaN |
| Gate v3 / Trial 09 | 历史日志 70 点 | neutrality + consistency 正则；其余缺失 | loss 0.1908 -> 0.1519；日志无 NaN |
| SCDA pilot | 512 条，160 step | 冻结主干；语义 adapter | loss 0.2076 -> 0.1516；residual 0 -> 0.0577；无 NaN |
| SCDA improved | 8,617 条，2,700 step | r=64, lr=5e-5, wd=0.03, dropout=0.1, s=0.25, 全分支 | final loss 0.132721；residual 0.407930；无 NaN/Inf |

### 5.2 多种子文本图像一致性

表中“总体均值”按 192 张图像直接平均；“图级标准差”刻画不同 prompt 的离散程度。由于每个方法只使用 3 个 seed，本文不把小数差距解释为统计显著性，而是将其作为当前协议下的方向性证据。

| 方法 | seed 43 | seed 44 | seed 45 | 总体 CLIP 均值 | 图级标准差 | 相对 baseline |
|---|---:|---:|---:|---:|---:|---:|
| Frozen PixArt baseline | 0.273054 | 0.266827 | 0.279745 | 0.273209 | 0.066340 | 0.00% |
| Trial 02 | 0.260878 | 0.258000 | 0.259319 | 0.259399 | 0.056447 | -5.06% |
| Trial 09 | 0.260015 | 0.254790 | 0.255296 | 0.256700 | 0.056175 | -6.05% |
| SCDA improved | 0.265700 | 0.264239 | 0.264648 | 0.264862 | 0.058155 | -3.06% |
| SCDA loss_reg | 0.271075 | 0.264239 | 0.269169 | 0.268161 | 0.065450 | -1.85% |
| SCDA global+object | 0.270117 | 0.263931 | 0.267111 | 0.267053 | 0.067150 | -2.25% |

在三个 seed 中，baseline 均高于 SCDA 版本；loss_reg 在 seed 43 接近 baseline，但 seed 44/45 仍低于 baseline。global+object 消融均值为 0.267053，低于四分支 loss_reg 的 0.268161，说明在当前设置下移除属性和关系分支没有带来整体 CLIP 增益。因此，当前结果不能支持“SCDA 整体 CLIP 优于 baseline”的假设；残差正则仍是目前最优的 SCDA 配置，但尚未恢复到基线水平。

## 6. 讨论

第一，稳定训练与有效生成并不等价。loss_reg 的最终 total/diffusion/semantic-reg loss 分别为 0.133204/0.133001/0.020297，残差范数为 0.136511，低于原始 SCDA 的 0.407930；CLIP 同步上升但仍未超过 baseline，说明“更小残差”是必要的稳定性因素，却不是充分的语义监督。标准扩散损失仅要求拟合训练噪声，并不直接约束对象计数、属性绑定或关系遵循。

第二，伪标签质量可能构成上限。对象、属性、关系由规则解析得到，其中属性和关系天然比名词更依赖上下文。错误切分、T5 子词对齐偏差或缺失标注都可能把不可靠信号广播到所有图像 token。当前只有掩码存在比例，没有人工标签质量估计，因而不能判断问题主要来自模型结构还是输入监督。

第三，当前完整方法把四分支同时打开，无法区分哪一支带来收益或退化。对象分支可能有助于主体显著性，而关系分支的噪声可能拉低整体分数；残差系数 0.25 只是保守首选，不是通过搜索确定的最优值。没有消融的“完整方法”不能证明各组件有效。

第四，CLIP 对整体风格和实体语义敏感，但对“红色球在蓝色盒子左侧”等可验证关系不够充分；64 条文本也混合了短文本、风格词堆叠和长叙述。仅报告总体 CLIP 容易掩盖特定能力的变化。更严谨的结论需补充对象、属性、关系分组的 GenEval/T2I-CompBench/VQA 评分与人工盲评。

## 7. 后续实验方案

后续实验按以下顺序执行，并在每一步使用同一 64x3 快速筛选协议：

1. 残差强度扫描。测试 s=0.10、0.25、0.50、1.00，固定其余训练和推理设置。候选配置必须在至少两个 seed 上不低于 baseline，才进入扩大评测。
2. 语义分支消融。比较 object、object+attribute 与 all，使用上一步最优残差系数。关系分支只有在关系 prompt 子集存在清晰增益时才保留。
3. 伪标签审计。随机抽取至少 200 条 prompt，对对象、属性、关系标注 precision/recall/F1，并记录多词实体、否定、并列、介词短语和长文本的失败模式。
4. 评测扩展。构建对象、属性、关系三类各不少于 100 条的 prompt 集，每个配置至少使用 5 个 seed，报告均值、标准差和分组结果；同时引入 GenEval、T2I-CompBench 或可复核 VQA 评分。
5. 图像质量评估。准备固定的真实 COCO reference 和一致预处理后计算 FID/KID；只有在一致性和质量都不退化时才考虑进一步训练。
6. 目标函数改进。在完成上述诊断后，再考虑加入低权重、可控的语义一致性或偏好目标，并以小规模试验确认不会损伤图像质量。

## 8. 结论

本文实现了 SCDA，一种面向 PixArt 的参数高效文本语义分层条件适配器。方法利用规则伪标签构造对象、属性和关系条件，以零初始化 adapter、层级窗口和时间门控注入残差。实验确认该方案在 8,617 条样本上的训练稳定且可复现；加入残差正则后 CLIP 从 0.264862 提升至 0.268161，但仍低于 baseline 的 0.273209。故当前工作应被视为一个稳定、可用于消融的研究原型，而非已证实优于 PixArt baseline 的方法。下一阶段应优先缩小设计空间、审计伪标签、采用结构化分项指标并引入真实图像质量评价。

## 参考文献

[1] Peebles W, Xie S. Scalable Diffusion Models with Transformers. ICCV, 2023.

[2] Chen J, Wu Y, Luo S, et al. PixArt-alpha: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis. ICLR, 2024.

[3] Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.

[4] Hessel J, Holtzman A, Forbes M, et al. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. EMNLP, 2021.

[5] Gokhale T, et al. Benchmarking Spatial Relationships in Text-to-Image Generation. arXiv:2305.05418, 2023.

[6] Ghosh D, et al. Geneval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. NeurIPS, 2024.
