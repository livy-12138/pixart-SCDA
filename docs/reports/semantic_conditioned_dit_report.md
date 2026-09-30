# 面向文本语义分层的 DiT 条件嵌入方法

## 摘要

扩散 Transformer（DiT）通常将完整文本序列作为统一条件输入，使不同语义角色，如场景、对象、属性和对象间关系，在相同层级和相同去噪阶段参与图像生成。这种统一处理可能导致多对象遗漏、属性绑定错误以及复杂关系遵循不足。本文提出一种面向文本到图像生成的**文本语义分层条件适配方法**（Semantic-Conditioned DiT Adapter, SCDA）。在不引入边界框、分割掩码或关系图标注的条件下，SCDA 首先对英文 prompt 进行依存句法分析和规则伪标注，将 T5 token 划分为对象、属性和关系三类语义组；随后利用字符区间将词级标签对齐到 T5 SentencePiece 子 token，并对各组隐藏状态进行池化。四类条件，即全局、对象、属性和关系条件，经零初始化瓶颈适配器转化为残差，并按照 DiT 层级和扩散时间步动态注入图像 token。该方法保留原始 PixArt 的文本 cross-attention，不改变图像 token 数量，仅训练约 0.60M 新增参数。本文档给出问题定义、方法细节、训练流程、实验协议、消融设计和局限性，可作为后续论文初稿的技术依据。

**关键词：** 文本到图像生成；扩散 Transformer；PixArt；条件嵌入；文本一致性；弱监督

## 1. 引言

### 1.1 研究背景

DiT 将图像 latent 划分为 patch token，并通过自注意力建模图像内部依赖，通过 cross-attention 接收文本条件。PixArt-α 使用 T5-XXL 特征作为文本条件，具备清晰的 DiT 结构与可扩展的训练流程。然而，基线将文本 token 作为同质序列处理，模型需要从统一 token 序列中自行推断以下差异：

- 哪些 token 描述全局场景；
- 哪些 token 对应待生成对象；
- 哪些 token 是对象属性；
- 哪些 token 描述对象间的动作或空间关系。

在复杂 prompt 中，这种隐式分工容易导致对象竞争、属性串位和关系错误。现有空间可控方法经常依赖 bbox、mask 或注意力图，但这些额外标注与推理期优化会增加数据和部署成本。本文聚焦于**仅使用原始英文文本**的条件嵌入改进。

### 1.2 问题定义

给定文本 prompt $p$、高斯噪声 latent $z_t$ 和时间步 $t$，目标是训练去噪网络 $\epsilon_\theta$，使其预测噪声 $\epsilon$：

$$
\mathcal{L}_{\mathrm{diff}} = \mathbb{E}_{z_0,\epsilon,t,p}
\left[\left\|\epsilon - \epsilon_\theta(z_t,t,p)\right\|_2^2\right].
$$

与基线不同，本文将 $p$ 转换为全局、对象、属性和关系四类条件，并在保留基线文本 cross-attention 的同时，为不同语义角色提供显式、分层的残差路径。

### 1.3 贡献

本文的预期贡献如下，必须通过后续实验验证：

1. 提出一种不依赖 bbox、mask 或 attention-map 优化的 DiT 文本条件分层机制。
2. 提出词级依存分析到 T5 SentencePiece token 的字符区间对齐流程，可从未额外标注的英文 caption 生成对象、属性和关系伪标签。
3. 提出层级和时间步感知的零初始化语义残差适配器，在加载 PixArt 基线 checkpoint 时保持初始等价性。
4. 提供参数高效训练方案，仅优化约 0.60M 新增参数，并保留原始 PixArt 主干与 cross-attention。

## 2. 相关工作

### 2.1 Diffusion Transformer

DiT 在 latent 空间中使用 Transformer 进行扩散去噪。PixArt-α 采用 AdaLN-single 时间步调制、图像 token self-attention 与文本 cross-attention。该架构适合研究条件如何进入图像 token 表示，但基线没有显式区分文本语义角色。

### 2.2 文本一致性与组合生成

文本一致性方法通常通过更强的文本编码器、cross-attention 操作、推理期引导或额外结构控制提高 prompt adherence。推理期方法可能依赖 attention map 的空间可解释性，并带来额外采样成本。本文不修改 cross-attention 权重，也不在采样时迭代优化，而是在训练阶段学习轻量条件残差。

### 2.3 结构化文本条件

对象、属性和关系的结构化表示已被用于视觉关系理解、场景图生成和可控图像生成。与需要人工对象框或关系标注的方法不同，本文使用英文依存句法与规则产生弱监督角色标签，因此需特别评估伪标签噪声及其对生成质量的影响。

## 3. 方法

### 3.1 PixArt 基线

输入 latent $z_t \in \mathbb{R}^{B\times4\times H\times W}$ 经 patch embedding 变为图像 token：

$$
X_0 = \mathrm{PatchEmbed}(z_t) + P, \quad X_0 \in \mathbb{R}^{B\times Q\times D}.
$$

对 256x256 图像，latent 尺寸为 32x32，patch size 为 2，因此 $Q=16\times16=256$。T5 特征首先经 PixArt 的 CaptionEmbedder 映射到 $D=1152$：

$$
Y = \mathrm{CaptionEmbedder}(H_{\mathrm{T5}}), \quad Y\in\mathbb{R}^{B\times L\times D}.
$$

每个 PixArt block 执行：

$$
X \leftarrow X + \mathrm{SA}(\mathrm{AdaLN}(X,t)),
$$
$$
X \leftarrow X + \mathrm{CA}(X,Y),
$$
$$
X \leftarrow X + \mathrm{MLP}(\mathrm{AdaLN}(X,t)).
$$

SCDA 不替换上述计算，并继续把完整 token 序列 $Y$ 送入 cross-attention。

### 3.2 词级伪标签生成

给定英文 prompt，使用 spaCy 英文依存分析器生成词级结构：

- **对象（object）**：名词或专有名词短语的中心词；
- **属性（attribute）**：依存关系为 `amod`、`compound`、`nummod` 或 `poss` 的修饰词；
- **关系（relation）**：动词及其直接宾语/介词宾语，或名词短语中的介词关系，例如 `car beside bicycle`。

为了与特征提取阶段一致，caption 先经过 PixArt T5 预处理函数两次清洗。随后使用 T5 fast tokenizer 返回每个 SentencePiece token 的字符区间 $(a_j,b_j)$，并用 spaCy 词的字符区间 $(s_i,e_i)$ 进行对齐。当：

$$
\min(e_i,b_j) - \max(s_i,a_j) > 0,
$$

则将第 $j$ 个 T5 token 归入该词对应的语义组。每个样本保存三个二值掩码：

$$
M^{\mathrm{obj}},M^{\mathrm{attr}},M^{\mathrm{rel}}\in\{0,1\}^{L}.
$$

全局条件不依赖伪标签，始终使用原始 T5 attention mask。

### 3.3 语义组池化

对于语义组 $k\in\{g,o,a,r\}$，令 $m^k$ 表示对应 token 权重，$v$ 表示有效文本 token mask，组表示定义为：

$$
c^k = \frac{\sum_{j=1}^{L}m_j^k v_jY_j}
{\max(1,\sum_{j=1}^{L}m_j^k v_j)}.
$$

其中 $m^g=v$；$m^o=M^{\mathrm{obj}}$；$m^a=M^{\mathrm{attr}}$；$m^r=M^{\mathrm{rel}}$。当某类伪标签为空时，分母被截断为 1，该组输出为零向量，不会伪造语义信息。

### 3.4 语义残差适配器

每类条件使用一个共享瓶颈适配器：

$$
A_k(c^k)=W_k^{\mathrm{up}}\sigma(W_k^{\mathrm{down}}c^k),
$$

其中 $W_k^{\mathrm{down}}\in\mathbb{R}^{r\times D}$，$W_k^{\mathrm{up}}\in\mathbb{R}^{D\times r}$，$r=64$，$\sigma$ 为 SiLU。上投影 $W_k^{\mathrm{up}}$ 和 bias 均零初始化，因此：

$$
A_k(c^k)=0
$$

在训练开始时成立。该初始化保证加载预训练 PixArt 后，新增路径不会改变首次前向结果。

### 3.5 层级与时间步调制

令第 $l$ 个 DiT block 的固定语义可用掩码为 $u_{l,k}$，可学习层级尺度为 $\beta_{l,k}$。时间调制由时间 embedding $e_t$ 经线性层得到：

$$
\gamma_k(t)=\mathrm{sigmoid}(W_t e_t)_k.
$$

注入到第 $l$ 层前的残差为：

$$
R_l(t)=\sum_{k\in\{g,o,a,r\}}u_{l,k}\beta_{l,k}\gamma_k(t)A_k(c^k),
$$
$$
X_l\leftarrow X_l+R_l(t).
$$

残差在 token 维度广播，即同一语义条件加到该层所有图像 patch token。由于本文没有 bbox，该设计意在调节不同语义角色在不同去噪阶段的表达强度，不提供显式空间定位。

### 3.6 默认层级窗口

对 28 层 PixArt-XL/2，当前实现的默认窗口如下：

| 条件 | 注入 block | 设计意图 |
|---|---:|---|
| 全局 | 0-27 | 保持场景、风格与完整 prompt 语义 |
| 对象 | 0-13 | 强化主体类别、共存与粗结构 |
| 关系 | 9-18 | 强化动作、交互与相对关系 |
| 属性 | 14-23 | 强化颜色、材质、大小和后期细节 |
| 最后 4 层 | 不注入对象/属性/关系 | 尽量保留基线纹理生成能力 |

该划分是待验证的归纳偏置，不应在论文中表述为已证实结论。

### 3.7 条件 dropout

训练时，对象、属性和关系条件以概率 $p=0.1$ 独立置零；全局条件始终保留。该设计使模型在角色标签缺失或伪标签不可靠时仍依赖完整的 baseline 文本 cross-attention，减少对规则标签的过拟合。

### 3.8 训练目标与参数量

第一阶段仅使用标准噪声预测损失 $\mathcal{L}_{\mathrm{diff}}$，不额外使用属性或关系监督损失。冻结 PixArt 主干与 T5，仅优化：

- 四个语义瓶颈适配器；
- 28x4 个层级尺度；
- 1152->4 的时间步门控层。

在 $D=1152,r=64$ 时，参数量为：

| 模块 | 参数量 |
|---|---:|
| 单个适配器 | 148,672 |
| 四个适配器 | 594,688 |
| 层级尺度 | 112 |
| 时间步门控 | 4,612 |
| **总计** | **599,412** |

## 4. 实现与训练流程

### 4.1 数据预处理

每个已有 T5 特征文件原本包含：

```text
caption_feature: [1, 120, 4096]
attention_mask:  [1, 120]
```

预处理后额外写入：

```text
semantic_token_masks: [3, 120]
```

三个通道依次对应 object-root、attribute 和 relation。预处理脚本应使用与 T5 特征生成完全相同的 tokenizer 与 caption 清洗流程，否则字符区间对齐不成立。

### 4.2 训练配置

当前 COCO 配置采用：256x256、batch size 32、10 epochs、AdamW、初始学习率 $10^{-4}$、fp16、语义 dropout 0.1。训练入口为：

```bash
accelerate launch train_scripts/train.py configs/PixArt_xl2_coco2014_semantic.py
```

生成伪标签掩码的完整命令见 [SEMANTIC_TRAINING.md](../tools/SEMANTIC_TRAINING.md)。

### 4.3 训练监控

除扩散损失外，建议记录：

- 四类适配器输出范数；
- 四类适配器梯度范数；
- 层级尺度 $\beta_{l,k}$ 的均值和绝对值；
- 时间调制 $\gamma_k(t)$ 随时间步的曲线；
- 空对象/属性/关系掩码比例；
- 训练吞吐、显存与相对基线训练时间。

如果适配器输出持续为零、梯度接近零，说明模型可能忽略新增路径；若尺度迅速增大，则需降低学习率、提高 dropout 或增加残差范数约束。

## 5. 实验设计

### 5.1 研究问题

- RQ1：语义分层条件是否提高总体文本-图像一致性？
- RQ2：对象、属性和关系分支分别是否提升对应的组合能力？
- RQ3：层级/时间步调制是否优于所有层共享固定注入？
- RQ4：规则伪标签噪声是否会抵消语义分层带来的收益？
- RQ5：参数高效微调是否能接近或超过全量微调的条件一致性提升？

### 5.2 对比方法

所有方法使用相同 PixArt checkpoint、相同训练数据、相同采样器、CFG、采样步数和随机种子集合：

| ID | 方法 | 目的 |
|---|---|---|
| B0 | Frozen PixArt baseline | 原始生成能力 |
| B1 | 全量 PixArt 微调 | 区分容量增益与方法增益 |
| B2 | 仅全局 adapter | 检验仅增加参数的影响 |
| M1 | Global + object | 检验对象层级条件 |
| M2 | M1 + attribute | 检验属性条件 |
| M3 | M2 + relation | 检验关系条件 |
| M4 | M3 + fixed layer weights | 检验层级可学习尺度 |
| M5 | 完整 SCDA | 检验时间步调制与结构化 dropout |

### 5.3 评价指标

不要只报告 FID。建议至少报告：

| 指标 | 衡量维度 | 预期用途 |
|---|---|---|
| GenEval | 对象、计数、颜色、位置、关系 | 主组合生成指标 |
| T2I-CompBench | 属性绑定、组合、空间关系 | 主文本一致性指标 |
| TIFA 或 VQAScore | prompt 事实一致性 | 补充语义遵循 |
| CLIPScore | 整体语义相似度 | 辅助，不能单独解释 |
| FID/KID | 真实感与分布质量 | 确认未显著损害图像质量 |
| 人工偏好评估 | 多对象、属性、关系正确性 | 检查自动指标盲区 |

建议建立由短到长、由简单到复杂的 prompt 测试集，并按对象数、属性数和关系数分桶报告结果。

### 5.4 伪标签质量评估

从训练 caption 中随机抽取 200-500 条英文 prompt，人工标注对象中心词、属性归属和关系谓词。报告对象、属性、关系的 Precision、Recall 和 F1；还应报告 T5 对齐覆盖率：

$$
\mathrm{Coverage}=\frac{\#\{\text{语义词至少对应一个有效 T5 token}\}}
{\#\{\text{语义词}\}}.
$$

没有此项评估，无法区分“模型无效”和“伪标签错误”。

### 5.5 消融实验

建议执行以下消融：

1. 移除 object、attribute、relation 中的一类或多类条件。
2. 将语义条件注入所有层，与默认层级窗口比较。
3. 移除时间步门控，使用固定强度。
4. 将零初始化改为随机初始化，验证 checkpoint 稳定性。
5. 测试 $r\in\{16,32,64,128\}$ 的瓶颈维度。
6. 测试 $p\in\{0,0.05,0.1,0.2\}$ 的语义 dropout。
7. 使用人工标签子集替换规则伪标签，估计伪标签上限。

## 6. 预期结果与假设

以下内容是待验证假设，而非已获得结果：

- 对象分支可能主要提升对象存在率、对象组合和多对象 prompt 的一致性。
- 属性分支可能主要提升颜色、材质、大小等属性绑定指标。
- 关系分支可能主要提升动作和空间关系 prompt 的正确率，但受依存规则准确率影响最大。
- 全局条件与最后四层不注入局部语义的设置，可能有助于维持整体自然度和纹理质量。
- 由于缺少显式空间输入，该方法不应被宣称能保证精确对象位置或遮挡顺序。
- FID 或 CLIPScore 未必出现显著提升；更有价值的提升可能体现在 GenEval、T2I-CompBench 和人工组合一致性评估。

## 7. 局限性与风险

1. **伪标签噪声：** spaCy 对短 prompt、片段句、并列结构、长距离修饰和复杂介词链并不总是可靠。
2. **语义角色不完整：** 当前对象条件仅使用名词短语中心词；对象短语中的属性并未并入对象掩码，可能削弱“red car”整体概念。
3. **非空间化注入：** 所有残差广播给全部 image token，不能可靠控制对象出现位置，也无法处理显式遮挡。
4. **条件冗余：** 原始 cross-attention 已访问完整 T5 序列，新增分支可能被忽略，或仅产生参数量带来的非结构化增益。
5. **语言限制：** 当前规则仅面向英文；中文、代码混合文本和非语法化提示词不能直接使用。
6. **训练路径验证不足：** 当前实现已完成静态语法检查，但尚未在完整 Linux PyTorch/xFormers 环境完成端到端训练、恢复 checkpoint 和采样验证。
7. **推理接口：** 当前训练实现依赖预计算 semantic mask。部署推理时必须为输入 prompt 运行相同的 spaCy/T5 对齐逻辑，或扩展现有推理脚本；否则无法启用语义 adapter。

## 8. 结论

本文提出 SCDA：一种仅依赖文本的 DiT 语义分层条件适配方案。它从英文 prompt 自动构建对象、属性和关系伪标签，在 T5 子 token 空间完成对齐与池化，并通过零初始化、层级和时间步感知的残差适配器补充 PixArt 的统一文本 cross-attention。该方法的优势是无需 bbox 或关系图人工标注、参数量小、可安全加载基线 checkpoint；核心挑战是伪标签可靠性、语义残差是否被有效利用，以及该机制能否在组合一致性提升与图像质量保持之间取得平衡。后续工作应先完成伪标签质量评估和端到端训练，再基于严格消融结果确定论文中的最终主张。

## 附录 A：复现实验命令

```bash
pip install spacy
python -m spacy download en_core_web_sm

python tools/prepare_semantic_masks.py \
  --json-path /root/private_data/data/COCO2017Mini/partition/data_info.json \
  --feature-root /root/private_data/data/COCO2017Mini/caption_feature_wmask \
  --tokenizer /root/private_data/PixArt-alpha-attentiongate/output/pretrained_models/t5-v1_1-xxl

accelerate launch train_scripts/train.py \
  configs/PixArt_xl2_coco2014_semantic.py
```

## 附录 B：论文撰写检查表

- [ ] 明确使用的 PixArt checkpoint、训练样本数、分辨率、GPU 和总训练步数。
- [ ] 说明 T5 tokenizer、caption 清洗、spaCy 版本和英文模型版本。
- [ ] 报告伪标签对象/属性/关系的准确率与 T5 对齐覆盖率。
- [ ] 用同一采样器、CFG、步数和随机种子比较所有方法。
- [ ] 同时报告组合一致性与图像质量指标。
- [ ] 报告参数量、训练显存、训练时间和推理延迟。
- [ ] 提供失败案例，特别是并列对象、复杂关系、长 prompt 和错误句法分析。
