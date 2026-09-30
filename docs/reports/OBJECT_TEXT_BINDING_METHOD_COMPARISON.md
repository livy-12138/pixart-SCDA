# 文生图对象-文本绑定方法对比与 SCDA 改进路线

## 1. 研究问题定位

当前项目要解决的不是一般的 prompt-image 相似度，而是组合语义中的绑定关系：

- 对象是否出现：`a blue jay` 不能只生成一只泛化的 bird；
- 属性是否出现：`red`、`blue`、`glass` 等属性是否被生成；
- 属性是否绑定到正确对象：`a red ball and a blue cube` 不能变成蓝色球和红色立方体；
- 对象间关系是否满足：`a cat behind a chair`、`a person holding a cup`。

SCDA/token-pair 的当前定位是：**不使用框、分割图或人工场景图，只从文本中构造对象/属性/关系 token，并以轻量 adapter 注入冻结 PixArt。** 这使它比带空间标注的方法便宜，但也意味着监督信号弱、关系几何约束不足。

## 2. 代表性方法谱系

| 方法/方向 | 典型代表 | 需要的额外监督 | 介入位置 | 对象存在 | 属性绑定 | 空间/动作关系 | 与 SCDA 的关系 |
|---|---|---|---|---|---|---|---|
| 基线文本条件 | PixArt、Stable Diffusion 原生 cross-attention | 无 | 原始 cross-attention | 中等 | 弱 | 弱 | 当前 Frozen baseline |
| 推理时注意力重分配 | Attend-and-Excite | 无训练数据；推理时优化 latent | token attention / latent | 较强 | 中等 | 弱-中等 | 最接近的无标注外部基线 |
| 词级注意力评估/重加权 | DAAM、Prompt-to-Prompt | 无额外标注 | cross-attention 可视化或编辑 | 诊断为主 | 中等 | 弱 | 适合作为 SCDA 的诊断和 loss 信号 |
| 文本结构/场景图条件 | Structured Diffusion、Composable Diffusion | 文本解析或场景图 | 分阶段/分词条件 | 强 | 强 | 中等 | 与 SCDA 的伪标签路线最接近，但通常不是参数高效 adapter |
| 文本 token 对象级注意力 | SynGen、token-level attention control 类方法 | 无或弱监督 | token-to-image attention | 强 | 强 | 中等 | 与 token-pair SCDA 的机制最相近 |
| 框/区域约束 | GLIGEN、BoxDiff、MultiDiffusion | 框、布局或区域条件 | spatial grounding / attention | 强 | 强 | 强 | 能力更强，但不是同等输入条件，不能直接公平比较 |
| 实例级区域控制 | InstanceDiffusion、Instance-conditioned adapters | 实例框/掩码/参考图 | instance token + spatial module | 很强 | 强 | 很强 | 工程上限高，标注和推理成本高 |
| 外部视觉反馈 | Layout-guided/检测器引导采样 | 检测器或 VQA 反馈 | 采样循环 | 强 | 中等-强 | 强 | 可作为 SCDA 的后处理或验证器，不适合作为纯文本公平基线 |
| 参数高效微调 | LoRA、T2I-Adapter、ControlNet | 任务数据；ControlNet 常需结构条件 | 权重/条件分支 | 取决于数据 | 取决于数据 | 取决于条件 | 可用于匹配参数量的工程基线 |

### 2.1 Attend-and-Excite：最应该加入的外部基线

Attend-and-Excite 在推理过程中找出 prompt 中未被充分激活的名词 token，并优化潜变量，使对应 token 的 cross-attention 响应增强。它不需要重新训练模型，也不需要框标注，因此与 SCDA 的“只使用文本”约束最接近。

优点：

- 可直接作用于已有 PixArt/扩散模型；
- 对对象遗漏有明确目标；
- 能作为 token-level attention control 的强基线。

局限：

- 主要优化单个对象 token 的激活，不显式建模“属性属于哪个对象”；
- 多对象之间的关系和空间位置仍然缺少几何约束；
- 额外采样迭代会显著增加推理时间；
- 在 PixArt 上需要重新实现 attention hook，不能直接把 Stable Diffusion 代码结果当作公平对比。

对 SCDA 的启示：把 `object attention coverage` 作为推理诊断或训练正则，但不能只提高对象 token 的注意力总量，否则可能损害属性 token。

### 2.2 Structured Diffusion / SynGen：最接近 SCDA 的结构化文本方向

这类方法先从 prompt 中抽取对象、属性、关系或句法结构，再把结构化 token 分组送入扩散模型，或者在生成过程中约束 token attention。它们与 SCDA 都利用文本结构，但通常更强调推理时的显式分组、场景图和 token 级约束。

与 SCDA 的差异：

- SCDA 将对象/属性/关系压缩后注入 adapter；结构化扩散方法通常保留更细粒度的 token-to-image 交互；
- SCDA 只训练约 0.6M 级别的新增模块；推理时结构化方法可能几乎不训练但采样更慢；
- SCDA 当前没有直接的对象、属性或关系监督，结构化方法往往有显式 attention/scene-graph 约束。

对 SCDA 的启示：不要继续扩大 pooled residual，而应在 token-pair 路径保留 token 级交互，并只对相关 object-attribute token 对施加约束。

### 2.3 GLIGEN / BoxDiff / MultiDiffusion：能力上限但不是同条件对比

这些方法通过边界框、区域布局或空间 attention 把对象放到指定位置，通常在对象存在、属性绑定和空间关系上更强。它们适合回答“如果允许额外布局输入，SCDA 的能力差距有多大”，不适合作为当前纯文本 setting 的直接主基线。

建议把它们作为扩展对照，明确标注输入条件不同：

- **纯文本公平比较**：Frozen PixArt、Attend-and-Excite、Structured/SynGen 类方法、token-pair SCDA；
- **额外布局上限比较**：GLIGEN、BoxDiff、MultiDiffusion/InstanceDiffusion；
- **参数效率比较**：LoRA/T2I-Adapter 与同训练数据、同可训练参数量的 token-pair SCDA。

## 3. SCDA 与相关方向的直接比较

| 维度 | Frozen PixArt | Pooled SCDA | Token-pair SCDA | Attend-and-Excite | Structured/SynGen | GLIGEN/BoxDiff |
|---|---|---|---|---|---|---|
| 输入 | 文本 | 文本 + 伪标签 | 文本 + 伪标签 | 文本 | 文本 + 结构解析 | 文本 + 框/布局 |
| 是否训练 | 否 | 是 | 是 | 否 | 通常否或少量训练 | 是/需要训练 |
| 对象粒度 | 完整文本 | pooled object | token-level object | token attention | token/scene graph | instance/region |
| 属性归属 | 隐式 | 弱 | 显式 pair bias | 隐式 | 可显式 | 显式区域 |
| 关系建模 | 原生文本 | pooled relation | relation token，可扩展 pair | 弱 | 中等-强 | 强 |
| 额外推理开销 | 低 | 低 | 中 | 高 | 中-高 | 中-高 |
| 对标注依赖 | 无 | 句法伪标签 | 句法伪标签 | 无 | 句法/场景解析 | 框/掩码/布局 |
| 当前证据 | CLIP 0.273209 | CLIP 明显退化 | Distill CLIP 0.273675 | 待复现 | 待复现 | 不是同条件 |

当前结果已经支持一个明确判断：**pooled SCDA 的瓶颈不是“语义分支数量不够”，而是把对象/属性/关系压成全局残差后丢失了绑定结构。** 因此停止 pooled 路线是合理的；后续应只在 token-level binding 上做改进。

## 4. 最值得尝试的改进方向

### 方向 A：显式 object-attribute pair loss（最高优先级）

当前 pair bias 只通过扩散噪声预测损失间接学习。可从每个 prompt 构造正负 pair：

- 正样本：`red -> ball`、`blue -> cube`；
- 负样本：交换对象属性或随机错配；
- 在 token-pair attention 矩阵上使用 margin/InfoNCE 损失。

示意目标：

\[
L_{pair}=\max(0, m-a(o,a^+) + a(o,a^-)),
\]

其中 `a(o,a)` 是对象 token 与属性 token 的聚合 attention 或 pair score。该损失直接针对你的研究问题，比继续调大扩散 loss 中的 semantic residual 更合理。

风险：句法伪标签错误会把错误 pair 当正样本。因此必须加入 pair-label 置信度、仅训练高置信短 prompt，并对随机交换属性做人工抽样审计。

### 方向 B：保留 token 级 pair attention，取消 pooled residual（最高优先级）

当前已有 token-pair 路线，建议将其作为唯一主线：

1. object token 作为 query，attribute/relation token 作为 key/value；
2. pair bias 只作用于合法依存关系的 token 对；
3. 使用 `softplus` 保证 pair strength 非负；
4. 保留冻结 baseline cross-attention，并用 teacher distillation 限制整体行为偏移。

这一路线与 SynGen/Structured Diffusion 的 token-level 思路最接近，同时保持 SCDA 的参数高效和无框输入特点。

### 方向 C：关系从“词集合”升级为有向 pair/triple

当前 relation mask 主要聚合动词、介词和目标 token，仍然容易丢失谁作用于谁。建议显式保存：

```text
(subject_object_id, predicate_token_ids, target_object_id)
```

并构造有向关系矩阵 `R[i,j]`。对 `cat left of dog` 与 `dog left of cat` 使用相反的 pair 目标，避免模型只学到“left/of/dog 都出现”。

关系分支应优先在空间关系和动作关系 prompt 上评测，而不是用总体 CLIP 判断成败。

### 方向 D：attention coverage 与 exclusivity 正则

对每个对象 token，要求至少有一个图像 token 获得足够注意力；同时限制两个对象的注意力图完全重叠：

\[
L_{cov}=\sum_o \max(0,\tau-\max_p A_{p,o}),\quad
L_{excl}=\sum_{o\ne o'} \langle A_o,A_{o'}\rangle.
\]

它对应 Attend-and-Excite 的“激活不足修复”，但可在训练阶段作为轻量正则。必须对 attention 做归一化，否则模型可能通过整体放大数值投机。

### 方向 E：句法伪标签升级为名词短语和依存图

不要只把 noun root 当作对象。应保留完整 noun phrase、compound、数量词、否定和并列结构，并对每条 object-attribute/relation edge 保存 confidence。建议用规则解析结果和 LLM/VLM 抽样校正各自 200 条，报告 object/attribute/relation 的 precision、recall、F1。

这条方向对模型结构改动小，但可能比增加 adapter 宽度更有效，因为当前监督上限很可能受 mask 错误限制。

### 方向 F：检测器/VQA 只做验证器，不改变公平主设定

Grounding DINO、OWL-ViT、VQA 或人工标注可以用于：

- 离线验证 object presence、attribute binding、relation accuracy；
- 挑选高置信训练 prompt；
- 对采样结果做 rejection/reranking。

不建议把检测器直接作为主模型输入，否则 SCDA 就从“纯文本绑定”变成“视觉反馈控制”，应单列为扩展实验。

## 5. 推荐对比实验矩阵

### 5.1 最小公平主表

| 方法 | 训练/推理 | 输入 | 必测指标 |
|---|---|---|---|
| Frozen PixArt | 无训练/普通采样 | 文本 | CLIP、object/attribute/relation |
| Attend-and-Excite | 无训练/优化采样 | 文本 | 同上 + 推理时间 |
| Token-pair w/o pair loss | adapter 训练 | 文本+伪标签 | 同上 |
| Token-pair + pair loss | adapter 训练 | 文本+伪标签 | 同上 |
| Token-pair + pair loss + distill | adapter 训练 | 文本+伪标签 | 同上 + baseline 保真 |

### 5.2 能力上限扩展表

| 方法 | 额外输入 | 作用 |
|---|---|---|
| GLIGEN/BoxDiff | 框/布局 | 展示有空间监督时的上限 |
| Grounding/VQA reranking | 生成候选 + 检测/VQA | 展示视觉反馈代价 |
| LoRA/同参数 adapter | 文本 | 排除“仅仅因为增加可训练参数” |

不要把 GLIGEN/BoxDiff 的结果和纯文本 SCDA 合并排名；应在表格中单独标注输入条件。

## 6. 基于现有结果的优先级

| 优先级 | 实验 | 预期回答的问题 | 成功标准 |
|---|---|---|---|
| P0 | 复现 Frozen baseline + token-pair distill | token-pair 是否稳定保持 baseline | CLIP 不低于 baseline CI 下界 |
| P1 | token-pair 加 pair margin/InfoNCE loss | 是否真正提升绑定而非只提升总体相似度 | Binding Accuracy 和人工绑定率提升 |
| P1 | Attend-and-Excite 同协议复现 | 纯文本推理时 attention 优化是否优于 adapter | 同等 prompt/seed 下比较绑定和时间 |
| P1 | 伪标签 precision/recall/F1 审计 | 失败来自结构还是模型 | 高置信 prompt 的绑定性能提升 |
| P2 | relation directed pair/triple | 是否能改善空间/动作关系 | Relation Accuracy 提升且 CLIP 不下降 |
| P2 | coverage/exclusivity 正则 | 是否减少对象遗漏和注意力串位 | Object Recall、Binding Accuracy 提升 |
| P3 | GLIGEN/BoxDiff/检测器 reranking | 与额外布局/反馈方法的能力差距 | 单独作为上限或扩展表 |

## 7. 结论

与你当前工作最相近的不是 GLIGEN 或 ControlNet，而是 **Attend-and-Excite、Structured Diffusion、SynGen 以及 token-level attention control**。它们共同关注文本 token 与图像区域/注意力的对应关系；区别在于它们通常在推理时直接控制 attention，而你的 token-pair SCDA 通过轻量可训练模块学习这种关系，并用 teacher distillation 保持 PixArt 原始能力。

当前最有价值的改进不是继续增加 pooled 分支，而是：

1. 对合法 object-attribute pair 加显式对比损失；
2. 保留有向 object-relation pair/triple；
3. 加 attention coverage/exclusivity 正则；
4. 审计并提高伪标签质量；
5. 用 Attend-and-Excite 作为纯文本强基线，用 GLIGEN/BoxDiff 作为有额外布局输入的能力上限。

这条路线可以把论文贡献从“增加一个语义 adapter”提升为“在冻结 PixArt 上进行无框、弱监督、对象级文本绑定，并系统比较推理时控制与参数高效学习”。

## 参考方向

- Attend-and-Excite: 强化扩散模型中未充分激活的文本 token，推理时提升对象生成。
- Prompt-to-Prompt: 通过 cross-attention 替换/保持实现文本编辑和 token 对齐控制。
- DAAM: 使用 cross-attention 解释和评价文本 token 对图像区域的贡献。
- Structured Diffusion / Composable Diffusion: 将 prompt 拆解为结构化对象、属性和关系条件。
- SynGen: 利用句法结构和 token 关系改善组合文本到图像生成。
- GLIGEN: 在文本条件之外引入 grounding 输入，实现对象级空间控制。
- BoxDiff: 使用框/空间约束引导扩散采样中的对象布局。
- MultiDiffusion / InstanceDiffusion: 通过区域或实例条件实现多对象组合控制。
