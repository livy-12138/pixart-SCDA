# 可学习分支层训练报告

## 摘要

本实验在冻结的 PixArt-XL-2 主干上训练 token-level、pair-aware 语义模块，并为
`global`、`object`、`attribute`、`relation` 四个语义分支加入逐 Transformer 层的可学习 gate。
目标是让模型自行确定不同语义分支在何处注入最有效，同时通过冻结 PixArt teacher 的稀疏蒸馏避免破坏原始生成能力。

训练已完整结束：1 epoch、14,786 optimizer steps，无 NaN 或 OOM。最终 CLIP 为 `0.273597`，略高于 Frozen PixArt baseline `0.273209`，但略低于初始化所用的 step-2,500 蒸馏 checkpoint `0.273675`。各分支的 layer gate 在整个训练过程中几乎没有偏离初始化，说明当前 gate 参数化与学习率组合没有获得足够的任务驱动信号以学出不同的注入层位置。

## 实验目标与方法

### 语义模块

- 保留原始 PixArt cross-attention，因此每个图像 patch 仍可以直接关注完整 T5 global 文本序列。
- 语义模块使用角色标记的 token 序列，区分 global、object、attribute、relation。
- 使用 object-to-attribute pair-aware attention bias：对象 token 的注意力用于提升与其关联属性 token 的注意力 logit。
- 不使用 bounding box；对象与属性的对齐完全通过文本 token 与图像 patch 的 cross-attention 学习。
- 语义残差的全局 pooled 注入关闭（`semantic_residual_scale=0.0`）；使用 token-level attention 修正。
- 每层、每分支都有可学习 gate，并在训练日志中记录四个分支的层 gate 均值。

### 稳定性约束

- PixArt 主干冻结，仅训练语义 token/pair 模块和可学习 layer gate。
- 使用 Frozen PixArt teacher，蒸馏系数 `0.5`；每 8 个 micro-batch 执行一次 teacher 蒸馏，降低开销。
- token gate 最大幅度限制为 `0.08`，避免语义模块过强覆盖原始 PixArt 条件控制。
- 使用 semantic residual regularization，系数 `0.02`。

## 训练配置

| 项目 | 值 |
|---|---|
| 配置 | `configs/PixArt_xl2_coco2017_token_pair_learnable_layers.py` |
| 初始 checkpoint | `output/coco2017_token_pair_distill_gate008_sparse/checkpoints/epoch_1_step_2500.pth` |
| Teacher checkpoint | `output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth` |
| 数据 | COCO2017，118,286 个有效训练样本 |
| 分辨率 | 512 x 512 |
| epoch / steps | 1 / 14,786 |
| micro-batch | 8 |
| gradient accumulation | 4 |
| effective batch | 32 |
| 优化器 | AdamW，LR `1e-5`，weight decay `0.01`，eps `1e-10` |
| 随机种子 | 43 |
| 精度 | FP16 |
| 语义正则系数 | 0.02 |
| 蒸馏系数 / 间隔 | 0.5 / 每 8 个 micro-batch |
| token gate 上限 | 0.08 |

## 运行状态与吞吐

- 训练总时长约 3 小时 35 分钟。
- 训练日志记录的平均 `time_all` 为 `0.905 s/optimizer step`。
- 按有效 batch 32 折算，吞吐约 `35.36 images/s`。该数值包含稀疏蒸馏的平均开销。
- 训练期间 GPU 利用率约 100%，显存约 32 GB；没有 NaN、Inf 或 OOM。
- 最终 checkpoint：`output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth`。

## 训练轨迹

下表取日志中最接近目标 step 的记录。loss 为单次日志窗口值，不应将其与独立模型的单点 loss 直接比较。

| Step | diffusion loss | semantic reg loss | residual norm | token gate | pair strength | global gate | object gate | attribute gate | relation gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.1241 | 0.0950 | 0.2598 | 0.0160 | 0.0493 | 0.9820 | 0.0180 | 0.0180 | 0.0180 |
| 2,500 | 0.1376 | 0.0308 | 0.1547 | 0.0161 | 0.0493 | 0.9820 | 0.0180 | 0.0180 | 0.0180 |
| 5,000 | 0.1407 | 0.0192 | 0.1234 | 0.0161 | 0.0494 | 0.9820 | 0.0180 | 0.0180 | 0.0180 |
| 7,500 | 0.1335 | 0.0130 | 0.1034 | 0.0162 | 0.0494 | 0.9820 | 0.0179 | 0.0179 | 0.0179 |
| 10,000 | 0.1272 | 0.0102 | 0.0916 | 0.0163 | 0.0494 | 0.9820 | 0.0179 | 0.0179 | 0.0179 |
| 12,500 | 0.1352 | 0.0085 | 0.0837 | 0.0164 | 0.0494 | 0.9819 | 0.0179 | 0.0179 | 0.0179 |
| 14,750 | 0.1455 | 0.0073 | 0.0778 | 0.0164 | 0.0494 | 0.9819 | 0.0179 | 0.0179 | 0.0179 |

## 训练结论

1. 语义残差范数从 `0.2598` 降到 `0.0778`，正则与蒸馏共同将语义修正控制在保守范围。
2. token gate 从 `0.0160` 缓慢增加至 `0.0164`，远低于上限 `0.08`，说明语义模块没有采取激进干预。
3. pair strength 保持正值并稳定在约 `0.0494`，没有出现早期实验中负 pair strength 抑制属性的情况。
4. layer gate 从 `0.9820 / 0.0180 / 0.0180 / 0.0180` 仅变化到 `0.9819 / 0.0179 / 0.0179 / 0.0179`。本实验没有证据表明当前设计已学到 global、object、attribute、relation 的不同最优注入层。
5. 因最终评测分数低于 step-2,500 蒸馏初始化 checkpoint，不建议将完整 14,786-step checkpoint 作为当前最佳模型；应保留 step-2,500 checkpoint 作为整体 CLIP 的候选最佳。

## 后续改进建议

1. 为 layer gate 使用独立且更高的学习率，并从语义投影参数组中分离。
2. 将当前近似固定的 gate 改为跨层 softmax 分布或 Gumbel-Softmax 选择，使不同层之间显式竞争。
3. 对 object/attribute/relation gate 添加弱熵正则或 warm-up，避免 global 分支初始占优后梯度被压制。
4. 用更大、受控的双对象双属性测试集选择 checkpoint，而不是只用通用 CLIP 均值决定是否保留。

## 可复现性与产物

- 训练日志：`output/coco2017_token_pair_learnable_layers/train_log.log`
- 运行时配置快照：`output/coco2017_token_pair_learnable_layers/config.py`
- checkpoints：`output/coco2017_token_pair_learnable_layers/checkpoints/`
- 总体评测记录：`output/coco2017_eval/COMPARISON.md`
- 实验事件记录：`experiments/COCO2017_TOKEN_PAIR_RECORD.md`
