# B1 / B2：实现、验证与评测记录

> 目标：在**不改动 TP-SCDA 核心机制**（Token-Pair 机制、object–attribute relation、
> pair compatibility、attention bias、layer/timestep gate、T5 encoder、PixArt backbone）
> 的前提下，提高 TP-SCDA 在绑定类指标上的表现。
> 两个方案都作为**可选开关**接入，默认关闭，关闭时行为与历史完全一致。

---

## 1. 动机

门控优化（`results/GATE_OPTIMIZATION.md`）解决了"门控不动"的问题，
但也暴露出**训练目标本身缺少绑定信号**：

- 训练损失 = 扩散 MSE + 把语义残差往零推的正则项。
- 没有任何一项奖励"属性绑定到了正确的对象"。
- 实测后果：门控能动了之后，它**向下漂移**
  （有效强度 0.150 → 0.066，14000 步），即优化器判定该分支无用。

两个方案分别从**结构**和**目标**两侧补这个缺失的信号。

| | 改什么 | 一句话 |
|---|---|---|
| **B1** | 前向结构 | pair 亲和度只保留解析出的 object→attribute 边 |
| **B2** | 训练目标 | 加一项对比式对齐损失，奖励"看对象的 patch 也看它的属性" |

两者**互相独立**，可单独开启，也可同时开启。

---

## 2. B1：edge-gated pair affinity

**现状问题。** pair 亲和度由 `object_pair_proj / attribute_pair_proj`
的点积学到，是**纯内容相似度**，丢掉了 parser 已经给出的依存结构。
本语料上解析的对齐失败率为 **0.00%**（见 `results/parser_alignment/`），
即这份结构信息是免费且可靠的。

**做法。** 在 `SemanticTokenCrossAttention.forward` 里，
把已经按 role mask 限定的 pair 矩阵再乘上解析出的邻接矩阵：

```python
if pair_edge_gate and edge_pairs is not None:
    pair = pair * self._edge_matrix(edge_pairs, edge_count, length,
                                    pair.dtype, pair.device)
```

`_edge_matrix()` 构造 `(B, L, L)` 的 0/1 邻接矩阵，
行=object 子词、列=attribute 子词，取自解析出的 `(obj_token, attr_token)` 对。
边只做**乘性抑制**，不新增任何连接——即只能"收窄"注意力，不会凭空引入关系。

**训练数据从哪来。** 缓存的 `caption_feature_wmask/*.npz` 里只有三种 role mask，
没有边结构，也没有原文。逐样本重新生成 118 294 个特征档（79 GB）代价过高，
因此 `tools/build_edge_index.py` 从 `data_info.json` 的原始 caption 解析一次，
存成稀疏三元组索引：

```
/root/private_data/data/COCO2017Prepared/partition/edge_index.npz   2.07 MB
  samples            : 118 287
  edges              : 520 630
  samples with edges : 92 209   (78%)
  no object parsed   : 14
  object but no edge : 26 064
  平均 object token  : 4.40 / 样本
```

数据集侧 `InternalData` 在 `load_semantic_edges=True` 时懒加载该索引
（每个 DataLoader worker 只读一次），产出定长 `(24, 2)` 的 int64 边表
（不足补 −1）与 `semantic_edge_count`，默认 collate 即可堆叠。

**推理一致性（重要）。** 边门控在**前向传播内部**，
因此 B1 的 checkpoint 在推理时**必须**同样提供解析边，
否则会以"它被训练去避免的那种无约束亲和度"被评测 —— 见 ISSUE-018。
已为生成脚本补上 `build_semantic_edges()`，两者逐行一致：

```
checked 200 prompts, mismatches: 0
VERDICT: CONSISTENT
```

---

## 3. B2：binding alignment loss

**做法。** 对每条解析边 `(o, a)`，用 patch 对 object token 的注意力
给每个候选 attribute token 打分：

```
score[a'] = Σ_p  attn[p, o] · attn[p, a']
```

再在**parser 给出的 attribute token 集合内**做交叉熵，目标就是被绑定的那个 `a`。
语义即"看对象的 patch，应当也看向它的属性，而不是别的属性"。
温度 `semantic_binding_temperature = 1.0`，权重 `semantic_binding_coef = 0.1`。

**关键实现细节。**

- 打分矩阵按 `(B·E, P, L)` 分块矩阵乘算，避免 `(B, E, P, L)` 的四维大张量。
- 候选集用 `attribute_mask` 掩掉，`masked_fill(-inf)` 后再 `log_softmax`，
  保证梯度只流向"属性 vs 属性"的区分，不会去和无关 token 竞争。
- 无边的样本返回 `None`，不产生损失项；`coef=0` 时**完全不计算**（零开销）。
- 每层各自产生一个损失，`get_semantic_binding_loss()` 跨 28 个 block 取均值。

**只影响训练。** 前向输出与 gateopt 完全一致，推理时无需任何额外输入。

---

## 4. 代码改动清单

| 文件 | 改动 |
|---|---|
| `diffusion/model/nets/PixArt.py` | `SemanticTokenCrossAttention.forward` 新增 `edge_pairs / edge_count / pair_edge_gate / binding_loss_coef / binding_temperature`；新增 `_edge_matrix()`、`_binding_alignment_loss()`；模型级 `binding_loss()` / `get_semantic_binding_loss()`；新增构造参数 `semantic_pair_edge_gate`、`semantic_binding_coef`、`semantic_binding_temperature`（**默认全部关闭**） |
| `diffusion/data/datasets/InternalData.py` | 新增 `load_semantic_edges / semantic_edge_index / max_semantic_edges`；`edge_pairs_for()` 懒加载索引；`data_info` 增加 `semantic_edges`、`semantic_edge_count` |
| `train_scripts/train.py` | B2 损失接入总损失；新增可观测字段 `semantic_binding_loss` / `semantic_binding_coef`；config → model_kwargs 白名单补三个新键 |
| `tools/build_edge_index.py` | **新增**，构建边索引（一次运行） |
| `tools/prepare_semantic_masks.py` | **新增** `build_semantic_edges()`，与 mask 构建共用同一解析链 |
| `tools/generate_scda_samples.py` | 新增 `--semantic-pair-edge-gate` / `--max-semantic-edges`，在生成时提供边 |
| `results/_work/gen_compbench.py` | 新增 `b1` / `b2` 两个 method 条目；B1 时构造边；新增 `--checkpoint` 覆盖参数；修正 `mask_fix_applied_for_issue_011` 元数据表述 |
| `results/_work/run_eval.py` | `--method` 枚举加入 `b1` / `b2`（ISSUE-020） |

**未改动：** Token-Pair 机制、object–attribute relation、pair compatibility、
attention bias 的形式、layer/timestep gate、T5 encoder、PixArt backbone。
所有新增行为默认关闭，旧 checkpoint 的加载与推理路径不受影响。

---

## 5. 冒烟测试

`results/_work/smoke_b1b2.py` → 完整输出见 `results/b1b2/smoke_b1b2.log`。

```
--- dataset ---                      语义掩码/边张量形状、dtype、−1 填充      全 ok
--- edge matrix ---                  邻接矩阵与解析对逐位一致                  ok
--- B1: edge-gated pair ---          开启后模块输出改变 max|delta|=3.732e-04   ok
--- B2: binding alignment loss ---   损失有限、>0、梯度到达 q/k/两个 pair_proj  ok
                                     无 / 关闭系数两种退化情形均不产生损失      ok
--- full model forward ---           B0/B1/B2 前向有限；B2 聚合损失 1.0995       ok
--- generation-path edges ---        推理路径解析出边、张量形状 (1,24,2) int64   ok
SMOKE_OK
```

**为什么 B1 不看最终输出**（详见 ISSUE-019）：
`out_proj` 是零初始化（zero-conv），语义分支在初始状态下是 no-op，
此时**任何**注意力变化都不会传导到最终输出。
因此判据放在模块层面：填入固定随机 `out_proj` 后比较开关 B1 的模块输出。

### 5.1 训练安全性检查：每条边的 token 都落在对应 role mask 内

B2 的交叉熵在 `attribute_mask` 内做 `masked_fill`：
若某条边的 attribute token **不在**该 mask 内，其目标项会被掩成 `-inf`，
损失变 `inf`，3.6 小时的训练会在中途报废。
因此在正式训练前先做了这项检查（真实数据，2000 个样本）：

```
checked samples      : 2000
samples with edges   : 1580 (79.0%)
total edges          : 8257
max edges in a sample: 24  (cap 24)
object token NOT in object mask   : 0
attribute token NOT in attr mask  : 0
VERDICT: SAFE
```

同时确认边数上限 24 在真实数据上**恰好被触达**（存在边数 = 24 的样本），
说明该上限不是形同虚设的宽松值；两侧（训练索引、推理解析）都按同一顺序
截断到 24，行为一致。

---

## 6. 训练与评测

配置：`configs/PixArt_xl2_coco2017_token_pair_b1_edgegate.py`
和 `..._b2_bindingloss.py`。两者都以 gate 优化配置为基座，
**唯一的新变量**分别是边门控和绑定损失，因此与 gateopt 的对比是受控的。

- 训练长度：`num_epochs=1` = 14 786 微步，与产出论文当前 checkpoint 的那次**等长**；
- 基座：`PixArt-XL-2-512x512-native-gate-init.pth`（未蒸馏）；
- `distill_coef = 0.0`（本次工作不做任何蒸馏内容）；
- 每个变体存 3 个检查点（step 7000 / 14000 / 14786）。

评测走**两套协议**，与已有方法完全同源：

1. **T2I-CompBench++ 官方八项**：300 prompt × 1 图/类，官方评测器；
2. **64 提示词协议 × 3 seed**：CLIPScore + 三个绑定 proxy
   （`Object-presence proxy` / `Attribute-presence proxy` / `Binding proxy`，
   10 000 次 paired bootstrap 的 95% 区间）。

队列脚本：`results/_work/b1b2_queue.sh`
（等 gateopt 评测释放 GPU → B1 训练 → B1 两套评测 → B2 训练 → B2 两套评测 → 打分）。
日志：`/root/compbench_work/logs/b1b2_queue.log`。

### 判定口径

- **绑定类（颜色/形状/纹理）**一致为正、**非绑定类**不退化 → 达到参数高效适配的合理预期。
- 单张图的评测有噪声、每类仅 300 样本，不看单个类别的正负号。
- proxy 指标只在 **95% 区间下界 > 0** 时写"稳定提升"，下界为 0 只写"呈提升趋势"。
- 不期望八项全面超过公开 PixArt-α：冻结基线就是原始 PixArt-α 权重
  （逐张量验证 maxdiff=0），差距来自主干与训练数据，不是适配方法。

---

## 7. B1 的终止（2026-09-19 08:05）

**B1 在训练到 step 6050 时被终止，不进行评测。**

依据是 `results/b1b2/B1_EFFECT_SIZE.md` 中的两条独立测量：

1. **训练轨迹与 gateopt 几乎逐位相同**——loss 完全相同，
   门控有效强度相对差异 0.19%，参数位移差异 1e-4 量级；
2. **推理时效应量只有 0.04%~0.35%**——用训练好的 gateopt checkpoint
   切换边门控实测（无边样本为 0.000%，说明测的就是边门控本身）。

B1 基于的 gateopt 基座本身已比冻结基线低约 **0.03**，
而 B1 自身的效应量比这个差距小**两个数量级**，不可能补回来。
继续跑完训练与两套评测（约 4 小时 GPU）只会确认一个已知的空结果。

**根因是结构性的**：pair bias 的幅值被 `softplus(pair_strength) = 0.0486`
限制（`pair_strength = −3.0`），再乘语义强度 ~0.09 × 层门控 ~0.10，
到达残差流只剩 ~5e-4。而这个幅值是论文 Token-Pair 机制本身的设计，
属于任务简报中**明确禁止修改**的核心机制，**不能通过调大 `pair_strength` 增强 B1**。

**保留的证据**：`output/coco2017_token_pair_b1_edgegate/` 下的
`experiment_tables/training_metrics.csv`（122 步 × 18 门控可观测字段）
与 `train_log.log`，作为空结果的训练侧证据。未产生检查点（未到 step 7000）。

**B2 继续按原计划进行**：它是唯一改动**训练目标**而非前向结构的机制，
不受上述缩放链约束。队列见 `results/_work/b2_queue.sh`。

## 8. 时间线

| 时间 | 事件 |
|---|---|
| 2026-09-19 04:00 | 边索引构建完成（118 287 样本 / 520 630 边 / 2.07 MB） |
| 2026-09-19 04:15 | 冒烟测试全绿；推理一致性 200/200 通过 |
| 2026-09-19 04:20 | `b1b2_queue.sh` 入队，等待 gateopt 评测释放 GPU |
| 2026-09-19 06:36 | B1 训练启动 |
| 2026-09-19 08:00 | 测得 B1 效应量 0.04%~0.35%，训练轨迹与 gateopt 同步 |
| 2026-09-19 08:05 | **B1 终止于 step 6050（作者决定），转 B2** |
