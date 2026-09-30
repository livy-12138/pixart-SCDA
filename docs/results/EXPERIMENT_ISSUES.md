# 实验问题与偏差记录

本文件记录补实验过程中遇到的所有问题、所做的最小修改、以及与原始协议的偏差。
**不在此处修改任何 TP-SCDA 核心算法。**

---

## ISSUE-001  环境被评测依赖改变后，项目 `tools` 包被遮蔽（已修复）

**发现时间：** 2026-09-18
**严重程度：** 高 —— 修复前**整个生成管线无法运行**

**现象：** 安装 T2I-CompBench++ 官方 UniDet 评测器所需的 `detectron2` 之后，
`tools/generate_scda_samples.py` 报错：

```
ModuleNotFoundError: No module named 'tools.prepare_semantic_masks'
```

**根因：** `detectron2` 的 wheel 在
`/opt/conda/lib/python3.10/site-packages/tools/` 下安装了一个**顶层 `tools` 常规包**
（内容是 detectron2 的 CLI 脚本 `train_net.py` 等）。Python 的导入规则是
**常规包优先于命名空间包**：项目 `tools/` 目录内没有 `__init__.py`，因此只是
一个「命名空间包候选」，而 site-packages 中的常规包会胜出并遮蔽它。

**修复（最小改动，不涉及算法）：** 新建空文件
`tools/__init__.py`，使项目 `tools/` 成为常规包，并因其位于 `sys.path[0]`
（脚本自行 `sys.path.insert(0, ROOT)`）而优先匹配。

**验证：** 修复后重新生成 `asset/samples.txt` 的 group_01（8 张，seed 43），
与磁盘上已有图片**逐像素 8/8 完全一致**，证明修复未改变任何生成行为。

**影响范围提示：** 这是一个**环境级**问题。任何在此环境使用本仓库
`python tools/xxx.py` 的人都可能遇到；若在别处重建环境，需重新确认。

---

## ISSUE-002  历史评测产物缺少运行元数据，无法完全重建运行条件

**严重程度：** 中 —— 不影响本次补数据，但影响可复现性声明

`output/*/evaluation_manifest.csv` 只有字段
`model, seed, output_dir, prompts, steps, cfg_scale`，
**没有记录 checkpoint 路径、语义模块 flags（`semantic_token_attention`、
`semantic_token_gate_max`、`semantic_residual_scale`、`semantic_adapter_dim`）**。

这直接违反任务书第 3 节"每个正式实验必须保存的元数据"的要求。

**本次补救：** 新生成的每个实验目录都写入完整 `config.json`
（含 checkpoint、全部语义 flags、seed 方案、环境版本、时间戳）。

---

## ISSUE-003  Pooled SCDA 与 Token-pair SCDA 未能逐位复现

**严重程度：** 中 —— 相关数值已在论文表2/表3/表4中，本次不重跑

按 configs 链推断的 flags 重新生成 `group_01`（seed 43），与磁盘已有图片比对：

| 方法 | 推断的 checkpoint | 推断 flags | 复现结果 |
|---|---|---|---|
| Frozen PixArt | `PixArt-XL-2-512x512-native-gate-init.pth` | 无 | ✅ **8/8 逐位一致** |
| TP-SCDA (learnable layers) | `epoch_1_step_14786.pth` | `--semantic-conditioning --semantic-token-attention --semantic-token-gate-max 0.08 --semantic-residual-scale 0.0` | ✅ **8/8 逐位一致** |
| Full-span SCDA | `epoch_6_step_22182.pth` | `--semantic-conditioning --semantic-residual-scale 0.25` | ✅ **8/8 逐位一致** |
| Pooled SCDA | `epoch_5_step_18485.pth` | `--semantic-conditioning --semantic-residual-scale 0.25` | ❌ MAE≈1.5，13–44% 像素差 >2 |
| Token-pair SCDA | `epoch_5_step_60000.pth` | `--semantic-conditioning --semantic-token-attention --semantic-token-gate-max 1.0` | ❌ MAE≈4–14，35–85% 像素差 >2 |

**分析：** Frozen / Full-span / Learnable 三者逐位一致，说明生成管线本身是
**确定性**的，因此 Pooled 与 Token-pair 的差异**不是数值噪声，而是真实的配置差异**。
`experiments/COCO2017_TOKEN_PAIR_RECORD.md` 明确记载 token-pair 评测用的就是
`epoch_5_step_60000.pth`，与本文所用一致，故差异只能来自未记录的
语义模块 flags（见 ISSUE-002）。

**影响：** **本次需要补的两个核心数字（表5、表6）依赖的是
Frozen PixArt 与 TP-SCDA(=learnable layers) 两个模型，二者均已逐位确认**，
因此该问题不影响本次补数据。但论文若声明"完全可复现"，需先补齐这些 flags。

**未做：** 没有继续穷举 flags 组合去凑匹配。任务书禁止为了让结果好看而调参，
逆向凑配置同样不应做；如实记录即可。

---

## ISSUE-004  T2I-CompBench++ 官方评测协议为「每条 prompt 10 张图」，远超预估规模

**严重程度：** 高 —— 直接决定时间预算

任务书第 3 节的统一协议（64 prompt × 3 seed = 192 张/模型）是**本文自有的
小规模协议**，与 benchmark 官方协议不是一回事。下载官方数据后确认：

- 官方 val 集每类 **300 条 prompt**
- `3_in_1_eval/3_in_1.py` **硬编码 `num=10`**，即每条 prompt 10 张图
- 故每类 **3000 张**，8 类共 **24 000 张/方法**

实测生成吞吐约 **1.2–1.5 s/图**（token-pair 注意力路径比无掩码路径慢约一倍；
见 profile：solver 1.47 s/img，T5 编码 0.85s/16条、语义掩码 0.09s/16条，均可忽略），
故 24 000 张 ≈ **8–10 小时/方法**。

**应对（不改变协议，只改变生成顺序）：** 生成按 **repeat 分阶段 + 两方法交替**
推进，每完成一阶段，**两个方法都拥有一套完整的 k-图/提示词 覆盖**。
这样即使总时长不足，仍可在完整子集上给出诚实的评测结果，
并在 `config.json` 中如实记录实际的 `repeats_this_run`。

**明确的偏差（若最终未跑满 10 张）：** 将写入 `summary.json` 与报告，
说明实际 images-per-prompt 与官方的差异及其对均值的预期影响（仅影响方差，不改变期望）。

---

## ISSUE-005  并行评测任务与本任务争夺 GPU

**严重程度：** 低

用于调研 T2I-CompBench++ 评测流程的子任务在本机安装了 `detectron2`、
`fairscale`、`word2number`，并降级了 `Pillow`(→9.5.0)、`ruamel.yaml`(→0.17.32)、
`setuptools`(→<81)，以完成端到端验证。这些改动：

- 触发了 ISSUE-001（已修复并验证）
- 验证期间占用 GPU 显存，**使早期吞吐测量偏低**（一度测得 1.87 s/图）

已重新在无争用条件下测量真实吞吐。

---

## ISSUE-006  磁盘空间限制：原始图片不落在项目盘

**严重程度：** 中

`/root/private_data`（项目所在盘）**仅剩 22 GB**，而 2 方法 × 24 000 张
512×512 PNG 约需 24 GB，**放不下**。因此生成的原始图片存放在
`/root/compbench_work/`（overlay，可用 4.4 TB）。

`results/` 下保存的是**全部 per-sample 原始记录**（prompt、seed、图片路径、
每个指标的判定与分值），符合任务书"必须保留 per-sample 原始结果"的要求；
图片本体为可再生成的中间产物，其生成配方（checkpoint + flags + seed 方案）
已完整写入 `config.json`。

---

## ISSUE-007  论文正文的【待补】中，有若干不属于服务器任务范围

以下项**无法**由本服务器补全，属于作者本地正文写作问题，**未代为填写**：

- 通信作者标注与署名顺序（第 1 页）
- 中图分类号核定（第 1 页）
- 式(5) 中 `h_{p,o}` 的定义式（3.3 节）
- 式(8) 右侧缺少层下标 `l`，与"对每层设置 role gate"表述不一致（3.4 节）
- 式(2)(6) 中 `d` 与 `D` 的符号统一（3.3 节）
- 参考文献逐条核对卷期页码（文末）

**另：** 论文 4.1 节所写"token-pair 主 run 在 epoch 5 / step 60 000"与
4.1 节其它描述需与 `configs/PixArt_xl2_coco2017_token_pair_learnable_layers.py`
（1 epoch / 14 786 步）核对一致。

---

## ISSUE-008  可学习层门控在该 checkpoint 中几乎未学习；且 TP-SCDA 路径的门控不依赖时间步

**严重程度：** 高 —— 直接影响论文 3.4 节与 4.3.3 节的表述准确性

**(a) 门控未学到层间差异。** 从 `epoch_1_step_14786.pth` 读出
`semantic_token_layer_gate`（形状 28×4），取 `sigmoid` 后与代码中的初始化值比较
（初始化：`full((28,4), -4.0)`，再把第 0 列置 `4.0`）：

| 角色 | 初始化 λ | 训练后均值 | 平均偏移 | 层间标准差 | 最大绝对偏移 |
|---|---:|---:|---:|---:|---:|
| global | 0.982014 | 0.981925 | −0.000089 | 2.9e−05 | 1.6e−04 |
| object | 0.017986 | 0.017898 | −0.000088 | 1.4e−05 | 1.1e−04 |
| attribute | 0.017986 | 0.017885 | −0.000101 | 2.8e−05 | 1.3e−04 |
| relation | 0.017986 | 0.017896 | −0.000090 | 2.4e−05 | 1.2e−04 |

原始 logit 全程停留在 ±4.0 的千分之几以内。**结论：该 checkpoint 中
28 层的门控彼此几乎完全相同，基本停在保守初始化值上，并没有"学到注入位置"。**

这解释了训练日志中 `semantic_layer_gate_global/object/attribute/relation`
从第 1 步到第 14 750 步数值几乎不变的现象。

**对论文的影响：**
- 3.4 节"通过层级门控控制参数高效适配的调节强度"——在**这一 checkpoint**上，
  门控提供的是**一个近似恒定的保守缩放**，而非学到的层级区分。
- 4.3.3 节把 binding 由 26/39 提升到 27/39 归因于"层门控用于控制注入位置与强度"，
  证据不足：该提升更可能来自门控的**初始化设计**（global 通路开启、三个角色通路
  近乎关闭）本身，而非训练学到的层间调节。
- 建议正文如实表述为"门控在该设置下未被训练充分激活，其作用退化为保守的全局缩放"，
  或补充证据（如门控正则系数、梯度范数）说明为何未移动。

**(b) TP-SCDA 的门控不依赖时间步。** 代码核对（`diffusion/model/nets/PixArt.py`）：
`semantic_time_gate`（`Linear(hidden, 4)` → `sigmoid`）**只被
`_semantic_residual()` 与 `_collect_semantic_stats()` 使用**；而这两者仅在
`semantic_token_attention` 关闭（即 pooled 残差路径）时才会被调用——
第 278 行在 `semantic_token_attention_enabled` 为真时把 `semantic_conditions` 置为 `None`。

因此 **TP-SCDA 实际使用的门控是 `λ_l = σ(w_l)`，逐层、逐角色，与扩散时间步 t 无关**，
与论文 3.4 节"该门控由层参数和时间步嵌入产生""λ_l(t)"的表述不符。

这正是论文【待补】第 62 条要求"按代码核对实际形式并改写"的地方。
**本任务不修改论文**，仅提供核对结果与证据。

**(c) 有效注入强度很小。** `semantic_token_gate` 标量 = 0.2085，
有效系数 `semantic_token_gate_max · tanh(·)` = **0.08 × tanh(0.2085) = 0.01645**。
即 token-pair 分支的贡献被缩放到约 **1.6%**，与"总体 CLIPScore 与 Frozen 几乎持平"
（0.273597 vs 0.273209，+0.14%）的观察一致。

**产物：** `results/training/layer_gate.png`、`layer_gate_statistics.csv`、
`timestep_gate.png`（明确标注该模块在 TP-SCDA 路径中不参与计算）。

---

---

## ISSUE-009  安装 mmcv 时 numpy 被升级到 2.x（已回退）

**严重程度：** 高（潜在）—— 已及时发现并回退，未造成数据污染

为跑 GenEval 的 Mask2Former 检测器，需要编译带算子的 `mmcv`。执行
`pip install mmcv==2.1.0` 时，pip 的依赖解析把 **numpy 从 1.26.4 升级到 2.2.6**。
而 **PyTorch 2.1.2 不支持 numpy 2.x**，会破坏生成管线与所有依赖 numpy 的脚本。

**处置：** 立即 `pip install numpy==1.26.4` 回退，并验证
`numpy 1.26.4 / torch 2.1.2 / torchvision 0.16.2 / diffusers 0.30.3` 互操作正常、
`mmcv.ops` 仍可用。

**影响评估：** 当时正在运行的生成进程在启动时已载入 numpy 1.26.4，**不受影响**；
回退后新启动的进程也恢复正常。但**必须**在完成全部生成后重新做一次逐位一致性校验
（见下），以确认环境变更没有改变出图结果。

---

## ISSUE-010  GenEval 官方评测器要求 mmdet 2.x，本机为 mmdet 3.x

**严重程度：** 中 —— 只影响表6（GenEval Color Attribution）

GenEval 的 `evaluation/evaluate_images.py` 默认配置路径为
`configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py`（**mmdet 2.x** 命名），
且其 README 明确要求 `git checkout 2.x`。本机可安装的是 mmdet 3.3.0，配置命名为
`mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py`，API 亦不同（2.x 与 3.x 不兼容）。

**实测结论（2026-09-18）：无法在本机跑通，记为 `NOT COMPLETED`。** 具体阻断点：

1. **权重键名不兼容**：用 mmdet 3.x 的同名配置 `mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py`
   加载官方 2.x checkpoint，`init_detector` 报告大量键不匹配：
   - 源（2.x）：`...encoder.layers.0.attentions.0.sampling_offsets.weight`
   - 目标（3.x）：`...encoder.layers.0.self_attn.sampling_offsets.weight`
   即 3.x 把 `attentions.0/attentions.1/ffns.0` 改名为 `self_attn/cross_attn/ffn`。
   模型能"构建成功"，但**这些模块是随机初始化**，检测器无效。
2. **API 返回格式不兼容**：GenEval 的 `evaluate_images.py` 依赖 mmdet **2.x** 的
   `inference_detector` 返回「按 80 类分组的 bbox 列表」（`bbox[index]` 为 (N,5) 数组）；
   mmdet 3.x 返回 `DetDataSample`，`result[0] / result[1] / bbox[index]` 全部失效。
3. **装 mmdet 2.x 会破坏已验证的 CompBench 评测器**：mmdet 2.x 需要 mmcv **1.x**，
   而 T2I-CompBench++ 的 UniDet 评测器需要 mmcv **2.x**，两者不可共存于同一环境。

**处置：** 不修改官方评测器、不用替代检测器冒充官方结果。
`results/summary.json` 中 GenEval Color Attribution 记为 `"status": "not_completed"`。
**可行路径（供作者参考）**：在**独立 conda 环境**中装 mmdet 2.x + mmcv 1.x 后运行官方评测器；
本次因与 CompBench 评测窗口冲突未执行。

---

## ISSUE-011  【严重】交叉注意力掩码被静默关闭，导致短提示词出图退化

**严重程度：极高 —— 影响论文已有的全部生成结果**

### 现象

用 T2I-CompBench++ 官方 val 提示词出图时，图像呈现「熔融/海报化」的病态外观
（例如 `A bathroom with beige tile and a white toilet.` 生成彩色碎块而非浴室）。
而 `asset/samples.txt` 的长提示词出图正常。

### 定位过程（每一步都有对照）

| 测试 | 结果 | 结论 |
|---|---|---|
| 用 **diffusers 原生 `PixArtAlphaPipeline`** 加载 `models/PixArt-XL-2-512x512` 生成同一提示词 | **正常图像**（干净的绿色墙面长凳、正常的浴室） | 权重没问题，**问题在本仓库的推理路径** |
| 逐张量比对 `PixArt-XL-2-512x512-native-gate-init.pth` 与 HF `diffusion_pytorch_model.safetensors` | **maxdiff = 0.000e+00**（`x_embedder.proj`、`y_embedder.y_embedding`、`blocks.0.cross_attn.q_linear`、`mlp.fc1`、`final_layer.linear` 等全部逐位相同） | **checkpoint 完全正确** |
| 同一提示词 fp16 vs fp32 推理 | 两者**输出几乎相同，都退化** | 不是数值精度问题 |
| `git diff --ignore-cr-at-eol` 比对推理路径与上游 PixArt-alpha | `diffusion/model/nets/PixArt_blocks.py` 的 `MultiHeadCrossAttention.forward` 被改写 | 锁定改动点 |

### 根因

上游 PixArt-alpha 的写法是：

```python
q  = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
if mask is not None:
    attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
x = xformers.ops.memory_efficient_attention(q, k, v, p=..., attn_bias=attn_bias)
```

本仓库改写为「xformers 不可用时回退到 SDPA」，但在回退分支里：

```python
if mask is not None:
    if isinstance(mask, (list, tuple)):
        key_padding = None          # ← 掩码被静默丢弃
```

**关键**：`PixArt.forward` 传给 `PixArtBlock` 的并不是张量掩码，而是
`y_lens = mask.sum(dim=1).tolist()`，即一个 **Python list**：

```python
y_lens = mask.sum(dim=1).tolist()
...
x = auto_grad_checkpoint(block, x, y, t0, y_lens)   # 第 5 个位置参数 = PixArtBlock 的 mask
```

因此当 `B > 1` 时走 SDPA 回退分支，`mask` 是 list → `key_padding = None`
→ **图像 Patch 对全部 120 个 T5 token 做注意力，包括 padding token**。

> 注：本机**已安装 xformers 0.0.22**，但因为 `mask is not None and B == 1` 不成立，
> xformers 分支同样走不到，所以必然进入这个有缺陷的回退分支。

### 为什么长提示词看起来正常

T5 序列固定补零到 120。`asset/samples.txt` 的提示词很长（padding 少），
而 CompBench 的提示词很短（如 `a green bench and a blue bowl` 只有 8 个词，
超过 110 个 token 是 padding）。掩码失效对前者的影响可忽略，对后者是灾难性的。
这解释了本次实验**唯一一个此前未出现过的现象**。

### 影响

- **论文表2/表3/表4 以及 4.3.1—4.3.3 节的全部数值，都是在关闭掩码的条件下得到的**，
  即 CLIPScore `0.273 209` 等是在退化图像上计算的。
- 论文 4.3.4/4.3.5 节待补的 T2I-CompBench++ 与 GenEval 结果，若沿用当前代码，
  同样是在退化图像上评测，**与公开论文数值不可比**。

### 处置

**已于 2026-09-18 经作者确认后修复仓库代码。**

修改位置：`diffusion/model/nets/PixArt_blocks.py`
- 新增模块级函数 `cross_attn_key_padding(mask, key_len, device)`：把
  「每条样本有效 token 数的 list」展开为 SDPA 所需的 `(B,1,1,L)` 布尔 key-padding 掩码
  （True=保留）；张量掩码原样处理。
- `MultiHeadCrossAttention.forward` 的 SDPA 分支改为调用该函数，
  删除原先 `key_padding = None` 的错误回退。

**修复等价性验证**：修复后的仓库代码与修复前用运行时补丁得到的图像
在相同命令下 **20/20 逐位一致**；与已生成的全量 CompBench 图片比对时，
仅最后一批（prompt 16–19）因批组成不同出现 fp16 舍入级差异（MAE<2/255），
非实现差异。**因此本报告表5 的数值在修复后依然成立。**

**影响提示**：修复后**论文表2/表3/表4 的 CLIPScore 与 proxy 数值会改变**，
建议用修复后的代码重跑这三张表。


---

## ISSUE-012  complex 类需「每 prompt 10 张图」，与本次 1 图/prompt 的设置冲突

**严重程度：** 低—中 —— 只影响表5 的「复杂」一列

`3_in_1_eval/3_in_1.py` 第 63 行硬编码 `num = 10`（每条 prompt 的图片数），
且要求 `total_score = np.zeros(num * dataset_num)` 与图片数严格匹配。
因此当本次实验只生成 1 图/prompt 时，该脚本报
`ValueError: could not broadcast input array from shape (0,) into shape (10,)`，
complex 一列 300 张全部标记为 failed。

**处置（不修改官方评测器）：** 为 complex 类**单独**按官方要求生成
**每 prompt 10 张图**，并只取官方 val 集**前 100 条** prompt
（100 × 10 = 1000 张/方法），以 `--limit 100` 运行，
由驱动把官方 `complex_val.txt` 的**逐字前 100 行**写入运行目录
（官方文件本身不改动）。

**需在论文中注明的偏差：**
- complex 列基于 **100/300** 条官方 prompt（其余 7 列为全部 300 条）
- complex 列为 **10 图/prompt**，其余 7 列为 **1 图/prompt**


---

## ISSUE-013  门控优化改造与验证（已实施）

**严重程度：** — （这是对 ISSUE-008 的处置，非新问题）

**改动清单**（全部经作者确认后实施）：

| 文件 | 改动 |
|---|---|
| `diffusion/model/nets/PixArt.py` | 门控初始化可配置（默认 **±1.0**，线性区）；新增 `semantic_token_gate_activation`、`semantic_gate_max_warmup_steps`；`semantic_gate_scale()` 把两级压缩合并为**一个**可学习标量；新增 `set_semantic_gate_max()` 与 `semantic_gate_diagnostics()` |
| `diffusion/model/nets/PixArt_blocks.py` | ISSUE-011 的交叉注意力掩码修复 |
| `train_scripts/train.py` | `GateTrainingSchedule`（注入上限升温 / 正则退火 / 门控冻结-解冻）；门控**独立学习率组**；**逐参数组梯度范数**；非有限批次的**裁剪前**判定；门控健康检查告警；`max_train_steps` |
| `train_scripts/train.py` 日志 | `training_metrics.csv` 新增 18 列；控制台每 log 间隔输出 `GATE step=... eff=... drift_max=... layer_std_max=... grad=... grad_share=...` |
| `configs/PixArt_xl2_coco2017_token_pair_gate_opt.py` | 新配置，**不含任何蒸馏**（`distill_coef = 0.0`） |

**验证结果**（3000 步，前 2000 步门控冻结，43 分钟）：

| 指标 | 旧 run（14 786 步） | 新 run（解冻后 1 000 步） | 倍数 |
|---|---:|---:|---:|
| 门控位移 `|w-w_init|max` | 9.25e−03 | **3.57e−01** | **38.6×** |
| 层间标准差（λ 空间） | 2.95e−05 | **1.71e−03** | **58.1×** |
| 门控梯度范数 | ~4e−05 | 6–40 | ~10⁵ |

**结论：门控在解冻后 1 000 步内的位移就超过旧 run 全程 14 786 步的 38 倍，
且各层开始分化（层间标准差上升 58 倍）。ISSUE-008 的诊断得到证实。**

**遗留：** ① fp16 非有限批次 286/3000（9.5%），建议改 bf16 或降低 `semantic_token_gate_max`；
② 这只是可行性验证，门控仍在上升期，需跑满 14 786 步才能得到可用 checkpoint；
③ 详见 `results/GATE_OPTIMIZATION.md` 附录。

**未做：** 未改动任何蒸馏相关代码或系数（按要求）。

---

### ISSUE-013 补充：跑满 14 786 步后的实测结果 —— 显著退步

门控改造**在机制上成功、在指标上失败**，两者都要如实记录。

**训练侧（成功）**：门控确实学会并移动了。有效强度从解冻时的 0.150
收敛到 **0.0632**（`semantic_gate_raw` −1.320），层间标准差 3.51e−2，
`drift_max` 1.71。ISSUE-008 的诊断（门控因 sigmoid 饱和 + 学习率不足而不动）成立。

**指标侧（失败）**：把门控打开**显著损害了绑定类指标**。
300 prompt × 1 图/类、官方评测器、同一代码状态，逐 prompt 配对 10 000 次
bootstrap（脚本 `results/_work/compbench_paired_test.py`）：

| 类别 | 冻结基线 | 门控改造 | 差值 | 95% 区间 | 判定 |
|---|---:|---:|---:|---|---|
| 颜色 | 0.3956 | 0.3616 | **−0.0340** | **[−0.0653, −0.0029]** | **退步（显著）** |
| 形状 | 0.4158 | 0.3897 | **−0.0261** | **[−0.0502, −0.0023]** | **退步（显著）** |

**同一次检验还得到一个对论文更重要的结果**：论文当前的 TP-SCDA checkpoint
在三个**目标**绑定类上（颜色/形状/纹理）与冻结基线**无显著差异**，
唯一显著的类别是计数，且是退步（−0.0125，[−0.0239, −0.0020]）。

**有效强度与绑定类表现（三个点，单调向下）：**

| 有效强度 | 模型 | 绑定类 |
|---:|---|---|
| 0 | 冻结基线 | 基线 |
| 0.0164 | 论文当前 checkpoint | ≈ 基线（无显著差异） |
| 0.063 | 门控改造 | **显著低于基线** |

**结论：** 该语义分支在当前设计下**注入越强越差**。论文当前 checkpoint
与冻结基线持平，是因为其门控实际近乎关闭（0.0164），
而非因为分支提供了有效的绑定信号。论文 4.3.3 节不能写"门控打开后指标提升"。

**详细分析：** `results/improvement_report/GATEOPT_FINDING.md`；
完整检验表：`results/improvement_report/compbench_paired_tests.csv`。

**对 B1/B2 的影响：** 两者都以该退步基座为起点，因此**主要与自身基座比较**，
与 frozen 的直接比较会被基座退步污染。已在 `results/b1b2/B1_B2_LOG.md` 记录。


---

## ISSUE-014  掩码修复的代价：注意力内核变慢；xformers 快速路径不可用

**严重程度：** 低（性能，非正确性）

**现象：** ISSUE-011 修复后，生成吞吐从约 0.75 s/图降到约 1.9 s/图。

**原因：** 给 `torch.nn.functional.scaled_dot_product_attention` 传入显式
`attn_mask` 会**禁用 flash-attention 内核**，退回 memory-efficient/math 后端。

**尝试过的优化（失败，已回退）：** 改用上游 PixArt-alpha 的
`xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N]*B, mask)` 以保住融合内核。

**失败原因：** `BlockDiagonalMask` 要求 key 总数等于 `sum(kv_seqlen)`，
而 PixArt 传给 block 的是**补零到 120 的 key**（B×120），该恒等式不成立。
实测输出为**纯噪声**（高频能量 35.3 vs 正常图 10.2），已立即回退。

**当前实现：** 有掩码时走 SDPA 显式掩码（正确，约 1.9 s/图）；
无掩码时走 xformers 融合内核。

**因此：** `results/efficiency/efficiency.json` 中的推理耗时是在**修复前**测的，
已过期，需要重测。参数量与显存部分不受影响。

**可能的后续优化方向（未实施）：** 把 key 按各样本真实长度裁切后再做 block-diagonal
注意力（即真正实现变长序列批处理），而不是对补零序列做掩码。改动较大，本次未做。


---

## ISSUE-015  【已修复】门控默认初始化改动污染了旧 checkpoint（我引入的回归）

**严重程度：** 高 —— 曾导致一次表2/3/4 重跑结果作废

**经过：** 为做门控优化，我把模型 `semantic_gate_init` 的**默认值**从 ±4.0 改成 ±1.0。
但 `token_pair` 与 `distill` 两个 checkpoint **是在「可学习层门控」功能加入之前训练的，
其 state_dict 里根本没有 `semantic_token_layer_gate` 这个参数**（已核实：
`layer_gate_in_ckpt=False`）。加载时该参数取模型默认值 —— 于是从历史值
λ=0.018/0.982 变成 λ=0.269/0.731，**角色分支注入强度被放大 15 倍**。

**后果：** 第一次表2/3/4 重跑中，token_pair 的 CLIPScore 得到 0.1879
（Frozen 0.3438），绑定 proxy 的 object_accuracy 只有 0.24–0.41，明显是模型被破坏。

**修复：** 把模型默认恢复为 **±4.0**（保持历史行为），新的初始化只通过
`configs/PixArt_xl2_coco2017_token_pair_gate_opt.py` **显式**设置 ±1.0。
修复后验证：旧 checkpoint 加载后 `layer_gate global=0.9820 / roles=0.0180`，
与历史完全一致。

**影响范围：** 只有 token_pair 的表2/3/4 结果作废（已重跑）。
表5 不受影响——它用的 `learnable_layers` checkpoint **含有**该参数，可正确载入。
pooled / full-span 不使用 token attention，不受影响。

**教训：** 修改模型参数的**默认初始化**会静默改变所有"缺该参数"的旧 checkpoint 的行为。
默认值必须保持历史行为，新行为一律显式配置。

---

## ISSUE-016  现有 SCDA checkpoint 是在「掩码失效」的基座上训练的，不能直接用于修复后的基座

**严重程度：** 高 —— 影响表2/3/4 的解释

**观察（修复掩码后重跑，64 提示词 × 3 seed）：**

| 方法 | 修复前 CLIPScore | 修复后 CLIPScore | 相对 Frozen |
|---|---:|---:|---:|
| 冻结 PixArt | 0.273209 | **0.343811** | — |
| Pooled SCDA | 0.247061 | **0.281300** | −18.2% |
| Full-span SCDA | 0.247550 | **0.283517** | −17.5% |
| Token-pair SCDA | 0.266105 | *（ISSUE-015 污染，已重跑）* | — |
| Token-pair + learnable | 0.273597 | **0.343690** | −0.03% |

**两点：**
1. **修复掩码让冻结基线从 0.2732 提升到 0.3438（+25.8%）** —— 直接量化了 ISSUE-011
   对图像质量的破坏程度。
2. **Pooled / Full-span 的相对退化从 −9.6%/−9.4% 扩大到 −18.2%/−17.5%。**

**推断（基于两点证据，非直接证明）：** 这两个 SCDA 分支是在**掩码失效的基座**上训练的，
它们学到的残差实际上在"补偿"基座注意力的错误。当基座被修好后，这个补偿变成了偏差，
因此相对退化反而变大。

**对论文的影响：**
- 表2/3/4 若直接用旧 checkpoint 在修复后的代码上重跑，得到的**不是**各方法能力的公平对比，
  而是"在错误基座上训练的模型放到正确基座上"的表现。
- 若论文要以修复后的代码为准，**5 个消融方法都需要在修复后的代码上重新训练**。
- 若维持原 checkpoint，则应如实说明：所有生成结果都受 ISSUE-011 影响，
  数值来自掩码失效的推理，仅用于方法间相对比较。


---

## ISSUE-017  【已修复】门控激活函数默认值同样污染旧 checkpoint（ISSUE-015 的同类问题）

**严重程度：** 高 —— 曾使 token_pair 的重跑结果二次作废

**经过：** 门控优化时我把有效强度的公式从
`gate_max × tanh(semantic_token_gate)` 改成 `gate_max × sigmoid(semantic_token_gate)`，
并把**默认值**设为 `sigmoid`。这与 ISSUE-015 是同一类错误：
旧 checkpoint 里虽然**有** `semantic_token_gate`（值会被载入），
但**公式变了**，实际强度随之改变：

| checkpoint | gate_max | 历史 eff = g·tanh(raw) | 误用 sigmoid 后 | 放大 |
|---|---:|---:|---:|---:|
| token_pair | 1.0 | **0.199 454** | 0.550 429 | **2.76×** |
| learnable | 0.08 | **0.016 446** | 0.044 162 | 2.69× |
| distill | 0.08 | 0.016 005 | 0.044 024 | 2.75× |

**修复：** 激活函数默认值恢复为 **`tanh`**（历史行为）；
新设置只通过 `configs/PixArt_xl2_coco2017_token_pair_gate_opt.py`
**显式**声明 `semantic_token_gate_activation = 'sigmoid'`。
同时把 `semantic_token_gate_init` 默认值恢复为 **0.05**（历史值）。

**修复后审计**（加载各 checkpoint 实测有效强度）：

```
tokenpair   eff=0.199454  历史=0.199454  ✓
learnable   eff=0.016446  历史=0.016446  ✓
distill     eff=0.016005  历史=0.016005  ✓
baseline / pooled / fullspan 不使用 token attention，不受影响
```

**表5 不受影响**：它生成于 11:39，两次默认值改动都发生在 16:00 之后。

**两次回归的共同教训：**
修改模型参数的**默认值**（初始化或计算方式）会静默改变所有"缺该参数"或
"依赖该公式"的旧 checkpoint 的行为。正确做法是——
**默认值必须冻结在历史行为上，任何新行为都通过配置显式开启**。


## ISSUE-018  B1 的边约束会改变前向传播，推理时必须同样提供解析边（已处理）

**严重程度：** 高（若不处理，会使 B1 的训练/推理不一致，评测结果无效）

**问题：** B1 把 pair 亲和度乘上解析出的 object→attribute 邻接矩阵，
这一步在**前向传播内部**。训练时数据管道提供 `semantic_edges`；
但 CompBench 生成脚本原来的 `data_info` 里没有边，
`edge_pairs is None` → 边门控**被静默跳过** →
checkpoint 会以"它被训练去避免的那种无约束亲和度"被评测。

**处理：**

1. 在 `tools/prepare_semantic_masks.py` 增加 `build_semantic_edges()`，
   与 `build_semantic_masks()` 共用同一套解析链
   （`clean_caption` → spaCy → `build_labels` → `overlapping_indices`），
   保证配对顺序与 `tools/build_edge_index.py` 一致，截断上限同为 24。
2. `results/_work/gen_compbench.py` 与 `tools/generate_scda_samples.py`
   在 `semantic_pair_edge_gate=True` 时按生成脚本的方式构造边张量
   （`torch.from_numpy(np.stack(...))` → int64 → device），与数据集侧同构。
3. **一致性验证**：从 118 287 条 prompt 中随机抽 200 条，
   逐行比对"推理时解析"与"训练时索引"：

   ```
   checked 200 prompts, mismatches: 0
   VERDICT: CONSISTENT
   ```

**残留风险：** 边数超过 24 的样本在两侧都按同一顺序截断到 24，行为一致；
该比例已在 `edge_index.npz` 的构建日志中记录（平均 4.40 个 object token/样本）。


## ISSUE-019  冒烟测试用「输出是否改变」判断 B1 是否生效是无效判据（已修正）

**严重程度：** 中（会让人误判 B1 完全没接线）

**经过：** B1 冒烟测试最初比较 `B1 开/关` 两次前向的**最终输出**，
结果为 `differs from B0: False`，一度看起来像 B1 没生效。
实际原因是 `SemanticTokenCrossAttention.out_proj` 被**零初始化**
（`nn.init.zeros_`，ControlNet 式 zero-conv），
语义分支在初始状态下整体是 no-op ——
此时**任何**注意力变化都不会传导到最终输出，与 B1 是否接线无关。

**修正：** 判据改为在**模块层面**验证——
把 `out_proj.weight` 填入固定随机值使分支可观测，再比较开关 B1 的模块输出：

```
[ok] B1 changes the branch output  max|delta|=3.732e-04
[ok] edge matrix matches parsed pairs
```

**教训：** 零初始化的分支不能用"端到端输出差异"做接线验证，
必须在分支内部（或填充 out_proj 后）验证。


## ISSUE-020  `run_eval.py --method` 为硬编码枚举，新变体无法评测（已修复）

**严重程度：** 低（会直接报错，不会静默出错）

**问题：** `run_eval.py` 的 `--method` 是固定 `choices` 列表，
新增的 `b1` / `b2` 不在其中，评测会在参数解析阶段失败。

**修复：** 把 `'b1'`、`'b2'` 加入 `choices`。
输出目录约定不变（生成 → `/root/compbench_work/images/<method>/`，
评测 → `/root/compbench_work/results/<method>/<category>/`），
与已有方法完全一致。


## 偏差汇总表

| 编号 | 类型 | 是否影响本次表5/表6 | 状态 |
|---|---|---|---|
| ISSUE-001 | 环境遮蔽，已最小修复 | 否（已恢复逐位一致） | 已修复并验证 |
| ISSUE-002 | 历史元数据缺失 | 否 | 已记录，新实验已补全元数据 |
| ISSUE-003 | 2 个消融方法无法逐位复现 | 否 | 已记录，不重跑 |
| ISSUE-004 | 官方协议规模远大于预估 | 是（时间预算） | 分阶段推进，如实记录实际覆盖 |
| ISSUE-005 | GPU 争用导致早期测速偏低 | 否 | 已重新测量 |
| ISSUE-006 | 磁盘限制，图片外置 | 否 | 已记录 |
| ISSUE-007 | 正文写作项 | 否 | 未处理（非服务器任务） |
| ISSUE-008 | 层门控未学习 + TP-SCDA 门控不依赖时间步 | 影响正文 3.4 / 4.3.3 表述 | 已量化并记录，未改论文亦未改代码 |
| ISSUE-009 | pip 装 mmcv 时 numpy 被升到 2.x | 潜在，已回退 | 已回退并验证；将在生成后复核逐位一致性 |
| ISSUE-010 | GenEval 要求 mmdet 2.x，本机 3.x | 仅表6 | 评估中，跑不通则记 NOT COMPLETED |
| **ISSUE-011** | **交叉注意力掩码被静默关闭** | **影响论文全部生成结果** | **已定位并验证，未改代码，待作者决策** |
| ISSUE-012 | complex 需 10 图/prompt | 仅表5「复杂」列 | 单独生成前 100 条 × 10 图，已记录偏差 |
| ISSUE-013 | 门控优化改造与验证 | 论文 3.4 / 4.3.3 节 | **已实施并验证：门控学习速度提升约 570×** |
| ISSUE-014 | 掩码修复使注意力内核变慢 | 性能 | 已记录；xformers 快速路径实测不可用 |
| **ISSUE-015** | **门控默认初始化污染旧 checkpoint** | **曾使一次重跑作废** | **已修复并验证；token_pair 已重跑** |
| **ISSUE-016** | **旧 SCDA checkpoint 与修复后的基座不匹配** | **表2/3/4 解释** | 已量化，待作者决策 |
| **ISSUE-017** | **门控激活默认值污染旧 checkpoint** | **曾使重跑二次作废** | **已修复并审计通过；token_pair 已重跑** |
| ISSUE-018 | B1 边约束需在推理时重现 | 若忽略则 B1 评测无效 | 已接线，200/200 逐行一致 |
| ISSUE-019 | 冒烟测试判据被零初始化分支掩盖 | 否 | 已改为模块层面判据 |
| ISSUE-020 | `run_eval.py --method` 枚举缺新变体 | 否 | 已修复 |
