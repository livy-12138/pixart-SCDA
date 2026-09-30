# 面向文生图属性绑定的双门控交叉注意力替换

**Dual-gated cross-attention replacement for object–attribute binding in text-to-image generation**

面向 **PixArt-α / PixArt-XL-2-512** 的**免训练、推理期**注意力改写：在冻结主干上做 token-pair 交叉注意力重分配，并用**空间半径门控 + 时间去噪门控**把副作用限制在目标对象的支撑域与目标去噪窗口内。不新增可训练参数，不修改主干权重。

> **关于仓库名**：本仓库最早是早期 **TP-SCDA**（semantic-conditioned token-pair adaptation）线的实现与实验记录，那条线是**负结果**（八类指标与冻结基线持平），并暴露了两个实现缺陷。当前论文的工作是在**修复这些缺陷之后**完成的。旧版 README 已归档到 [`docs/legacy/`](docs/legacy/README_2026-09-30_TP-SCDA-negative.md)，负结果与缺陷记录完整保留在 [`docs/results/`](docs/results)。

---

## 1. 方法要点

主干（冻结）：PixArt-XL-2-512（28 层 DiT，hidden 1152，16 heads）+ T5-xxl 文本编码器，全部实验共用同一份冻结检查点。

- **token-pair 注意力重分配**：在 DiT 每个 block 的交叉注意力前向中，把对象 token 与其配对属性 token 的注意力权重向该属性 token 集中、其余列按行归一化等比稀释。这是 softmax **之后**的逐行重分配，不引入加性偏置；实现见 `diffusion/model/nets/PixArt_blocks.py`（支持 `replace` / `reweight` / `raise` / `equalize` / `outside` / `sink` 等模式）。
- **半径渐变的空间门控**：以对象锚点为中心、沿半径由中心强度衰减到对象支撑域边界，边界外强度为 0，从而只在该对象附近生效。实现见 `diffusion/model/nets/lcar.py`。
- **时间去噪门控**：只在归一化去噪进度 p ∈ [0.25, 0.5] 的窗口内施加替换，窗口外**退化为恒等映射**（不替换）。
- 提示词解析不出对象-属性对时掩码为空、算子自动跳过——这也是 2D 空间与 3D 空间两类上算子不激活的原因。

论文口径的参数：替换强度 0.9→0.1、半径倍率 ρ = 1、竞争抑制系数 μ = 0.5、时间窗 [0.25, 0.5]，全部写在配置文件里，无隐式默认值。

## 2. 主要结果

T2I-CompBench++ 官方评测器，每类 300 条提示词、每提示词 1 张（n = 300）：

| 设置 | 颜色 | 计数（numeracy） |
|---|---:|---:|
| 冻结基线（不施加干预） | 0.3956 | 0.5155 |
| 半径替换（全程施加） | **0.4679**（Δ +0.0722，显著） | 0.1692（Δ −0.3464，**显著下降**） |
| 半径替换 + 时间门控 [0.25, 0.5] | **0.4411**（Δ +0.0455，显著） | 0.4207（Δ −0.0948，代价比全程降低 **72.63%**） |

- 时间门控把颜色增益的计数代价从 −0.3464 压到 −0.0948，颜色增益仍显著：两类串扰可以被**分别切断**。
- **代价如实标注**：计数在两种设置下都低于冻结基线，本文的定位是"缓解而非消除"；其余类别（形状/纹理/2D/3D 空间/非空间/复杂）见论文表 2、表 3。
- 同协议（每类前 100 条，复杂类每提示词 10 张）与现有方法对比：颜色 0.4050 > 冻结 PixArt-α 0.3086 > SynGen 迁移实现 0.2969 > DreamRenderer 适配实现 0.2475；形状 0.5359 为同组最高。该协议与 300 条协议**不可直接比较**。
- 推理开销实测（NVIDIA L20 46 GB / CUDA 12.1 / PyTorch 2.1.2 / fp16 / batch 4 / DPM-Solver 20 步 / CFG 4.0 / 512×512）：冻结主干 0.75 s/图、峰值显存 22.62 GB；本文方法 1.24 s/图、峰值 22.63 GB。

## 3. 代码位置

| 路径 | 作用 |
|---|---|
| `diffusion/model/nets/lcar.py` | 半径渐变 + 锚点空间门控（LCAR） |
| `diffusion/model/nets/PixArt_blocks.py` | 交叉注意力改写入口与各替换模式；含 ISSUE-011 的 key-padding mask 修复 |
| `diffusion/model/nets/PixArt.py` | 模型主体、条件注入与门控挂钩 |
| `docs/results/_work/gen_compbench.py` | 生成 + 评测驱动（类别以命令行参数传入；随机种子按图像身份确定性派生） |
| `docs/results/_work/run_lcar_twaxis.py` | 半径 × 时间双轴扫描（论文时间谱一组实验） |
| `configs/` | 实验配置；配置之间以 `exec()` 链式继承（改基类会带动所有派生配置） |
| `tools/` `scripts/` `train_scripts/` `app/` | 数据准备、评测、训练入口（`train_scripts/train.py`）、Gradio demo |
| `docs/` `docs/results/` `docs/reports/` | 运行说明、逐提示词结果、配对检验、人工标注、效率实测、机制可视化素材 |

## 4. 复现

1. 环境：Linux + CUDA 见 [`docs/RUNNING_LINUX.md`](docs/RUNNING_LINUX.md)（`environment-linux.yml` / `requirements-linux.txt`）；DCU 见 [`docs/RUNNING_DCU.md`](docs/RUNNING_DCU.md)。
2. 数据与权重：T2I-CompBench++ 官方 val 提示词、PixArt-XL-2-512 与 T5-xxl 公开权重，**三者均未修改**。
3. 生成：用 `docs/results/_work/gen_compbench.py` 逐类生成（每类 300 条、每提示词 1 张）。
4. 评测：官方 T2I-CompBench++ 评测器逐类评测，得到 `per_prompt.csv`。
5. 统计：逐提示词配对差 + 提示词级 bootstrap 95% 区间（口径见论文 3.2 节）。

## 5. 数据可用性

**仓库内现有**

| 目录 | 内容 |
|---|---|
| `docs/results/t2i_compbench_pp/` | 冻结基线与 TP-SCDA 的逐提示词结果（每类 300 条） |
| `docs/results/t2i_compbench_pp_sameprotocol/` | 同协议（每类前 100 条）方法对比 |
| `docs/results/tables234_fixed/` | 汇总指标与 bootstrap 区间（论文表 2/3/4 口径） |
| `docs/results/binding_swap/`、`human_binding*/` | 绑定交换数据与人工标注材料 |
| `docs/results/efficiency/` | 推理耗时与显存实测 |
| `docs/results/visualization/` | 机制可视化素材 |
| `docs/results/PAIR_REPLACE_AND_NODISTILL_RESULTS.md`、`NAMING_KEY.md` | token-pair 替换结果说明、命名对照表 |

**说明**：生成图像与逐提示词评测结果体量较大，未随仓库发布；本文方法的实现、实验脚本与方法参数已随仓库公开——关键运行的参数预设固化在 `docs/results/_work/gen_compbench.py` 的 `METHODS` 中，可配合 `docs/results/_work/verify_repro.py` 与公开的提示词/权重/评测器重新生成（与论文"数据与代码可用性"一节表述一致）。

## 6. 早期 TP-SCDA 线（负结果，完整保留）

- 结果：TP-SCDA 在八类 T2I-CompBench++ 指标上与冻结基线**持平**，未观察到提升。
- 两个实现缺陷（当前论文全部实验以修复后的代码为前提）：
  - **ISSUE-011**：改写后的 `MultiHeadCrossAttention` 丢掉了 key 的 padding mask，导致每个图像 patch 都会把注意力分配给 T5 的 padding token（已修复，`cross_attn_key_padding`）。
  - **ISSUE-008**：可学习层门控初始化落在 sigmoid 饱和区（±4.0，dλ/dw ≈ 0.0177），整段训练几乎没学动（已诊断并重做，单步有效学习率提升约 570×）。
- 详细记录：[`docs/results/PROJECT_SUMMARY.md`](docs/results/PROJECT_SUMMARY.md)、[`docs/results/EXPERIMENT_ISSUES.md`](docs/results/EXPERIMENT_ISSUES.md)、[`docs/results/GATE_OPTIMIZATION.md`](docs/results/GATE_OPTIMIZATION.md)、[`docs/results/improvement_report/`](docs/results/improvement_report)；旧版 README 见 [`docs/legacy/`](docs/legacy/README_2026-09-30_TP-SCDA-negative.md)。

---

## 引用

向长幸，黄天云. 面向文生图属性绑定的双门控交叉注意力替换. 中国图象图形学报（审稿中）.

## 许可

本仓库是 [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) 的研究分支，基线版本 `cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892`，继承 **Apache-2.0** 许可（见 [`LICENSE`](LICENSE)）。
