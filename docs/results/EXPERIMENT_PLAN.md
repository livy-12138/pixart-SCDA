# TP-SCDA 实验补全计划

生成时间：2026-09-18（服务器本地时间）
执行环境：`/root/private_data/PixArt-alpha-attentiongate`

---

## 1. 发现了什么（体检结果）

### 1.1 硬件与软件环境

| 项 | 实测值 |
|---|---|
| GPU | NVIDIA L20, 46068 MiB (48 GB), 驱动 550.142 |
| CUDA (驱动) | 12.4 |
| PyTorch | 2.1.2+cu121 |
| Python | 3.10.21 |
| diffusers | 0.30.3 |
| transformers | 4.44.2 |
| 网络 | 可达 huggingface.co / pypi / github.com（无 `curl`，用 python urllib） |
| `clip` / `open_clip` | **未安装**；CLIP 通过 `transformers.CLIPModel` 使用，权重已在本地 HF 缓存 |
| `detectron2` | **未安装**（影响 T2I-CompBench++ 部分子项，见 §5） |

### 1.2 代码与权重

- 主干：`PixArt_XL_2`（`diffusion/model/nets/PixArt.py`，depth=28, hidden=1152, patch=2, heads=16）。
- 文本编码器：T5-v1.1-XXL，本地于 `output/pretrained_models/t5_ckpts/t5-v1_1-xxl`。
- VAE：`sd-vae-ft-ema`。
- 生成入口：`tools/generate_scda_samples.py`（batch 生成，512px，DPM-Solver 20 步）。
- 语义模块开关（`PixArt.__init__`）：
  - `semantic_conditioning` —— pooled SCDA 残差路线
  - `semantic_token_attention` —— token-pair 路线
  - `semantic_token_layer_gate`（形状 `(depth, 4)`）—— 可学习层门控
  - `semantic_residual_scale`、`semantic_token_gate_max`

### 1.3 **关键发现：生成管线已逐位复现验证**

用 `tools/generate_scda_samples.py` 对 `asset/samples.txt` 的 group_01（8 条 prompt，seed 43）重新生成，与磁盘上已有图片逐像素比对：

| 方法 | 复现用 checkpoint | flags | 结果 |
|---|---|---|---|
| **Frozen PixArt** | `output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth` | （无语义 flags） | **8/8 逐位一致** ✅ |
| **TP-SCDA（= token-pair + learnable layers）** | `output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth` | `--semantic-conditioning --semantic-token-attention --semantic-token-gate-max 0.08 --semantic-residual-scale 0.0` | **8/8 逐位一致** ✅ |
| Full-span SCDA | `output/coco2017_scda_fullspan/checkpoints/epoch_6_step_22182.pth` | `--semantic-conditioning --semantic-residual-scale 0.25` | **8/8 逐位一致** ✅ |
| Pooled SCDA | `output/coco2017_scda/checkpoints/epoch_5_step_18485.pth` | `--semantic-conditioning --semantic-residual-scale 0.25` | ✗ MAE≈1.5，13–44% 像素差 >2 |
| Token-pair SCDA | `output/coco2017_scda_token_pair_positive_mb8_acc4/checkpoints/epoch_5_step_60000.pth` | `--semantic-conditioning --semantic-token-attention --semantic-token-gate-max 1.0` | ✗ MAE≈4–14 |

**CLIP 打分协议已验证**：对已对齐的 prompt 复算 CLIPScore，与 `clip_scores.csv` 记录值 **差值 = 0.000000**（完全一致）。协议为本地 CLIP ViT-B/32、图像与完整 prompt 的 cosine similarity、prompt 截断至 77 token。

> **意义**：本次要补的两个核心数字（表5 的 TP-SCDA 八项、表6 的 Color Attribution）所需的两个模型——**Frozen PixArt 与 TP-SCDA**——配方均已逐位确认，可以放心用于新 benchmark。

### 1.4 已完成、不需要重跑的实验

- 统一 CLIPScore（64 prompt × 3 seed）：Frozen `0.273209`、Pooled `0.247061`、Full-span `0.247550`、Token-pair `0.266105`、Learnable-layers `0.273597`。原始数据：`output/*/clip_scores.csv`。
- 对象/属性/绑定 proxy（17 案例，51+39 判断）与 10 000 次 paired bootstrap 区间。原始数据：`output/attribute_binding_metrics_all*.csv`。
- 上述已填入论文表2、表3、表4。**本次不再重跑。**

---

## 2. 哪些缺失（本次要补的）

按任务书第 7 节回填对照表：

| # | 论文位置 | 需要的数 | 优先级 | 状态 |
|---|---|---|---|---|
| 1 | 表5 最后一行 TP-SCDA | T2I-CompBench++ 八项 | **1** | 缺 |
| 2 | 表6 最后一行 TP-SCDA | GenEval Color Attribution | **2** | 缺 |
| 3 | 图3 | Frozen vs TP-SCDA 同 seed 成对出图 | 3 | 缺 |
| 4 | 4.3.6 节 | 机制可视化热力图（层×时间步 Patch 响应、结构化偏置 B） | 3 | 缺 |
| 5 | 4.1 节末 | 参数量/训练时长/显存/推理时间/学习率/weight decay/loss 曲线 | 4–5 | 缺 |
| 6 | 3.2 节末 | parser/token 对齐 P/R/F1 | 6 | 缺（无人工真值则只做描述性统计） |
| 7 | 4.4 节 | 人工绑定标注 | 7 | 缺（只生成标注表，不伪造结果） |

**已测量规模**（实际下载官方 prompt 文件后统计）：

- T2I-CompBench++：**7 856 条**（color 860、shape 1000、texture 1000、spatial 1000、3d_spatial 1000、numeracy 1000、non_spatial 1000、complex 1000）
- GenEval：553 条总 prompt，其中 **color_attr 100 条**

---

## 3. 准备如何执行

### 3.1 统一协议（逐字遵守任务书第 3 节）

分辨率 512×512；采样器 DPM-Solver；20 步；CFG 4.0；CLIP ViT-B/32。benchmark 官方协议优先于本文 64-prompt 协议，两套分开报告。

### 3.2 执行顺序

```
Step 0  复现验证（已完成）                       ✅
Step 1  下载并固化官方 prompt 集 + 评测器代码     进行中
Step 2  小样本端到端冒烟（20 条，验证评测器可用） 待做
Step 3  全量生成 T2I-CompBench++ 图片            待做
Step 4  BLIP-VQA / 官方评测器打分                待做
Step 5  GenEval color_attr 100 条生成 + 评测     待做
Step 6  成对可视化 + 机制热力图                  待做
Step 7  训练日志 / gate 统计 / 效率指标          待做
Step 8  parser 描述性统计                       待做
Step 9  汇总 summary.json + EXPERIMENT_REPORT.md 待做
```

### 3.3 生成吞吐实测

group_01 共 8 张 512px 图（20 步 DPM-Solver）采样耗时 **8.0 s ≈ 1.0 图/秒**（不含约 100 s 的模型/T5 加载）。据此估算：

- T2I-CompBench++ 7 856 张 ≈ **2.2 h / 模型**
- GenEval color_attr 100 张 ≈ **2 min / 模型**

### 3.4 待用户确认的范围决策

是否在 TP-SCDA 之外，**同时跑 Frozen PixArt 的 T2I-CompBench++**（额外 ≈2.2 h）作为同协议对照：

- **赞成**：论文表5 的 PixArt-α 公开值为颜色 0.6690 / 形状 0.4927 / 纹理 0.6477。若我们自己的 Frozen PixArt 在同协议下复现出接近值，即可**验证评测管线正确**，并让 TP-SCDA 那一行有可解释的对照；否则单个数字无法判断好坏。
- **反对**：论文表5 只预留了 TP-SCDA 一行，多出的 Frozen 行需要作者决定是否入表。

### 3.5 已知风险

- **UniDet 评测器**：T2I-CompBench++ 的 2D 空间、3D 空间、计数三项官方使用 UniDet 检测器（`UniDet_eval/`），通常依赖 `detectron2`（当前未安装且编译困难）。若不可用，将采用官方 repo 中可运行的替代路径；若仍不可行，该项记为 `NOT COMPLETED` 并写入 `EXPERIMENT_ISSUES.md`，**不估计、不插值**。
- **评测器版本一致性**：必须全流程固定同一份评测器代码与权重版本，并写入 `config.json`。

---

## 4. 预计输出文件

```
results/
├── t2i_compbench_pp/   raw/ per_prompt.csv summary.csv summary.json config.json README.md
├── geneval/            raw/ per_prompt.csv summary.csv summary.json config.json README.md
├── visualization/      binding_examples/ multi_object_binding/ complex_prompts/ mechanism/
├── training/           loss.csv loss_curve.png config.json trainable_params.txt
│                       checkpoint_info.json layer_gate_statistics.csv layer_gate.png timestep_gate.png
├── efficiency/         efficiency.csv efficiency.json README.md
├── parser_alignment/   raw/ per_prompt.csv summary.csv summary.json README.md
├── human_binding/      annotation_sheet.csv
├── summary.json
├── EXPERIMENT_REPORT.md
└── EXPERIMENT_ISSUES.md
```

per-sample 原始行至少包含：`prompt_id, prompt, seed, generated_image, metric, prediction, success/failure, error_message`。

---

## 5. 已记录的偏差与问题（详见 EXPERIMENT_ISSUES.md）

1. **历史 eval 产物缺少运行元数据**：`output/*/evaluation_manifest.csv` 只有 `model,seed,output_dir,prompts,steps,cfg_scale`，**没有记录 checkpoint 路径与语义模块 flags**，导致无法从产物完全重建运行条件。
2. **Pooled SCDA 与 Token-pair SCDA 未能逐位复现**（Frozen / Full-span / Learnable-layers 均可逐位复现）。这两个方法的论文数值已存在（表2/表3/表4），本次不需要重跑；但该复现缺口已如实记录。
3. 论文第 4.1、3.2、3.3、3.4 节仍有若干 **【待补】** 属于作者本地正文写作问题（公式符号定义、中图分类号、通信作者标注、参考文献核对），不在本服务器任务范围内，不代为填写。
