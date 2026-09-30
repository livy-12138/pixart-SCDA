# 文件清理记录（2026-09-18）

执行前私盘可用 **16 GB**，执行后 **121 GB**（释放 105 GB）。

判据：**只删除未被任何 `configs/` `experiments/` `reports/` `tools/` `results/`
文件引用的中间 checkpoint，以及崩溃转储**。所有被引用的、论文复现所需的权重一律保留。

---

## 保留（6 个，25.5 GB）

| 文件 | 大小 | 用途 |
|---|---:|---|
| `output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth` | 2.3 G | 冻结基线；所有 SCDA 训练的初始化权重 |
| `output/coco2017_scda/checkpoints/epoch_5_step_18485.pth` | 4.6 G | 论文表2/3/4 —— Pooled SCDA |
| `output/coco2017_scda_fullspan/checkpoints/epoch_6_step_22182.pth` | 4.6 G | 论文表2/3/4 —— Full-span SCDA |
| `output/coco2017_scda_token_pair_positive_mb8_acc4/checkpoints/epoch_5_step_60000.pth` | 4.7 G | 论文表2/3/4 —— Token-pair SCDA |
| `output/coco2017_token_pair_distill_gate008_sparse/checkpoints/epoch_1_step_2500.pth` | 4.7 G | 被 `configs/PixArt_xl2_coco2017_token_pair_learnable_layers.py` 引用 |
| `output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth` | 4.7 G | **TP-SCDA 主结果（论文表5）** |

> 注意：`*.pth` 每个约 4.7 GB，因为 `save_checkpoint` 同时写入
> 模型权重 + EMA 权重 + 优化器状态。若后续需要更多磁盘，可以把中间 checkpoint
> 改为只存 `state_dict`（可压到约 2.4 GB）。

## 已删除（21 个 checkpoint + 3 个 core dump，约 105 GB）

| 项目 | 大小 | 说明 |
|---|---:|---|
| `coco2017_scda_token_pair_positive_mb8_acc4/checkpoints/` 的 15 个中间 checkpoint | 71 G | 仅保留 `epoch_5_step_60000.pth`（论文引用） |
| `coco2017_token_pair_learnable_layers/checkpoints/` 的 5 个中间 checkpoint | 24 G | 仅保留 `epoch_1_step_14786.pth`（论文引用） |
| `coco2017_token_pair_gate_opt/checkpoints/epoch_1_step_3000.pth` | 4.7 G | 门控优化可行性验证 run 的 checkpoint；验证时门控仍在上升期，不是可用模型。**日志与 metrics CSV 已保留**在 `output/coco2017_token_pair_gate_opt/experiment_tables/` |
| `core.25683` `core.31793` `core.6349` | 6.5 G | 崩溃转储，无保留价值 |

**未改动**：所有 `output/*/` 的非 checkpoint 产物（CLIPScore、proxy 指标、
评测图片、训练 metrics CSV、日志）、`experiments/`、`reports/`、`results/`、
`asset/`、`models/`、`configs/`、`train_scripts/`、`diffusion/`。

## 清理后功能验证

用保留的 TP-SCDA checkpoint 重新出图（`tools/generate_scda_samples.py`，group_01 共 8 张）：

```
checkpoint loaded; missing=1 unexpected=0
group_01: 8 images
```

并目视确认 `A small cactus with a happy face in the Sahara desert.` 生成了正确的
仙人掌图像 —— 即 ISSUE-011 的掩码修复在清理后依然生效。

## 未清理项（按作者要求全部保留）

临时工作区位于 overlay 分区（不占私盘），按作者指示 **全部保留**：

- `/root/compbench_work/images`（9 600 张修复版 CompBench 图片）
- `/root/compbench_work/images_buggy`（4 800 张掩码失效版，ISSUE-011 对照用）
- `/root/compbench_work/eval`、`weights`、`geneval`、`results`、`results_buggy`、`logs`
- `/tmp/cbresearch`、`/tmp/cb`、`/tmp/ge`、`/tmp/smoke` 等
