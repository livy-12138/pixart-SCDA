# PixArt 迭代训练记录

固定评测提示词为 `asset/samples.txt`（64 条）。每轮图像和 `metrics.json` 位于 `experiments/run_002_gate_v2/`、`experiments/run_003_gate_v3/`。

## 结果

| 轮次 | loss 首/末/最低 | 观察 |
|---|---:|---|
| run_002_gate_v2 | 0.1907 / 0.1513 / 0.1334 | 主损失下降；原始日志 step 2028 出现 `grad_norm:nan`。 |
| run_003_gate_v3 | 0.1908 / 0.1519 / 0.1337 | neutrality + consistency 正则约 0.002，日志未见 NaN；尚无证据证明生成质量提升。 |

## 问题和下一轮方案

- 工作区缺少 `COCO2014Prepared10K/data_info.json`，T5 权重目录为空，当前不能安全启动训练。
- v2 可能有 fp16 溢出；沿用 v3 正则，增加 `isfinite` 梯度检查、异常时降学习率并保存故障 checkpoint。
- 资源恢复后从 v3 `epoch_10_step_3130.pth` 续训；每轮固定 64 条提示词生成样图，并计算 CLIP、FID/KID（有真实集时）及对象/属性/关系遵循评分。

checkpoint 保留在回收站路径 `/root/private_data/.Trash-0/files/output/`，未复制 131GB 大文件。
