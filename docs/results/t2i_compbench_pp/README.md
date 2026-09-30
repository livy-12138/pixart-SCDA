# T2I-CompBench++ 评测（论文表5）

**评测器**：T2I-CompBench++ 官方评测器（`BLIPvqa_eval/BLIP_vqa.py`、`UniDet_eval/*.py`、
`CLIPScore_eval/CLIP_similarity.py`、`3_in_1_eval/3_in_1.py`）。官方仓库文件未改动。

**数据来源**：官方 val 集，每类 300 条 prompt。除 complex 外为 **1 图/prompt**；
complex 为 **前 100 条 prompt × 10 图**（原因见 `../EXPERIMENT_ISSUES.md` ISSUE-012）。

**协议**：512×512，DPM-Solver 20 步，CFG 4.0，seed 43（每图独立确定性种子）。

**目录结构**：
```
<method>/<category>/summary.json    官方均值与计数
<method>/<category>/raw/per_prompt.csv  逐图原始记录（prompt/seed/图片/判定/分值）
<method>/<category>/failures.csv    失败样本（status=failed, error=...）
<method>/<category>/config.json     运行元数据
summary.csv                          全部 (method, category) 汇总
table5_tpscda_row.json              表5 可直接填入的行
```

**重要**：`method = tpscda` 使用的是**修复掩码后**（见 ISSUE-011）生成的图片。
`/root/compbench_work/results_buggy/` 保留了**修复前**（当前仓库代码）的结果，
两者不可混用。公开论文数值（PixArt-α 等）另表报告，协议不同，不构成统一排名。
