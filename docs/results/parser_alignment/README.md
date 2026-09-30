# Parser / token 对齐（论文 3.2 节）

**仅描述性统计。** 本服务器**没有人工真值**，因此按要求**不输出 P（精确率）/R（召回率）/F1**，
只统计：各 split 中解析出对象/属性/关系的 prompt 比例、每 prompt 平均元素数、
字符区间→T5 子词**对齐失败率**、T5 截断比例。

`raw/per_prompt.csv` 保存逐 prompt 原始计数，`summary.csv` / `summary.json` 为汇总。
