# 人工绑定标注（v7，论文 4.4 节）

对比模型：`frozen_current`（冻结基线） vs `frozen_ebcol020`（本文主模型）。
两列在表里已按行随机盲化为 **图 A / 图 B**，对照表单独存放。

## 怎么做

1. 找**两名**标注者，各自独立完成，不要互相商量；全程不要打开
   `annotation_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv`。
2. 打开 `缩略图_*.png`（每张 25 行，左=图A、右=图B，标着 #item_id），
   对照 `annotation_sheet.csv` 逐行填写。
3. 每行对 **图A 和图B 各判一次**，取值三选一（SynGen 定义）：

   | 取值 | 含义 |
   |---|---|
   | `proper` | 提示词中的对象都出现，且属性绑定到正确的对象 |
   | `improper` | 对象都出现，但至少一个属性绑到了错误的对象 |
   | `neglect` | 至少一个提示词中的对象没有出现 |

   即 `annotator1` 列填 `proper/improper`（图A的判定）… 见下面的列格式。

4. 列格式：`annotator1` 与 `annotator2` 两列，每列写
   `A:proper B:improper` 这样的形式（A、B 是图 A / 图 B 的判定），
   `notes` 记录疑难案例。
5. 两人都填完后，跑：
   `python3 score_annotation.py <填好的annotation_sheet.csv>`
   它会给出各模型三类比例、Cohen's kappa（含 95% 自助区间）、
   以及需要裁决的条目数。**分歧条目由第三人裁决或两人讨论定稿**，
   定稿后再跑一次即可。

## 统计口径

- 每条提示词的两次判定（图A、图B）分别计入两个模型；
- kappa 按**提示词聚类**做自助重采样（每条提示词贡献 A/B 两个判定，不能当独立样本）；
- 该脚本不生成任何标注，只做统计。
