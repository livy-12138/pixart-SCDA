# 人工绑定标注（论文 4.4 节）

**本目录不含任何人工标注结果。** 标注必须由两名标注者完成。

- `annotation_sheet.csv` —— 工作表，200 行；**每行一条提示词，含两张图**（图 A / 图 B 为两个方法，已盲化）
- `annotation_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv` —— 盲化对照表，**标注完成前不要打开**
- 图片路径为服务器绝对路径：`/root/compbench_work/images`

## 标注说明（SynGen 三分类）

对每一行，给定提示词与图 A、图 B，分别判断：

| 取值 | 含义 |
|---|---|
| `proper` | 提示词中的对象都出现，且属性绑定到正确的对象 |
| `improper` | 对象都出现，但至少一个属性绑定到了错误的对象 |
| `neglect` | 至少一个提示词中的对象没有出现（实体遗漏） |

在 `annotator1` / `annotator2` 两列分别填写；`notes` 可记录疑难案例。

## 统计（标注完成后）

本脚本**不计算** Proper / Improper / Entity Neglect 比例与 Cohen's kappa，
也不生成 95% 置信区间——这些必须在真实标注完成后由作者计算。
