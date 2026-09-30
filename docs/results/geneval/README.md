# GenEval Color Attribution（论文表6）

**状态：NOT COMPLETED**

**原因（见 `../EXPERIMENT_ISSUES.md` ISSUE-010）**：
官方评测器要求 **mmdet 2.x**，本机仅可安装 mmdet 3.x，二者存在
① 权重键名系统性不兼容（`attentions.0`→`self_attn` 等）、
② `inference_detector` 返回格式不兼容（DetDataSample vs 2.x 的按类列表）。
改装有风险且会与已可用的 CompBench UniDet 评测器（需 mmcv 2.x）冲突。

**未做**：没有用替代检测器或自实现流程冒充官方结果。
建议在**独立环境**中安装 mmdet 2.x + mmcv 1.x 后运行官方评测器。
