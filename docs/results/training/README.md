# 训练与门控统计（论文 4.1 / 4.3.6 节）

- `loss.csv` / `loss_curve.png` —— 14 786 步训练损失
- `trainable_params.txt` —— 总参数 / 可训练参数及判据（与 `train_scripts/train.py` 一致）
- `config.json` —— 训练超参快照
- `checkpoint_info.json` —— checkpoint、训练时长、训练峰值显存
- `layer_gate_statistics.csv` / `layer_gate.png` —— 逐层角色门控
- `timestep_gate.png` —— 时间步门控（**该模块在 TP-SCDA 路径中不参与计算**，见 ISSUE-008）
