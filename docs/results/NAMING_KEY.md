# 命名对照表（务必先读这份）

评测脚本 `results/_work/run_eval.py` / `results/_work/gen_compbench.py` 里的
method 键名与论文术语**不一致**，容易混淆。以下为权威对照，
checkpoint 中的层门控参数已逐一验证。

| 评测键名 | 层门控 (28,4) | checkpoint | 实际是什么 | 论文中的位置 |
|---|---|---|---|---|
| `frozen` | 无 | `pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth` | 原始 PixArt-α，零适配 | **基线** |
| `pooled` | 无 | `coco2017_scda/epoch_5_step_18485.pth` | 池化 SCDA | 表4 消融 |
| `fullspan` | 无 | `coco2017_scda_fullspan/epoch_6_step_22182.pth` | 全跨度 SCDA | 表4 消融 |
| `tokenpair` | **无** | `coco2017_scda_token_pair_positive_mb8_acc4/epoch_5_step_60000.pth` | token-pair **去掉层门控** | 表4 消融 "Token-pair" 行 |
| **`tpscda`** | **有** | `coco2017_token_pair_learnable_layers/epoch_1_step_14786.pth` | **token-pair + 可学习层门控 = 论文的 TP-SCDA** | **论文主方法** |
| `gateopt` | 有 | `coco2017_token_pair_gate_opt/epoch_1_step_14786.pth` | **本次新增的实验变体**（门控训练控制改造） | **不在论文里** |
| `b1` | — | `coco2017_token_pair_b1_edgegate/` | 本次新增：边门控（已终止，空干预） | 不在论文里 |
| `b2` | 有 | `coco2017_token_pair_b2_bindingloss/` | 本次新增：绑定对齐损失 | 不在论文里 |
| `b2_neutral` | 有 | `coco2017_token_pair_b2_neutralgate/` | 本次新增：绑定损失 + 中性门控初值 | 不在论文里 |
| `b2_frozengate` | 有 | `coco2017_token_pair_b2_frozengate/` | 本次新增：绑定损失 + 门控冻结 | 不在论文里 |
| `probe_rolegate` | 有 | `.../probe_rolegate_historical.pth` | **探针，非训练模型**：role 门控置回历史值 | 不在论文里 |

## 三条必须记住的结论

1. **`tpscda` 才是论文的方法。** 所有"TP-SCDA vs 基线"的结论都指这一行。
2. **`tokenpair` 不是论文的方法**，是表4 的消融对照（少了层门控）。
   它比 `tpscda` 差是预期内的。
3. **`gateopt` / `b1` / `b2` 等都是本次工作新增的实验，不属于论文，
   且都是负结果或空结果。** 论文原有的方法与结果未被改动。

## 门控配置的差异（关键，容易看错）

| 运行 | `semantic_token_gate_max` | 激活 | 有效强度 |
|---|---:|---|---:|
| `tpscda`（论文） | 0.08 | tanh | **0.0164** |
| `tokenpair` | 1.0 | tanh | 0.1995 |
| `gateopt` | 0.3 | sigmoid | 0.1500 → 0.0632（训练中下漂） |

**`gate_max` 与激活函数是配置、不是权重**——checkpoint 里没有记录。
评测时若用错，等于用不同的注入强度去评同一个模型。
