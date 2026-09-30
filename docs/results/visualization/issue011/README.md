# ISSUE-011 掩码开关对照图

同一提示词、同一确定性 seed，仅切换 `results/_work/mask_fix.py` 的运行时补丁：
- BEFORE：当前仓库 `MultiHeadCrossAttention` 的 SDPA 回退分支（mask 为 list 时 key_padding=None）
- AFTER：按上游 PixArt-alpha 语义恢复 key-padding 掩码

生成脚本：`results/_work/gen_compbench.py --fix-mask`；对照脚本见 `results/_work/`。
仓库文件未被修改，补丁仅在运行时生效。
