"""DreamRenderer 在 PixArt-α 上的适配实现 —— **运行时 monkey-patch，仓库文件一行不改**。

原方法：Zhou et al., "DreamRenderer: Taming Multi-Instance Attribute Control in
Large-Scale Text-to-Image Models"（ICCV 2025）。官方构建在 **FLUX** 上。

## 迁移分析：哪些能迁移、哪些不能

原方法有两个创新：

**(1) Bridge Image Tokens —— ❌ 无法迁移。**
它依赖 FLUX 的 **Joint Attention**：文本 token 与图像 token **拼接成一条序列**做注意力，
于是可以在序列里插入"桥接图像 token"让 T5 文本绑定到正确实例。
PixArt-α 是**分离**的图像自注意力 + 图像→文本交叉注意力，**没有拼接序列这个宿主结构**。

**(2) Hard/Soft Attribute Binding —— ✅ 已适配。**
```
原版（Joint Attention 掩码）：
    M_hard^i = 1  若 q ∈ T_i 且 k ∈ T_i ∪ I_i，否则 0
适配到 PixArt（交叉注意力掩码）：
    patch p 归属实例 i 时，只允许关注「实例 i 的文本 token + 全局 token」
    （阻止跨实例属性泄漏 —— 这正是原方法的核心目的）
```
原论文强调 **hard binding 只用在中间层**（输入/输出层负责全局信息），本实现沿用该结论。

## 实例区域从哪来（不需要外部检测器 / LLM）

原方法用**用户提供的边界框或掩码**。本实现改为**从模型自身的交叉注意力导出**：

    第 1 遍：算未掩码的注意力 → 每个 patch 在对象 token 上的响应
    归属：  owner(p) = argmax_o  A[p, o]        （每个 patch 归属响应最强的对象）
    第 2 遍：按 owner 施加硬掩码，再算最终输出

这与本项目"自导出区域"的思路一致，且**每步都重新估计**（区域随去噪演化）。

## 用法

    import dreamrenderer
    dreamrenderer.set_prompt_state(obj_token_idx=[...], attr_of={obj: [attrs]},
                                   global_token_idx=[...])
    dreamrenderer.apply(model, mid_only=True)
    ... 生成 ...
    dreamrenderer.revert(model)

**不 import 本模块时，主代码行为与原来完全一致。**
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention, cross_attn_key_padding

# 当次 prompt 的解析结果（由生成脚本在采样前设置；batch=1 时用模块级状态是安全的）
_STATE = {
    'enable': False,
    'obj_idx': [],        # 对象 token 下标（T5 子词级）
    'attr_of': {},        # 对象 token -> 该对象的属性 token 列表
    'global_idx': [],     # 全局/场景 token（背景、关系、冠词等），始终允许关注
    'mid_lo': 0,          # 应用 hard binding 的中间层区间
    'mid_hi': 10 ** 9,
}

_ORIG_FORWARD = None
_BLOCK_INDEX = {}          # module id -> block 下标


def set_prompt_state(obj_idx, attr_of, global_idx):
    """设置当前 prompt 的句法解析结果。"""
    _STATE['obj_idx'] = list(obj_idx)
    _STATE['attr_of'] = {int(k): list(v) for k, v in attr_of.items()}
    _STATE['global_idx'] = list(global_idx)


def _instance_bias(cfg, num_patches, num_tokens, dtype, device, attn_probs):
    """按"每个 patch 的归属对象"构造掩码偏置：允许=0，禁止=-inf。

    ``attn_probs`` : [B, N, L] 未掩码的注意力（用于估计归属）。
    """
    B, N, L = attn_probs.shape
    obj = cfg['obj_idx']
    if not obj:
        return None
    obj_t = torch.as_tensor(obj, device=device, dtype=torch.long)
    # 每个 patch 归属响应最强的对象 token
    sub = attn_probs[:, :, obj_t]                       # [B, N, |O|]
    owner = sub.argmax(dim=-1)                          # [B, N]

    allow = torch.zeros(B, N, L, dtype=torch.bool, device=device)
    if cfg['global_idx']:
        g = torch.as_tensor(cfg['global_idx'], device=device, dtype=torch.long)
        allow[:, :, g] = True
    for k, o in enumerate(obj):
        m = (owner == k)                                # [B, N]
        allow[:, :, o] |= m
        for a in cfg['attr_of'].get(o, ()):
            allow[:, :, a] |= m
    # 未归属任何对象的 patch（|O| 为空等）不施加约束
    return allow


def _patched_forward(self, x, cond, mask=None):
    """替换 MultiHeadCrossAttention.forward：在中间层对交叉注意力施加实例硬掩码。"""
    cfg = _STATE
    bi = _BLOCK_INDEX.get(id(self), -1)
    if (not cfg['enable']) or (not cfg['obj_idx']) or \
            not (cfg['mid_lo'] <= bi < cfg['mid_hi']):
        return _ORIG_FORWARD(self, x, cond, mask)

    B, N, C = x.shape
    q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
    k, v = kv.unbind(2)
    k, v = k.transpose(1, 2), v.transpose(1, 2)
    # 显式算权重必须升 fp32 —— 见 MultiHeadCrossAttention.forward 的注释：
    # patch 状态来的 q 范数可达数百，q·k 在 120 个 key 上累加到 1e3 量级，
    # fp16 下溢出成 inf，softmax(inf)=NaN。NaN 会顺着 VAE 解码 survive，
    # 表现为出图整体偏暗发灰。正常路径的融合 SDPA 内核内部就是 fp32，不会踩到。
    q, k, v = q.float(), k.float(), v.float()
    logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
    if mask is not None:
        kp = cross_attn_key_padding(mask, k.shape[-2], k.device)
        logits = logits.masked_fill(~kp, torch.finfo(logits.dtype).min)

    # 第 1 遍：算未掩码归属
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1).mean(dim=1)              # [B,N,L]
        allow = _instance_bias(cfg, N, k.shape[-2], logits.dtype, logits.device, probs)
    if allow is None:
        return _ORIG_FORWARD(self, x, cond, mask)
    # 第 2 遍：硬掩码后重算
    logits = logits.masked_fill(~allow[:, None, :, :], torch.finfo(logits.dtype).min)
    attn = torch.softmax(logits, dim=-1)
    out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(out.to(x.dtype)))


def apply(model, mid_lo_frac=0.25, mid_hi_frac=0.75, verbose=True):
    """挂上补丁。hard binding 只用于中间层（沿用原论文结论）。"""
    global _ORIG_FORWARD
    if _ORIG_FORWARD is None:
        _ORIG_FORWARD = MultiHeadCrossAttention.forward
    MultiHeadCrossAttention.forward = _patched_forward
    _BLOCK_INDEX.clear()
    n = len(model.blocks)
    _STATE['mid_lo'] = int(n * mid_lo_frac)
    _STATE['mid_hi'] = int(n * mid_hi_frac) + 1
    for i, blk in enumerate(model.blocks):
        _BLOCK_INDEX[id(blk.cross_attn)] = i
    _STATE['enable'] = True
    if verbose:
        print(f'[DreamRenderer] 已挂载：hard binding 用于第 {_STATE["mid_lo"]}~'
              f'{_STATE["mid_hi"]-1} 层（共 {n} 层）；Bridge Image Tokens 未迁移'
              f'（PixArt 无 Joint Attention）', flush=True)
    return _STATE['mid_lo'], _STATE['mid_hi']


def revert(model=None):
    """摘掉补丁，恢复原始实现。"""
    global _ORIG_FORWARD
    if _ORIG_FORWARD is not None:
        MultiHeadCrossAttention.forward = _ORIG_FORWARD
    _STATE['enable'] = False
    if model is not None:
        print('[DreamRenderer] 已卸下补丁', flush=True)
