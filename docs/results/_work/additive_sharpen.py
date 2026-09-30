"""两轴注意力干预 —— 运行时 monkey-patch，**仓库文件一行不改**。

## 动机

本项目此前的"乘法替换 + 行归一化"是**零和**的：角色词（对象/属性）目标占比 t=0.9
意味着同一 patch 上其余 token 被压到 0.1（7 倍压制）——背景与空间上下文被压扁，
计数类 62% 退化为噪声。

文献证据表明**代价不是注意力干预的固有属性**：
  * Divide & Bind：TV 损失把各主体注意力图在**空间上铺开**，计数 +3.58%
  * Attend-and-Excite：最大化各主体注意力峰值，计数 +6.42%
两者都同时提升颜色与计数 —— 因为它们在**空间轴**操作，而我们在 **token 轴**做零和再分配。

## 本模块提供两条轴（可独立开关）

**token 轴（加性抬升）**：只抬角色词，其他 token 逐位不动
```
w' = w + (rise / n_role) · 1_C        rise ∈ [0, δ]，行和 = 1 + rise ≤ 1+δ
```
与乘法版的区别：**不减**，因此不会压扁背景。（等价于原先的"路线 1"）

**空间轴（对象竞争锐化）**：对每个 patch，在对象 token 之间做幂次锐化
```
A_o ← A_o^γ / Σ_o' A_o'^γ             保持该 patch 的对象总质量不变
```
效果：每个 patch 更明确地归属某一个对象 → 各对象的注意力图在空间上互相分离
→ 对象可数。这是 D&B 的 TV 损失想达到的 "divide" 效果，但**不需要每步梯度优化**。

## 用法

    import additive_sharpen as AS
    AS.set_role_state(role_idx=[...], obj_idx=[...], delta=0.4, gamma=2.0)
    AS.apply(model)
    ... 生成 ...
    AS.revert()
"""
from __future__ import annotations

import torch

from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention, cross_attn_key_padding

_STATE = {
    'enable': False,
    'role_idx': [],      # 角色 token（对象 + 属性）
    'obj_idx': [],       # 对象 token（锐化只作用于这些）
    'delta': 0.0,        # 加性抬升的整行增量上限（0 = 关闭）
    'gamma': 1.0,        # 对象竞争锐化指数（1.0 = 关闭）
    'capture_layers': None,
}

_ORIG = None
_IDX = {}


def set_role_state(role_idx, obj_idx, delta=0.0, gamma=1.0):
    _STATE['role_idx'] = list(role_idx)
    _STATE['obj_idx'] = list(obj_idx)
    _STATE['delta'] = float(delta)
    _STATE['gamma'] = float(gamma)


def _patched_forward(self, x, cond, mask=None):
    cfg = _STATE
    if not cfg['enable'] or (cfg['delta'] <= 0 and cfg['gamma'] <= 1.0):
        return _ORIG(self, x, cond, mask)

    B, N, C = x.shape
    q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
    k, v = kv.unbind(2)
    k, v = k.transpose(1, 2), v.transpose(1, 2)
    logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
    if mask is not None:
        kp = cross_attn_key_padding(mask, k.shape[-2], k.device)
        logits = logits.masked_fill(~kp, torch.finfo(logits.dtype).min)

    L = k.shape[-2]
    dev, dt = logits.device, logits.dtype

    # ── token 轴：加性抬升（作用在 logit 上，等价于 softmax 后按比例抬升）──
    if cfg['delta'] > 0 and cfg['role_idx']:
        r = torch.as_tensor(cfg['role_idx'], device=dev, dtype=torch.long)
        mask_role = torch.zeros(L, device=dev, dtype=dt)
        mask_role[r] = 1.0
        n_role = float(len(cfg['role_idx']))
        logits = logits + (cfg['delta'] / n_role) * mask_role.view(1, 1, 1, L)

    attn = torch.softmax(logits, dim=-1)                      # [B,H,N,L]

    # ── 空间轴：对象竞争锐化（作用在 softmax 后的权重上）────────────────
    if cfg['gamma'] > 1.0 and cfg['obj_idx']:
        o = torch.as_tensor(cfg['obj_idx'], device=dev, dtype=torch.long)
        A = attn[:, :, :, o]                                  # [B,H,N,|O|]
        tot = A.sum(-1, keepdim=True)
        Ap = A.clamp_min(1e-9).pow(cfg['gamma'])
        A = Ap / Ap.sum(-1, keepdim=True).clamp_min(1e-9) * tot
        attn = attn.clone()
        attn[:, :, :, o] = A

    out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(out))


def apply(model=None, verbose=True):
    global _ORIG
    if _ORIG is None:
        _ORIG = MultiHeadCrossAttention.forward
    MultiHeadCrossAttention.forward = _patched_forward
    _STATE['enable'] = True
    if verbose:
        print(f'[两轴干预] 已挂载：δ={_STATE["delta"]} γ={_STATE["gamma"]} '
              f'角色 token={len(_STATE["role_idx"])} 对象 token={len(_STATE["obj_idx"])}',
              flush=True)


def revert(model=None):
    global _ORIG
    if _ORIG is not None:
        MultiHeadCrossAttention.forward = _ORIG
    _STATE['enable'] = False
