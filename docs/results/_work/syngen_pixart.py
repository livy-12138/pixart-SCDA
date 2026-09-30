"""SynGen 在 PixArt-α（T5 + DiT）上的迁移实现。

原方法：Rassin et al., "Linguistic Binding in Diffusion Models: Enhancing
Attribute Correspondence through Attention Map Alignment"（NeurIPS 2023, oral）。
官方实现：https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models

## 迁移要点（与原实现的差异，全部记录在案）

| 方面 | 原实现（SD 1.4/1.5） | 本实现（PixArt-α） |
|---|---|---|
| 主干 | UNet，cross-attn 在 attention_res=16 层 | DiT，`MultiHeadCrossAttention`，全部 block |
| 文本编码 | CLIP tokenizer（77 token） | **T5**（120 token，含 padding） |
| (名词, 修饰词) 抽取 | spaCy 依存分析（amod/nmod/compound…） | **复用本项目的 `build_semantic_edges`**，直接给出 (对象, 属性) 对 |
| 注意力图分辨率 | 16×16 = 256 | **32×32 = 1024**（patch=2，512px） |
| 采样器 | DDIM（官方） | **DDIM**（保持一致，便于与官方对照） |

## 损失（逐字移植自官方 `compute_loss.py`）

    L_pos = Σ_i Σ_{(m,n)∈P(S_i)} symKL(A_m, A_n)      # 让修饰词与其名词重叠
    L_neg = (L_neg_modifier + L_neg_noun) / 2          # 推离无关词
    L     = L_pos + λ_neg · L_neg
    symKL(p,q) = (KL(p‖q) + KL(q‖p)) / 2

对潜变量做梯度步：`z ← z − step_size · ∇_z L`，只在前若干步施加。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence


# --------------------------------------------------------------------------
# 损失（移植自官方 compute_loss.py）
# --------------------------------------------------------------------------
def _symmetric_kl(attn_map1, attn_map2):
    """对称 KL 散度。两张图都先归一化成空间概率分布。

    ⚠ 官方实现对传进来的图直接构造 Categorical(probs=...)，要求 probs 和为 1。
    本实现显式归一化，避免因上游口径不同而静默产出错误数值。
    """
    p = attn_map1.reshape(-1).float()
    q = attn_map2.reshape(-1).float()
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    p = p / p.sum()
    q = q / q.sum()
    return (kl_divergence(Categorical(probs=p), Categorical(probs=q)) +
            kl_divergence(Categorical(probs=q), Categorical(probs=p))) / 2.0


def calculate_positive_loss(attention_maps, modifier_idx, noun_idx):
    """正损失：让修饰词与其名词的注意力图尽量重叠（最小化对称 KL）。"""
    return _symmetric_kl(attention_maps[modifier_idx], attention_maps[noun_idx])


def calculate_outside_loss(attention_maps, src_idx, outside_indices):
    """负损失：把修饰词/名词的注意力图推离句法上无关的词。

    返回 (逐无关词的最大对称 KL 列表, 计数)，与原实现同构。
    """
    negative_loss = []
    for outside_idx in outside_indices:
        negative_loss.append(_symmetric_kl(attention_maps[src_idx],
                                           attention_maps[outside_idx]))
    return negative_loss, len(negative_loss)


def syngen_loss(attention_maps, pairs, all_indices, lam_neg=1.0):
    """总损失。

    pairs      : [(修饰词 token 下标, 名词 token 下标), ...]（来自解析器）
    all_indices: 句法实体集合内的全部 token（用于界定"无关词"）
    """
    if not pairs:
        return None
    # 抓取到的图是 [patch, token]（CrossAttnCapture.averaged 的输出），而官方
    # compute_loss.py 里的 attention_maps 是 [token, patch]。官方那套 attention_maps[idx]
    # 取的是"某个 token 的空间图"；不转置的话这里取到的是"某个 patch 在 120 个 token
    # 上的分布"，语义完全不对（形状也从 (1024,) 变成 (120,)）。
    attention_maps = attention_maps.transpose(0, 1)
    inside = set()
    for m, n in pairs:
        inside.add(m)
        inside.add(n)
    outside = [t for t in all_indices if t not in inside]

    pos = sum(calculate_positive_loss(attention_maps, m, n) for m, n in pairs) / len(pairs)

    neg_terms = []
    for m, n in pairs:
        for idx in (m, n):
            if not outside:
                continue
            losses, cnt = calculate_outside_loss(attention_maps, idx, outside)
            if losses:
                neg_terms.append(sum(losses) / len(losses))
    neg = sum(neg_terms) / len(neg_terms) if neg_terms else pos.new_zeros(())

    return pos + lam_neg * neg


# --------------------------------------------------------------------------
# 注意力抓取（可微）
# --------------------------------------------------------------------------
class CrossAttnCapture:
    """在 PixArt 的 `MultiHeadCrossAttention` 上挂前向钩子，重算可微注意力图。

    与 `mechanism_viz.py` 同一套注册方式，但**不加 .detach()** —— SynGen 需要
    从损失反传到潜变量。
    """

    def __init__(self, model, layers=None):
        self.model = model
        self.layers = layers or list(range(len(model.blocks)))
        self.maps = {}
        self.handles = []

    def __enter__(self):
        for li in self.layers:
            ca = self.model.blocks[li].cross_attn

            def make_hook(li_):
                def hook(module, args):
                    # args = (x, cond, mask)；先清掉本层上一轮存的图，避免引用泄漏
                    self.maps.pop(li_, None)
                    x, cond = args[0], args[1]
                    mk = args[2] if len(args) > 2 else None
                    B, N, _ = x.shape
                    q = module.q_linear(x).view(B, N, module.num_heads, module.head_dim).transpose(1, 2)
                    kv = module.kv_linear(cond).view(B, -1, 2, module.num_heads, module.head_dim)
                    k, _ = kv.unbind(2)
                    k = k.transpose(1, 2)
                    # 必须升 fp32 再 matmul。MultiHeadCrossAttention.forward 的注释已写明：
                    # patch 状态来的 q 范数可达数百，q·k 在 120 个 key 上累加到 1e3 量级，
                    # fp16 下直接溢出成 inf，softmax(inf) 得到 NaN（NaN 会顺着 VAE 解码
                    # 一路survive，最终被 save_image 映射成黑图）。正常路径用的融合 SDPA
                    # 内核内部本来就是 fp32 算的，所以只有"自己显式算权重"的地方会踩到。
                    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (module.head_dim ** -0.5)
                    if mk is not None and not isinstance(mk, (list, tuple)):
                        kp = mk.bool()[:, None, None, :]
                        logits = logits.masked_fill(~kp, torch.finfo(logits.dtype).min)
                    # 保持可微；按 head 平均得到 [B, N, L]
                    self.maps[li_] = torch.softmax(logits, dim=-1).mean(dim=1)
                return hook

            self.handles.append(ca.register_forward_pre_hook(make_hook(li)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.maps = {}
        return False

    def averaged(self):
        """把各层的图按 patch 维平均，得到 [B, P, L]（与官方"多分辨率聚合"等价）。"""
        if not self.maps:
            return None
        return torch.stack(list(self.maps.values()), dim=0).mean(dim=0)


# --------------------------------------------------------------------------
# 采样与引导
# --------------------------------------------------------------------------
# 关键设计：**不改采样器**。
#
# `DPM_Solver.sample()` 支持 `correcting_xt_fn(x, t, step)` —— 每个采样步结束时
# 会被调用一次，可以修改 xt。官方工厂函数 `diffusion/dpm_solver.py: DPMS()`
# 没有传这个参数，所以这里复制它并补上。
#
# 好处：**协议完全不变** —— 仍是 DPM-Solver 20 步、CFG 4.0，与其他所有方法一致，
# 不需要像 SynGen 官方那样换成 DDIM（那会造成采样器口径差异）。
def guided_DPMS(model, condition, uncondition, cfg_scale, model_kwargs,
                corrector, noise_schedule="linear", diffusion_steps=1000,
                model_type='noise', guidance_type='classifier-free'):
    """与 `diffusion.dpm_solver.DPMS` 等价，但额外挂上 `correcting_xt_fn`。"""
    from diffusion.model import gaussian_diffusion as gd
    from diffusion.model.dpm_solver import model_wrapper, DPM_Solver, NoiseScheduleVP

    betas = torch.tensor(gd.get_named_beta_schedule(noise_schedule, diffusion_steps))
    ns = NoiseScheduleVP(schedule='discrete', betas=betas)
    # corrector 直接调 model，绕过了 noise_pred_fn 里的时间换算；把**同一份**换算
    # 注入给它。公式不在两处各写一遍，避免将来改 schedule 时漂移
    # （get_model_input_time 是 model_wrapper 内部的闭包，import 不到）。
    if hasattr(corrector, 'set_input_time_fn'):
        def _to_input_time(t):
            if ns.schedule == 'discrete':
                return (t - 1. / ns.total_N) * 1000.
            return t
        corrector.set_input_time_fn(_to_input_time)
    model_fn = model_wrapper(
        model, ns, model_type=model_type, model_kwargs=model_kwargs,
        guidance_type=guidance_type, condition=condition,
        unconditional_condition=uncondition, guidance_scale=cfg_scale)
    return DPM_Solver(model_fn, ns, algorithm_type="dpmsolver++",
                      correcting_xt_fn=corrector)


class SynGenCorrector:
    """在采样步内对潜变量做 SynGen 梯度步。

    每个被引导的步：
      1. 以 grad 打开的方式跑一次模型（条件路 + 无条件路拼批）
      2. 用钩子抓到**可微**的交叉注意力图 [B, P, L]
      3. 算 L_pos + λ·L_neg
      4. 反传得到 ∇_z L，做一步 z ← z − step_size · ∇_z L
    """

    def __init__(self, model_fn, condition, uncondition, model_kwargs,
                 pairs_by_batch, num_guided_steps=5, step_size=20.0,
                 lam_neg=1.0, capture_layers=None, verbose=False):
        self.model_fn = model_fn            # 通常是 model.forward_with_dpmsolver
        self.condition = condition
        self.uncondition = uncondition
        self.model_kwargs = dict(model_kwargs or {})
        self.pairs_by_batch = pairs_by_batch
        self.K = int(num_guided_steps)
        self.step_size = float(step_size)
        self.lam_neg = float(lam_neg)
        self.layers = capture_layers
        self.verbose = verbose
        self.calls = 0
        self._to_input_time = None          # 由 guided_DPMS 注入，见 set_input_time_fn

    def set_input_time_fn(self, fn):
        """由 `guided_DPMS` 注入：把 solver 的连续时间换成**模型输入时间**。

        正常路径走 `dpm_solver.noise_pred_fn`，它内部做了
        `t_input = get_model_input_time(t_continuous)`；本 corrector 为拿到可微的
        注意力图而直接调 `model.forward_with_dpmsolver`，必须自己补上这一步。
        漏掉的话模型会在错误的噪声水平上被求值（第 0 步会把 t_continuous≈1.0 当成
        timestep=1，而正确值是 (1.0-1/1000)*1000≈999），抓到的注意力图与真实生成步
        对不上，梯度也作用在错误的步上。
        """
        self._to_input_time = fn

    def __call__(self, x, t, step):
        # 只在前 K 步引导；其余步原样返回（真正的"不干预"）
        if step >= self.K or not any(self.pairs_by_batch):
            return x
        B = x.shape[0]
        # t 可能是标量张量，展成批
        t_b = t.reshape(-1)
        if t_b.numel() == 1:
            t_b = t_b.expand(B)
        # 送进模型前必须换成"模型输入时间"。正常路径里 dpm_solver.noise_pred_fn 会做
        # t_input = get_model_input_time(t_continuous)，而这里直接调
        # model.forward_with_dpmsolver，把 solver 的连续时间当成了模型时间步。
        # 不换算的话模型在错误的噪声水平上被求值（第 0 步 1.0 会被当成 timestep=1，
        # 正确值是 (1.0-1/1000)*1000≈999），抓到的注意力图与真实生成步对不上，
        # 梯度也就作用在错误的步上。换算函数由 guided_DPMS 注入，与 solver 同源。
        if self._to_input_time is not None:
            t_b = self._to_input_time(t_b)

        with torch.enable_grad():
            xg = x.clone().detach().requires_grad_(True)
            x_in = torch.cat([xg, xg], dim=0)
            # 批序必须与 dpm_solver.model_wrapper 的 c_in = cat([uncond, cond]) 一致。
            # PixArt.forward 对 data_info['semantic_token_masks'] 的 factor==2 展开
            # 写死了「前一半零掩码、后一半角色掩码」；把 cond 放前一半会让语义支路
            # 整条落空（role 全零 → token_delta 恒零），抓到的注意力图就不是真实
            # 生成路径上的那一张，SynGen 的梯度会指向一个不该被优化的方向。
            y_in = torch.cat([self.uncondition, self.condition], dim=0)
            # 时间步也要跟着翻倍。漏掉这一行时 x 是 2B、t 是 B，
            # 块内 `t.reshape(B, 6, -1)` 会把 6*hidden 摊成 6*(hidden/2)，
            # 于是 scale_shift_table(1152) 与 t(576) 对不上而崩在第一个 block。
            t_in = torch.cat([t_b, t_b], dim=0)
            with CrossAttnCapture(self.model_fn.__self__, self.layers) as cap:
                self.model_fn(x_in, t_in, y_in, **self.model_kwargs)
                maps = cap.averaged()                 # [2B, P, L]
                maps = maps[B:]                       # 取条件路（后一半）
                loss = None
                for b, pairs in enumerate(self.pairs_by_batch):
                    if not pairs:
                        continue
                    idx = sorted({q for pr in pairs for q in pr})
                    lb = syngen_loss(maps[b], pairs, idx, self.lam_neg)
                    if lb is not None:
                        loss = lb if loss is None else loss + lb
            if loss is not None and loss.requires_grad:
                grad = torch.autograd.grad(loss, xg, retain_graph=False)[0]
                out = (xg - self.step_size * grad).detach()
                if self.verbose:
                    print(f'  [syngen] step={step} loss={float(loss):.5f} '
                          f'|grad|={float(grad.norm()):.4g}', flush=True)
                self.calls += 1
                return out
            return xg.detach()
