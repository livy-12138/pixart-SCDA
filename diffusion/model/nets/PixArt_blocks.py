# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import os
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, Attention as Attention_
from einops import rearrange, repeat
try:
    import xformers.ops
    _HAS_XFORMERS = True
except Exception:
    xformers = None
    _HAS_XFORMERS = False

from diffusion.model.utils import add_decomposed_rel_pos

# Set PAIR_DEBUG=1 to print, per cross-attention call, how much attention mass
# the parsed object/attribute tokens actually carry.  Used to diagnose why the
# pair-replacement rewrite can execute and still leave the output bit-identical.
_PAIR_DEBUG = os.environ.get('PAIR_DEBUG') == '1'

# Set PAIR_CAPTURE=1 (or PAIR_BOX['on']=True from a capture script) to record
# the cross-attention weights the *pair-replacement* path itself computes.
# That path never calls scaled_dot_product_attention -- it builds `w` by hand
# and does torch.matmul(w, vf) -- so wrapping SDPA (what capture_mech.py does
# for the normal path) sees nothing at all when pair_replace is on.  Off by
# default; recording costs one .cpu() copy per cross-attention per call.
_PAIR_BOX = {'on': os.environ.get('PAIR_CAPTURE') == '1', 'maps': []}


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def cross_attn_key_padding(mask, key_len, device):
    """Boolean key-padding mask for scaled_dot_product_attention (True = keep).

    ``PixArt.forward`` computes ``y_lens = mask.sum(dim=1).tolist()`` and hands
    that python list of per-sample valid-token counts down to the blocks, so the
    ``mask`` reaching a cross-attention layer is normally a *list of lengths*,
    not a tensor.  Upstream PixArt-alpha expresses this with
    ``xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)``; the
    SDPA equivalent is an explicit ``(B, 1, 1, L)`` padding mask.

    Treating the list case as "no mask" would make every image patch attend to
    the T5 padding tokens, which degrades short prompts severely.
    """
    if mask is None:
        return None
    if isinstance(mask, (list, tuple)):
        lengths = torch.as_tensor([int(m) for m in mask], device=device).reshape(-1, 1)
        index = torch.arange(key_len, device=device).reshape(1, -1)
        return (index < lengths)[:, None, None, :]
    key_padding = mask.bool()
    if key_padding.dim() == 2:
        key_padding = key_padding[:, None, None, :]
    return key_padding


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., **block_kwargs):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.pair_replace = False   # PROBE-ONLY, default off
        self.pair_roles = None
        self.pair_edge_mat = None
        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        # NOTE ON THE FAST PATH: xformers' BlockDiagonalMask would keep the
        # fused kernel AND mask padding, but it requires the total key count to
        # equal sum(kv_seqlen).  PixArt hands the blocks padded keys (B x 120),
        # so that identity does not hold and the result is garbage -- verified
        # experimentally.  Whenever a padding mask is present we therefore take
        # the SDPA branch below, which applies the mask explicitly.  The cost is
        # roughly 1.4x end-to-end throughput; correctness wins.
        if _HAS_XFORMERS and mask is None:
            q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
            kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(2)
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p)
        else:
            q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
            kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            key_padding = cross_attn_key_padding(mask, k.shape[-2], k.device)
            # PROBE-ONLY (default off): replace the DiT's own cross-attention
            # weights at the parsed object->attribute pairs with the OBJECT's
            # weight, so an attribute token carries the same weight as the noun
            # it is bound to.  Operates on the weights the DiT already computes,
            # so there is no separate q/k and no additive bias to drown in a
            # huge logit scale.
            roles = getattr(self, 'pair_roles', None)
            if getattr(self, 'pair_replace', False) and roles is not None:
                # fp32 throughout: q comes from patch states with norm ~668, so
                # q.k accumulates to ~2.8e3 over 120 keys.  In fp16 that
                # overflows to inf and softmax(inf) is NaN -- measured: the
                # training loss is nan from step 1 and inference produces pure
                # black images (NaN survives VAE decode and save_image maps it
                # to 0).  The fused SDPA kernel the normal path uses does this
                # arithmetic in fp32 internally, so the original code never hit
                # it; computing the weights explicitly means upcasting here.
                if _PAIR_DEBUG:
                    print(f'[dbg_in] x_nan={torch.isnan(x).any().item()} cond_nan={torch.isnan(cond).any().item()} '
                          f'q_nan={torch.isnan(q).any().item()} v_nan={torch.isnan(v).any().item()} '
                          f'x_max={float(x.abs().max()):.3g} q_max={float(q.abs().max()):.3g} '
                          f'v_max={float(v.abs().max()):.3g}', flush=True)
                # `autocast(enabled=False)` is load-bearing: under the training
                # loop's autocast, torch.matmul would cast these fp32 operands
                # straight back down to fp16, and q.k over head_dim then
                # overflows fp16 (measured: q_max 533, k_max 11.7 -> logits inf
                # -> softmax NaN).  .float() alone does NOT survive autocast.
                with torch.autocast(device_type=q.device.type, enabled=False):
                    qf, kf, vf = q.float(), k.float(), v.float()
                    logits = torch.matmul(qf, kf.transpose(-2, -1)) * (self.head_dim ** -0.5)
                    if _PAIR_DEBUG:
                        print(f'[dbg_logits] k_fp16_max={float(k.abs().max()):.4g} k_isinf={bool(torch.isinf(k).any())} '
                              f'logits_max={float(logits.max()):.4g} logits_isinf={bool(torch.isinf(logits).any())} '
                              f'logits_isnan={bool(torch.isnan(logits).any())}', flush=True)
                    if key_padding is not None:
                        logits = logits.masked_fill(~key_padding, torch.finfo(logits.dtype).min)
                    w = torch.softmax(logits, dim=-1)              # (B,H,P,L) fp32
                    em = getattr(self, 'pair_edge_mat', None)
                    if em is not None:
                        am = roles.to(w.dtype)[:, 1]
                        if _PAIR_DEBUG:
                            _obj = (roles.to(w.dtype)[:, 0] > 0)
                            _att = (am > 0)
                            print(f'[pairdbg] w_max={float(w.max()):.4g} '
                                  f'w@obj_max={float(w[_obj[:, None, None, :].expand_as(w)].max()) if _obj.any() else 0:.3g} '
                                  f'w@attr_max={float(w[_att[:, None, None, :].expand_as(w)].max()) if _att.any() else 0:.3g} '
                                  f'w@attr_mean={float(w[_att[:, None, None, :].expand_as(w)].mean()) if _att.any() else 0:.3g} '
                                  f'em_max={float(em.max()):.3g}', flush=True)
                        # (w @ em)[..., a] = sum_o w[..., o] * edge[o, a] -- i.e. the
                        # attribute takes the noun's OWN (unnormalised) DiT weight.
                        # em is (B, L, L) and w is (B, H, P, L): without the head
                        # axis the batch dim of em would align against H (silently
                        # wrong whenever B == H, a hard error otherwise), so it is
                        # inserted explicitly to broadcast across heads.
                        pair_weight = torch.matmul(w, em.unsqueeze(1).to(w.dtype))
                        if _PAIR_DEBUG:
                            # Natural (pre-edit) relation between an attribute and
                            # the noun(s) it is bound to: how often does a patch
                            # already read the ATTRIBUTE more strongly than those
                            # nouns, and how much lift would a raise have to add
                            # where it does not?
                            _n = em.sum(dim=1).clamp_min(1).view(-1, 1, 1, em.shape[-1])
                            _mean_noun = pair_weight / _n
                            _mask = ((am > 0) & (em.sum(dim=1) > 0))
                            _mask = _mask[:, None, None, :].expand_as(w)
                            # Share of each attention row held by the parsed
                            # roles together: if noun + attribute only carry a
                            # few percent, the mechanism has almost no mass to
                            # work with, whatever the edit rule is.
                            _objm = (roles.to(w.dtype)[:, 0] > 0)[:, None, None, :].expand_as(w)
                            _attm = (am > 0)[:, None, None, :].expand_as(w)
                            _pairm = (_objm.bool() | _attm.bool())
                            _s_obj = (w * _objm).sum(-1)
                            _s_att = (w * _attm).sum(-1)
                            _s_pair = (w * _pairm.to(w.dtype)).sum(-1)
                            _top = w.max(-1).values
                            print(f'[dbg_share] 名词和={float(_s_obj.mean()):.4f} '
                                  f'形容词和={float(_s_att.mean()):.4f} '
                                  f'两者并集={float(_s_pair.mean()):.4f} '
                                  f'最大单token={float(_top.mean()):.4f} '
                                  f'整行和={float(w.sum(-1).mean()):.4f}', flush=True)
                            # WHO owns the row?  Identify the argmax token and say
                            # whether the parse marked it at all -- if the winner
                            # is a noun sub-token the mask missed, the role
                            # matching is broken; if it is unrelated, the parsed
                            # structure and the model's attention are disjoint.
                            _ti = w.argmax(-1)                        # (B,H,P)
                            _rd = roles.to(w.dtype)                   # (B,3,L)
                            def _mark(ch):
                                # role value at each row's argmax position
                                return (_rd[:, ch].unsqueeze(1).unsqueeze(1)
                                        .expand(w.shape[0], w.shape[1], w.shape[2], -1)
                                        .gather(-1, _ti.unsqueeze(-1)).squeeze(-1) > 0)
                            _is_obj, _is_att, _is_rel = _mark(0), _mark(1), _mark(2)
                            _n = float(_ti.numel())
                            _mode = int(torch.mode(_ti.flatten()).values)
                            _tf = _ti.float()
                            print(f'[dbg_ti] idx_mean={float(_tf.mean()):.1f} min={int(_ti.min())} '
                                  f'max={int(_ti.max())} frac0={float((_ti == 0).float().mean()):.3f} '
                                  f'frac<10={float((_ti < 10).float().mean()):.3f}', flush=True)
                            print(f'[dbg_winner] obj={float(_is_obj.sum())/_n:.3f} '
                                  f'att={float(_is_att.sum())/_n:.3f} '
                                  f'rel={float(_is_rel.sum())/_n:.3f} '
                                  f'other={float((~(_is_obj|_is_att|_is_rel)).sum())/_n:.3f} '
                                  f'最常见下标={_mode} '
                                  f'行最大权重均值={float(w.max(-1).values.mean()):.4f}', flush=True)
                            _sel, _nat = w[_mask], _mean_noun[_mask]
                            if _sel.numel():
                                print(f'[dbg_natural] attr>noun 占比='
                                      f'{float((_sel > _nat).float().mean()):.3f} '
                                      f'平均需抬升={float(torch.clamp(_nat - _sel, min=0).mean()):.4f} '
                                      f'attr_mean={float(_sel.mean()):.4f} '
                                      f'noun_mean={float(_nat.mean()):.4f}', flush=True)
                        mode = getattr(self, 'pair_replace_mode', 'replace')
                        if mode == 'reweight':
                            # Global re-weighting TOWARDS the parsed elements.
                            # Every role-marked token (object / attribute /
                            # relation -- the "图元") is scaled up together and
                            # the rest of the row is diluted, until the elements
                            # hold `content_target` of the mass.  Because the
                            # SAME factor multiplies all of them, the ratios
                            # inside the group are preserved -- noun:attribute
                            # does not change -- which is what the per-pair
                            # modes got wrong: they lifted the attribute at the
                            # noun's expense and the object disappeared from the
                            # picture.  The row still sums to 1.
                            content = (roles.to(w.dtype).sum(1) > 0)[:, None, None, :].to(w.dtype)
                            # LCAR：把「非外观属性」的词（数词/方位词/关系虚词/冠词）
                            # 排除在**抬升目标**之外 —— 抬这些词会把计数、深度结构、
                            # 关系语义压垮。
                            #
                            # ⚠ 关键：只能缩小「抬谁」（content_lift），**不能**缩小
                            # 「抬多少」的分母预算（content）。若把排除词从 content
                            # 里一并剔除，s 会大幅变小，alpha=(t/(1-t))·((1-s)/s)
                            # 随之爆炸（实测把 "six airplanes" 的 content 缩到只剩
                            # airplanes 后 alpha≈171），注意力被病态压到单个 token 上，
                            # 生成直接崩成噪声。
                            _exc = getattr(self, 'pair_replace_exclude', None)
                            if _exc is not None:
                                content_lift = content * (1.0 - _exc.to(w.dtype)[:, None, None, :])
                            else:
                                content_lift = content
                            s = (w * content).sum(-1, keepdim=True)
                            # content_target 支持两种形态：
                            #   标量（现有 rw25/50/80）—— 全图统一；
                            #   [B, P] 空间场（LCAR 渐变替换式）—— 锚点处最高、
                            #   沿半径线性下降、半径外为 0。
                            # t=0 时 alpha=(0/1)·((1-s)/s)=0 被 clamp(min=1) 钳成 1，
                            # w 逐位不变 → 半径外**不做任何替换**。
                            t_map = getattr(self, 'pair_replace_content_target_map', None)
                            if t_map is not None:
                                t = t_map.to(s.dtype)[:, None, :, None]        # (B,1,P,1)
                                t = t.clamp(0.0, 0.99)
                            else:
                                _t = float(getattr(self, 'content_target', 0.5))
                                t = s.new_full((), min(max(_t, 0.01), 0.99))
                            _additive = bool(getattr(self, 'pair_replace_additive', False))
                            if _additive:
                                # ── 加性提升（不重归一化）──────────────────
                                # 原有做法是"乘法抬升 + 行归一化"，抬多少就必须从
                                # 别处扣多少 —— 角色词涨到 90% 意味着其他词被压到
                                # 10%（7 倍压制），背景与空间上下文被压扁，计数崩坏。
                                #
                                # 这里改成**只加不减**：角色词得到绝对提升，其他
                                # token 逐位不动。整行的增量由 δ_max 直接控制
                                # （除以角色词个数，使整行只涨 rise），避免出现
                                # "整行和 7 倍" 那种幅值爆炸。
                                _dmax = float(getattr(self, 'pair_replace_delta_max', 0.2))
                                if t_map is not None:
                                    # t_map 是 (B,P)，必须重塑成 (B,1,P,1) —— 直接用它
                                    # 会与 content_lift 的 (B,1,1,L) 在第 3 维对不上
                                    # （1024 vs 120），报 size mismatch。
                                    _tm = t_map.to(s.dtype)[:, None, :, None]
                                    _peak = _tm.max().clamp_min(1e-6)
                                    rise = (_tm / _peak) * _dmax        # (B,1,P,1) ∈ [0,δ]
                                else:
                                    rise = torch.full_like(s, _dmax)   # (B,H,P,1)
                                n_role = content_lift.sum(-1, keepdim=True).clamp_min(1.0)
                                w = w + (rise / n_role) * content_lift
                            else:
                                alpha = (t / (1.0 - t)) * ((1.0 - s) / s.clamp_min(1e-9))
                                alpha = torch.clamp(alpha, min=1.0)     # raise only
                                w = w * (1.0 + (alpha - 1.0) * content_lift)
                                w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
                            if _PAIR_DEBUG:
                                share = float((w * content).sum(-1).mean())
                                print(f'[dbg_out] mode=reweight target={t:.2f} '
                                      f'实际图元占比={share:.4f} 行和={float(w.sum(-1).mean()):.4f} '
                                      f'alpha_mean={float(alpha.mean()):.2f}', flush=True)
                        elif mode == 'outside':
                            # Protect the noun: lift the attribute as usual, but
                            # fund it from the tokens OUTSIDE the pair, taken in
                            # proportion to what each of them holds.  The noun's
                            # absolute weight -- and therefore its share of the
                            # row -- is untouched, so whatever the patch was
                            # reading for the object's shape stays intact; only
                            # unrelated tokens give up mass.  The row still sums
                            # to 1.
                            pair_m = ((em.sum(dim=-1) > 0) | (am > 0))[:, None, None, :]
                            gain = torch.clamp(pair_weight - w, min=0) * (am > 0)[:, None, None, :].to(w.dtype)
                            w_out = w * (~pair_m).to(w.dtype)
                            avail = w_out.sum(-1, keepdim=True)
                            need = gain.sum(-1, keepdim=True)
                            fund = getattr(self, 'pair_replace_fund', 'proportional')
                            if fund == 'tail':
                                # water-filling from the BOTTOM: take the smallest
                                # outsider weights in full, then the next ones,
                                # until the lift is paid for.  Spares the largest
                                # (most informative) unmarked tokens.
                                ws, _ = torch.sort(w_out, dim=-1)
                                cum = ws.cumsum(-1)
                                k = (cum < need).sum(-1, keepdim=True).clamp(max=w_out.shape[-1] - 1)
                                lam = ws.gather(-1, k)
                                pay = torch.minimum(w_out, lam)
                            elif fund == 'uniform':
                                # same share from every outsider that holds mass
                                n_pos = (w_out > 0).sum(-1, keepdim=True).clamp_min(1)
                                pay = torch.minimum(w_out, need / n_pos)
                            else:
                                pay = w_out * (need / avail.clamp_min(1e-9))
                            # never take more than they hold, and make the lift
                            # exactly what was actually collected
                            pay = torch.minimum(pay, w_out)
                            collected = pay.sum(-1, keepdim=True)
                            gain = gain * torch.clamp(collected / need.clamp_min(1e-9), max=1.0)
                            w = w + gain - pay.clamp_min(0)
                            if _PAIR_DEBUG:
                                print(f'[dbg_out] mode=outside rowsum_max={float(w.sum(-1).max()):.4g} '
                                      f'gain_max={float(gain.max()):.4g} w_min={float(w.min()):.4g}', flush=True)
                        elif mode == 'sink':
                            # ONLY the sink is touched: find each row's argmax
                            # (the boundary token our measurement showed owns
                            # ~96% of the mass) and scale just that position,
                            # then renormalise.  Nothing else moves -- unlike
                            # `reweight`, which diluted every other token too and
                            # produced noise, leaving it unclear which effect
                            # was responsible.
                            _fac = float(getattr(self, 'sink_factor', 0.0))
                            _i = w.argmax(-1, keepdim=True)
                            _m = torch.ones_like(w).scatter_(-1, _i, _fac)
                            w = (w * _m)
                            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
                            if _PAIR_DEBUG:
                                print(f'[dbg_out] mode=sink factor={_fac} '
                                      f'rowsum={float(w.sum(-1).mean()):.4f}', flush=True)
                        elif mode == 'gated':
                            # Soft, patch-aware gate.  The lift for an attribute
                            # is proportional to how much THIS patch actually
                            # looks at that attribute's noun, relative to the
                            # other parsed objects:
                            #     pref[o] = w[o] / sum_{o'} w[o']   (over objects)
                            #     target[a] = sum_o pref[o] * w[o] * em[o, a]
                            # so a patch depicting the apple lifts "blue" (its
                            # noun is dominant here) and barely touches "green"
                            # (that noun is nearly absent).  Without this the
                            # edit raises EVERY attribute whose own noun happens
                            # to outrank it -- which is how a patch drawing the
                            # apple ended up reading "green" from the vase pair.
                            # The lift is still paid for by the pair's own nouns,
                            # so the row keeps summing to 1.
                            obj_has_edge = (em.sum(dim=-1) > 0)[:, None, None, :].to(w.dtype)
                            w_obj = w * obj_has_edge
                            gamma = float(getattr(self, 'pair_replace_gate_gamma', 1.0))
                            if gamma != 1.0:
                                w_obj = w_obj.clamp_min(0) ** gamma
                            pref = w_obj / w_obj.sum(-1, keepdim=True).clamp_min(1e-9)
                            target = torch.matmul(pref * w, em.unsqueeze(1).to(w.dtype))
                            n_safe = em.sum(dim=1).clamp_min(1).view(B, 1, 1, -1)
                            gain = torch.clamp(target - w, min=0) * (am > 0)[:, None, None, :].to(w.dtype)
                            w = w + gain
                            w = w - torch.matmul(gain / n_safe,
                                                 em.transpose(-2, -1).unsqueeze(1).to(w.dtype))
                            if _PAIR_DEBUG:
                                print(f'[dbg_out] mode=gated gamma={gamma:.2f} '
                                      f'rowsum_max={float(w.sum(-1).max()):.4g} '
                                      f'gain_max={float(gain.max()):.4g} '
                                      f'w_min={float(w.min()):.4g}', flush=True)
                        elif mode == 'equalize':
                            # Local renormalisation: the attribute is lifted to
                            # the mean weight of the nouns it is bound to, and
                            # exactly that amount is taken back FROM THOSE NOUNS
                            # (split evenly between them).  The pair's total is
                            # therefore conserved, so the row still sums to 1 --
                            # but, unlike a whole-row rescale, every token
                            # outside the pair keeps its weight untouched, and a
                            # pair whose attribute already outranks its noun is
                            # left alone.
                            n_obj = em.sum(dim=1)                       # (B, L)
                            n_safe = n_obj.clamp_min(1).view(B, 1, 1, -1)
                            pair_mean = pair_weight / n_safe
                            gain = torch.clamp(pair_mean - w, min=0)
                            gain = gain * (am > 0)[:, None, None, :].to(gain.dtype)
                            w = w + gain
                            # nouns pay gain/n each, gathered back through emᵀ
                            pay = torch.matmul(
                                gain / n_safe,
                                em.transpose(-2, -1).unsqueeze(1).to(w.dtype))
                            w = w - pay
                            if _PAIR_DEBUG:
                                print(f'[dbg_out] mode=equalize '
                                      f'rowsum_max={float(w.sum(-1).max()):.4g} '
                                      f'gain_max={float(gain.max()):.4g} '
                                      f'w_min={float(w.min()):.4g}', flush=True)
                        elif mode == 'raise':
                            # Raise-only, no renormalisation.  For every parsed
                            # (noun, attribute) pair the attribute is lifted to
                            # the weight its noun already carries, but only where
                            # it sits BELOW it: an attribute the patch already
                            # reads strongly is left untouched.  The row is
                            # deliberately NOT renormalised afterwards, so the
                            # edit stays local to the pair and every token
                            # outside it keeps its weight.
                            w = torch.where(am[:, None, None, :] > 0,
                                            torch.maximum(w, pair_weight), w)
                            if getattr(self, 'pair_replace_renorm', False):
                                # Requested explicitly: bring the row back to
                                # sum 1 after the raise, which scales EVERY
                                # token down proportionally -- including those
                                # outside the pair -- to pay for the mass the
                                # raise added.  (The alternative, 'equalize',
                                # keeps the sum at 1 by debiting only the pair's
                                # own nouns, leaving the rest of the row alone.)
                                w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
                        else:
                            new_a = pair_weight
                            strength = float(getattr(self, 'pair_replace_strength', 1.0))
                            if strength < 1.0:
                                # Partial replacement: interpolate towards the
                                # rewritten weights instead of taking them
                                # wholesale.  Full strength moves ~half the
                                # attention mass of every object-attending patch
                                # onto the attribute token; how much a model
                                # tolerates is what the lambda sweep measures.
                                new_a = w + strength * (new_a - w)
                            w = torch.where(am[:, None, None, :] > 0, new_a, w)
                            # remaining weights scale proportionally so the row sums to 1
                            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
                    if _PAIR_BOX['on']:
                        # 记录的是替换/重分配之后的权重（此处的 w），
                        # 形状 (B,H,N,L)，与 SDPA 侧抓取的口径一致。
                        _PAIR_BOX['maps'].append(w.detach().float().cpu())
                    x = torch.matmul(w, vf).transpose(1, 2).to(v.dtype)
                    if _PAIR_DEBUG:
                        # Is the cross-attention output even patch-specific?
                        # If every row reads the same sink token, this is nearly
                        # constant across patches and carries no spatial/
                        # binding information -- which would mean NO weight-
                        # editing mechanism can ever steer binding.
                        _xf = x.float()                                  # (B,P,H,D)
                        _across = _xf.std(dim=1).mean()                  # variation over patches
                        _scale = _xf.abs().mean().clamp_min(1e-9)
                        print(f'[dbg_patchvar] 跨patch标准差/幅度='
                              f'{float(_across/_scale):.4f} 幅度={float(_xf.abs().mean()):.4g}',
                              flush=True)
                        print(f'[dbg_out] w_nan={torch.isnan(w).any().item()} x_nan={torch.isnan(x).any().item()} '
                              f'x_max={float(x.abs().max()):.3g} x_fp16_inf={bool(torch.isinf(x).any())}', flush=True)
            else:
                x = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop.p if self.training else 0.,
                    attn_mask=key_padding)
                x = x.transpose(1, 2)
        x = x.reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class WindowAttention(Attention_):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
        **block_kwargs,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **block_kwargs)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        if use_fp32_attention := getattr(self, 'fp32_attention', False):
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))
        if _HAS_XFORMERS:
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0., attn_mask=(mask.squeeze(1)[:, None, None, :].bool() if mask is not None else None)).transpose(1, 2)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#   AMP attention with fp32 softmax to fix loss NaN problem during training     #
#################################################################################
class Attention(Attention_):
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        use_fp32_attention = getattr(self, 'fp32_attention', False)
        if use_fp32_attention:
            q, k = q.float(), k.float()
        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MaskFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DecoderLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, decoder_hidden_size):
        super().__init__()
        self.norm_decoder = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_decoder(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(t_freq)

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs//s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0)
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class CaptionEmbedderDoubleBr(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0)
        self.embedding = nn.Parameter(torch.randn(1, in_channels) / 10 ** 0.5)
        self.y_embedding = nn.Parameter(torch.randn(token_num, in_channels) / 10 ** 0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, global_caption, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(global_caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        global_caption = torch.where(drop_ids[:, None], self.embedding, global_caption)
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return global_caption, caption

    def forward(self, caption, train, force_drop_ids=None):
        assert caption.shape[2: ] == self.y_embedding.shape
        global_caption = caption.mean(dim=2).squeeze()
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            global_caption, caption = self.token_drop(global_caption, caption, force_drop_ids)
        y_embed = self.proj(global_caption)
        return y_embed, caption
