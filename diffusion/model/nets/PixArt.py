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
import torch
import torch.nn as nn
import os
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, LabelEmbedder, FinalLayer
from diffusion.utils.logger import get_root_logger


class SemanticConditionAdapter(nn.Module):
    """A zero-initialized bottleneck residual for one semantic condition type."""

    def __init__(self, hidden_size, bottleneck_size):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.SiLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, condition):
        return self.up(self.act(self.down(condition)))


def build_edge_matrix(edge_pairs, edge_count, length, dtype, device):
    """(B, L, L) 0/1 adjacency over T5 sub-tokens from parsed (object, attribute) pairs.

    Standalone so the pair-replacement path does not depend on the semantic
    module existing.
    """
    batch, n_edges = edge_pairs.shape[0], edge_pairs.shape[1]
    matrix = torch.zeros(batch, length, length, dtype=dtype, device=device)
    valid = edge_pairs[..., 0] >= 0
    if edge_count is not None:
        counts = edge_count.to(device)
        valid = valid & (torch.arange(n_edges, device=device)[None, :] < counts[:, None])
    obj = edge_pairs[..., 0].clamp(min=0, max=length - 1).to(device)
    attr = edge_pairs[..., 1].clamp(min=0, max=length - 1).to(device)
    b_idx = torch.arange(batch, device=device)[:, None].expand_as(obj)
    matrix[b_idx[valid], obj[valid], attr[valid]] = 1.0
    return matrix


class SemanticTokenCrossAttention(nn.Module):
    """Per-patch attention over role-marked T5 tokens with weak pair binding."""

    def __init__(self, hidden_size, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.object_pair_proj = nn.Linear(hidden_size, hidden_size)
        self.attribute_pair_proj = nn.Linear(hidden_size, hidden_size)
        self.role_bias = nn.Parameter(torch.zeros(3))
        # Keep the pair-aware contribution non-negative so it can only
        # reinforce object-to-attribute attention. A negative logit gives a
        # small initial softplus scale while still allowing it to grow.
        self.pair_strength = nn.Parameter(torch.tensor(-3.0))
        # Set by forward() when the binding objective is enabled (B2).
        self._last_binding_loss = None
        # PROBE-ONLY, default off.  Replaces the learned pair affinity with the
        # parsed edge indicator (0/1), and optionally overrides its scale.
        # Never set during training.
        self.pair_edge_hard = False
        self.pair_strength_override = None
        # LCAR（推理期局部竞争注意重分配）。保持 None 或 enable=False 时，
        # 本模块的行为与打补丁之前完全一致。
        self.lcar = None          # diffusion.model.nets.lcar.LCARConfig
        self.lcar_state = None    # diffusion.model.nets.lcar.LCARState
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, image_tokens, text_tokens, role_masks, text_valid_mask=None,
                return_role_outputs=False, edge_pairs=None, edge_count=None,
                pair_edge_gate=False, binding_loss_coef=0.0, binding_temperature=1.0,
                lcar_block=None, lcar_step=None):
        # lcar_block / lcar_step 只在 LCAR 开启时由 DiT 主循环传入，用于按层分键
        # 缓存锚点，并按去噪步判断「前 calc_ratio 比例的早期步」窗口；默认 None
        # 时下面的 LCAR 分支整体不执行。
        batch, patches, hidden = image_tokens.shape
        length = text_tokens.shape[1]
        q = self.q_proj(image_tokens).view(batch, patches, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(text_tokens).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(text_tokens).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        roles = role_masks.to(logits.dtype)
        logits = logits + torch.einsum('brl,r->bl', roles, self.role_bias)[:, None, None, :]
        object_mask, attribute_mask = roles[:, 0], roles[:, 1]
        # Infer object-attribute affinity from token representations (no boxes required).
        object_repr = self.object_pair_proj(text_tokens)
        attribute_repr = self.attribute_pair_proj(text_tokens)
        pair = torch.sigmoid(torch.matmul(object_repr, attribute_repr.transpose(1, 2)) * self.scale)
        pair = pair * object_mask[:, :, None] * attribute_mask[:, None, :]

        # ---- B1: restrict the pair affinity to the PARSED object->attribute
        # edges.  Without this the bias is driven purely by a learned content
        # similarity, discarding the dependency information the parser already
        # provides (measured 0.00% alignment failure rate on this corpus).
        if self.pair_edge_hard and edge_pairs is not None:
            # probe: the pair weight IS the parsed edge, not a learned affinity
            pair = self._edge_matrix(edge_pairs, edge_count, length,
                                     pair.dtype, pair.device)
        elif pair_edge_gate and edge_pairs is not None:
            pair = pair * self._edge_matrix(edge_pairs, edge_count, length,
                                            pair.dtype, pair.device)

        object_logits = logits.masked_fill(object_mask[:, None, None, :] <= 0, torch.finfo(logits.dtype).min)
        object_attention = torch.softmax(object_logits, dim=-1).mean(dim=1)
        pair_attribute_bias = torch.bmm(object_attention, pair)
        _ps = self.pair_strength if self.pair_strength_override is None else pair_attribute_bias.new_tensor(float(self.pair_strength_override))
        _beta = torch.nn.functional.softplus(_ps)
        _bias = _beta * pair_attribute_bias
        # ---- LCAR：局部竞争注意重分配 --------------------------------------
        # 唯一改动点：只改写**支路自身**的 bias。主干 cross-attention 的 QKV、
        # 权重、输出投影全程不动，也不新增任何可训练参数（beta 复用现有强度）。
        # self.lcar 为 None 或 enable=False 时下面的分支不执行，_bias 与改前
        # 的 softplus(_ps) * pair_attribute_bias 逐位相同。
        if self.lcar is not None and self.lcar.enable and lcar_block is not None:
            from .lcar import lcar_delta
            # 显式对齐 dtype：LCAR 内部用 float32 累加以保证数值稳定，
            # 但支路本身跑 fp16，直接相加会把整条链路提升成 float32，
            # 随后与 fp16 的 v 做 matmul 会报 "expected Half but found Float"。
            _delta = lcar_delta(
                logits.mean(dim=1), object_attention, pair, object_mask,
                attribute_mask, _beta, self.lcar, self.lcar_state, lcar_block,
                lcar_step, edge_pairs=edge_pairs, edge_count=edge_count)
            _bias = _bias + _delta.to(_bias.dtype)
        logits = logits + _bias[:, None, :, :]
        # ---- LCAR 空间渐变替换式 ------------------------------------------
        # 产出一个 [B, M] 的 content_target 空间场（锚点处最高，沿半径线性下降，
        # 半径外为 0 = 不替换），由 DiT 主循环转交给主干注意力使用。这里只**算**，
        # 不改本模块的任何输出，所以对支路本身没有副作用。
        self._lcar_target_field = None
        if self.lcar is not None and self.lcar.replace_graded:
            from .lcar import spatial_content_target
            # 锚点版渐变场（已由 8 类完整评测验证有效）。这里多传 pair /
            # attribute_mask / 解析边，是为了让锚点选择用上"对象—属性绑定"。
            self._lcar_target_field = spatial_content_target(
                logits.mean(dim=1), object_attention, pair, object_mask,
                attribute_mask, self.lcar, self.lcar_state, lcar_block,
                lcar_step, edge_pairs=edge_pairs, edge_count=edge_count)
        if text_valid_mask is not None:
            logits = logits.masked_fill(~text_valid_mask[:, None, None, :].bool(), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)

        # ---- B2: explicit attention-alignment objective.  Nothing in the
        # diffusion loss rewards "attribute bound to the right object", so the
        # branch has no gradient telling it to bind correctly; the gate simply
        # drifts towards switching the branch off.  This term supplies that
        # gradient directly.
        self._last_binding_loss = None
        if binding_loss_coef > 0.0 and edge_pairs is not None:
            self._last_binding_loss = self._binding_alignment_loss(
                weights, object_attention, attribute_mask, edge_pairs, edge_count,
                binding_temperature)

        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, patches, hidden)
        output = self.out_proj(output)
        if not return_role_outputs:
            return output
        # Produce role-specific residuals from the same attention map.  The
        # learnable per-layer branch gates decide where object/attribute/
        # relation information is injected; the all-token output is the
        # conservative global branch.
        role_outputs = []
        for role_index in range(3):
            role = roles[:, role_index].to(weights.dtype)[:, None, None, :]
            role_weights = weights * role
            denom = role_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            role_feature = torch.matmul(role_weights, v) / denom
            role_feature = role_feature.transpose(1, 2).reshape(batch, patches, hidden)
            role_outputs.append(self.out_proj(role_feature))
        return output, torch.stack(role_outputs, dim=2)

    def _edge_matrix(self, edge_pairs, edge_count, length, dtype, device):
        """(B, L, L) adjacency over T5 sub-tokens from parsed (object, attribute) pairs."""
        batch, n_edges = edge_pairs.shape[0], edge_pairs.shape[1]
        matrix = torch.zeros(batch, length, length, dtype=dtype, device=device)
        valid = edge_pairs[..., 0] >= 0
        if edge_count is not None:
            counts = edge_count.to(device)
            valid = valid & (torch.arange(n_edges, device=device)[None, :] < counts[:, None])
        obj = edge_pairs[..., 0].clamp(min=0, max=length - 1).to(device)
        attr = edge_pairs[..., 1].clamp(min=0, max=length - 1).to(device)
        b_idx = torch.arange(batch, device=device)[:, None].expand_as(obj)
        matrix[b_idx[valid], obj[valid], attr[valid]] = 1.0
        return matrix

    def _binding_alignment_loss(self, weights, object_attention, attribute_mask,
                                edge_pairs, edge_count, temperature):
        """Contrastive objective for object->attribute binding.

        For every parsed edge (o, a) we score each candidate attribute token a'
        by ``sum_p attn[p, o] * attn[p, a']`` -- i.e. "how much do the patches
        that look at object o also look at a'?" -- and take the cross entropy
        against the bound token a, restricted to the parser's attribute tokens.
        """
        device = weights.device
        batch, heads, patches, length = weights.shape
        edge_pairs = edge_pairs.to(device)
        n_edges = edge_pairs.shape[1]
        valid = edge_pairs[..., 0] >= 0
        if edge_count is not None:
            counts = edge_count.to(device)
            valid = valid & (torch.arange(n_edges, device=device)[None, :] < counts[:, None])
        if not bool(valid.any()):
            return None

        wm = weights.float().mean(dim=1)                                  # (B, P, L)
        obj = edge_pairs[..., 0].clamp(min=0, max=length - 1)
        attr = edge_pairs[..., 1].clamp(min=0, max=length - 1)
        index = obj[:, None, :].expand(batch, patches, n_edges)
        weight_o = object_attention.gather(2, index).permute(0, 2, 1).float()   # (B, E, P)
        # (B*E, 1, P) x (B*E, P, L) -> (B, E, L)
        score = torch.bmm(weight_o.reshape(batch * n_edges, 1, patches),
                          wm[:, None].expand(batch, n_edges, patches, length)
                            .reshape(batch * n_edges, patches, length)).reshape(batch, n_edges, length)
        score = score / max(float(temperature), 1e-6)
        candidate_ok = attribute_mask.to(device).float()[:, None, :] > 0
        score = score.masked_fill(~candidate_ok, torch.finfo(score.dtype).min)
        logp = torch.log_softmax(score, dim=-1)
        target = logp.gather(2, attr[..., None]).squeeze(-1)              # (B, E)
        loss = -target[valid].mean()
        return loss if bool(torch.isfinite(loss)) else None

    def binding_loss(self):
        """Last alignment loss computed in forward(), or None."""
        return self._last_binding_loss

    def stats(self):
        """Return detached scalars for monitoring pair-aware training."""
        return {
            'semantic_pair_strength': torch.nn.functional.softplus(self.pair_strength).detach(),
            'semantic_role_bias_object': self.role_bias[0].detach(),
            'semantic_role_bias_attribute': self.role_bias[1].detach(),
            'semantic_role_bias_relation': self.role_bias[2].detach(),
        }


class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False,
                 time_condition_gate=True, condition_gate_scale=0.05, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        self.time_condition_gate = time_condition_gate
        self.gate_enabled = True
        self.condition_gate_scale = nn.Parameter(torch.tensor(float(condition_gate_scale))) if time_condition_gate else None
        if time_condition_gate:
            self.condition_gate = nn.Linear(6 * hidden_size, 1)
            nn.init.zeros_(self.condition_gate.weight)
            nn.init.zeros_(self.condition_gate.bias)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        cross_attn = self.cross_attn(x, y, mask)
        if self.time_condition_gate and self.gate_enabled:
            gate = 1 + self.condition_gate_scale * torch.tanh(self.condition_gate(t)).view(B, 1, 1)
            cross_attn = gate * cross_attn
        x = x + cross_attn
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


#############################################################################
#                                 Core PixArt Model                                #
#################################################################################
@MODELS.register_module()
class PixArt(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=True, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=4096, lewei_scale=1.0, config=None, model_max_length=120, semantic_conditioning=False, semantic_adapter_dim=64, semantic_dropout=0.1, time_condition_gate=True, condition_gate_scale=0.05, semantic_residual_scale=1.0, semantic_active_branches=None, semantic_token_attention=False, semantic_token_gate_max=1.0,
                 semantic_gate_init=-4.0, semantic_gate_init_global=4.0,
                 semantic_token_gate_init=0.05,
                 semantic_token_gate_activation='tanh',
                 semantic_gate_max_warmup_steps=0,
                 semantic_pair_edge_gate=False,
                 semantic_binding_coef=0.0,
                 semantic_binding_temperature=1.0,
                 semantic_time_window=None,
                 semantic_random_probe=False,
                 semantic_pair_edge_hard=False,
                 semantic_pair_strength_override=None,
                 cross_attn_pair_replace=False,
                 cross_attn_pair_replace_strength=1.0,
                 cross_attn_pair_replace_mode='replace',
                 cross_attn_pair_replace_renorm=False,
                 cross_attn_pair_replace_gate_gamma=1.0,
                 cross_attn_pair_replace_fund='proportional',
                 cross_attn_pair_replace_content_target=0.5,
                 cross_attn_pair_replace_sink_factor=0.0, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,
        self.semantic_conditioning = semantic_conditioning
        self.semantic_dropout = semantic_dropout
        self.semantic_residual_scale = float(semantic_residual_scale)
        self.semantic_token_attention_enabled = bool(semantic_token_attention)
        self.semantic_token_gate_max = float(semantic_token_gate_max)
        self.semantic_active_branches = set(semantic_active_branches or ('global', 'object', 'attribute', 'relation'))
        self._last_semantic_stats = None
        self._last_semantic_regularization = None

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(input_size // patch_size, input_size // patch_size),
                          window_size=window_size if i in window_block_indexes else 0,
                          use_rel_pos=use_rel_pos if i in window_block_indexes else False,
                          time_condition_gate=time_condition_gate,
                          condition_gate_scale=condition_gate_scale)
            for i in range(depth)
        ])
        if semantic_conditioning:
            self.semantic_adapters = nn.ModuleDict({
                name: SemanticConditionAdapter(hidden_size, semantic_adapter_dim)
                for name in ('global', 'object', 'attribute', 'relation')
            })
            # Global semantics remain available throughout the denoising stack.
            # The remaining windows match layout, interaction, and detail stages.
            layer_mask = torch.zeros(depth, 4)
            layer_mask[:, 0] = 1.
            layer_mask[:depth // 2, 1] = 1.
            layer_mask[depth // 3:(2 * depth) // 3 + 1, 3] = 1.
            layer_mask[depth // 2:max(depth // 2 + 1, depth - 4), 2] = 1.
            for index, name in enumerate(('global', 'object', 'attribute', 'relation')):
                if name not in self.semantic_active_branches:
                    layer_mask[:, index] = 0.
            self.register_buffer('semantic_layer_mask', layer_mask)
            self.semantic_layer_scale = nn.Parameter(torch.ones(depth, 4))
            self.semantic_time_gate = nn.Linear(hidden_size, 4)
            nn.init.zeros_(self.semantic_time_gate.weight)
            nn.init.zeros_(self.semantic_time_gate.bias)
            if self.semantic_token_attention_enabled:
                self.semantic_token_cross_attention = SemanticTokenCrossAttention(hidden_size, num_heads)
                # ---- Effective strength of the token-pair branch ------------
                # Exactly ONE squashed, learnable scalar scales the branch.
                # It used to be `semantic_token_gate_max * tanh(gate)` nested
                # with the per-layer sigmoid gates; two nested squashing ops
                # pushed the role branches down to ~3e-4 and their gradients
                # into the saturated region of sigmoid, so they never moved.
                # Default MUST stay 'tanh': it is the formula every existing
                # token-pair checkpoint was trained with.  Switching it changes
                # the effective injection strength of those checkpoints by ~2.8x
                # (1.0*tanh(0.2022)=0.1995 vs 1.0*sigmoid(0.2022)=0.5504).
                # The gate-optimisation config opts in to 'sigmoid' explicitly.
                self.semantic_token_gate_activation = str(semantic_token_gate_activation)
                self.semantic_token_gate = nn.Parameter(torch.tensor(float(semantic_token_gate_init)))
                # Per-layer role gates.  The DEFAULT (+-4.0) must stay exactly
                # as it always was: checkpoints trained before this parameter
                # existed (e.g. the token-pair run) do not contain it and would
                # otherwise silently pick up a different initialisation --
                # measured to change the effective role strength by ~15x and
                # wreck those models.  The gate-optimisation config sets +-1.0
                # explicitly, which moves the gate into the linear part of the
                # sigmoid (|w| = 1 -> lambda ~ 0.27 / 0.73, dlambda/dw ~ 0.2).
                self.semantic_token_layer_gate = nn.Parameter(torch.full((depth, 4), float(semantic_gate_init)))
                self.semantic_token_layer_gate.data[:, 0] = float(semantic_gate_init_global)
                # Snapshot of the intended initialisation, for drift monitoring.
                # persistent=False keeps it out of every state_dict.
                self.register_buffer('semantic_gate_init_snapshot',
                                     self.semantic_token_layer_gate.detach().clone(),
                                     persistent=False)
                self.semantic_gate_init_global = float(semantic_gate_init_global)
                self.semantic_gate_max_target = float(semantic_token_gate_max)
                self.semantic_gate_max_warmup_steps = int(semantic_gate_max_warmup_steps)
                self.semantic_gate_max_current = float(semantic_token_gate_max)
                # B1 / B2 controls (see results/TP_SCDA_IMPROVEMENT_PLAN.md).
                self.semantic_pair_edge_gate = bool(semantic_pair_edge_gate)
                self.semantic_binding_coef = float(semantic_binding_coef)
                self.semantic_binding_temperature = float(semantic_binding_temperature)
                # PROBE ONLY (default None = off).  Restricts the semantic
                # branch to a timestep range, to test whether injecting it
                # during only part of the sampling trajectory is better than
                # injecting it throughout.  Never set during training.
                self.semantic_time_window = (
                    tuple(float(v) for v in semantic_time_window)
                    if semantic_time_window is not None else None)
                if self.semantic_token_attention_enabled:
                    self.semantic_token_cross_attention.pair_edge_hard = bool(semantic_pair_edge_hard)
                    self.semantic_token_cross_attention.pair_strength_override = semantic_pair_strength_override
                # PROBE ONLY (default False = off).  Replaces the semantic
                # branch's output with a random vector of MATCHED magnitude, to
                # separate "the branch's learned content is harmful" from "any
                # perturbation of that size is harmful".  Never set in training.
                self.semantic_random_probe = bool(semantic_random_probe)
        # Pair replacement rewrites the DiT's OWN cross-attention, so unlike the
        # semantic-branch probes above it must be armed whether or not that
        # branch is enabled.  Nesting these two under
        # `semantic_token_attention_enabled` is what silently disabled the
        # mechanism for the entire 2026-09-20 pair_replace run -- training and
        # evaluation both ran with the flag unset, i.e. with no mechanism at all.
        self.cross_attn_pair_replace = bool(cross_attn_pair_replace)
        # 1.0 = the attribute takes the noun's weight outright; below 1.0 blends
        # towards the DiT's own weights (see PixArt_blocks.py).
        self.cross_attn_pair_replace_strength = float(cross_attn_pair_replace_strength)
        # 'replace': attribute weight is overwritten with its nouns' weight and
        #            the attention row is renormalised (the 2026-09-20 variant).
        # 'raise'  : attribute is lifted to its nouns' weight only where it sits
        #            below it, and the row is left un-normalised so nothing
        #            outside the pair moves.
        self.cross_attn_pair_replace_mode = str(cross_attn_pair_replace_mode)
        # 'raise' only: whether to rescale the whole row back to sum 1 afterwards
        # (proportional shrink of every other token) or leave the row un-normalised.
        self.cross_attn_pair_replace_renorm = bool(cross_attn_pair_replace_renorm)
        self.cross_attn_pair_replace_gate_gamma = float(cross_attn_pair_replace_gate_gamma)
        # 'outside' mode: who pays for the lift -- 'proportional' (each
        # outsider in proportion to its weight), 'tail' (smallest first),
        # 'uniform' (equal share).
        self.cross_attn_pair_replace_fund = str(cross_attn_pair_replace_fund)
        self.cross_attn_pair_replace_content_target = float(cross_attn_pair_replace_content_target)
        # 'sink' mode: how much of the row's argmax weight survives (0 = removed)
        self.cross_attn_pair_replace_sink_factor = float(cross_attn_pair_replace_sink_factor)
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        if config:
            logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
            logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        else:
            print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')

    def forward(self, x, timestep, y, mask=None, data_info=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
        semantic_mask = mask
        if semantic_mask is not None and semantic_mask.shape[0] != y.shape[0]:
            if y.shape[0] % semantic_mask.shape[0] != 0:
                raise ValueError('text mask batch size must divide caption batch size')
            repeat_shape = [y.shape[0] // semantic_mask.shape[0]] + [1] * (semantic_mask.ndim - 1)
            semantic_mask = semantic_mask.repeat(*repeat_shape)
        _needs_pair_structure = (self.semantic_conditioning
                                 or getattr(self, 'cross_attn_pair_replace', False))
        if _needs_pair_structure and isinstance(data_info, dict):
            role_masks = data_info.get('semantic_token_masks')
            if torch.is_tensor(role_masks) and role_masks.shape[0] != y.shape[0]:
                if y.shape[0] % role_masks.shape[0] != 0:
                    raise ValueError('semantic mask batch size must divide caption batch size')
                factor = y.shape[0] // role_masks.shape[0]
                # The solver builds c_in = cat([unconditional, condition]), so the
                # FIRST half of a doubled batch has no text to mark -- the zeros
                # belong there and the roles on the conditional half.  This used
                # to be written the other way round (roles first), which put the
                # whole semantic branch on the null-text half; measured on the
                # real generation path: uncond half 4.0 attribute-role mass,
                # cond half 0.0.  The branch then reached the image through
                # CFG as uncond + scale*(cond - uncond), i.e. sign-flipped and
                # multiplied by (1 - scale).
                if factor == 2:
                    expanded = torch.cat((torch.zeros_like(role_masks), role_masks), dim=0)
                else:
                    expanded = role_masks.repeat(factor, *([1] * (role_masks.ndim - 1)))
                data_info = dict(data_info)
                data_info['semantic_token_masks'] = expanded
                # The parsed edges index the SAME batch -- pair replacement builds
                # its adjacency matrix from them -- so they must follow it.  If
                # they do not, the rewrite broadcasts against the wrong axis
                # instead of failing.  Under CFG the unconditional half is first
                # and carries zero role masks, so its edge content is inert.
                for _key in ('semantic_edges', 'semantic_edge_count'):
                    _val = data_info.get(_key)
                    if not torch.is_tensor(_val):
                        continue
                    data_info[_key] = (torch.cat((_val, _val), dim=0) if factor == 2
                                       else _val.repeat(factor, *([1] * (_val.ndim - 1))))
        semantic_conditions = None if self.semantic_token_attention_enabled else self._build_semantic_conditions(y, semantic_mask, data_info)
        semantic_adapter_outputs = None
        semantic_residual_norms = []
        if semantic_conditions is not None:
            semantic_adapter_outputs = self._semantic_adapter_outputs(semantic_conditions)
        role_masks = data_info.get('semantic_token_masks') if isinstance(data_info, dict) else None
        role_info = data_info if isinstance(data_info, dict) else {}
        # LCAR 替换目标的排除掩码 [B, L]：True 的位置不参与替换与渐变
        # （数词/方位词/关系虚词/冠词），其余 token 照常。
        lcar_exclude = role_info.get('lcar_replace_exclude')
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                repeat_shape = [y.shape[0] // mask.shape[0]] + [1] * (mask.ndim - 1)
                mask = mask.repeat(*repeat_shape)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1)
        binding_losses = []
        # PROBE-ONLY: hand the parsed structure to the DiT's own cross-attention
        if getattr(self, 'cross_attn_pair_replace', False) and role_masks is not None:
            _em = None
            if isinstance(role_info, dict) and role_info.get('semantic_edges') is not None:
                _em = build_edge_matrix(
                    role_info['semantic_edges'], role_info.get('semantic_edge_count'),
                    role_masks.shape[-1], torch.float32, role_masks.device)
            for _b in self.blocks:
                _b.cross_attn.pair_replace = True
                _b.cross_attn.pair_roles = role_masks
                _b.cross_attn.pair_edge_mat = _em
                _b.cross_attn.pair_replace_strength = self.cross_attn_pair_replace_strength
                _b.cross_attn.pair_replace_mode = self.cross_attn_pair_replace_mode
                _b.cross_attn.pair_replace_renorm = self.cross_attn_pair_replace_renorm
                _b.cross_attn.pair_replace_gate_gamma = self.cross_attn_pair_replace_gate_gamma
                _b.cross_attn.pair_replace_fund = self.cross_attn_pair_replace_fund
                _b.cross_attn.content_target = self.cross_attn_pair_replace_content_target
                _b.cross_attn.sink_factor = self.cross_attn_pair_replace_sink_factor
        for block_index, block in enumerate(self.blocks):
            if semantic_adapter_outputs is not None:
                residual, residual_norm = self._semantic_residual(
                    semantic_adapter_outputs, t, block_index
                )
                x = x + residual.unsqueeze(1)
                semantic_residual_norms.append(residual_norm)
            if self.semantic_token_attention_enabled and role_masks is not None:
                # NOTE: the probe flag must be here too, or the parsed edges never reach
                # the module and the probe silently does nothing.
                _lc = getattr(self.semantic_token_cross_attention, 'lcar', None)
                _need_edges = (self.semantic_pair_edge_gate or self.semantic_binding_coef > 0
                               or getattr(self.semantic_token_cross_attention, 'pair_edge_hard', False)
                               # LCAR 要用解析器给出的对象—属性边来决定锚点与抑制集
                               or getattr(_lc, 'enable', False)
                               or getattr(_lc, 'replace_graded', False))
                edges = role_info.get('semantic_edges') if _need_edges else None
                edge_count = role_info.get('semantic_edge_count') if edges is not None else None
                token_delta, role_deltas = self.semantic_token_cross_attention(
                    x, y.squeeze(1), role_masks,
                    semantic_mask.squeeze(1).squeeze(1) if semantic_mask is not None else None,
                    return_role_outputs=True,
                    edge_pairs=edges, edge_count=edge_count,
                    pair_edge_gate=self.semantic_pair_edge_gate,
                    binding_loss_coef=self.semantic_binding_coef,
                    binding_temperature=self.semantic_binding_temperature,
                    lcar_block=block_index, lcar_step=timestep)
                layer_gates = torch.sigmoid(self.semantic_token_layer_gate[block_index]).view(1, 1, 4, 1)
                token_delta = layer_gates[:, :, 0] * token_delta
                token_delta = token_delta + (layer_gates[:, :, 1:] * role_deltas).sum(dim=2)
                scale = self.semantic_gate_scale()
                if self.semantic_time_window is not None:
                    lo, hi = self.semantic_time_window
                    keep = ((timestep >= lo) & (timestep <= hi)).to(scale.dtype)
                    keep = keep.reshape(-1, *([1] * (token_delta.ndim - 1)))
                    scale = scale * keep
                if self.semantic_random_probe:
                    token_delta = torch.randn_like(token_delta) * token_delta.std()
                x = x + scale * token_delta
                semantic_residual_norms.append(token_delta.float().norm(dim=-1).mean())
            if self.semantic_token_attention_enabled:
                block_binding = self.semantic_token_cross_attention.binding_loss()
                if block_binding is not None:
                    binding_losses.append(block_binding)
            # 把支路上算出的 LCAR content_target 空间场转交给本层的主干注意力。
            # 没开渐变替换时这里恒为 None，主干注意力行为与打补丁前完全一致。
            _tf = (getattr(self.semantic_token_cross_attention, '_lcar_target_field', None)
                   if self.semantic_token_attention_enabled else None)
            # 时间窗：窗外置全零（t=0 → 下游 alpha 被 clamp 到 1 → 逐位不改 w）。
            # 不能置 None —— 那会让 reweight 回退到标量 content_target，反而全周期替换。
            # 必须和上面 _tf 一样加守卫：semantic_token_cross_attention 只在
            # semantic_token_attention=True 时才创建，未开时它根本不存在，
            # 不加守卫会让**所有 semantic_token_attention=False 的配置**
            # （冻结基线 frozen/frozen_current*、整个嵌入注入 eb* 族）直接抛
            # AttributeError 而跑不起来。未开时本就无 LCAR，取 None 语义正确。
            _lcc = (getattr(self.semantic_token_cross_attention, 'lcar', None)
                    if self.semantic_token_attention_enabled else None)
            if _lcc is not None:
                from .lcar import in_replace_window
                if not in_replace_window(timestep, _lcc):
                    # 窗外一律不给替换。渐变模式下 _tf 是 [B, M] 的场，置全零即可；
                    # 但**标量模式**（replace_graded=False）下 _tf 恒为 None，只把
                    # None 传下去会走下游的 content_target 标量回退分支，等于窗外
                    # 照样替换——所以这里必须显式补一个全零场。
                    # 全零场使 t=0 → alpha 被 clamp 到 1 → w 逐位不变，即恒等操作。
                    _tf = (torch.zeros_like(_tf) if _tf is not None
                           else torch.zeros(x.shape[0], x.shape[1],
                                            device=x.device, dtype=torch.float32))
            block.cross_attn.pair_replace_content_target_map = _tf
            # 排除掩码来自 data_info，批维是提示词数；主干注意力在 CFG 下批维翻倍，
            # 按与 role_masks 相同的方式 tile 展开。
            _lx = lcar_exclude
            if (_lx is not None and role_masks is not None
                    and _lx.shape[0] != role_masks.shape[0]):
                _lx = _lx.repeat(role_masks.shape[0] // _lx.shape[0], 1)
            block.cross_attn.pair_replace_exclude = _lx
            # 加性替换开关（由 LCARConfig 传入）。
            # 与 _lcc 同样必须加守卫：semantic_token_cross_attention 只在
            # semantic_token_attention=True 时才存在，未开时它根本没有该属性。
            _lc2 = (getattr(self.semantic_token_cross_attention, 'lcar', None)
                    if self.semantic_token_attention_enabled else None)
            block.cross_attn.pair_replace_additive = bool(getattr(_lc2, 'replace_additive', False))
            block.cross_attn.pair_replace_delta_max = float(getattr(_lc2, 'replace_delta_max', 0.2))
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)  # (N, T, D) #support grad checkpoint
        if semantic_adapter_outputs is not None:
            self._last_semantic_stats = self._collect_semantic_stats(
                semantic_conditions, semantic_adapter_outputs, t, semantic_residual_norms
            )
        elif self.semantic_token_attention_enabled and role_masks is not None:
            self._last_semantic_stats = self.semantic_token_cross_attention.stats()
            self._last_semantic_stats['semantic_token_gate'] = self.semantic_gate_scale().detach()
            for key, value in self.semantic_gate_diagnostics().items():
                self._last_semantic_stats[key] = value.detach()
            layer_gate_means = torch.sigmoid(self.semantic_token_layer_gate).mean(dim=0)
            for index, name in enumerate(('global', 'object', 'attribute', 'relation')):
                self._last_semantic_stats[f'semantic_layer_gate_{name}'] = layer_gate_means[index].detach()
            self._last_semantic_stats['semantic_residual_norm'] = torch.stack(semantic_residual_norms).mean().detach()
        else:
            self._last_semantic_stats = None
        self._last_semantic_regularization = (torch.stack(semantic_residual_norms).pow(2).mean()
                                              if semantic_residual_norms else x.new_zeros(()))
        # Mean over DiT blocks; None when the binding objective is disabled.
        self._last_binding_loss = (torch.stack(binding_losses).mean()
                                   if binding_losses else None)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def get_semantic_binding_loss(self):
        """Mean object->attribute alignment loss over blocks, or None (B2)."""
        return getattr(self, '_last_binding_loss', None)

    def semantic_gate_scale(self):
        """Effective strength of the token-pair branch (a single learnable scalar).

        ``sigmoid`` keeps the coefficient in (0, gate_max) and, unlike the
        previous ``tanh`` form, is not multiplied by a second per-layer sigmoid
        gate -- that double squashing is what drove the role branches to ~3e-4.
        """
        raw = self.semantic_token_gate
        if self.semantic_token_gate_activation == 'tanh':
            squashed = torch.tanh(raw)
        else:
            squashed = torch.sigmoid(raw)
        return self.semantic_gate_max_current * squashed

    def set_semantic_gate_max(self, value):
        """Called by the training loop to apply the warm-up ramp."""
        self.semantic_gate_max_current = float(value)

    def semantic_gate_diagnostics(self):
        """Scalars for the training log -- everything the gate needs to be
        debuggable from the metrics CSV alone."""
        with torch.no_grad():
            w = self.semantic_token_layer_gate.detach().float()
            lam = torch.sigmoid(w)
            snapshot = self.semantic_gate_init_snapshot.detach().float()
            names = ('global', 'object', 'attribute', 'relation')
            out = {}
            for index, name in enumerate(names):
                column = lam[:, index]
                out[f'semantic_gate_mean_{name}'] = column.mean()
                # Spread across layers: ~0 means the gate never differentiated
                # the injection depth, which is exactly how the previous run
                # failed while still looking "trained" in the CSV.
                out[f'semantic_gate_layer_std_{name}'] = column.std(unbiased=False)
                out[f'semantic_gate_drift_{name}'] = (w[:, index] - snapshot[:, index]).abs().max()
            out['semantic_gate_effective'] = self.semantic_gate_scale().detach()
            out['semantic_gate_max_current'] = torch.as_tensor(
                float(self.semantic_gate_max_current), device=w.device)
            out['semantic_gate_raw'] = self.semantic_token_gate.detach().float()
            out['semantic_gate_drift_max'] = (w - snapshot).abs().max()
            out['semantic_gate_layer_std_max'] = torch.stack(
                [out[f'semantic_gate_layer_std_{n}'] for n in names]).max()
            return out

    def _build_semantic_conditions(self, text_tokens, text_mask, data_info):
        """Pool projected T5 tokens according to offline semantic masks."""
        if not self.semantic_conditioning:
            return None
        if not isinstance(data_info, dict) or 'semantic_token_masks' not in data_info:
            raise ValueError(
                'semantic_conditioning=True requires data_info["semantic_token_masks"]. '
                'Run tools/prepare_semantic_masks.py and set load_semantic_masks=True.'
            )
        semantic_masks = data_info['semantic_token_masks'].to(
            device=text_tokens.device, dtype=text_tokens.dtype
        )
        if semantic_masks.ndim != 3 or semantic_masks.shape[1] != 3:
            raise ValueError('semantic_token_masks must have shape [B, 3, text_length]')
        token_features = text_tokens.squeeze(1)
        if semantic_masks.shape[0] != token_features.shape[0] or semantic_masks.shape[2] != token_features.shape[1]:
            raise ValueError('semantic_token_masks batch size or text length does not match caption features')
        if text_mask is None:
            valid_mask = torch.ones_like(semantic_masks[:, 0])
        else:
            valid_mask = text_mask.squeeze(1).squeeze(1).to(token_features.dtype)
            if valid_mask.shape[0] != token_features.shape[0]:
                repeat_shape = [token_features.shape[0] // valid_mask.shape[0]] + [1] * (valid_mask.ndim - 1)
                valid_mask = valid_mask.repeat(*repeat_shape)

        def pool(weights):
            weights = weights * valid_mask
            return (token_features * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.)

        conditions = torch.stack((
            pool(valid_mask),
            pool(semantic_masks[:, 0]),
            pool(semantic_masks[:, 1]),
            pool(semantic_masks[:, 2]),
        ), dim=1)
        if self.training and self.semantic_dropout > 0:
            keep = torch.rand(conditions.shape[:2], device=conditions.device) >= self.semantic_dropout
            keep[:, 0] = True
            conditions = conditions * keep.unsqueeze(-1).to(conditions.dtype)
        return conditions

    def _semantic_adapter_outputs(self, conditions):
        names = ('global', 'object', 'attribute', 'relation')
        return torch.stack([
            self.semantic_adapters[name](conditions[:, index])
            for index, name in enumerate(names)
        ], dim=1)

    def _semantic_residual(self, adapter_outputs, timestep_embedding, block_index):
        time_scale = torch.sigmoid(self.semantic_time_gate(timestep_embedding)).unsqueeze(-1)
        layer_scale = (self.semantic_layer_mask[block_index] * self.semantic_layer_scale[block_index])
        residual = self.semantic_residual_scale * (adapter_outputs * time_scale * layer_scale.view(1, -1, 1)).sum(dim=1)
        return residual, residual.float().norm(dim=1).mean()

    def _collect_semantic_stats(self, conditions, adapter_outputs, timestep_embedding, residual_norms):
        names = ('global', 'object', 'attribute', 'relation')
        time_gates = torch.sigmoid(self.semantic_time_gate(timestep_embedding)).float()
        stats = {}
        for index, name in enumerate(names):
            stats[f'semantic_condition_norm_{name}'] = conditions[:, index].float().norm(dim=1).mean().detach()
            stats[f'semantic_adapter_norm_{name}'] = adapter_outputs[:, index].float().norm(dim=1).mean().detach()
            stats[f'semantic_time_gate_{name}'] = time_gates[:, index].mean().detach()
            active = self.semantic_layer_mask[:, index].bool()
            if active.any():
                stats[f'semantic_layer_scale_{name}'] = self.semantic_layer_scale[:, index][active].float().mean().detach()
            else:
                stats[f'semantic_layer_scale_{name}'] = conditions.new_zeros(()).float()
        stats['semantic_residual_norm'] = torch.stack(residual_norms).mean().detach()
        return stats

    def get_semantic_stats(self):
        """Return detached monitoring statistics from the latest forward pass."""
        return self._last_semantic_stats

    def get_semantic_regularization(self):
        """Differentiable residual penalty from the latest training forward."""
        return self._last_semantic_regularization

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask=mask, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        data_info = kwargs.get('data_info')
        if self.semantic_conditioning and isinstance(data_info, dict):
            data_info = dict(data_info)
            for key, value in list(data_info.items()):
                if not torch.is_tensor(value) or value.ndim == 0:
                    continue
                if value.shape[0] != len(combined) // 2:
                    continue
                if key == 'semantic_token_masks':
                    value = torch.cat((value, torch.zeros_like(value)), dim=0)
                else:
                    value = torch.cat((value, value), dim=0)
                data_info[key] = value
            kwargs = dict(kwargs)
            kwargs['data_info'] = data_info
        model_out = self.forward(combined, timestep, y, mask=mask, **kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        if self.semantic_conditioning:
            for adapter in self.semantic_adapters.values():
                nn.init.zeros_(adapter.up.weight)
                nn.init.zeros_(adapter.up.bias)
            nn.init.zeros_(self.semantic_time_gate.weight)
            nn.init.zeros_(self.semantic_time_gate.bias)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


#################################################################################
#                                   PixArt Configs                                  #
#################################################################################
@MODELS.register_module()
def PixArt_XL_2(**kwargs):
    return PixArt(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
