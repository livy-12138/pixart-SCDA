#!/usr/bin/env python3
"""Generate T2I-CompBench++ images for one method.

Official CompBench protocol: 300 val prompts per category x 10 images = 3000
images per category.  Filenames follow the official convention

    <prompt verbatim, spaces preserved, no underscores>_<global index %06d>.png

where the counter is global within a category (official
GORS_finetune/inference_eval.py uses `base_count` incremented across the whole
category), i.e. index = prompt_index * images_per_prompt + repeat.

Design notes
------------
* Every image has its own deterministic seed derived from
  (base seed, category index, prompt index, repeat).  A given image therefore
  depends only on its own identity, so generation is resume-safe and
  stage-safe: generating repeats 0..4 and later repeats 5..9 produces exactly
  the same files as generating 0..9 in one pass.
* Images are produced in repeat-major order so that after every completed stage
  there is a COMPLETE `k`-images-per-prompt set covering all 300 prompts.  That
  lets the benchmark be scored on a complete subset if the full 10-image run
  cannot finish.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

# 注意：PixArt 专有模块**延迟到用到时才导入**。
# 跨骨干对比（--backbone sigma/sana）在独立 venv 里跑，那里没有 diffusion/tools，
# 顶层导入会让入口直接崩。PixArt 主线（--method）行为完全不变。
DPMS = None
PixArt_XL_2 = None
T5Embedder = None
AutoencoderKL = None
save_image = None
build_semantic_edges = None
build_semantic_masks = None
color_attribute_indices = None


def _lazy_pixart_imports():
    global DPMS, PixArt_XL_2, T5Embedder, AutoencoderKL, save_image
    global build_semantic_edges, build_semantic_masks, color_attribute_indices
    if DPMS is not None:
        return
    from diffusers.models import AutoencoderKL as _AE
    from torchvision.utils import save_image as _si
    from diffusion import DPMS as _D
    from diffusion.model.nets import PixArt_XL_2 as _P
    from diffusion.model.t5 import T5Embedder as _T
    from tools.prepare_semantic_masks import (
        build_semantic_edges as _e, build_semantic_masks as _m,
        color_attribute_indices as _c)
    DPMS, PixArt_XL_2, T5Embedder = _D, _P, _T
    AutoencoderKL, save_image = _AE, _si
    build_semantic_edges, build_semantic_masks = _e, _m
    color_attribute_indices = _c

CATEGORIES = ['color', 'shape', 'texture', 'spatial', '3d_spatial',
              'numeracy', 'non_spatial', 'complex']

# Verified by bit-exact reproduction of the existing evaluation images --
# see results/EXPERIMENT_PLAN.md section 1.3.
METHODS = {
    'tpscda': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0,
    ),
    'frozen': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
    ),
    # --- ablation variants, so their CompBench rows come from the SAME
    # --- protocol and the SAME (mask-fixed) code as TP-SCDA and Frozen.
    'pooled': dict(
        checkpoint=str(ROOT / 'output/coco2017_scda/checkpoints/epoch_5_step_18485.pth'),
        semantic_conditioning=True, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=0.25,
    ),
    'fullspan': dict(
        checkpoint=str(ROOT / 'output/coco2017_scda_fullspan/checkpoints/epoch_6_step_22182.pth'),
        semantic_conditioning=True, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=0.25,
    ),
    'tokenpair': dict(
        checkpoint=str(ROOT / 'output/coco2017_scda_token_pair_positive_mb8_acc4/checkpoints/epoch_5_step_60000.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=1.0, semantic_residual_scale=0.0,
    ),
    # Gate-optimised TP-SCDA.  NOTE: gate_max and the gate activation are
    # CONFIG, not weights -- they must be reproduced exactly at inference or the
    # checkpoint is evaluated with a different injection strength than it was
    # trained with.
    'gateopt': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_gate_opt/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.3, semantic_residual_scale=0.0,
        semantic_token_gate_activation='sigmoid',
    ),
    # --- improvements over the gate-optimised run -------------------------------
    # B1: pair-aware bias restricted to the parsed object->attribute edges.
    # The edge gate is part of the FORWARD PASS, so the parsed edges must be
    # supplied at inference as well -- otherwise the checkpoint is scored with
    # the unrestricted affinity it was trained to avoid.  The edges come from
    # build_semantic_edges(), verified row-for-row against the training-time
    # index (200/200 prompts identical).
    'b1': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_b1_edgegate/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.3, semantic_residual_scale=0.0,
        semantic_token_gate_activation='sigmoid',
        semantic_pair_edge_gate=True,
    ),
    # B2 adds a TRAINING-ONLY objective; the forward pass is identical to
    # gateopt, so there is nothing extra to reproduce here.
    'b2': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_b2_bindingloss/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.3, semantic_residual_scale=0.0,
        semantic_token_gate_activation='sigmoid',
    ),
    # TP-SCDA retrained with NO distillation at all, so the ablation can be
    # reported without it.  Same config as `tpscda` otherwise; the only changes
    # are distill_coef=0 and a non-distilled starting checkpoint.
    'tpscda_nodistill': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers_nodistill/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0,
    ),
    # B2 with the gate frozen at the neutral strength for the entire run, so the
    # branch cannot shrink its way out of the alignment objective.
    'b2_frozengate': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_b2_frozengate/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.3, semantic_residual_scale=0.0,
        semantic_token_gate_activation='sigmoid',
    ),
    # TRAINED model: the injection mechanism replaced by rewriting the DiT's own
    # cross-attention weights at the parsed pairs (attribute <- noun's weight).
    # The separate semantic branch is switched off entirely.
    'pair_replace': dict(
        checkpoint=str(ROOT / 'output/coco2017_pair_replace/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=0.0,
        cross_attn_pair_replace=True,
    ),
    # PROBE (not a trained model): instead of a separate injection module, the
    # DiT's OWN cross-attention weights at the parsed object->attribute pairs
    # are replaced by the noun's weight, and the rest of the row rescaled
    # proportionally.  This is the formulation that matches the design intent
    # ("把 pair 内所有 token 的权重设为名词的权重") without a separate q/k and
    # without any additive bias to be drowned by the logit scale.
    'probe_ditpair': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0,
        cross_attn_pair_replace=True,
    ),
    # PROBE (not a trained model): value-space binding tag.  Adds a share of
    # each noun's embedding to its bound attribute token BEFORE the DiT runs.
    # Unlike every attention-weight rule this leaves the sink token, the row
    # distribution and the model's own spatial assignment untouched -- measured
    # on 12 prompts at alpha=0.15: three prompts fixed, none broken.
    'probe_embedbind': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=0.0,
        embed_bind_alpha=0.15,
    ),
    'frozen_eb010': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.1,
    ),
    'frozen_eb020': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.2,
    ),
    'frozen_eb030': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.3,
    ),
    'frozen_ebn015': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.15, embed_bind_mode='normalize',
    ),
    'frozen_ebn030': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.3, embed_bind_mode='normalize',
    ),
    'frozen_ebc015': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.15, embed_bind_mode='contrast',
    ),
    'frozen_ebc030': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.3, embed_bind_mode='contrast',
    ),
    'frozen_ebp2': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.15, embed_bind_position_scale=2.0,
    ),
    'frozen_ebp3': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.15, embed_bind_position_scale=3.0,
    ),
    'frozen_eb040': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.4,
    ),
    'frozen_eb050': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.5,
    ),
    # --- seed replication of the one positive result (colour) ---------------
    # The alpha sweep is single-seed, and its dose-response is not monotone
    # (0.15 -> +0.008, 0.20 -> +0.030), so the +0.03 needs an independent
    # seed before it can be claimed.  Image filenames do not carry the seed,
    # so a distinct method name is what keeps the two runs apart; each entry
    # is an exact copy of its seed-43 counterpart.
    'frozen_current_s44': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
    ),
    'frozen_current_s45': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
    ),
    'frozen_eb020_s44': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.2,
    ),
    'frozen_eb020_s45': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.2,
    ),
    # --- the REPLACEMENT scheme at the routing shares used in the qualitative
    # --- sheet (reweight 25/50/80% of each attention row).  Those images were
    # --- generated from the distilled checkpoint and never scored on CompBench;
    # --- these entries score the same mechanism on the model this paper
    # --- reports, so the replacement row has aggregate numbers of its own.
    'tpscda_nd_rw25': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers_nodistill/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0,
        cross_attn_pair_replace=True, cross_attn_pair_replace_mode='reweight',
        cross_attn_pair_replace_content_target=0.25,
    ),
    'tpscda_nd_rw50': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers_nodistill/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0,
        cross_attn_pair_replace=True, cross_attn_pair_replace_mode='reweight',
        cross_attn_pair_replace_content_target=0.5,
    ),
    'tpscda_nd_rw80': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers_nodistill/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0,
        cross_attn_pair_replace=True, cross_attn_pair_replace_mode='reweight',
        cross_attn_pair_replace_content_target=0.8,
    ),
    # (2) colour-only tagging: only colour attributes get the tag, so shape /
    # texture / spatial structure is untouched.
    'frozen_ebcol020': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.20, embed_bind_filter='color',
    ),
    # (3) the norm lever: pure magnitude inflation, no content mixing.
    'frozen_ebs020': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.20, embed_bind_mode='scale',
    ),
    'frozen_ebs030': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.30, embed_bind_mode='scale',
    ),
    # Weight-editing variant never benchmarked on the official protocol:
    # raise each attribute to its nouns' level and fund it ONLY from the tokens
    # outside the pair (articles/prepositions/other words) -- the noun's own
    # weight is untouched, so the object's evidence stays intact.
    'frozen_outside': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        cross_attn_pair_replace=True, cross_attn_pair_replace_mode='outside',
        cross_attn_pair_replace_fund='proportional',
    ),
    'frozen_sink50': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        cross_attn_pair_replace=True, cross_attn_pair_replace_mode='sink',
        cross_attn_pair_replace_sink_factor=0.5,
    ),
    'frozen_sink00': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        cross_attn_pair_replace=True, cross_attn_pair_replace_mode='sink',
        cross_attn_pair_replace_sink_factor=0.0,
    ),
    # Control: the untouched base model regenerated with TODAY's code, so any
    # score difference against the 2026-09-18 `frozen` baseline is attributable
    # to the code (mask-fix path) rather than to an intervention.
    'frozen_current': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
    ),
    # SAME value-space tag, but on the FROZEN base model: no semantic branch to
    # interfere -- isolates whether the tag itself does anything.
    'frozen_embedbind': dict(
        checkpoint=str(ROOT / 'output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=1.0,
        embed_bind_alpha=0.15,
    ),
    # PROBES (not trained models): the paper's TP-SCDA scored with the
    # pair-replacement mechanism applied at inference under each edit rule.
    #   raise    = attribute lifted to its nouns' weight where it sits below it,
    #              row left un-normalised (the row sum grows by the lift).
    #   equalize = same lift, but the amount is debited from the pair's own
    #              nouns, so the row still sums to 1 and no token outside the
    #              pair moves at all.
    'probe_raise': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=0.0,
        cross_attn_pair_replace=True,
        cross_attn_pair_replace_mode='raise',
    ),
    'probe_equalize': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=False, semantic_token_attention=False,
        semantic_token_gate_max=1.0, semantic_residual_scale=0.0,
        cross_attn_pair_replace=True,
        cross_attn_pair_replace_mode='equalize',
    ),
    # PROBE (not a trained model): the gate-optimised checkpoint with the
    # semantic branch's output replaced by a random vector of MATCHED magnitude.
    # Separates "the learned content is harmful" from "any perturbation of that
    # size is harmful".  Same checkpoint and strength as `gateopt`, which is the
    # regime where the branch demonstrably hurts (colour -0.034, significant).
    'probe_random': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_gate_opt/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.3, semantic_residual_scale=0.0,
        semantic_token_gate_activation='sigmoid',
        semantic_random_probe=True,
    ),
    # PROBES (not trained models): the paper's TP-SCDA scored with the semantic
    # branch active during only PART of the sampling trajectory.  Tests whether
    # injecting throughout is what costs the counting metric -- binding is
    # decided in the early high-noise steps, while late steps only refine
    # appearance.  Same checkpoint as `tpscda`; only the forward pass differs.
    'probe_early': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0,
        semantic_time_window=(500.0, 10000.0),
    ),
    'probe_late': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0,
        semantic_time_window=(0.0, 499.0),
    ),
    # PROBE (not a trained model): the gate-optimised checkpoint with the ROLE
    # gates written back to their historical level at build time.  Tests whether
    # the regression comes from the role branches, which grew 23-27x, rather
    # than from the global branch, which grew only 1.55x.  Must be reported as
    # a probe, never mixed into a table of trained checkpoints without saying so.
    'probe_rolegate': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_gate_opt/checkpoints/probe_rolegate_historical.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.3, semantic_residual_scale=0.0,
        semantic_token_gate_activation='sigmoid',
    ),
    # B2 started from the historical NEUTRAL gate level instead of an open one.
    # Same forward pass as b2 -- the trained gate weight carries the difference.
    'b2_neutral': dict(
        checkpoint=str(ROOT / 'output/coco2017_token_pair_b2_neutralgate/checkpoints/epoch_1_step_14786.pth'),
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.3, semantic_residual_scale=0.0,
        semantic_token_gate_activation='sigmoid',
    ),
}

MAX_SEMANTIC_EDGES = 24     # must match max_semantic_edges in the training config


def image_seed(base, cat_idx, prompt_idx, repeat):
    """Deterministic per-image seed. Unique across (category, prompt, repeat)."""
    return (base * 1_000_003 + cat_idx * 10_007 + prompt_idx * 101 + repeat) % (2 ** 31 - 1)


# ==========================================================================
# 跨骨干对比：diffusers pipeline 生成分支
# ==========================================================================
# 用于论文的"同协议跨骨干"对照。**完全复刻** PixArt 分支的：
#   提示词读取 → 逐类循环 → 文件名 <prompt>_<global_index:06d>.png →
#   种子 image_seed(seed, cat_idx, prompt_idx, repeat) → manifest_<cat>.csv
# 只把"模型怎么跑"换成 diffusers pipeline，其余一律不动。
BACKBONES = {
    'sd15':  'stable-diffusion-v1-5/stable-diffusion-v1-5',
    'sd21':  'Manojb/stable-diffusion-2-1-base',          # 社区镜像（官方仓库 gated）
    'sdxl':  'stabilityai/stable-diffusion-xl-base-1.0',
    'ssd1b': 'segmind/SSD-1B',
    # 以下两个需要 diffusers>=0.27/0.32，在独立 venv 里跑（见 run_pub_compare.sh）
    'sigma': 'PixArt-alpha/PixArt-Sigma-XL-2-512-MS',
    'sana':  'Efficient-Large-Model/Sana_600M_512px_diffusers',
    # PixArt-α 官方 diffusers 版：用于**验证本对比管线本身**。
    # 它与本项目原生路径（--method frozen）是同一模型；若两者在相同提示词上
    # 结果不一致，说明本管线有问题，SD 系的对比也就不可信。
    'pixart': 'PixArt-alpha/PixArt-XL-2-512x512',
}


def _local_snapshot(repo_id):
    """把 repo id 解析成本地缓存快照路径。

    直接用 repo id 会走 hub 的组件解析：该 checkpoint 的 model_index.json 写的是
    旧类名 ``Transformer2DModel``（0.28 时代），新版 diffusers 无法解析，会报
    "no file named scheduler_config.json" 这类误导性错误。传本地路径可绕开。
    缓存不存在时原样返回 repo id（走网络）。
    """
    hub = Path(os.environ.get('HF_HOME', '')) / 'hub'
    d = hub / ('models--' + repo_id.replace('/', '--'))
    snaps = sorted((d / 'snapshots').glob('*')) if (d / 'snapshots').is_dir() else []
    return str(snaps[-1]) if snaps else repo_id


def _build_backbone_pipe(name, dtype, cache_dir):
    """按名字构造 diffusers pipeline（返回已 to('cuda') 的 pipe）。"""
    import torch as _t
    from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline,
                           DPMSolverMultistepScheduler)

    if name in ('sd15', 'sd21'):
        pipe = StableDiffusionPipeline.from_pretrained(
            _local_snapshot(BACKBONES[name]), torch_dtype=dtype, safety_checker=None,
            requires_safety_checker=False, cache_dir=cache_dir)
    elif name in ('sdxl', 'ssd1b'):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            _local_snapshot(BACKBONES[name]), torch_dtype=dtype, cache_dir=cache_dir)
    elif name == 'sigma':
        # 512 仓库只有 transformer；VAE/scheduler/T5 从 1024 仓库取（组件共享）
        from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel
        pipe = PixArtSigmaPipeline.from_pretrained(
            _local_snapshot('PixArt-alpha/PixArt-Sigma-XL-2-1024-MS'),
            torch_dtype=dtype, cache_dir=cache_dir)
        pipe.transformer = PixArtTransformer2DModel.from_pretrained(
            _local_snapshot(BACKBONES[name]), subfolder='transformer',
            torch_dtype=dtype, cache_dir=cache_dir)
        return pipe.to('cuda')
    elif name == 'sana':
        from diffusers import SanaPipeline
        pipe = SanaPipeline.from_pretrained(
            _local_snapshot(BACKBONES[name]), torch_dtype=dtype, cache_dir=cache_dir)
        return pipe.to('cuda')
    elif name == 'pixart':
        # PixArt-α 官方 diffusers 版（完整仓库，含自己的 T5 与 VAE）
        from diffusers import PixArtAlphaPipeline
        pipe = PixArtAlphaPipeline.from_pretrained(
            _local_snapshot(BACKBONES[name]), torch_dtype=dtype, cache_dir=cache_dir)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe.to('cuda')
    else:
        raise ValueError(f'未知骨干 {name}')

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to('cuda')


def run_diffusers_backbone(args):
    """按与 PixArt 分支相同的协议，用 diffusers pipeline 生成。"""
    import os as _os
    from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline,
                           DPMSolverMultistepScheduler)

    print(f'[backbone] {args.backbone} <- {BACKBONES[args.backbone]}', flush=True)
    pipe = _build_backbone_pipe(args.backbone, torch.float16,
                                _os.environ.get('HF_HOME'))
    pipe.set_progress_bar_config(disable=True)

    end_repeat = args.end_repeat if args.end_repeat is not None else args.images_per_prompt
    out_root = Path(args.out_root) / args.backbone
    out_root.mkdir(parents=True, exist_ok=True)

    cat_list = [c.strip() for c in args.categories.split(',') if c.strip()]
    for cat_idx, category in enumerate(cat_list):
        cat_out = out_root / category
        cat_out.mkdir(parents=True, exist_ok=True)
        prompts = [l.strip() for l in
                   (Path(args.dataset_dir) / f'{category}_val.txt').read_text().splitlines()
                   if l.strip()]
        if args.limit_prompts:
            prompts = prompts[:args.limit_prompts]

        jobs = [(r, p_i, p) for r in range(args.start_repeat, end_repeat)
                for p_i, p in enumerate(prompts)]
        todo = [j for j in jobs
                if not (cat_out / f'{j[2]}_{j[1] * args.images_per_prompt + j[0]:06d}.png').exists()]
        print(f'[backbone] {category}: {len(prompts)} prompts x repeats '
              f'{args.start_repeat}..{end_repeat-1} = {len(jobs)} images; '
              f'{len(jobs)-len(todo)} present, {len(todo)} to generate', flush=True)

        rows = []
        for s in range(0, len(todo), max(1, args.batch_size)):
            chunk = todo[s:s + args.batch_size]
            for (rep, p_i, prompt) in chunk:
                seed = image_seed(args.seed, cat_idx, p_i, rep)
                gen = torch.Generator('cuda').manual_seed(seed)
                img = pipe(prompt, num_inference_steps=args.steps,
                           guidance_scale=args.cfg_scale,
                           height=args.image_size, width=args.image_size,
                           generator=gen).images[0]
                gidx = p_i * args.images_per_prompt + rep
                name = f'{prompt}_{gidx:06d}.png'
                img.save(cat_out / name)
                rows.append(dict(category=category, prompt=prompt, prompt_index=p_i,
                                 repeat=rep, global_index=gidx, image=name))
            print(f'[backbone]   {category} {min(s+args.batch_size,len(todo))}/{len(todo)}',
                  flush=True)

        man = out_root / f'manifest_{category}.csv'
        write_header = not man.exists()
        with open(man, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['category', 'prompt', 'prompt_index',
                                               'repeat', 'global_index', 'image'])
            if write_header:
                w.writeheader()
            w.writerows(rows)
    print('[backbone] 完成', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default=None, choices=sorted(BACKBONES),
                    help='跨骨干对比：用 diffusers pipeline 生成（与 PixArt 分支同协议）')
    ap.add_argument('--method', required=True, choices=sorted(METHODS))
    ap.add_argument('--dataset-dir', default='/tmp/cb/T2I-CompBench/examples/dataset')
    ap.add_argument('--out-root', default='/root/compbench_work/images')
    ap.add_argument('--images-per-prompt', type=int, default=10,
                    help='total images per prompt (index space, for filenames)')
    ap.add_argument('--start-repeat', type=int, default=0)
    ap.add_argument('--end-repeat', type=int, default=None, help='exclusive')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--cfg-scale', type=float, default=4.0)
    ap.add_argument('--categories', default=','.join(CATEGORIES))
    ap.add_argument('--limit-prompts', type=int, default=0,
                    help='use only the first N prompts of each category (0 = all)')
    ap.add_argument('--embed-bind-alpha', type=float, default=None,
                    help='override the method config value (value-space binding tag)')
    ap.add_argument('--fix-mask', action='store_true',
                    help='ISSUE-011: restore the cross-attention key-padding mask '
                         'that the repo SDPA fallback silently drops.')
    ap.add_argument('--checkpoint', default=None,
                    help='override the METHODS checkpoint (e.g. an intermediate step)')
    ap.add_argument('--t5-cache', default=str(ROOT / 'output/pretrained_models/t5_ckpts'))
    ap.add_argument('--vae-path', default=str(ROOT / 'output/pretrained_models/sd-vae-ft-ema'))

    # ---- LCAR：局部竞争注意重分配（推理期，只改写支路自身 bias） -------------
    # 默认 lcar_enable=0，即完全不介入；所有默认值与现有行为等价，
    # 保证不带这些开关的老命令逐位不变。
    ap.add_argument('--lcar_enable', type=int, default=0)
    ap.add_argument('--lcar_mu', type=float, default=0.5)
    ap.add_argument('--lcar_rho', type=float, default=1.0,
                    help='半径倍率。半径定义为「锚点到支撑域 Ω_o 边界的距离」，'
                         'rho=1.0 即恰好取到对象边界（原 r=rho*sqrt(|Ω_o|) 在 '
                         '32x32 网格下会算出比整图还大的半径，已废弃）')
    ap.add_argument('--lcar_tau_rel', type=float, default=0.1)
    ap.add_argument('--lcar_lam', type=float, default=1.0)
    ap.add_argument('--lcar_kernel', default='gauss', choices=['gauss', 'hard'])
    ap.add_argument('--lcar_assign', default='soft', choices=['soft', 'hard'])
    ap.add_argument('--lcar_suppress_set', default='obj+attr',
                    choices=['obj', 'obj+attr', 'none'])
    ap.add_argument('--lcar_anchor_src', default='attr',
                    choices=['attr', 'obj', 'random'])
    ap.add_argument('--lcar_anchor_update', default='once',
                    choices=['once', 'every5', 'ema'])
    ap.add_argument('--lcar_mass_conserv', type=int, default=0)
    ap.add_argument('--lcar_layers', default='all', choices=['all', 'mid'])
    ap.add_argument('--lcar_calc_ratio', type=float, default=0.25)
    ap.add_argument('--lcar_eta', type=float, default=1.0)

    # ---- LCAR 空间渐变**替换式**（主干注意力重加权，与上面的加性偏置是两套）----
    # 把 reweight 模式下全图统一的 content_target 换成按锚点渐变的场：
    #   锚点处 t_center，沿半径线性降到 r 处的 t_edge，半径之外为 0（不替换）。
    # 挂在 --method tpscda_nd_rw80 上时，可直接与 rw80 的标量版本对照。
    ap.add_argument('--lcar_replace_graded', type=int, default=0)
    ap.add_argument('--lcar_replace_t_center', type=float, default=0.8)
    ap.add_argument('--lcar_replace_t_edge', type=float, default=0.1)
    ap.add_argument('--lcar_replace_exclude_nonattr', type=int, default=0,
                    help='从替换目标里剔除非外观属性的词（数词/方位词/关系虚词/冠词）；'
                         '其余 token 照常参与替换与渐变')
    ap.add_argument('--lcar_replace_additive', type=int, default=0,
                    help='路2：加性替换——只抬角色 token、不重归一化（其他 token 逐位不动）')
    ap.add_argument('--lcar_replace_delta_max', type=float, default=0.2,
                    help='加性替换的整行增量上限 δ：行和变为 1+rise，rise∈[0,δ]')
    ap.add_argument('--lcar_replace_window', default=None,
                    help='替换的时间窗 "lo,hi"，用归一化进度表示：p = 1 − t/t_max，'
                         '0=最早期(噪声最大)、1=最末期(接近成图)。默认全程。'
                         '例：--lcar_replace_window 0.5,1.0 表示只在后半程替换')
    ap.add_argument('--lcar_tag', default='',
                    help='本次 run 的标签，用于写 lcar_configs/<tag>.json')
    args = ap.parse_args()
    if args.backbone:
        return run_diffusers_backbone(args)

    # 走到这里说明是 PixArt 主线，此时才需要那批专有模块
    _lazy_pixart_imports()

    cfg = METHODS[args.method]
    if args.embed_bind_alpha is not None:
        cfg = dict(cfg, embed_bind_alpha=args.embed_bind_alpha)
    if args.checkpoint:
        cfg = dict(cfg, checkpoint=args.checkpoint)
        print(f'checkpoint override -> {args.checkpoint}')
    if args.fix_mask:
        sys.path.insert(0, str(ROOT / 'results/_work'))
        import mask_fix
        mask_fix.apply()
        print('ISSUE-011 mask fix APPLIED (runtime monkey-patch, repo unmodified)')
    end_repeat = args.end_repeat if args.end_repeat is not None else args.images_per_prompt
    device = 'cuda'
    dtype = torch.float16

    out_root = Path(args.out_root) / args.method
    out_root.mkdir(parents=True, exist_ok=True)
    log = open(out_root / 'generation.log', 'a', buffering=1)

    def say(msg):
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{stamp}] {msg}', flush=True)
        log.write(f'[{stamp}] {msg}\n')

    say(f'=== method={args.method} repeats {args.start_repeat}..{end_repeat-1} '
        f'of {args.images_per_prompt}, batch={args.batch_size} ===')
    say(f'checkpoint={cfg["checkpoint"]}')
    torch.manual_seed(args.seed)
    latent_size = args.image_size // 8

    model = PixArt_XL_2(
        input_size=latent_size,
        lewei_scale=0.5 if args.image_size == 256 else 1,
        semantic_conditioning=cfg['semantic_conditioning'],
        semantic_adapter_dim=64,
        semantic_dropout=0,
        semantic_residual_scale=cfg['semantic_residual_scale'],
        semantic_token_attention=cfg['semantic_token_attention'],
        semantic_token_gate_max=cfg['semantic_token_gate_max'],
        semantic_token_gate_activation=cfg.get('semantic_token_gate_activation', 'tanh'),
        semantic_pair_edge_gate=cfg.get('semantic_pair_edge_gate', False),
        semantic_time_window=cfg.get('semantic_time_window'),
        semantic_random_probe=cfg.get('semantic_random_probe', False),
        cross_attn_pair_replace=cfg.get('cross_attn_pair_replace', False),
        cross_attn_pair_replace_strength=cfg.get('cross_attn_pair_replace_strength', 1.0),
        cross_attn_pair_replace_mode=cfg.get('cross_attn_pair_replace_mode', 'replace'),
        # These four were set in METHOD entries but never forwarded, so every
        # such run silently used the model defaults instead (content_target
        # 0.5, fund proportional, renorm on, sink_factor 0).  Forward them so a
        # METHOD entry actually means what it says.
        cross_attn_pair_replace_fund=cfg.get('cross_attn_pair_replace_fund', 'proportional'),
        cross_attn_pair_replace_content_target=cfg.get('cross_attn_pair_replace_content_target', 0.5),
        cross_attn_pair_replace_renorm=cfg.get('cross_attn_pair_replace_renorm', True),
        cross_attn_pair_replace_gate_gamma=cfg.get('cross_attn_pair_replace_gate_gamma', 1.0),
        cross_attn_pair_replace_sink_factor=cfg.get('cross_attn_pair_replace_sink_factor', 0.0),
    ).to(device, dtype=dtype).eval()
    state = torch.load(cfg['checkpoint'], map_location='cpu')
    state = state.get('state_dict', state)
    state.pop('pos_embed', None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    say(f'checkpoint loaded; missing={len(missing)} unexpected={len(unexpected)}')
    del state
    torch.cuda.empty_cache()

    # ---- LCAR：挂到支路模块上 ------------------------------------------------
    # 只改支路自身的 pair bias；主干 QKV/权重不动，不新增可训练参数。
    # lcar_enable=0（默认）时不挂载，行为与打补丁前逐位相同。
    lcar_cfg = None
    lcar_state = None
    if args.lcar_enable or args.lcar_replace_graded:
        from diffusion.model.nets.lcar import LCARConfig, LCARState, attach
        lcar_cfg = LCARConfig(
            enable=bool(args.lcar_enable), mu=args.lcar_mu, rho=args.lcar_rho,
            tau_rel=args.lcar_tau_rel, lam=args.lcar_lam,
            kernel=args.lcar_kernel, assign=args.lcar_assign,
            suppress_set=args.lcar_suppress_set, anchor_src=args.lcar_anchor_src,
            anchor_update=args.lcar_anchor_update,
            mass_conserv=bool(args.lcar_mass_conserv), eta=args.lcar_eta,
            layers=args.lcar_layers,
            # 时间窗实验：让锚点场在任何步都能算出（否则窗在中后段时场根本不会被计算），
            # 由 PixArt.py 的 in_replace_window 决定该步是否真的施加。
            calc_ratio=(1.0 if args.lcar_replace_window else args.lcar_calc_ratio),
            replace_additive=bool(args.lcar_replace_additive),
            replace_delta_max=args.lcar_replace_delta_max,
            replace_window=(tuple(float(x) for x in args.lcar_replace_window.split(','))
                            if args.lcar_replace_window else None),
            replace_graded=bool(args.lcar_replace_graded),
            replace_t_center=args.lcar_replace_t_center,
            replace_t_edge=args.lcar_replace_t_edge)
        n_attached, lcar_state = attach(model, lcar_cfg)
        say(f'LCAR attached to {n_attached} branch modules: {lcar_cfg.as_dict()}')
        # run 的完整配置落盘，供 lcar_configs/*.json 与回填核对
        tag = args.lcar_tag or ('lcar_mu%03d_rho%d'
                                % (int(round(args.lcar_mu * 100)), args.lcar_rho))
        cfg_dir = Path('/root/compbench_work/汇总/lcar_configs')
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / f'{tag}.json').write_text(json.dumps(
            dict(tag=tag, method=args.method, categories=args.categories,
                 seed=args.seed, steps=args.steps, cfg_scale=args.cfg_scale,
                 images_per_prompt=args.images_per_prompt,
                 replace_exclude_nonattr=int(args.lcar_replace_exclude_nonattr),
                 checkpoint=cfg['checkpoint'], **lcar_cfg.as_dict()),
            ensure_ascii=False, indent=2), encoding='utf-8')

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, dtype=dtype).eval()
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_cache,
                    torch_dtype=torch.float, model_max_length=120)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.t5_cache + '/t5-v1_1-xxl',
                                              use_fast=True, local_files_only=True)
    import spacy
    nlp = spacy.load('en_core_web_sm')

    cat_list = [c.strip() for c in args.categories.split(',') if c.strip()]
    t_start = time.time()
    for cat_idx, category in enumerate(cat_list):
        cat_out = out_root / category
        cat_out.mkdir(parents=True, exist_ok=True)
        prompts = [l.strip() for l in
                   (Path(args.dataset_dir) / f'{category}_val.txt').read_text().splitlines()
                   if l.strip()]
        if args.limit_prompts:
            prompts = prompts[:args.limit_prompts]

        # repeat-major: complete k-image coverage after every stage
        jobs = [(r, p_i, p) for r in range(args.start_repeat, end_repeat)
                for p_i, p in enumerate(prompts)]
        todo = [j for j in jobs
                if not (cat_out / f'{j[2]}_{j[1] * args.images_per_prompt + j[0]:06d}.png').exists()]
        say(f'{category}: {len(prompts)} prompts x repeats '
            f'{args.start_repeat}..{end_repeat-1} = {len(jobs)} images; '
            f'{len(jobs) - len(todo)} present, {len(todo)} to generate')
        if not todo:
            continue

        rows = []
        with torch.inference_mode():
            for s in range(0, len(todo), args.batch_size):
                chunk = todo[s:s + args.batch_size]
                chunk_prompts = [c[2] for c in chunk]
                emb, text_mask = t5.get_text_embeddings(chunk_prompts)
                emb = emb[:, None].to(device=device, dtype=dtype)
                _eb_alpha = float(cfg.get('embed_bind_alpha', 0.0))
                if _eb_alpha > 0:
                    # Value-space binding tag (see tools/generate_scda_samples.py):
                    # add a share of the noun's embedding to each bound attribute
                    # token, before the DiT sees it.  Attention weights, the sink
                    # token and the model's spatial assignment are all untouched.
                    _eb_mode = str(cfg.get('embed_bind_mode', 'add'))
                    for _b, _p in enumerate(chunk_prompts):
                        _edges, _n = build_semantic_edges(_p, nlp, tokenizer,
                                                          emb.shape[2], MAX_SEMANTIC_EDGES)
                        _objs = {int(o) for o, _ in _edges[:_n]}
                        _order = sorted(_objs)
                        _rank = {o: k for k, o in enumerate(_order)}
                        _pscale = float(cfg.get('embed_bind_position_scale', 1.0))
                        _filter = str(cfg.get('embed_bind_filter', 'all'))
                        _cidx = (color_attribute_indices(_p, nlp, tokenizer, emb.shape[2])
                                 if _filter == 'color' else None)
                        for _o, _a in _edges[:_n]:
                            _o, _a = int(_o), int(_a)
                            if _cidx is not None and _a not in _cidx:
                                continue
                            _a_eff = _eb_alpha * (_pscale ** _rank.get(_o, 0))
                            _vec = emb[_b, 0, _a].clone()
                            if _eb_mode == 'normalize':
                                _mix = (1 - _a_eff) * _vec + _a_eff * emb[_b, 0, _o]
                                emb[_b, 0, _a] = _mix * (_vec.norm() / _mix.norm().clamp_min(1e-6))
                            elif _eb_mode == 'contrast':
                                _push = emb[_b, 0, _o].clone()
                                for _j in _objs:
                                    if _j != _o:
                                        _push = _push - emb[_b, 0, _j]
                                emb[_b, 0, _a] = _vec + _a_eff * _push
                            elif _eb_mode == 'scale':
                                emb[_b, 0, _a] = _vec * (1.0 + _a_eff)
                            else:
                                emb[_b, 0, _a] = _vec + _a_eff * emb[_b, 0, _o]
                null = model.y_embedder.y_embedding[None].repeat(
                    len(chunk_prompts), 1, 1)[:, None].to(dtype)
                hw = torch.full((len(chunk_prompts), 2), args.image_size,
                                device=device, dtype=dtype)
                ar = torch.ones((len(chunk_prompts), 1), device=device, dtype=dtype)
                data_info = {'img_hw': hw, 'aspect_ratio': ar}
                # LCAR 替换目标 token 的排除掩码 [B, L]：把非外观属性的词
                # （数词、方位词、关系虚词、冠词）从替换目标里剔除。
                # 机制上：这些词被抬升会把计数 / 深度结构 / 关系语义压垮，
                # 而 color/shape/texture 的角色 token 全是真正的外观属性，
                # 不受影响。其余 token 照常参与替换与渐变。
                if args.lcar_replace_exclude_nonattr:
                    from diffusion.model.nets.lcar import build_exclude_mask
                    _exc = []
                    for _p in chunk_prompts:
                        _ids = tokenizer(_p, max_length=emb.shape[2],
                                         truncation=True)['input_ids']
                        _toks = tokenizer.convert_ids_to_tokens(_ids)
                        _exc.append(build_exclude_mask(_toks, emb.shape[2]))
                    data_info['lcar_replace_exclude'] = torch.stack(_exc).to(device)
                # Role masks are the pair-replace mechanism's INPUT, not an
                # optional extra of the semantic branch: gating them on
                # `semantic_conditioning` alone is what silently disabled the
                # mechanism for the whole 2026-09-20 pair_replace evaluation.
                if cfg['semantic_conditioning'] or cfg.get('cross_attn_pair_replace'):
                    masks = torch.stack([
                        torch.from_numpy(
                            build_semantic_masks(p, nlp, tokenizer, emb.shape[2]))
                        for p in chunk_prompts
                    ]).to(device=device, dtype=dtype)
                    data_info['semantic_token_masks'] = masks
                if cfg.get('semantic_pair_edge_gate') or cfg.get('cross_attn_pair_replace'):
                    pairs = [build_semantic_edges(p, nlp, tokenizer, emb.shape[2],
                                                  MAX_SEMANTIC_EDGES)
                             for p in chunk_prompts]
                    data_info['semantic_edges'] = torch.from_numpy(
                        np.stack([e for e, _ in pairs])).to(device=device)
                    data_info['semantic_edge_count'] = torch.from_numpy(
                        np.asarray([c for _, c in pairs])).to(device=device)

                # Deterministic per-image noise.
                noise = torch.stack([
                    torch.randn(4, latent_size, latent_size, device=device, dtype=dtype,
                                generator=torch.Generator(device=device).manual_seed(
                                    image_seed(args.seed, cat_idx, c[1], c[0])))
                    for c in chunk
                ])
                # LCAR 的锚点缓存在一次去噪链内复用，换图必须清空，
                # 否则会串用上一张图的锚点。
                if lcar_state is not None:
                    lcar_state.reset()
                solver = DPMS(model.forward_with_dpmsolver, condition=emb,
                              uncondition=null, cfg_scale=args.cfg_scale,
                              model_kwargs=dict(data_info=data_info, mask=text_mask))
                latents = solver.sample(noise, steps=args.steps, order=2,
                                        skip_type='time_uniform', method='multistep')
                images = vae.decode((latents / 0.18215).to(dtype)).sample
                for (rep, p_i, prompt), image in zip(chunk, images):
                    gidx = p_i * args.images_per_prompt + rep
                    name = f'{prompt}_{gidx:06d}.png'
                    save_image(image.float(), cat_out / name,
                               normalize=True, value_range=(-1, 1))
                    rows.append(dict(category=category, prompt=prompt,
                                     prompt_index=p_i, repeat=rep,
                                     global_index=gidx, image=name))
                done = min(s + args.batch_size, len(todo))
                if (s // args.batch_size) % 10 == 0 or done == len(todo):
                    el = time.time() - t_start
                    rate = el / max(done, 1)
                    say(f'  {category} {done}/{len(todo)}  elapsed {el/60:.1f} min  '
                        f'{rate:.2f} s/img  eta_cat {(len(todo)-done)*rate/60:.0f} min')

        man = out_root / f'manifest_{category}.csv'
        write_header = not man.exists()
        with open(man, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['category', 'prompt', 'prompt_index',
                                               'repeat', 'global_index', 'image'])
            if write_header:
                w.writeheader()
            w.writerows(rows)

        meta = dict(
            experiment_name=f'compbench_{args.method}',
            model_name='PixArt-XL-2-512x512',
            checkpoint=cfg['checkpoint'],
            method=args.method,
            semantic_conditioning=cfg['semantic_conditioning'],
            semantic_token_attention=cfg['semantic_token_attention'],
            semantic_token_gate_max=cfg['semantic_token_gate_max'],
            semantic_token_gate_activation=cfg.get('semantic_token_gate_activation', 'tanh'),
            semantic_residual_scale=cfg['semantic_residual_scale'],
            semantic_pair_edge_gate=bool(cfg.get('semantic_pair_edge_gate', False)),
            semantic_time_window=cfg.get('semantic_time_window'),
            semantic_random_probe=bool(cfg.get('semantic_random_probe', False)),
            max_semantic_edges=MAX_SEMANTIC_EDGES if cfg.get('semantic_pair_edge_gate') else None,
            prompt_file=str(Path(args.dataset_dir)),
            prompt_version='T2I-CompBench++ val splits (300 prompts per category)',
            seed=args.seed,
            seed_scheme='per-image: (seed*1000003 + cat_idx*10007 + prompt_idx*101 + repeat) % (2^31-1)',
            resolution=args.image_size,
            sampler='DPM-Solver (order=2, multistep, time_uniform)',
            num_steps=args.steps,
            CFG=args.cfg_scale,
            images_per_prompt_target=args.images_per_prompt,
            repeats_this_run=[args.start_repeat, end_repeat],
            batch_size=args.batch_size,
            T5_version='t5-v1_1-xxl',
            evaluator='T2I-CompBench++ official evaluators',
            mask_fix_applied_for_issue_011='permanent repo fix in '
                                            'diffusion/model/nets/PixArt_blocks.py '
                                            '(cross_attn_key_padding); --fix-mask '
                                            f'runtime patch {bool(args.fix_mask)}',
            GPU=torch.cuda.get_device_name(0),
            CUDA_version=torch.version.cuda,
            PyTorch_version=torch.__version__,
            Python_version=sys.version.split()[0],
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
            total_seconds=round(time.time() - t_start, 1),
        )
        (out_root / f'config_run_{args.start_repeat}_{end_repeat-1}.json').write_text(
            json.dumps(meta, indent=2, ensure_ascii=False))
    say(f'DONE {args.method} repeats {args.start_repeat}..{end_repeat-1} '
        f'in {(time.time() - t_start)/3600:.2f} h')


if __name__ == '__main__':
    main()
