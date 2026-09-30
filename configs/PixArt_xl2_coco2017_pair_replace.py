"""TP-SCDA with the injection mechanism REPLACED by direct rewriting of the
DiT's own cross-attention weights.

Why the original injection could not work
-----------------------------------------
The original mechanism adds a pair-aware bias inside a SEPARATE attention
module (`SemanticTokenCrossAttention`).  That module builds its query from the
patch hidden states, whose norm is ~668 (the residual stream accumulates over 28
blocks), so `q_proj(x)` has std ~46 and the attention logits reach ~2766 while
the pair bias contributes 0.049.  The softmax is therefore one-hot and any
additive bias is annihilated -- measured: changing the pair matrix, and even
forcing `pair_strength` to -20, changes the model output by EXACTLY zero.

The design intent -- "give every token in an object's pair the same weight as
the noun" -- cannot exist at that logit scale.

What this config does instead
-----------------------------
The parsed object->attribute structure is applied to the weights the DiT ALREADY
computes:

    w = softmax(DiT cross-attention logits)
    new_a = w @ edge_matrix            # attribute takes the NOUN's own weight
    w[attribute positions] = new_a
    w = w / w.sum(-1)                  # remaining weights rescaled proportionally

No separate q/k, so no logit-scale problem; no additive bias, so nothing to
drown.  The object and its bound attributes end up with the same attention
weight, which is the stated design intent.

NOTE ON SCOPE: this modifies the DiT's original cross-attention, which the paper
describes as kept unchanged ("在保持原始交叉注意力不变的前提下").  It is a
deliberate departure, requested explicitly, and is recorded as such.

The separate semantic token branch is switched off entirely (both the token
path and the pooled path), so this run isolates the replacement mechanism.
Trained from the untouched PixArt-alpha base, 1 epoch, no distillation, so it is
comparable with `PixArt_xl2_coco2017_scda_token_pair` and the gate-optimised run.
"""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_learnable_layers.py', encoding='utf-8').read(),
             '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_learnable_layers.py', 'exec'))

experiment_name = 'coco2017_pair_replace'
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_pair_replace'

# --- the mechanism change ----------------------------------------------------
cross_attn_pair_replace = True      # apply the pair structure to the DiT's own weights
semantic_conditioning = False       # switch off the pooled path
semantic_token_attention = False    # switch off the separate token branch

# The replacement mechanism adds NO parameters of its own -- it rewrites weights
# the DiT already computes -- so there are no adapters to train.  Freezing
# everything (train_semantic_only) would leave nothing trainable, and unfreezing
# everything would silently turn this into a full PixArt fine-tune, a different
# experiment from every other row in the table (all of which adapted a frozen
# backbone).  Adapt exactly the modules the mechanism touches instead.
train_semantic_only = False
trainable_contains = ('cross_attn.',)

# --- keep the run otherwise comparable ---------------------------------------
distill_coef = 0.0
load_from = '/root/private_data/PixArt-alpha-attentiongate/output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'
num_epochs = 1

# the dataset must supply the parsed edges
data = dict(data, load_semantic_edges=True,
            semantic_edge_index='/root/private_data/data/COCO2017Prepared/partition/edge_index.npz',
            max_semantic_edges=24)

save_model_steps = 7000
