#!/bin/bash
# Re-run tables 2/3/4 (64 fixed prompts x 3 seeds x 5 methods) with the
# ISSUE-011 cross-attention mask fix in place.  Only the mask changes -- every
# checkpoint and flag is exactly the one used for the original tables, so the
# difference in the numbers is attributable to the bug fix alone.
#
# Results land in output/multiseed_eval_fixed/<label>/seed_<seed>/.
set -u
cd /root/private_data/PixArt-alpha-attentiongate

CK_ROOT=output
GEN=tools/generate_scda_samples.py
GRPDIR=output/coco2017_eval_learnable_layers/prompt_groups_64
OUT=output/multiseed_eval_fixed

run () {
  local label="$1"; shift
  local ckpt="$1"; shift
  for seed in 43 44 45; do
    echo "===== $label seed=$seed $(date -Is) ====="
    python3 "$GEN" --checkpoint "$ckpt" --groups-dir "$GRPDIR" \
        --output-dir "$OUT/$label/seed_$seed" \
        --image-size 512 --steps 20 --cfg-scale 4.0 --seed "$seed" "$@" 2>&1 \
      | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
  done
}

run baseline     $CK_ROOT/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth

run coco2017_scda $CK_ROOT/coco2017_scda/checkpoints/epoch_5_step_18485.pth \
    --semantic-conditioning --semantic-residual-scale 0.25

run fullspan_final $CK_ROOT/coco2017_scda_fullspan/checkpoints/epoch_6_step_22182.pth \
    --semantic-conditioning --semantic-residual-scale 0.25

run token_pair   $CK_ROOT/coco2017_scda_token_pair_positive_mb8_acc4/checkpoints/epoch_5_step_60000.pth \
    --semantic-conditioning --semantic-token-attention --semantic-token-gate-max 1.0 \
    --semantic-residual-scale 0.0

run learnable_layers $CK_ROOT/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth \
    --semantic-conditioning --semantic-token-attention --semantic-token-gate-max 0.08 \
    --semantic-residual-scale 0.0

echo "TABLES234_GEN_DONE"
find "$OUT" -name '*.png' | wc -l
