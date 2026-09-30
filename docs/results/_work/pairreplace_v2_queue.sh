#!/bin/bash
# RE-RUN of the pair-replace experiment, 2026-09-20.
#
# The first run (12:06-16:17) trained and evaluated a model whose mechanism was
# NEVER ACTIVE: `self.cross_attn_pair_replace` was assigned inside
# `if self.semantic_token_attention_enabled:` in PixArt.py, and this config turns
# that branch off, so the flag was never set and the rewrite never ran -- during
# training or evaluation.  Those numbers describe a plain cross-attention
# fine-tune; that checkpoint is kept as the control
# (output/coco2017_pair_replace_ft_only).
#
# Fixed since: the flag is armed unconditionally, the edge matrix carries the
# head axis it needs, and it follows the CFG batch.  Smoke-tested with
# PAIR_DEBUG=1 to confirm the rewrite executes in the training path.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs
LOG="$L/pairreplace_v2_queue.log"
exec >> "$LOG" 2>&1

CK=output/coco2017_pair_replace/checkpoints/epoch_1_step_14786.pth

echo "########## TRAIN pair_replace (mechanism ACTIVE) $(date -Is) ##########"
PYTHONPATH=. accelerate launch --num_processes 1 train_scripts/train.py \
    configs/PixArt_xl2_coco2017_pair_replace.py
echo "########## TRAIN rc=$? $(date -Is) ##########"
if [ ! -f "$CK" ]; then echo "ABORT: no checkpoint ($CK) $(date -Is)"; exit 1; fi

# Freshness guard.  The generator SKIPS images that already exist, so a rerun
# that leaves the previous model's pictures in place silently evaluates the
# OLD model: the 2026-09-20 rerun hit exactly this, with 3100 stale images from
# the mechanism-off run waiting to be re-scored.
IMG=/root/compbench_work/images/pair_replace
if [ -d "$IMG" ] && [ -n "$(find "$IMG" -name '*.png' -print -quit 2>/dev/null)" ]; then
    echo "ABORT: $IMG already contains images -- archive them first $(date -Is)"
    exit 1
fi

echo "########## COMPBENCH GENERATE (7 categories, 1 img/prompt) $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method pair_replace --batch-size 16 \
    --categories color,shape,texture,spatial,3d_spatial,numeracy,non_spatial \
    --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

# complex is scored by 3_in_1.py, which hardcodes num=10 images per prompt.
echo "########## COMPBENCH GENERATE complex (100 prompts x 10) $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method pair_replace --batch-size 16 \
    --categories complex --limit-prompts 100 --start-repeat 0 --end-repeat 10 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## COMPBENCH EVALUATE $(date -Is) ##########"
cd results/_work
for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  echo "----- pair_replace / $c $(date -Is) -----"
  # complex MUST be evaluated with --limit 100: its images are 100 prompts x 10,
  # while 3_in_1.py derives the needed count from the prompt list.  Without the
  # limit it is handed 300 prompts for 1000 images and fails outright.
  if [ "$c" = "complex" ]; then LIMIT="--limit 100"; else LIMIT=""; fi
  timeout 10800 python3 run_eval.py --method pair_replace --category "$c" $LIMIT \
      --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_pairreplace_v2_${c}.log" 2>&1
  echo "  rc=$? $(date -Is)"
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## RESULT $(date -Is) ##########"
python3 results/_work/compbench_paired_test.py --baseline frozen --methods pair_replace 2>&1 | tail -11
python3 results/_work/compbench_paired_test.py --baseline tpscda --methods pair_replace 2>&1 | tail -11
echo "PAIRREPLACE_V2_DONE $(date -Is)"
