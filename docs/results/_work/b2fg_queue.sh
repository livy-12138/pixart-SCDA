#!/bin/bash
# b2_frozengate: the B2 binding objective with the gate FROZEN at the neutral
# strength for the whole run, so the branch cannot shrink its way out of the job.
#
# Rationale: the semantic branch is a STRONG intervention (turning it off moves
# the DiT output by ~15% of its RMS -- results/branch_effect_measurement.md), so
# it has ample influence; the problem is direction, not magnitude.  Yet the gate
# drifts DOWN in every run so far (0.150 -> 0.063 without B2, 0.150 -> 0.068 with
# B2): the optimiser shrinks the branch as its easiest route to a lower diffusion
# loss.  If that freedom is what lets the branch evade the job, remove it.  Here
# the gate is frozen at the paper checkpoint's neutral strength (0.0160) for the
# entire run while the alignment objective stays on, so the branch must either
# become useful for binding or learn to emit something harmless.
#
# The pair (b2_neutral, b2_frozengate) isolates one variable: whether the gate
# may move at all.
#
# Waits for the b2_neutral chain to finish first.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

until grep -q "TSPROBE_DONE" "$L/tsprobe_queue.log" 2>/dev/null; do
  if ! pgrep -f "[t]sprobe_queue.sh" >/dev/null 2>&1; then
    echo "WARNING: the timestep-probe chain is gone without TSPROBE_DONE $(date -Is)" >&2
    break
  fi
  sleep 300
done
echo "########## starting b2_frozengate $(date -Is) ##########"
sleep 60

CKPT=output/coco2017_token_pair_b2_frozengate/checkpoints/epoch_1_step_14786.pth
echo "########## TRAIN b2_frozengate start $(date -Is) ##########"
PYTHONPATH=. accelerate launch --num_processes 1 train_scripts/train.py \
    configs/PixArt_xl2_coco2017_token_pair_b2_frozengate.py
echo "########## TRAIN b2_frozengate rc=$? $(date -Is) ##########"

if [ ! -f "$CKPT" ]; then
  echo "########## ABORT: no final checkpoint ($CKPT) $(date -Is) ##########"
  exit 1
fi

echo "########## COMPBENCH GENERATE b2_frozengate $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method b2_frozengate --batch-size 16 \
    --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## COMPBENCH EVALUATE b2_frozengate $(date -Is) ##########"
cd results/_work
for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  echo "----- b2_frozengate / $c $(date -Is) -----"
  timeout 10800 python3 run_eval.py --method b2_frozengate --category "$c" \
      --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_b2_frozengate_${c}.log" 2>&1
  echo "  rc=$? $(date -Is)"
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## 64-PROMPT GENERATE b2_frozengate $(date -Is) ##########"
for seed in 43 44 45; do
  python3 tools/generate_scda_samples.py \
    --checkpoint "$CKPT" \
    --groups-dir output/coco2017_eval_learnable_layers/prompt_groups_64 \
    --output-dir "output/multiseed_eval_fixed/b2_frozengate/seed_${seed}" \
    --image-size 512 --steps 20 --cfg-scale 4.0 --seed "$seed" \
    --semantic-conditioning --semantic-token-attention \
    --semantic-token-gate-max 0.3 --semantic-token-gate-activation sigmoid \
    --semantic-residual-scale 0.0 2>&1 \
    | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
done

echo "########## SCORE + PAIRED TEST $(date -Is) ##########"
EV="baseline=output/multiseed_eval_fixed coco2017_scda=output/multiseed_eval_fixed \
    fullspan_final=output/multiseed_eval_fixed token_pair=output/multiseed_eval_fixed \
    learnable_layers=output/multiseed_eval_fixed gateopt=output/multiseed_eval_fixed \
    b2=output/multiseed_eval_fixed b2=output/multiseed_eval_fixed b2_neutral=output/multiseed_eval_fixed b2_frozengate=output/multiseed_eval_fixed"
python3 tools/score_clip_multiseed.py --evaluations $EV \
  --output output/tables234_fixed/final_multiseed_clip_scores.csv 2>&1 | tail -2
python3 tools/score_attribute_binding.py --evaluations $EV \
  --output output/tables234_fixed/attribute_binding_metrics_all.csv 2>&1 | tail -2
python3 tools/summarize_binding_comparison.py \
  --input output/tables234_fixed/attribute_binding_metrics_all.csv \
  --output output/tables234_fixed/attribute_binding_metrics_all_bootstrap.csv 2>&1 | tail -1
python3 results/_work/compbench_paired_test.py --methods tpscda gateopt b2 b2_neutral b2_frozengate 2>&1 | tail -30

echo "PROBE_DONE $(date -Is)"
