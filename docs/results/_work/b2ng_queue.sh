#!/bin/bash
# b2_neutral: the B2 binding objective started from the historical (neutral)
# gate level instead of the open one.
#
# Rationale: injection strength and the binding metrics are monotonically
# related in the wrong direction (0 -> baseline, 0.0164 -> ~baseline,
# 0.063 -> significantly worse).  The first B2 run starts at 0.150, i.e. deep
# inside the range that has been measured to hurt, so any benefit from the
# binding objective would be fighting a handicap the config imposed on itself.
# This run starts at 0.0160 and keeps the ceiling and the gate learning rate,
# so the optimiser may RAISE the gate if the objective makes the branch useful.
# The direction of the gate drift is the readout.
#
# Waits for the first B2 queue (training + both evaluations) to finish.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

until grep -q "B2_QUEUE_DONE" "$L/b2_queue.log" 2>/dev/null; do
  if ! pgrep -f "[b]2_queue.sh" >/dev/null 2>&1; then
    echo "WARNING: b2_queue.sh is gone without B2_QUEUE_DONE $(date -Is)" >&2
    break
  fi
  sleep 120
done
echo "########## starting b2_neutral $(date -Is) ##########"
sleep 60

CKPT=output/coco2017_token_pair_b2_neutralgate/checkpoints/epoch_1_step_14786.pth
echo "########## TRAIN b2_neutralgate start $(date -Is) ##########"
PYTHONPATH=. accelerate launch --num_processes 1 train_scripts/train.py \
    configs/PixArt_xl2_coco2017_token_pair_b2_neutralgate.py
echo "########## TRAIN b2_neutralgate rc=$? $(date -Is) ##########"

if [ ! -f "$CKPT" ]; then
  echo "########## ABORT: no final checkpoint ($CKPT) $(date -Is) ##########"
  exit 1
fi

echo "########## COMPBENCH GENERATE b2_neutral $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method b2_neutral --batch-size 16 \
    --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## COMPBENCH EVALUATE b2_neutral $(date -Is) ##########"
cd results/_work
for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  echo "----- b2_neutral / $c $(date -Is) -----"
  timeout 10800 python3 run_eval.py --method b2_neutral --category "$c" \
      --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_b2_neutral_${c}.log" 2>&1
  echo "  rc=$? $(date -Is)"
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## 64-PROMPT GENERATE b2_neutral $(date -Is) ##########"
for seed in 43 44 45; do
  python3 tools/generate_scda_samples.py \
    --checkpoint "$CKPT" \
    --groups-dir output/coco2017_eval_learnable_layers/prompt_groups_64 \
    --output-dir "output/multiseed_eval_fixed/b2_neutral/seed_${seed}" \
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
    b2=output/multiseed_eval_fixed b2_neutral=output/multiseed_eval_fixed"
python3 tools/score_clip_multiseed.py --evaluations $EV \
  --output output/tables234_fixed/final_multiseed_clip_scores.csv 2>&1 | tail -2
python3 tools/score_attribute_binding.py --evaluations $EV \
  --output output/tables234_fixed/attribute_binding_metrics_all.csv 2>&1 | tail -2
python3 tools/summarize_binding_comparison.py \
  --input output/tables234_fixed/attribute_binding_metrics_all.csv \
  --output output/tables234_fixed/attribute_binding_metrics_all_bootstrap.csv 2>&1 | tail -1
python3 results/_work/compbench_paired_test.py --methods tpscda gateopt b2 b2_neutral 2>&1 | tail -30

echo "B2NG_QUEUE_DONE $(date -Is)"
