#!/bin/bash
# B2 (binding alignment loss) only.
#
# B1 was cancelled on 2026-09-19: its measured effect on the model output is
# 0.04%-0.35%, and its training trajectory is numerically identical to the
# gateopt run it is based on.  See results/b1b2/B1_EFFECT_SIZE.md.  B1 cannot
# close a 0.03 gap to the frozen baseline when its own effect is two orders of
# magnitude smaller than that, so running its full evaluation would only spend
# GPU time to confirm a known null.
#
# B2 is the one mechanism that changes the TRAINING OBJECTIVE rather than the
# forward-pass structure, so it is not subject to the softplus(pair_strength) x
# gate-scale chain that makes B1 weak.  It is trained for one full epoch
# (14 786 micro-steps, same length as every other run) and scored under both
# standard protocols.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

B2=output/coco2017_token_pair_b2_bindingloss/checkpoints/epoch_1_step_14786.pth

echo "########## TRAIN b2_bindingloss start $(date -Is) ##########"
PYTHONPATH=. accelerate launch --num_processes 1 train_scripts/train.py \
    configs/PixArt_xl2_coco2017_token_pair_b2_bindingloss.py
echo "########## TRAIN b2_bindingloss rc=$? $(date -Is) ##########"

if [ ! -f "$B2" ]; then
  echo "########## ABORT: B2 produced no final checkpoint ($B2); skipping evaluation $(date -Is) ##########"
  exit 1
fi

echo "########## COMPBENCH GENERATE b2 $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method b2 --batch-size 16 \
    --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## COMPBENCH EVALUATE b2 $(date -Is) ##########"
cd results/_work
for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  echo "----- b2 / $c $(date -Is) -----"
  timeout 10800 python3 run_eval.py --method b2 --category "$c" \
      --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_b2_${c}.log" 2>&1
  echo "  rc=$? $(date -Is)"
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## 64-PROMPT GENERATE b2 $(date -Is) ##########"
for seed in 43 44 45; do
  python3 tools/generate_scda_samples.py \
    --checkpoint "$B2" \
    --groups-dir output/coco2017_eval_learnable_layers/prompt_groups_64 \
    --output-dir "output/multiseed_eval_fixed/b2/seed_${seed}" \
    --image-size 512 --steps 20 --cfg-scale 4.0 --seed "$seed" \
    --semantic-conditioning --semantic-token-attention \
    --semantic-token-gate-max 0.3 --semantic-token-gate-activation sigmoid \
    --semantic-residual-scale 0.0 2>&1 \
    | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
done

echo "########## SCORE b2 $(date -Is) ##########"
EV="baseline=output/multiseed_eval_fixed coco2017_scda=output/multiseed_eval_fixed \
    fullspan_final=output/multiseed_eval_fixed token_pair=output/multiseed_eval_fixed \
    learnable_layers=output/multiseed_eval_fixed gateopt=output/multiseed_eval_fixed \
    b2=output/multiseed_eval_fixed"
python3 tools/score_clip_multiseed.py --evaluations $EV \
  --output output/tables234_fixed/final_multiseed_clip_scores.csv 2>&1 | tail -2
python3 tools/score_attribute_binding.py --evaluations $EV \
  --output output/tables234_fixed/attribute_binding_metrics_all.csv 2>&1 | tail -2
python3 tools/summarize_binding_comparison.py \
  --input output/tables234_fixed/attribute_binding_metrics_all.csv \
  --output output/tables234_fixed/attribute_binding_metrics_all_bootstrap.csv 2>&1 | tail -1

echo "########## PAIRED TEST $(date -Is) ##########"
python3 results/_work/compbench_paired_test.py --methods tpscda gateopt b2 2>&1 | tail -22

echo "B2_QUEUE_DONE $(date -Is)"
