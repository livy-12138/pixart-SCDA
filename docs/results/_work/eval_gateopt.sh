#!/bin/bash
# Evaluate the gate-optimised TP-SCDA checkpoint under BOTH protocols so we can
# see exactly which metrics moved.
#
# CRITICAL: gate_max and the gate activation are CONFIG, not weights.  They are
# reproduced here exactly as they were during training (0.3 / sigmoid), otherwise
# the checkpoint would be scored at a different injection strength than it was
# trained with.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs
CKPT=output/coco2017_token_pair_gate_opt/checkpoints/epoch_1_step_14786.pth

# wait for the training job to finish
until [ -f "$CKPT" ]; do sleep 120; done
echo "########## gateopt checkpoint found $(date -Is) ##########"
sleep 60

echo "########## G1 CompBench generation (8 x 300 = 2400 imgs) $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method gateopt --batch-size 16 \
    --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## G2 CompBench evaluation $(date -Is) ##########"
cd results/_work
for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  echo "----- gateopt / $c $(date -Is) -----"
  timeout 10800 python3 run_eval.py --method gateopt --category $c \
      --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_gateopt_${c}.log" 2>&1
  echo "  rc=$? $(date -Is)"
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## G3 64-prompt protocol generation (192 imgs) $(date -Is) ##########"
for seed in 43 44 45; do
  python3 tools/generate_scda_samples.py \
    --checkpoint "$CKPT" \
    --groups-dir output/coco2017_eval_learnable_layers/prompt_groups_64 \
    --output-dir output/multiseed_eval_fixed/gateopt/seed_$seed \
    --image-size 512 --steps 20 --cfg-scale 4.0 --seed $seed \
    --semantic-conditioning --semantic-token-attention \
    --semantic-token-gate-max 0.3 --semantic-token-gate-activation sigmoid \
    --semantic-residual-scale 0.0 2>&1 \
    | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
done

echo "########## G4 scoring + comparison $(date -Is) ##########"
python3 tools/score_clip_multiseed.py \
  --evaluations baseline=output/multiseed_eval_fixed coco2017_scda=output/multiseed_eval_fixed \
                fullspan_final=output/multiseed_eval_fixed token_pair=output/multiseed_eval_fixed \
                learnable_layers=output/multiseed_eval_fixed gateopt=output/multiseed_eval_fixed \
  --output output/tables234_fixed/final_multiseed_clip_scores.csv 2>&1 | tail -2
python3 tools/score_attribute_binding.py \
  --evaluations baseline=output/multiseed_eval_fixed coco2017_scda=output/multiseed_eval_fixed \
                fullspan_final=output/multiseed_eval_fixed token_pair=output/multiseed_eval_fixed \
                learnable_layers=output/multiseed_eval_fixed gateopt=output/multiseed_eval_fixed \
  --output output/tables234_fixed/attribute_binding_metrics_all.csv 2>&1 | tail -2
python3 tools/summarize_binding_comparison.py \
  --input output/tables234_fixed/attribute_binding_metrics_all.csv \
  --output output/tables234_fixed/attribute_binding_metrics_all_bootstrap.csv 2>&1 | tail -1
echo "GATEOPT_EVAL_DONE $(date -Is)"
