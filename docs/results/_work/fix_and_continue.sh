#!/bin/bash
# Correction pass: the gate-initialisation default had briefly been changed,
# which silently altered checkpoints that do NOT contain
# `semantic_token_layer_gate` (token-pair).  The default is now back to the
# historical +-4.0, so only the token-pair results need regenerating.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

echo "########## C1 re-run token_pair tables 2/3/4 $(date -Is) ##########"
rm -rf output/multiseed_eval_fixed/token_pair
for seed in 43 44 45; do
  python3 tools/generate_scda_samples.py \
    --checkpoint output/coco2017_scda_token_pair_positive_mb8_acc4/checkpoints/epoch_5_step_60000.pth \
    --groups-dir output/coco2017_eval_learnable_layers/prompt_groups_64 \
    --output-dir output/multiseed_eval_fixed/token_pair/seed_$seed \
    --image-size 512 --steps 20 --cfg-scale 4.0 --seed $seed \
    --semantic-conditioning --semantic-token-attention \
    --semantic-token-gate-max 1.0 --semantic-residual-scale 0.0 2>&1 \
    | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
done

echo "########## C2 re-score all five $(date -Is) ##########"
python3 tools/score_clip_multiseed.py \
  --evaluations baseline=output/multiseed_eval_fixed coco2017_scda=output/multiseed_eval_fixed \
                fullspan_final=output/multiseed_eval_fixed token_pair=output/multiseed_eval_fixed \
                learnable_layers=output/multiseed_eval_fixed \
  --output output/tables234_fixed/final_multiseed_clip_scores.csv 2>&1 | tail -2
python3 tools/score_attribute_binding.py \
  --evaluations baseline=output/multiseed_eval_fixed coco2017_scda=output/multiseed_eval_fixed \
                fullspan_final=output/multiseed_eval_fixed token_pair=output/multiseed_eval_fixed \
                learnable_layers=output/multiseed_eval_fixed \
  --output output/tables234_fixed/attribute_binding_metrics_all.csv 2>&1 | tail -2

echo "########## C3 ablation CompBench generation $(date -Is) ##########"
bash results/_work/run_ablation_compbench.sh

echo "########## C4 ablation CompBench evaluation $(date -Is) ##########"
cd results/_work
for m in pooled fullspan tokenpair; do
  for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
    echo "----- $m / $c $(date -Is) -----"
    timeout 10800 python3 run_eval.py --method $m --category $c \
        --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_${m}_${c}.log" 2>&1
    echo "  rc=$? $(date -Is)"
  done
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## C5 gate optimisation full run $(date -Is) ##########"
PYTHONPATH=. accelerate launch --num_processes 1 train_scripts/train.py \
    configs/PixArt_xl2_coco2017_token_pair_gate_opt.py
echo "FIX_CONTINUE_DONE $(date -Is)"
