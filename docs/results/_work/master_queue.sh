#!/bin/bash
# Full remaining queue, ordered so that the mask-fix baseline is produced
# BEFORE any method change.  Nothing here touches distillation.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

echo "########## 1/5 tables 2/3/4 generation $(date -Is) ##########"
bash results/_work/rerun234.sh

echo "########## 2/5 tables 2/3/4 scoring $(date -Is) ##########"
mkdir -p output/tables234_fixed
python3 tools/score_clip_multiseed.py \
  --evaluations baseline=output/multiseed_eval_fixed \
                 coco2017_scda=output/multiseed_eval_fixed \
                 fullspan_final=output/multiseed_eval_fixed \
                 token_pair=output/multiseed_eval_fixed \
                 learnable_layers=output/multiseed_eval_fixed \
  --output output/tables234_fixed/final_multiseed_clip_scores.csv 2>&1 | tail -5
python3 tools/score_attribute_binding.py \
  --evaluations baseline=output/multiseed_eval_fixed \
                coco2017_scda=output/multiseed_eval_fixed \
                fullspan_final=output/multiseed_eval_fixed \
                token_pair=output/multiseed_eval_fixed \
                learnable_layers=output/multiseed_eval_fixed \
  --output output/tables234_fixed/attribute_binding_metrics_all.csv 2>&1 | tail -5

echo "########## 3/5 ablation CompBench generation $(date -Is) ##########"
bash results/_work/run_ablation_compbench.sh

echo "########## 4/5 ablation CompBench evaluation $(date -Is) ##########"
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

echo "########## 5/5 gate optimisation full run $(date -Is) ##########"
PYTHONPATH=. accelerate launch --num_processes 1 train_scripts/train.py \
    configs/PixArt_xl2_coco2017_token_pair_gate_opt.py
echo "MASTER_QUEUE_DONE $(date -Is)"
