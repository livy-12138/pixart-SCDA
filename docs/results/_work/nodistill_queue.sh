#!/bin/bash
# Retrain TP-SCDA with NO distillation, then evaluate it under both protocols.
#
# The paper's TP-SCDA checkpoint is not distillation-free: it continues from a
# distilled checkpoint AND keeps distilling (distill_coef = 0.5) during its own
# 14 786 steps.  That means the ablation row "Token-pair -> Token-pair + layer
# gate" changes distillation, initialisation, injection strength and training
# length all at once, so its improvement cannot be attributed to the gate.
#
# This run removes distillation from that lineage.  Everything else is inherited
# unchanged (1 epoch, batch 8 x accum 4, gate_max 0.08, token attention), so the
# only difference from the original run is the absence of distillation.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

until grep -q "PROBE_DONE" "$L/probe_queue.log" 2>/dev/null; do
  if ! pgrep -f "[p]robe_queue.sh" >/dev/null 2>&1; then
    echo "WARNING: role-gate probe chain gone without PROBE_DONE $(date -Is)" >&2
    break
  fi
  sleep 120
done
echo "########## NODISTILL start $(date -Is) ##########"
sleep 60

CKPT=output/coco2017_token_pair_learnable_layers_nodistill/checkpoints/epoch_1_step_14786.pth
echo "########## TRAIN tpscda_nodistill $(date -Is) ##########"
PYTHONPATH=. accelerate launch --num_processes 1 train_scripts/train.py \
    configs/PixArt_xl2_coco2017_token_pair_learnable_layers_nodistill.py
echo "########## TRAIN rc=$? $(date -Is) ##########"
if [ ! -f "$CKPT" ]; then
  echo "########## ABORT: no checkpoint ($CKPT) $(date -Is) ##########"; exit 1
fi

echo "########## COMPBENCH GENERATE $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method tpscda_nodistill --batch-size 16 \
    --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## COMPBENCH EVALUATE $(date -Is) ##########"
cd results/_work
for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  echo "----- tpscda_nodistill / $c $(date -Is) -----"
  timeout 10800 python3 run_eval.py --method tpscda_nodistill --category "$c" \
      --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_nodistill_${c}.log" 2>&1
  echo "  rc=$? $(date -Is)"
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## 64-PROMPT GENERATE $(date -Is) ##########"
for seed in 43 44 45; do
  python3 tools/generate_scda_samples.py \
    --checkpoint "$CKPT" \
    --groups-dir output/coco2017_eval_learnable_layers/prompt_groups_64 \
    --output-dir "output/multiseed_eval_fixed/tpscda_nodistill/seed_${seed}" \
    --image-size 512 --steps 20 --cfg-scale 4.0 --seed "$seed" \
    --semantic-conditioning --semantic-token-attention \
    --semantic-token-gate-max 0.08 --semantic-residual-scale 0.0 2>&1 \
    | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
done

echo "########## SCORE + PAIRED TEST $(date -Is) ##########"
EV="baseline=output/multiseed_eval_fixed coco2017_scda=output/multiseed_eval_fixed \
    fullspan_final=output/multiseed_eval_fixed token_pair=output/multiseed_eval_fixed \
    learnable_layers=output/multiseed_eval_fixed tpscda_nodistill=output/multiseed_eval_fixed"
python3 tools/score_clip_multiseed.py --evaluations $EV \
  --output output/tables234_fixed/final_multiseed_clip_scores.csv 2>&1 | tail -2
python3 results/_work/compbench_paired_test.py --methods tpscda tpscda_nodistill 2>&1 | tail -22
echo "NODISTILL_DONE $(date -Is)"
