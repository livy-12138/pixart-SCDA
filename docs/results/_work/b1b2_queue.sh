#!/bin/bash
# B1 (edge-gated pair affinity) and B2 (binding alignment loss).
#
# Each variant is trained for one full epoch (14 786 micro-steps, the same
# length as the run that produced the paper's current checkpoint), then scored
# under BOTH protocols so the comparison against the frozen baseline is
# like-for-like:
#   * T2I-CompBench++ 8 categories x 300 prompts x 1 image (same as every other
#     method in the same-protocol table), official evaluators;
#   * the 64-prompt protocol (3 seeds) for CLIPScore and the binding proxies.
#
# Gate settings (gate_max / activation) are CONFIG, not weights: they are
# reproduced exactly at inference.  B1's edge gate is part of the forward pass,
# so the parsed edges are supplied at generation time too.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

until grep -q "GATEOPT_EVAL_DONE" "$L/gateopt_eval.log" 2>/dev/null; do sleep 120; done
echo "########## GPU released by the gateopt evaluation $(date -Is) ##########"
sleep 60

train () {
  local name=$1 ckpt=$2
  echo "########## TRAIN $name start $(date -Is) ##########"
  PYTHONPATH=. accelerate launch --num_processes 1 train_scripts/train.py \
      "configs/PixArt_xl2_coco2017_token_pair_${name}.py"
  local rc=$?
  echo "########## TRAIN $name rc=$rc $(date -Is) ##########"
  if [ ! -f "$ckpt" ]; then
    echo "########## ABORT: $name produced no final checkpoint ($ckpt); skipping its evaluation $(date -Is) ##########"
    return 1
  fi
  return 0
}

compbench () {
  local method=$1
  echo "########## COMPBENCH GENERATE $method $(date -Is) ##########"
  python3 -u results/_work/gen_compbench.py --method "$method" --batch-size 16 \
      --start-repeat 0 --end-repeat 1 2>&1 \
    | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
  echo "########## COMPBENCH EVALUATE $method $(date -Is) ##########"
  cd results/_work
  for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
    echo "----- $method / $c $(date -Is) -----"
    timeout 10800 python3 run_eval.py --method "$method" --category "$c" \
        --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_${method}_${c}.log" 2>&1
    echo "  rc=$? $(date -Is)"
  done
  cd /root/private_data/PixArt-alpha-attentiongate
}

fixed64 () {
  local name=$1 ckpt=$2
  echo "########## 64-PROMPT GENERATE $name $(date -Is) ##########"
  for seed in 43 44 45; do
    python3 tools/generate_scda_samples.py \
      --checkpoint "$ckpt" \
      --groups-dir output/coco2017_eval_learnable_layers/prompt_groups_64 \
      --output-dir "output/multiseed_eval_fixed/${name}/seed_${seed}" \
      --image-size 512 --steps 20 --cfg-scale 4.0 --seed "$seed" \
      --semantic-conditioning --semantic-token-attention \
      --semantic-token-gate-max 0.3 --semantic-token-gate-activation sigmoid \
      --semantic-residual-scale 0.0 \
      --semantic-pair-edge-gate --max-semantic-edges 24 2>&1 \
      | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
  done
}

# ---------------------------------------------------------------- B1
B1=output/coco2017_token_pair_b1_edgegate/checkpoints/epoch_1_step_14786.pth
B2=output/coco2017_token_pair_b2_bindingloss/checkpoints/epoch_1_step_14786.pth

if train b1_edgegate "$B1"; then
  compbench b1
  fixed64 b1 "$B1"
else
  echo "########## B1 evaluation SKIPPED $(date -Is) ##########"
fi

# ---------------------------------------------------------------- B2
if train b2_bindingloss "$B2"; then
  compbench b2
  fixed64 b2 "$B2"
else
  echo "########## B2 evaluation SKIPPED $(date -Is) ##########"
fi

echo "########## SCORE both variants $(date -Is) ##########"
EV="baseline=output/multiseed_eval_fixed coco2017_scda=output/multiseed_eval_fixed \
    fullspan_final=output/multiseed_eval_fixed token_pair=output/multiseed_eval_fixed \
    learnable_layers=output/multiseed_eval_fixed gateopt=output/multiseed_eval_fixed \
    b1=output/multiseed_eval_fixed b2=output/multiseed_eval_fixed"
python3 tools/score_clip_multiseed.py --evaluations $EV \
  --output output/tables234_fixed/final_multiseed_clip_scores.csv 2>&1 | tail -2
python3 tools/score_attribute_binding.py --evaluations $EV \
  --output output/tables234_fixed/attribute_binding_metrics_all.csv 2>&1 | tail -2
python3 tools/summarize_binding_comparison.py \
  --input output/tables234_fixed/attribute_binding_metrics_all.csv \
  --output output/tables234_fixed/attribute_binding_metrics_all_bootstrap.csv 2>&1 | tail -1
echo "B1B2_QUEUE_DONE $(date -Is)"
