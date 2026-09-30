#!/bin/bash
# Resume the pair-replace re-run AFTER training: generation + evaluation only.
#
# Why a separate script: the first attempt at this stage hit the generator's
# skip-existing behaviour -- images/pair_replace still held the 3100 images from
# the 2026-09-20 mechanism-OFF run, so the generator would have skipped every
# one and the evaluators would have scored the OLD model's pictures, producing
# convincing numbers for the wrong model.  Those are archived as
# images/pair_replace_mech_off (the control); this script refuses to start if the
# output directory is not empty.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs
LOG="$L/pairreplace_v2_resume.log"
exec >> "$LOG" 2>&1

CK=output/coco2017_pair_replace/checkpoints/epoch_1_step_14786.pth
IMG=/root/compbench_work/images/pair_replace

[ -f "$CK" ] || { echo "ABORT: no checkpoint $CK"; exit 1; }

# Freshness guard: never evaluate images this run did not produce.
if [ -d "$IMG" ] && [ -n "$(find "$IMG" -name '*.png' -print -quit 2>/dev/null)" ]; then
    echo "ABORT: $IMG already contains images -- archive them first, otherwise"
    echo "       the generator skips them and the evals score an older model."
    exit 1
fi

echo "########## GENERATE (7 categories, 1 img/prompt) $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method pair_replace --batch-size 16 \
    --categories color,shape,texture,spatial,3d_spatial,numeracy,non_spatial \
    --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## GENERATE complex (100 prompts x 10) $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method pair_replace --batch-size 16 \
    --categories complex --limit-prompts 100 --start-repeat 0 --end-repeat 10 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## EVALUATE $(date -Is) ##########"
echo "  images produced: $(find "$IMG" -name '*.png' | wc -l) (expect 3100)"
cd results/_work
for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  echo "----- pair_replace / $c $(date -Is) -----"
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
