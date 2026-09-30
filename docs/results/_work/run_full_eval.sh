#!/bin/bash
# Full T2I-CompBench++ evaluation for both methods, all 8 categories.
# Run with an EXCLUSIVE GPU (no generation job) for official batch sizes.
set -u
cd /root/private_data/PixArt-alpha-attentiongate/results/_work
LOGDIR=/root/compbench_work/logs
mkdir -p "$LOGDIR"
MAIN="$LOGDIR/full_eval.log"
echo "===== FULL EVAL START $(date -Is) =====" >> "$MAIN"

for m in tpscda frozen; do
  for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
    echo "----- $m / $c $(date -Is) -----" >> "$MAIN"
    timeout 10800 python3 run_eval.py --method "$m" --category "$c" \
        --max-batch 8 --blip-max-batch 32 --min-free-gb 4 \
        >> "$LOGDIR/eval_${m}_${c}.log" 2>&1
    rc=$?
    echo "  rc=$rc $(date -Is)" >> "$MAIN"
    tail -3 "$LOGDIR/eval_${m}_${c}.log" >> "$MAIN"
  done
done
echo "===== FULL EVAL DONE $(date -Is) =====" >> "$MAIN"
