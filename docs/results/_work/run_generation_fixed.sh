#!/bin/bash
set -u
cd /root/private_data/PixArt-alpha-attentiongate
mkdir -p /root/compbench_work/logs
for r in 0 1 2; do
  e=$((r+1))
  for m in tpscda frozen; do
    echo "===== FIXED STAGE repeat=$r method=$m $(date -Is) ====="
    python3 -u results/_work/gen_compbench.py --method $m --batch-size 16 --fix-mask \
        --start-repeat $r --end-repeat $e 2>&1 \
      | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
    echo "===== rc=$? $(date -Is) ====="
  done
done
echo "ALL_FIXED_GENERATION_DONE"
