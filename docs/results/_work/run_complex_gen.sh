#!/bin/bash
set -u
cd /root/private_data/PixArt-alpha-attentiongate
for m in tpscda frozen; do
  echo "===== COMPLEX $m $(date -Is) ====="
  python3 -u results/_work/gen_compbench.py --method $m --batch-size 16 --fix-mask \
      --categories complex --limit-prompts 100 --start-repeat 0 --end-repeat 10 2>&1 \
    | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
  echo "===== rc=$? $(date -Is) ====="
done
echo "COMPLEX_GEN_DONE"
