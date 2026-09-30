#!/bin/bash
# T2I-CompBench++ for the ablation variants, so the comparison table is
# same-protocol / same-code (mask fix) instead of public reported numbers.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
for m in pooled fullspan tokenpair; do
  echo "===== ABLATION $m $(date -Is) ====="
  python3 -u results/_work/gen_compbench.py --method $m --batch-size 16 \
      --start-repeat 0 --end-repeat 1 2>&1 \
    | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
  echo "===== rc=$? $(date -Is) ====="
done
echo "ABLATION_COMPBENCH_GEN_DONE"
