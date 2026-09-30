#!/bin/bash
# PROBE: inject the object->attribute structure by REPLACING the DiT's own
# cross-attention weights at the parsed pairs -- the attribute token takes the
# noun's own (unnormalised) DiT weight, and the rest of the row is rescaled
# proportionally.  No separate q/k, no additive bias.
#
# Why: the original injection adds a bias inside a SEPARATE attention module
# whose logits reach ~2766 (q comes from patch states with norm ~668), so its
# softmax is one-hot and any additive bias is annihilated (measured: exactly
# zero effect).  Replacing the DiT's existing weights avoids that entirely.
#
# NOT a trained model -- the checkpoint is the paper's TP-SCDA, unmodified.  The
# DiT was not trained under this operation, so this measures whether the
# mechanism CAN act, not what a trained version would score.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

# RE-RUN 2026-09-20: the first attempt died on a shape bug in the replacement
# path (the edge matrix was matmul'd against the head axis instead of the sample
# axis).  Fixed in PixArt_blocks.py / PixArt.py; this now waits for the chain
# that currently owns the GPU instead of starting on top of it.
source /root/private_data/PixArt-alpha-attentiongate/results/_work/queue_wait.sh
wait_for_new_marker "$L/b2fg_queue.log" "PROBE_DONE" "[b]2fg_queue.sh"
echo "########## DITPAIR PROBE start $(date -Is) ##########"
sleep 60

# The 288 colour images left by the crashed run came from the buggy broadcast
# (every head read a different sample's edges), and the generator skips images
# that already exist, so they would silently survive into the new run.
rm -rf /root/compbench_work/images/probe_ditpair
echo "########## cleared stale probe_ditpair images $(date -Is) ##########"

echo "########## GENERATE probe_ditpair (color, texture) $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method probe_ditpair --batch-size 16 \
    --categories color,texture --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## EVALUATE probe_ditpair $(date -Is) ##########"
cd results/_work
for c in color texture; do
  echo "----- probe_ditpair / $c $(date -Is) -----"
  timeout 10800 python3 run_eval.py --method probe_ditpair --category "$c" \
      --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_ditpair_${c}.log" 2>&1
  echo "  rc=$? $(date -Is)"
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## RESULT $(date -Is) ##########"
python3 - <<'PY'
import sys
sys.path.insert(0,'results/_work')
from compbench_paired_test import load, paired_bootstrap, CN
for cat in ('color','texture'):
    frozen=load('frozen',cat); tp=load('tpscda',cat); pr=load('probe_ditpair',cat)
    if pr is None: print(f'{cat}: NOT COMPLETED'); continue
    for nm,ref in (('冻结基线',frozen),('TP-SCDA(原机制)',tp)):
        if ref is None: continue
        s=paired_bootstrap(ref,pr)
        v=('提升(显著)' if s['significant'] and s['delta']>0 else
           '退步(显著)' if s['significant'] and s['delta']<0 else '无显著差异')
        print(f"{CN[cat]:<6} 新机制 vs {nm:<14} {s['mean_a']:.4f} -> {s['mean_b']:.4f}  "
              f"delta={s['delta']:+.4f} [{s['ci_low']:+.4f},{s['ci_high']:+.4f}]  {v}")
PY
echo "DITPAIR_DONE $(date -Is)"
