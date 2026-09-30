#!/bin/bash
# Timestep-window probe on the paper's TP-SCDA checkpoint.
#
# The paper states the injection strength is modulated "by network layer AND
# timestep", but the token-pair path in the code has NO timestep dependence --
# `t` is only used by the pooled residual path.  Meanwhile the measured weak
# point is the counting metric (numeracy, -0.0125, significant, concentrated on
# two-object prompts), and binding is decided in the EARLY high-noise steps
# while late steps mostly refine appearance.
#
# Hypothesis: injecting the semantic branch throughout the whole trajectory is
# what costs counting, and restricting it to the early steps keeps whatever
# binding benefit exists while removing the harm.
#
# This is an INFERENCE-TIME PROBE on one fixed checkpoint -- no training, and
# prompts/seeds/resolution/sampler/steps/CFG are untouched.  The only change is
# which portion of the trajectory the branch is switched on for.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

until grep -q "PAIRREPLACE_DONE" "$L/pairreplace_queue.log" 2>/dev/null; do
  if ! pgrep -f "[p]airreplace_queue.sh" >/dev/null 2>&1; then
    echo "WARNING: pairreplace chain gone without PAIRREPLACE_DONE $(date -Is)" >&2
    break
  fi
  sleep 300
done
echo "########## TIMESTEP PROBE start $(date -Is) ##########"
sleep 60

for m in probe_early probe_late; do
  echo "########## GENERATE $m $(date -Is) ##########"
  python3 -u results/_work/gen_compbench.py --method "$m" --batch-size 16 \
      --categories color,numeracy --start-repeat 0 --end-repeat 1 2>&1 \
    | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"
  echo "########## EVALUATE $m $(date -Is) ##########"
  cd results/_work
  for c in color numeracy; do
    echo "----- $m / $c $(date -Is) -----"
    timeout 10800 python3 run_eval.py --method "$m" --category "$c" \
        --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_${m}_${c}.log" 2>&1
    echo "  rc=$? $(date -Is)"
  done
  cd /root/private_data/PixArt-alpha-attentiongate
done

echo "########## COMPARE $(date -Is) ##########"
python3 - <<'PY'
import sys
sys.path.insert(0, 'results/_work')
from compbench_paired_test import load, paired_bootstrap, CN
for cat in ('color', 'numeracy'):
    ref = load('tpscda', cat)
    if ref is None:
        print(f'{cat}: reference tpscda missing'); continue
    for m in ('probe_early', 'probe_late'):
        other = load(m, cat)
        if other is None:
            print(f'{cat} / {m}: NOT COMPLETED'); continue
        s = paired_bootstrap(ref, other)
        v = ('提升(显著)' if s['significant'] and s['delta'] > 0 else
             '退步(显著)' if s['significant'] and s['delta'] < 0 else '无显著差异')
        print(f"{CN[cat]:<6} {m:<12} 全trajectory={s['mean_a']:.4f} -> {s['mean_b']:.4f}  "
              f"delta={s['delta']:+.4f} [{s['ci_low']:+.4f},{s['ci_high']:+.4f}]  {v}")
PY

echo "TSPROBE_DONE $(date -Is)"
