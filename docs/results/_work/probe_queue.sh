#!/bin/bash
# Diagnostic probe: is the gateopt regression caused by the ROLE branches?
#
# The decomposition in results/improvement_report/gate_decomposition.csv shows
# the two kinds of injection moved very differently:
#     global (all-token) branch        x1.55
#     object / attribute / relation    x23.0 .. x27.2
# so the role branches are the prime suspect.  Retraining to test that costs
# 3.6 h; writing a PROBE checkpoint costs nothing, because the gates are plain
# parameters.  This script restores the role gates to their historical level on
# the gate-optimised checkpoint and leaves everything else (including the global
# gate and the overall scale) exactly as trained, then scores the three binding
# categories under the standard protocol.
#
# NOT a trained model.  Report as "门控改造 + 推理时 role 门控置回历史值".
#
# Runs only after the b2_neutral chain, so it never delays a requested result.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

until grep -q "RANDOMPROBE_DONE" "$L/randomprobe_queue.log" 2>/dev/null; do
  if ! pgrep -f "[r]andomprobe_queue.sh" >/dev/null 2>&1; then
    echo "WARNING: random-probe chain gone without RANDOMPROBE_DONE $(date -Is)" >&2
    break
  fi
  sleep 300
done
echo "########## PROBE start $(date -Is) ##########"
sleep 60

python3 results/_work/make_gate_probe_checkpoint.py || exit 1

echo "########## PROBE GENERATE (binding categories) $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method probe_rolegate --batch-size 16 \
    --categories color,shape,texture --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## PROBE EVALUATE $(date -Is) ##########"
cd results/_work
for c in color shape texture; do
  echo "----- probe_rolegate / $c $(date -Is) -----"
  timeout 10800 python3 run_eval.py --method probe_rolegate --category "$c" \
      --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_probe_${c}.log" 2>&1
  echo "  rc=$? $(date -Is)"
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## PROBE PAIRED TEST vs frozen AND vs gateopt $(date -Is) ##########"
python3 - <<'PY'
import csv, random, sys
from pathlib import Path
sys.path.insert(0, 'results/_work')
from compbench_paired_test import load, paired_bootstrap, CN, BINDING
rows = []
for base in ('frozen', 'gateopt'):
    for c in ('color', 'shape', 'texture'):
        a, b = load(base, c), load('probe_rolegate', c)
        if a is None or b is None:
            print(f'{c} vs {base}: NOT COMPLETED'); continue
        s = paired_bootstrap(a, b)
        verdict = ('提升(显著)' if s['significant'] and s['delta'] > 0 else
                   '退步(显著)' if s['significant'] and s['delta'] < 0 else '无显著差异')
        print(f"{CN[c]:<6} probe vs {base:<8} {s['mean_a']:.4f} -> {s['mean_b']:.4f}  "
              f"delta={s['delta']:+.4f} [{s['ci_low']:+.4f},{s['ci_high']:+.4f}]  {verdict}")
PY

echo "PROBE_DONE $(date -Is)"
