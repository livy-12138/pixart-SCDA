#!/bin/bash
# Does the semantic branch hurt because of its CONTENT, or because ANY
# perturbation of that magnitude hurts?
#
# The branch's output is nearly orthogonal to the DiT's own cross-attention
# output (cos ~ 0.001 over all 28 blocks), so it is not re-delivering the same
# information -- the "redundant pathway" framing was wrong.
#
# Three conditions on the SAME checkpoint (gateopt, where the branch
# demonstrably hurts: colour -0.034, texture -0.044, both significant):
#   A  as trained, text content          (already measured)
#   B  output replaced by a random vector of MATCHED magnitude   <- this script
#   C  branch switched off               (already measured, ~= frozen)
#
#   B ~= A  -> any perturbation of that size hurts; content-level fixes are useless
#   B ~= C  -> the learned content is what hurts; improving the content is worth trying
#
# Inference-time probe on one fixed checkpoint.  Prompts/seeds/resolution/
# sampler/steps/CFG untouched; only the branch's output vector is replaced.
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

until grep -q "B2NG_QUEUE_DONE" "$L/b2ng_queue.log" 2>/dev/null; do
  if ! pgrep -f "[b]2ng_queue.sh" >/dev/null 2>&1; then
    echo "WARNING: b2ng chain gone without B2NG_QUEUE_DONE $(date -Is)" >&2
    break
  fi
  sleep 120
done
echo "########## RANDOM-VECTOR PROBE start $(date -Is) ##########"
sleep 60

echo "########## GENERATE probe_random (color, texture) $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method probe_random --batch-size 16 \
    --categories color,texture --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## EVALUATE probe_random $(date -Is) ##########"
cd results/_work
for c in color texture; do
  echo "----- probe_random / $c $(date -Is) -----"
  timeout 10800 python3 run_eval.py --method probe_random --category "$c" \
      --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_probe_random_${c}.log" 2>&1
  echo "  rc=$? $(date -Is)"
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## VERDICT $(date -Is) ##########"
python3 - <<'PY'
import sys
sys.path.insert(0, 'results/_work')
from compbench_paired_test import load, paired_bootstrap, CN
for cat in ('color', 'texture'):
    txt, rnd = load('gateopt', cat), load('probe_random', cat)
    off = load('frozen', cat)
    if rnd is None:
        print(f'{cat}: NOT COMPLETED'); continue
    base = txt if txt else off
    s = paired_bootstrap(base, rnd)
    label = 'gateopt(文本)' if txt else 'frozen'
    print(f"{CN[cat]:<6} 参考={label:<12} {s['mean_a']:.4f}")
    print(f"       随机向量      {s['mean_b']:.4f}   delta={s['delta']:+.4f} "
          f"[{s['ci_low']:+.4f},{s['ci_high']:+.4f}]")
    r = paired_bootstrap(off, rnd) if off else None
    if r:
        print(f"       vs 冻结基线   {r['mean_a']:.4f} -> {r['mean_b']:.4f}  "
              f"delta={r['delta']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]")
PY
echo "RANDOMPROBE_DONE $(date -Is)"
