"""B2 with the gate FROZEN at the neutral strength -- remove the escape hatch.

Motivation
----------
Measured: the semantic branch is a STRONG intervention.  Turning it off changes
the DiT output by ~15% of its RMS (results/branch_effect_measurement.md), so the
branch has ample influence to change the metrics.  The problem is direction, not
magnitude.

In every run so far the gate drifts DOWN once unfrozen (0.150 -> 0.063 without
B2, 0.150 -> 0.068 with B2).  That is the optimiser taking the easiest route to a
lower diffusion loss: shrink the branch.  Adding the binding objective (B2) did
not stop it.

If the gate's freedom is what lets the branch evade the job, take the freedom
away.  Here the gate is frozen at the historical neutral strength of the paper's
checkpoint (0.3 * sigmoid(-2.88) = 0.0160) for the WHOLE run, while B2's
alignment objective stays on.  The branch cannot shrink its way out: at a fixed
strength it must either become useful for binding or learn to emit something
harmless that still satisfies the alignment loss.

This is a genuinely different experiment from b2 and b2_neutral:
  b2           gate starts open (0.150), may drift
  b2_neutral   gate starts neutral (0.016), may drift
  this run     gate starts neutral (0.016), may NOT drift
so the pair (b2_neutral, b2_frozengate) isolates exactly one variable: whether the
gate is allowed to move at all.
"""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_b2_bindingloss.py', encoding='utf-8').read(),
             '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_b2_bindingloss.py', 'exec'))

experiment_name = 'coco2017_token_pair_b2_frozengate'
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_token_pair_b2_frozengate'

# --- the two changes relative to the B2 run ---------------------------------
semantic_token_gate_init = -2.88     # 0.3 * sigmoid(-2.88) = 0.0160, neutral
semantic_gate_freeze_steps = 1000000  # never unfreeze within a 14 786-step run

save_model_steps = 7000
