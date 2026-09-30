"""B2 (binding objective) started from a NEUTRAL gate instead of an open one.

Why this variant
----------------
The measured relation between injection strength and the binding metrics is
monotone downwards, with three points:

    strength 0        frozen baseline
    strength 0.0164   the paper's checkpoint          ~= baseline (n.s.)
    strength 0.063    the gate-optimised run          significantly WORSE

The gate-optimised run starts at strength 0.150, because `sigmoid(0) = 0.5`
times the 0.3 ceiling.  That means the branch spends the whole run inside the
range that has been measured to hurt, so any benefit from the binding objective
(B2) would be fighting a handicap the configuration itself imposed.

This variant starts the gate at the historical neutral level and lets it move
either way:

    semantic_token_gate_init = -2.88   ->  0.3 * sigmoid(-2.88) = 0.0160

The ceiling is deliberately left at 0.3 and the boosted gate learning rate is
kept, so if the binding objective really does make the branch useful, the
optimiser is free to RAISE the gate -- which is the opposite of the downwards
drift observed in every run so far (0.150 -> 0.066 without B2).  The direction
of that drift is the readout: up means the branch became worth opening, down
means it did not.

Everything else is identical to the B2 run, so the pair (b2, b2_neutralgate)
isolates a single variable: the starting strength of the branch.
"""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_b2_bindingloss.py', encoding='utf-8').read(),
             '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_b2_bindingloss.py', 'exec'))

experiment_name = 'coco2017_token_pair_b2_neutralgate'
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_token_pair_b2_neutralgate'

# --- the only change relative to the B2 run ---------------------------------
# 0.3 * sigmoid(-2.88) = 0.0160, the historical (paper checkpoint) strength
semantic_token_gate_init = -2.88

save_model_steps = 7000
