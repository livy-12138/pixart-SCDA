"""TP-SCDA token-pair training with the gate controls from GATE_OPTIMIZATION.md.

Measured problem in the previous run: `semantic_token_layer_gate` was
initialised at +-4.0, i.e. in the saturated tail of the sigmoid where
dlambda/dw ~= 0.018, and it shared the 3.5e-6 learning rate of the 8M-parameter
projections.  Over 14 786 steps a parameter can travel at most ~0.05 in logit
space, while ~1.0 is needed just to leave saturation -- so the gate could not
move, and the object/attribute/relation branches stayed at an effective
strength of ~3e-4.

A controlled probe (same checkpoint, data, optimiser, 150 steps) showed that
only changing the initialisation to +-1.0 already increases the gate's travel
by 10.7x with an identical parameter displacement.

NOTE: this run uses NO distillation (distill_coef = 0), so it starts from the
untouched PixArt-alpha conversion rather than from a distilled checkpoint.
"""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_scda_token_pair.py', encoding='utf-8').read(),
             '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_scda_token_pair.py', 'exec'))

experiment_name = 'coco2017_token_pair_gate_opt'
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_token_pair_gate_opt'
load_from = '/root/private_data/PixArt-alpha-attentiongate/output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'

# --- no distillation in this run -------------------------------------------
distill_coef = 0.0

# --- gate optimisation ------------------------------------------------------
# 1) initialise in the LINEAR part of the sigmoid, not in its tail
semantic_gate_init = -1.0          # role branches: lambda ~ 0.269
semantic_gate_init_global = 1.0    # global branch: lambda ~ 0.731
# 2) a single squashed scalar instead of gate_max * tanh(...) nested with the
#    per-layer sigmoid; sigmoid(0) = 0.5 starts at half the ceiling
semantic_token_gate_init = 0.0
semantic_token_gate_activation = 'sigmoid'
semantic_token_gate_max = 0.3
# 3) ramp the ceiling up over the first steps instead of starting at full
#    strength while the projections are still random
semantic_gate_max_warmup_steps = 1500
# 4) the 113 gate scalars get their own learning rate
semantic_gate_lr_mult = 100.0
# 5) do not let the gates chase noise gradients from an untrained projection
semantic_gate_freeze_steps = 2000
# 6) anneal the semantic L2 pressure towards zero instead of holding it.
#    The coefficient MUST be rescaled: opening the gates ~10x makes the logged
#    residual norm grow from ~0.27 to ~7.9, i.e. its square grows ~835x.  With
#    the old 0.02 the penalty term alone reached 0.02*63.2 = 1.26 against a
#    diffusion loss of 0.33 -- the optimiser would have spent its step reducing
#    the regulariser instead of the diffusion objective.  5e-5 puts the penalty
#    at ~1% of the diffusion loss, the same order as the previous run's 0.66%.
#    The achieved share is logged every step as `semantic_reg_share`.
semantic_reg_coef = 5e-5
semantic_reg_coef_final = 0.0
semantic_reg_anneal_steps = 10000

# --- run length -------------------------------------------------------------
# Full run: num_epochs=1 is 14 785 micro-steps, the same length as the run that
# produced the paper's current checkpoint, so the two are directly comparable.
# This config was first used for a 3 000-step feasibility probe; that probe
# showed the gate finally moving (38.6x the drift of the whole previous run in
# 1 000 post-unfreeze steps), which is why the full run is being done now.
num_epochs = 1
max_train_steps = 0            # 0 = run to the end of the epoch
save_model_steps = 3000        # 5 checkpoints x ~4.7 GB = ~24 GB
save_model_epochs = 1
