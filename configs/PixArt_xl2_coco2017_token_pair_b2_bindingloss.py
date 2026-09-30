"""TP-SCDA + B2: explicit object->attribute attention-alignment objective.

Rationale (results/TP_SCDA_IMPROVEMENT_PLAN.md section B2): nothing in the
training objective rewards "attribute bound to the right object" -- the loss is
diffusion MSE plus a term that pushes the residual towards zero.  Measured
consequence: with the gate finally able to move, it drifts DOWN (effective
strength 0.150 -> 0.069 over 10k steps), i.e. the optimiser decides the branch
is useless.  B2 supplies the missing gradient: for every parsed edge (o, a),
patches that attend to object o must attend to a more than to other attribute
tokens.

Keeps every gate control from the gate-optimisation run, so the only new
variable relative to that run is the binding loss.
"""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_gate_opt.py', encoding='utf-8').read(),
             '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_gate_opt.py', 'exec'))

experiment_name = 'coco2017_token_pair_b2_bindingloss'
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_token_pair_b2_bindingloss'

# --- B2 ---
semantic_binding_coef = 0.1        # weight of the alignment loss
semantic_binding_temperature = 1.0
semantic_pair_edge_gate = False

data = dict(data, load_semantic_edges=True,
            semantic_edge_index='/root/private_data/data/COCO2017Prepared/partition/edge_index.npz', max_semantic_edges=24)

num_epochs = 1
max_train_steps = 0
save_model_steps = 7000
