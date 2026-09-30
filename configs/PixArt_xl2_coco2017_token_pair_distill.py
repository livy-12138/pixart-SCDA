"""Conservative token-pair refinement around the frozen PixArt teacher.

Only the semantic token/pair module is trainable.  A frozen baseline teacher
keeps the student close to PixArt while a bounded gate learns useful
object/attribute corrections.
"""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_scda_token_pair.py', encoding='utf-8').read(), '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_scda_token_pair.py', 'exec'))

experiment_name = 'coco2017_token_pair_distill_gate008'
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_token_pair_distill_gate008_sparse'

# Start from the best completed token-pair checkpoint, but preserve the
# original PixArt behavior with a frozen teacher during refinement.
load_from = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_scda_token_pair_positive_mb8_acc4/checkpoints/epoch_5_step_60000.pth'
distill_teacher_checkpoint = '/root/private_data/PixArt-alpha-attentiongate/output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'
distill_coef = 0.5
# One teacher pass every 8 micro-batches; the normal student update remains
# dense, reducing the teacher overhead while retaining a persistent anchor.
distill_interval = 8

# Conservative semantic intervention.  The learned raw gate is capped to 8%
# of the token delta, and the residual norm is regularized more strongly.
semantic_token_gate_max = 0.08
semantic_reg_coef = 0.02
semantic_residual_scale = 0.0

optimizer = dict(type='AdamW', lr=1e-5, weight_decay=0.01, eps=1e-10)
num_epochs = 1
save_model_steps = 2500
save_model_epochs = 1
seed = 43
