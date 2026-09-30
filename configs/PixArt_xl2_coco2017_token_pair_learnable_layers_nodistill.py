"""TP-SCDA (token-pair + learnable layer gates) WITHOUT ANY DISTILLATION.

Why this exists
---------------
The paper's current TP-SCDA checkpoint is NOT distillation-free.  Its training
lineage is

    base PixArt-alpha -> token-pair (5 epochs, 60 000 steps)
                      -> +distillation 2 500 steps (distill_coef = 0.5)
                      -> +learnable layer gates, 14 786 steps, STILL distilling

so the ablation row "Token-pair -> Token-pair + layer gate" changes several
things at once: distillation, the initialisation (a distilled checkpoint rather
than the base), the injection strength (12x) and the training length.  The
improvement in that row therefore cannot be attributed to the layer gate.

This config removes distillation from that lineage entirely, so the ablation can
be reported without it:

    - distill_coef = 0.0   (no teacher, no distillation term)
    - load_from the NON-distilled token-pair checkpoint instead of a distilled one

Everything else is inherited unchanged, so the only difference from the original
run is the removal of distillation.
"""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_learnable_layers.py', encoding='utf-8').read(),
             '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_learnable_layers.py', 'exec'))

experiment_name = 'coco2017_token_pair_learnable_layers_nodistill'
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_token_pair_learnable_layers_nodistill'

# --- the two changes that remove distillation ---------------------------------
distill_coef = 0.0
load_from = ('/root/private_data/PixArt-alpha-attentiongate/output/'
             'coco2017_scda_token_pair_positive_mb8_acc4/checkpoints/epoch_5_step_60000.pth')

save_model_steps = 7000
