"""Token-level, pair-aware SCDA attention experiment.

The original PixArt full-caption cross-attention remains active. SCDA no
longer pools semantic roles in this mode: each image patch queries the
role-marked T5 token sequence, with an inferred object-to-attribute bias.
"""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_scda_fullspan.py', encoding='utf-8').read(), '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_scda_fullspan.py', 'exec'))

experiment_name = 'coco2017_scda_token_pair_positive_mb8_acc4'
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_scda_token_pair_positive_mb8_acc4'
semantic_token_attention = True
semantic_residual_scale = 0.0
semantic_reg_coef = 0.001
# Token-level attention retains large patch-to-text intermediates during
# backpropagation; checkpoint each transformer block to keep the 512px batch
# within the available GPU memory.
# A smaller micro-batch avoids activation OOM without recomputing every block;
# accumulation preserves the original effective batch size of 32.
grad_checkpointing = False
train_batch_size = 8
gradient_accumulation_steps = 4
