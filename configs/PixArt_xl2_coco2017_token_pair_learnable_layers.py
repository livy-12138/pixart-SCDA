"""Token-pair refinement with learnable per-branch layer positions."""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_distill.py', encoding='utf-8').read(), '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_distill.py', 'exec'))

experiment_name = 'coco2017_token_pair_learnable_layers'
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_token_pair_learnable_layers'
load_from = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_token_pair_distill_gate008_sparse/checkpoints/epoch_1_step_2500.pth'

# Keep the same conservative teacher anchor and bounded semantic gate.
num_epochs = 1
save_model_steps = 2500
save_model_epochs = 1
optimizer = dict(type='AdamW', lr=1e-5, weight_decay=0.01, eps=1e-10)
seed = 43
