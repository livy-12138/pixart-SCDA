"""Improved SCDA experiment: conservative residual plus branch ablations."""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2014_scda.py', encoding='utf-8').read(), '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2014_scda.py', 'exec'))
data_root = '/root/private_data/data'
semantic_residual_scale = 0.25
semantic_active_branches = ('global', 'object', 'attribute', 'relation')
optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.03, eps=1e-10)
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/scda_improved_full'
