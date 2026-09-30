"""SCDA object and attribute semantic ablation."""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2014_scda_improved.py', encoding='utf-8').read(), 'PixArt_xl2_coco2014_scda_improved.py', 'exec'))
semantic_residual_scale = 0.10
semantic_reg_coef = 0.01
semantic_active_branches = ('global', 'object', 'attribute')
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/scda_ablate_attr'
experiment_name = 'scda_ablate_global_object_attribute'
