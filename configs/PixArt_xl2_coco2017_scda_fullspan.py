"""COCO2017 MS-SCDA with complete noun-phrase object masks."""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_scda.py', encoding='utf-8').read(), '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_scda.py', 'exec'))

work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_scda_fullspan'
experiment_name = 'coco2017_scda_fullspan'
semantic_residual_scale = 0.25
semantic_reg_coef = 0.0
