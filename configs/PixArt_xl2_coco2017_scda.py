"""COCO2017 full-scale MS-SCDA training configuration.

Feature extraction and semantic-mask generation must populate the prepared
directory before this configuration is launched on a GPU host.
"""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2014_scda_improved.py', encoding='utf-8').read(), '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2014_scda_improved.py', 'exec'))

data_root = '/root/private_data/data'
data = dict(type='InternalData', root='COCO2017Prepared',
            image_root='/root/private_data/data/coco2017',
            image_list_json=['data_info.json'], transform='default_train',
            load_vae_feat=True, load_semantic_masks=True)
num_epochs = 5
train_batch_size = 32
# Cached training data is I/O-bound.  These settings preserve model, batch,
# sampling, and optimization behavior while keeping host-side batches ready.
num_workers = 10
persistent_workers = True
prefetch_factor = 4
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_scda'
experiment_name = 'coco2017_scda_full'
