"""SCDA semantic adapters on prepared COCO captions and image features."""

data_root = '/root/private_data/PixArt-alpha-attentiongate/data/coco2014'
data = dict(type='InternalData', root='COCO2014Prepared10K', image_root='images',
            image_list_json=['data_info.json'], transform='default_train',
            load_vae_feat=True, load_semantic_masks=True)
image_size = 512
train_batch_size = 32
eval_batch_size = 16
use_fsdp = False
valid_num = 0
model = 'PixArt_XL_2'
aspect_ratio_type = None
multi_scale = False
window_block_indexes = []
window_size = 0
use_rel_pos = False
fp32_attention = False
lewei_scale = 1.0
num_workers = 4
train_sampling_steps = 1000
eval_sampling_steps = 250
model_max_length = 120

semantic_conditioning = True
semantic_adapter_dim = 64
semantic_dropout = 0.1
train_semantic_only = True
# Compatibility with the native-gate checkpoint format. The gate is
# zero-initialized and therefore neutral at model initialization.
time_condition_gate = True
condition_gate_scale = 0.05

num_epochs = 10
gradient_accumulation_steps = 1
grad_checkpointing = False
gradient_clip = 1.0
gc_step = 1
auto_lr = dict(rule='sqrt')
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=3e-2, eps=1e-10)
lr_schedule = 'constant'
lr_schedule_args = dict(num_warmup_steps=500)
save_image_epochs = 1
save_model_epochs = 1
save_model_steps = 5000
sample_posterior = True
mixed_precision = 'fp16'
scale_factor = 0.18215
ema_rate = 0.9999
log_interval = 50
cfg_scale = 4
mask_type = 'null'
num_group_tokens = 0
mask_loss_coef = 0.0
load_mask_index = False
vae_pretrained = '/root/private_data/PixArt-alpha-attentiongate/output/pretrained_models/sd-vae-ft-ema'
load_from = '/root/private_data/PixArt-alpha-attentiongate/output/pretrained_models/PixArt-XL-2-512x512-native-gate-init.pth'
resume_from = dict(checkpoint=None, load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
snr_loss = False
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2014_scda_512'
s3_work_dir = None
seed = 43
