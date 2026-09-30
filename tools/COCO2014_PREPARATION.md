# COCO 2014 Gate Training Preparation

Assume the downloaded COCO files are laid out as follows:

```text
/datasets/coco2014/
  train2014/
  val2014/
  annotations/captions_train2014.json
  annotations/captions_val2014.json
```

Create a 20,000-image training subset in the format consumed by `InternalData`:

```bash
python tools/prepare_coco2014.py \
  --annotations /datasets/coco2014/annotations/captions_train2014.json \
  --image-root /datasets/coco2014 \
  --output-root /datasets/COCO2014Prepared \
  --max-items 20000 \
  --verify-images
```

Create T5 and 256px VAE features. The feature roots must be inside the prepared
directory because `InternalData` reads them from there.

```bash
python tools/extract_features.py \
  --json_path /datasets/COCO2014Prepared/partition/data_info.json \
  --dataset_root /datasets/coco2014 \
  --t5_save_root /datasets/COCO2014Prepared/caption_feature_wmask \
  --vae_save_root /datasets/COCO2014Prepared/img_vae_features \
  --img_size 256 \
  --pretrained_models_dir /path/to/pretrained_models
```

This produces:

```text
/datasets/COCO2014Prepared/
  partition/data_info.json
  caption_feature_wmask/train2014_COCO_train2014_*.npz
  img_vae_features_256resolution/noflip/train2014_COCO_train2014_*.npy
```

Update the absolute paths in `configs/PixArt_xl2_coco2014_gate.py`, then start
gate-only training from a PixArt checkpoint:

```bash
accelerate launch train_scripts/train.py configs/PixArt_xl2_coco2014_gate.py
```

Use `load_from` for the base PixArt checkpoint. Do not use `resume_from` unless
resuming a checkpoint produced by this gate-training run.
