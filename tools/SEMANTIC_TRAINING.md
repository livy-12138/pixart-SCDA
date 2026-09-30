# Text-Semantic Adapter Training

This workflow trains only the added semantic adapters. PixArt's original DiT
blocks and its T5 feature encoder remain frozen.

## 1. Generate semantic masks

Run this in the same Linux environment that contains the tokenizer used when
the caption features were extracted. The tokenizer must be a fast tokenizer;
the script uses its character offsets to align spaCy word labels to T5
sub-tokens.

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

```bash
python tools/prepare_semantic_masks.py \
  --json-path /root/private_data/data/COCO2017Mini/partition/data_info.json \
  --feature-root /root/private_data/data/COCO2017Mini/caption_feature_wmask \
  --tokenizer /root/private_data/PixArt-alpha-attentiongate/output/pretrained_models/t5-v1_1-xxl
```

Each feature archive receives `semantic_token_masks[3, 120]`:

1. complete noun-phrase tokens (including modifiers and compounds);
2. attribute tokens;
3. relation predicate tokens.

The command is resumable. Add `--overwrite` only when captions, tokenizer, or
label rules have changed.

## 2. Train

```bash
accelerate launch train_scripts/train.py configs/PixArt_xl2_coco2014_semantic.py
```

The configuration enables `load_semantic_masks`, `semantic_conditioning`, and
`train_semantic_only`. The adapters are zero initialized, so loading a baseline
PixArt checkpoint is safe: its first forward pass remains the baseline until
the adapters are optimized.

## 3. Collect experiment tables

With `save_experiment_tables=True`, the main training process writes
spreadsheet-friendly files to `<work_dir>/experiment_tables/`:

- `training_metrics.csv`: loss, learning rate, gradient norm, GPU memory,
  semantic-mask presence ratios, condition/adapter norms, time gates, and
  layer scales at `metrics_log_interval`;
- `epoch_summary.csv`: epoch averages of the same numerical fields;
- `<experiment_name>_metadata.json`: the resolved experiment configuration.

Use a distinct `experiment_name` for every method and keep the resulting CSV
files alongside generated images and evaluation outputs.

## 4. Verify one archive

```bash
python - <<'PY'
import numpy as np
feature = np.load('/path/to/a/feature.npz')
print(feature['caption_feature'].shape)
print(feature['semantic_token_masks'].shape)
print(feature['semantic_token_masks'].sum(axis=1))
PY
```

An all-zero object, attribute, or relation row is valid for prompts that do not
contain that role. The global condition is always derived from the original
T5 attention mask and is never replaced by pseudo labels.
