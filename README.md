# TP-SCDA — Semantic-Conditioned Token-Pair Adaptation for PixArt-α

A research fork of [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) that injects
text-semantic structure (objects, attributes, relations) into a **frozen** DiT backbone
through lightweight adapters, a token-pair cross-attention bias, and learnable layer gates.

Base revision: `cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892` (upstream, 2024-10-31).
License: Apache 2.0 (inherited from upstream — see `LICENSE`).

---

## Read this first: the headline result is negative

**TP-SCDA ties the frozen PixArt-α baseline on all eight T2I-CompBench++ metrics.**
On the checkpoints available here, the method does not measurably improve prompt
adherence. This is reported as-is rather than papered over.

Two defects were found along the way, and they are arguably the more useful
contribution of this work:

| ID | Finding | Status |
|---|---|---|
| **ISSUE-011** | The rewritten `MultiHeadCrossAttention` silently dropped the cross-attention key-padding mask, so every image patch attended to T5 padding tokens. All previously computed generation metrics were measured on degraded images. | **Fixed** (`cross_attn_key_padding` in `diffusion/model/nets/PixArt_blocks.py`) |
| **ISSUE-008** | The learnable layer gates never learned anything. Initialization at `±4.0` sits in the sigmoid saturation region, where `dλ/dw = 0.0177`. The maximum reachable parameter distance over the whole run (~0.052) was ~20× short of the ~1.0 needed to escape. | **Diagnosed + reworked**; effective learning rate improved ~570× per step |

Details: `docs/results/PROJECT_SUMMARY.md` (summary), `docs/results/EXPERIMENT_ISSUES.md`
(all 13 issues), `docs/results/GATE_OPTIMIZATION.md` (ISSUE-008 diagnosis).

### Table 5 — T2I-CompBench++ (8 categories, official evaluator)

Measured **after** the ISSUE-011 mask fix. 300 prompts/category, 1 image per prompt.

| Category | TP-SCDA | Frozen PixArt-α |
|---|---:|---:|
| Color | 0.3922 | 0.3956 |
| Shape | 0.4135 | 0.4158 |
| Texture | 0.4697 | 0.4701 |
| 2D Spatial | 0.1969 | 0.2040 |
| 3D Spatial | 0.3366 | 0.3388 |
| Numeracy | 0.4961 | 0.5086 |
| Non-spatial | 0.3093 | 0.3095 |
| Complex | 0.3312 | 0.3314 |

Table 6 (GenEval) was **not completed** — the official evaluator requires `mmdet` 2.x,
which is incompatible with the `mmcv` 2.x needed by the CompBench++ UniDet evaluator.
No substitute implementation was passed off as the official result.

---

## Method

Everything below trains **only** the added parameters. The PixArt DiT blocks and the
T5 encoder stay frozen.

**Trainable parameters: 8,569,065 / 619,618,753 (1.383%).**
Training: 3.72 h, peak 30.38 GB. Inference: 1.238 s/image vs 0.747 s/image for the
frozen baseline (peak 22.63 GB).

### SCDA — Semantic-Conditioned DiT Adapter (pooled)

T5 tokens are pooled by semantic role (global / object / attribute / relation) into four
condition vectors, mapped through a zero-initialized bottleneck adapter
(`down → SiLU → up`, with `up` zeroed so the first forward pass is exactly the baseline),
then injected per DiT layer with a timestep-dependent gate.

Implemented in `diffusion/model/nets/PixArt.py`:
`_build_semantic_conditions`, `_semantic_adapter_outputs`, `_semantic_residual`.

### TP-SCDA — token-pair SCDA (main method)

Instead of pooling, each image patch cross-attends directly to the role-marked T5 token
sequence, with an object→attribute pairing bias:

1. learned per-role logit bias (`role_bias`);
2. learned object↔attribute affinity (`object_pair_proj` / `attribute_pair_proj`);
3. a softplus-parameterized `pair_strength` weighting that bias;
4. four per-role residual outputs;
5. per-block injection weighted by `semantic_token_layer_gate` (a 28×4 learnable gate)
   and a single learnable strength scalar.

Implemented in `SemanticTokenCrossAttention` (`diffusion/model/nets/PixArt.py`).

### Attention gate

`PixArtBlock.forward` scales the cross-attention output by
`gate = 1 + condition_gate_scale * tanh(condition_gate(t))`, with `condition_gate`
zero-initialized so the model starts neutral. Note this is a *separate* mechanism from
the TP-SCDA semantic injection gates.

### Token-pair attention rewriting

With `pair_replace=True`, `MultiHeadCrossAttention` rewrites attention weights using the
`(object, attribute)` edge matrix, so attribute tokens inherit their noun's weights.
Seven modes are supported: `replace` (default), `raise`, `reweight`, `gated`, `equalize`,
`outside`, `sink`. See `diffusion/model/nets/PixArt_blocks.py`.

### LCAR (experimental, not in the paper)

`diffusion/model/nets/lcar.py` — local competitive attention redistribution. Modifies only
the token-pair branch's own logit bias (boosting the anchor attribute, suppressing other
objects); the backbone is untouched and no new trainable parameters are added.

---

## Repository layout

```
diffusion/            model code; TP-SCDA and LCAR live in model/nets/
  model/nets/PixArt.py          SCDA + TP-SCDA core
  model/nets/PixArt_blocks.py   token-pair attention rewriting, ISSUE-011 fix
  model/nets/lcar.py            LCAR
configs/            23 experiment configs (chained via exec() inheritance)
tools/              data preparation, training drivers, evaluation scripts
train_scripts/      train.py — the real training entry point
scripts/            inference / interface scripts
app/                Gradio demo (app_512.py, app_offline.py)
asset/              samples.txt (fixed 64-prompt eval set), binding test cases
docs/               experiment records, run instructions, paper drafts
  results/          per-issue records, metrics, raw CSVs
  reports/          paper drafts (markdown + docx) and figures
```

### Config inheritance

Configs chain via `exec()` rather than imports, e.g.
`token_pair_gate_opt → scda_token_pair → scda_fullspan → scda → scda_improved → scda`.
Editing a base config changes every config derived from it.

`docs/results/NAMING_KEY.md` is the authoritative mapping between evaluation key names
(`frozen`, `pooled`, `fullspan`, `tokenpair`, `tpscda`, `gateopt`, …) and the method each
one actually denotes. **Read it before comparing any numbers** — `tokenpair` is a Table 4
ablation, not the paper method; `tpscda` is the paper method.

---

## Setup

### CUDA

```bash
conda env create -f environment-linux.yml
conda activate pixart-linux
```

Python 3.10, PyTorch 2.1.1, CUDA 12.1 wheels; host driver ≥ 530. `mmcv` is pinned to
1.7.2 because the native training scripts import `mmcv.runner`, which `mmcv` 2.x removed.
See `docs/RUNNING_LINUX.md`.

### Hygon DCU

```bash
conda env create -f environment-dcu.yml
conda activate pixart-dcu
source /opt/dtk/env.sh
./setup_dcu.sh
```

See `docs/RUNNING_DCU.md`. `xformers` is deliberately omitted on DCU.

---

## Data preparation

```bash
python tools/prepare_coco2014.py \
  --annotations /datasets/coco2014/annotations/captions_train2014.json \
  --image-root  /datasets/coco2014 \
  --output-root /datasets/COCO2014Prepared \
  --max-items 20000 --verify-images

python tools/extract_features.py \
  --json_path  /datasets/COCO2014Prepared/partition/data_info.json \
  --dataset_root /datasets/coco2014 \
  --t5_save_root  /datasets/COCO2014Prepared/caption_feature_wmask \
  --vae_save_root /datasets/COCO2014Prepared/img_vae_features \
  --img_size 256 --pretrained_models_dir /path/to/pretrained_models
```

Semantic masks (role labels from spaCy dependency parsing, aligned to T5 sub-tokens via
fast-tokenizer character offsets):

```bash
pip install spacy && python -m spacy download en_core_web_sm

python tools/prepare_semantic_masks.py \
  --json-path     /datasets/COCO2014Prepared/partition/data_info.json \
  --feature-root  /datasets/COCO2014Prepared/caption_feature_wmask \
  --tokenizer     /path/to/t5-v1_1-xxl
```

Resumable; add `--overwrite` only when captions, tokenizer, or label rules change.

Full details: `tools/COCO2014_PREPARATION.md` and `tools/SEMANTIC_TRAINING.md`.

---

## Training

The real entry point is `train_scripts/train.py`:

```bash
accelerate launch train_scripts/train.py configs/PixArt_xl2_coco2014_semantic.py
```

With `save_experiment_tables=True`, the main process writes `training_metrics.csv`,
`epoch_summary.csv`, and `<experiment_name>_metadata.json` to
`<work_dir>/experiment_tables/`.

> **Note:** the upstream `train.sh` is **not** included — it referenced
> `train_scripts/train_controlnet.py` and ControlNet configs, all of which were removed in
> this fork. Use the `accelerate launch` command above.

---

## Evaluation

```bash
python tools/generate_scda_samples.py     # generate images from a run
python tools/evaluate_multiseed.py        # multi-seed scoring driver
python tools/score_clip_distribution.py
python tools/score_attribute_binding.py
python tools/score_binding_swap.py
python tools/score_image_distribution.py
python tools/summarize_binding_comparison.py
```

The CompBench++ evaluation scripts live in `docs/results/_work/`
(`run_eval.py`, `gen_compbench.py`).

---

## Before you try to reproduce — read this

1. **21 of the 23 configs contain hardcoded absolute paths** (`/root/...`, `/public/...`)
   pointing at the original machine's dataset, feature, and checkpoint locations. Every
   config you use must be edited. This is the single biggest obstacle to reproduction.

2. **No checkpoints are included.** Model weights, training outputs, and the HuggingFace
   cache are excluded from this repository. The paper's baseline is a frozen PixArt-XL-2
   checkpoint; the TP-SCDA results come from
   `coco2017_token_pair_learnable_layers/epoch_1_step_14786.pth`, which is not
   distributed here.

3. **No dataset is included.** COCO 2014/2017 and the extracted T5/VAE features must be
   regenerated with the tools above.

4. **The gate configuration is a config value, not a weight.** `semantic_token_gate_max`
   and its activation function are *not* recorded in checkpoints. Evaluating a checkpoint
   with the wrong gate setting silently means evaluating a different injection strength.
   See the table in `docs/results/NAMING_KEY.md`.

5. **Re-running the CompBench++ evaluator requires ~186 GB of baseline models**
   (SDXL, SD2.1, PixArt-Sigma, Sana, …) downloaded from HuggingFace. They are not in
   this repository.

6. **Line endings were normalized to LF** in this export. The original working tree had
   mixed CRLF/LF; shell scripts in particular would not have executed with CRLF.

7. **Visualization figures are excluded** to keep the repository small (the PNG/JPG/HTML
   outputs under `results/` total ~78 MB). The underlying arrays, CSVs, and JSON summaries
   *are* included under `docs/results/`.

8. **Internal document cross-references are stale.** Records under `docs/results/` were
   written when they lived at `results/`, so paths inside them may not match this layout.
   Likewise, some configs reference `/root/compbench_work/...`, which was the evaluation
   workspace on the original machine.

---

## Acknowledgements

Built on [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) (Apache 2.0).
Evaluation uses [T2I-CompBench / T2I-CompBench++](https://github.com/Karine-Huang/T2I-CompBench).
