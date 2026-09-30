#!/usr/bin/env python3
"""Run the OFFICIAL T2I-CompBench++ evaluation metrics on generated images.

This driver calls the upstream evaluator scripts from a pristine copy of
https://github.com/Karine-Huang/T2I-CompBench unmodified.  It only prepares a
per-run working directory (symlinked images + a dataset/ dir with verbatim line
subsets of the official prompt files) and parses the evaluators' own
``vqa_result.json`` outputs into per-image CSVs.

Metric names use the ``compbench_official_*`` prefix on purpose: these numbers
come from the official T2I-CompBench++ evaluators (BLIP-VQA / UniDet /
CLIPScore / 3-in-1), not from any local CLIP-based proxy.

Usage
-----
    python run_eval.py --method tpscda --category color [--limit 20]

``--limit N`` limits the run to the FIRST N prompts of the category
(N * 10 images, because 10 images per prompt is the official protocol).  When
``--limit`` covers all 300 prompts the run is a full official evaluation.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CATEGORIES = ['color', 'shape', 'texture', 'spatial', '3d_spatial',
              'numeracy', 'non_spatial', 'complex']

# 10 images per prompt is the official T2I-CompBench protocol
IMAGES_PER_PROMPT = 10

REPO = Path('/root/compbench_work/eval/T2I-CompBench')
IMAGES_ROOT = Path('/root/compbench_work/images')
RUN_ROOT = Path('/root/compbench_work/eval/runs')
RESULTS_ROOT = Path('/root/compbench_work/results')
BLIP_CKPT = Path('/root/compbench_work/weights/model_base_vqa_capfilt_large.pth')
UNIDET_WEIGHTS = REPO / 'UniDet_eval/experts/expert_weights'

# category -> (list of evaluator steps, official metric slug, human name)
CATEGORY_SPEC = {
    'color': (['blip'], 'compbench_official_blipvqa',
              'T2I-CompBench++ official Disentangled BLIP-VQA (attribute binding: color)'),
    'shape': (['blip'], 'compbench_official_blipvqa',
              'T2I-CompBench++ official Disentangled BLIP-VQA (attribute binding: shape)'),
    'texture': (['blip'], 'compbench_official_blipvqa',
                'T2I-CompBench++ official Disentangled BLIP-VQA (attribute binding: texture)'),
    'spatial': (['unidet2d'], 'compbench_official_unidet_2d_spatial',
                'T2I-CompBench++ official UniDet-based 2D spatial relationship metric'),
    '3d_spatial': (['unidet3d'], 'compbench_official_unidet_3d_spatial',
                   'T2I-CompBench++ official UniDet + depth 3D spatial relationship metric'),
    'numeracy': (['unidet_num'], 'compbench_official_unidet_numeracy',
                 'T2I-CompBench++ official UniDet-based numeracy metric'),
    'non_spatial': (['clip'], 'compbench_official_clipscore',
                    'T2I-CompBench++ official CLIPScore (non-spatial relationships)'),
    'complex': (['blip', 'unidet2d_complex', 'clip_complex', 'three_in_one'],
                'compbench_official_3in1',
                'T2I-CompBench++ official 3-in-1 metric (mean of CLIPScore, '
                'Disentangled BLIP-VQA and UniDet) for complex compositions'),
}

# Files whose hashes identify the exact evaluator code used.
EVALUATOR_FILES = [
    'BLIPvqa_eval/BLIP_vqa.py',
    'BLIPvqa_eval/configs/vqa.yaml',
    'UniDet_eval/2D_spatial_eval.py',
    'UniDet_eval/3D_spatial_eval.py',
    'UniDet_eval/numeracy_eval.py',
    'UniDet_eval/experts/model_bank.py',
    'UniDet_eval/experts/model_bank_3d.py',
    'UniDet_eval/experts/obj_detection/generate_dataset.py',
    'UniDet_eval/experts/obj_detection/generate_dataset_3d.py',
    'CLIPScore_eval/CLIP_similarity.py',
    '3_in_1_eval/3_in_1.py',
]

CHECKPOINTS = {
    'blip_vqa': {
        'model': 'BLIP w/ ViT-B, CapFilt-L, fine-tuned on VQA',
        'file': str(BLIP_CKPT),
        'source': ('https://storage.googleapis.com/sfr-vision-language-research/'
                   'BLIP/models/model_base_vqa_capfilt_large.pth'),
    },
    'unidet_rs200': {
        'model': 'UniDet (Unified learned OCIM) RS200 6x+2x',
        'file': str(UNIDET_WEIGHTS / 'Unified_learned_OCIM_RS200_6x+2x.pth'),
        'source': ('https://huggingface.co/shikunl/prismer/resolve/main/'
                   'expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth'),
    },
    'unidet_r50': {
        'model': 'UniDet (Unified learned OCIM) R50 6x+2x',
        'file': str(UNIDET_WEIGHTS / 'Unified_learned_OCIM_R50_6x+2x.pth'),
        'source': 'https://drive.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq',
    },
    'depth_dpt': {
        'model': 'DPT-Hybrid (MiDaS) depth estimator',
        'file': str(UNIDET_WEIGHTS / 'dpt_hybrid-midas-501f0c75.pt'),
        'source': ('https://huggingface.co/lllyasviel/ControlNet/resolve/main/'
                   'annotator/ckpts/dpt_hybrid-midas-501f0c75.pt'),
    },
    'clip_vitb32': {
        'model': 'OpenAI CLIP ViT-B/32',
        'file': '~/.cache/clip/ViT-B-32.pt',
        'source': ('https://openaipublic.azureedge.net/clip/models/'
                   '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/'
                   'ViT-B-32.pt'),
    },
    'bert_tokenizer': {
        'model': 'bert-base-uncased tokenizer (used by BLIP text encoder)',
        'file': 'HF hub cache',
        'source': 'https://huggingface.co/bert-base-uncased',
    },
}

# Minimum free VRAM (GiB) required before starting each evaluator step, with
# the evidence behind each number.  "measured" means an actual reported peak;
# anything not measured falls back to the --min-free-gb floor and MUST be
# refined from the per_step_peak recorded in config.json after the first real
# run in the dedicated window.
#
# Evidence notes:
#   blip      : MEASURED 1753 MB peak.  BLIPvqa_eval/BLIP/train_vqa_func.py
#               prints "max mem: 1753" at inference, captured in the Phase-1
#               transcript (/tmp/cbresearch/bliprun2.log).  The OOMs seen later
#               were NOT caused by BLIP: at the moment of the OOM the failing
#               process had only "56.34 MiB allocated by PyTorch" while a
#               *different* process held 7.34 GiB.  BLIP is cheap; the default
#               --blip-max-batch 0 therefore leaves its batch_size_test=32 alone.
#   unidet2d  : MEASURED only as a lower bound -- OOM'd at the official
#               batch_size=64, succeeded with ~7.7 GiB free at a capped batch.
#   unidet3d  : MEASURED lower bound -- OOM'd inside the DPT depth upsampling
#               (a single 2.75 GiB tensor) with ~7.9 GiB free, even at batch 1.
#   unidet_num: same lower bound as unidet2d (~7.7 GiB free was enough).
#   clip      : same lower bound as unidet2d (~7.7 GiB free was enough).
STEP_MIN_FREE_GB = {
    'blip': 3.0,            # measured peak 1753 MB + headroom
    'unidet2d': 6.0,        # floor; peak unmeasured
    'unidet2d_complex': 6.0,
    'unidet3d': 9.0,        # known to exceed 7.9 GiB free at batch 1
    'unidet_num': 6.0,      # floor; peak unmeasured
    'clip': 6.0,            # floor; peak unmeasured
    'clip_complex': 6.0,
    'three_in_one': 0.0,    # pure numpy, no GPU
}

# --------------------------------------------------------------------------
# GPU helpers
# --------------------------------------------------------------------------


def gpu_mem_mb():
    """(used_mb, free_mb) from nvidia-smi, or (None, None) if unavailable."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free',
             '--format=csv,noheader,nounits', '-i', '0'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=20).stdout.decode().strip().splitlines()[0]
        used, free = [int(x.strip()) for x in out.split(',')]
        return used, free
    except Exception:  # noqa: BLE001
        return None, None


class VramSampler(threading.Thread):
    """Poll nvidia-smi while an evaluator subprocess runs."""

    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self.min_free_mb = None
        self.max_used_mb = None
        self.samples = 0

    def run(self):
        while not self._stop_evt.is_set():
            used, free = gpu_mem_mb()
            if used is not None:
                self.samples += 1
                self.max_used_mb = (used if self.max_used_mb is None
                                    else max(self.max_used_mb, used))
                self.min_free_mb = (free if self.min_free_mb is None
                                    else min(self.min_free_mb, free))
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=15)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def log(msg, fh=None):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                if limit and f.tell() > limit:
                    break
    except OSError:
        return None
    return h.hexdigest()


# Batch-size cap machinery.
#
# Several official evaluators hardcode batch_size = 64 (UniDet 2D / 3D /
# numeracy) or read batch_size_test: 32 from configs/vqa.yaml (BLIP).  On a
# shared GPU that does not fit.  Rather than edit the official evaluator .py
# files, run_eval.py runs them through a tiny shim that clamps
# torch.utils.data.DataLoader's batch_size.  This is numerically safe for every
# evaluator used here: each of them yields one independent score per image
# (UniDet's collate_fn returns an unpadded per-image list; nothing is
# batch-normalised across images at inference), and all of them concatenate
# results in loader order.  The shim lives in the run dir and is only active
# when --max-batch is set.
SHIM_SOURCE = '''
import os, runpy, sys
cap = int(os.environ.get('COMPBENCH_MAX_BATCH', '0') or 0)
if cap:
    import torch.utils.data as _tud
    _orig = _tud.DataLoader.__init__
    if not getattr(_orig, '_compbench_capped', False):
        def _patched(self, dataset, batch_size=1, *a, **kw):
            if isinstance(batch_size, int) and batch_size > cap:
                batch_size = cap
            return _orig(self, dataset, batch_size, *a, **kw)
        _patched._compbench_capped = True
        _tud.DataLoader.__init__ = _patched
target = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(target))
sys.argv = [target] + sys.argv[2:]
runpy.run_path(target, run_name='__main__')
'''

_SHIM = {'path': None, 'cap': 0, 'blip_cap': 0}


def write_shim(run_dir):
    p = run_dir / '_compbench_batchcap_shim.py'
    p.write_text(SHIM_SOURCE)
    return p


def wrap_with_shim(cmd):
    """Prefix cmd with the batch-cap shim, if a cap is configured."""
    if _SHIM['path'] and _SHIM['cap']:
        return [sys.executable, str(_SHIM['path'])] + list(cmd)
    return list(cmd)


def run_cmd(cmd, cwd, log_path, timeout=None, env=None):
    """Run a subprocess, tee its output to log_path. Returns (rc, seconds)."""
    t0 = time.time()
    with open(log_path, 'a') as fh:
        fh.write('\n' + '=' * 78 + '\n')
        fh.write(f'$ (cd {cwd}) {" ".join(str(c) for c in cmd)}\n')
        if env:
            fh.write(f'  extra env: {env}\n')
        fh.write('=' * 78 + '\n')
        fh.flush()
        run_env = dict(os.environ)
        # reduces allocator fragmentation when the GPU is shared; does not
        # change any numerics
        run_env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        if env:
            run_env.update({k: str(v) for k, v in env.items()})
        try:
            p = subprocess.run([str(c) for c in cmd], cwd=str(cwd),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout, env=run_env)
            out = p.stdout.decode('utf-8', 'replace')
            rc = p.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b'').decode('utf-8', 'replace')
            out += f'\n*** TIMEOUT after {timeout}s ***\n'
            rc = -9
        fh.write(out)
        fh.flush()
    return rc, time.time() - t0


# --------------------------------------------------------------------------
# Manifest handling
# --------------------------------------------------------------------------

class Sample:
    __slots__ = ('category', 'prompt', 'prompt_index', 'repeat',
                 'global_index', 'image', 'path')

    def __init__(self, category, prompt, prompt_index, repeat, global_index,
                 image, path):
        self.category = category
        self.prompt = prompt
        self.prompt_index = prompt_index
        self.repeat = repeat
        self.global_index = global_index
        self.image = image
        self.path = path

    @property
    def prompt_id(self):
        return f'{self.category}+{self.prompt_index}'


def official_prompts(category):
    """Read the official val prompt list, verbatim, in file order."""
    path = REPO / 'examples/dataset' / f'{category}_val.txt'
    with open(path) as fh:
        return [ln.rstrip('\n') for ln in fh if ln.strip()]


def load_samples(method, category, limit, images_dir=None):
    """Return the selected Sample list.

    Prefers the generator's manifest_<category>.csv; falls back to deriving the
    manifest from the image filenames (the official convention is
    ``<prompt>_<global index %06d>.png`` with a global counter, so
    ``prompt_index = global_index // 10`` and ``repeat = global_index % 10``).

    Both paths return one Sample per EXPECTED image, not per existing file, so
    images that were never generated are reported as failures rather than
    silently dropped from the denominator.
    """
    prompts = official_prompts(category)
    n_prompts = len(prompts) if limit is None else min(limit, len(prompts))

    img_root = Path(images_dir) if images_dir else (IMAGES_ROOT / method)
    manifest = img_root / f'manifest_{category}.csv'
    img_dir = img_root / category
    if not img_dir.is_dir():
        raise SystemExit(f'no image directory: {img_dir}')

    if manifest.is_file():
        source = f'manifest:{manifest}'
        rows = []
        with open(manifest, newline='') as fh:
            for r in csv.DictReader(fh):
                if r.get('category', category) != category:
                    continue
                gi = int(r['global_index'])
                rows.append(Sample(category, r['prompt'],
                                   int(r.get('prompt_index', gi // IMAGES_PER_PROMPT)),
                                   int(r.get('repeat', gi % IMAGES_PER_PROMPT)),
                                   gi, r['image'],
                                   img_dir / r['image']))
    else:
        source = (f'derived-from-filenames:{img_dir} '
                  '(generator manifest not written yet)')
        present = {}
        for name in sorted(os.listdir(img_dir)):
            if not name.endswith('.png'):
                continue
            stem = name[:-4]
            prompt, _, tail = stem.rpartition('_')
            if not prompt or not tail.isdigit():
                raise SystemExit(
                    f'unparseable image name (needs <prompt>_<%06d>.png): {name}')
            gi = int(tail)
            pi = gi // IMAGES_PER_PROMPT
            if pi < len(prompts) and prompt != prompts[pi]:
                raise SystemExit(
                    f'filename/global-index mismatch: {name!r} implies '
                    f'prompt_index {pi} but the official prompt there is '
                    f'{prompts[pi]!r}')
            present[gi] = name
        # Emit a row for EVERY expected image, whether or not the file exists,
        # so an unfinished generation job shows up as explicit failures instead
        # of being silently dropped.
        rows = []
        for pi in range(n_prompts):
            for rep in range(IMAGES_PER_PROMPT):
                gi = pi * IMAGES_PER_PROMPT + rep
                name = present.get(gi, f'{prompts[pi]}_{gi:06d}.png')
                rows.append(Sample(category, prompts[pi], pi, rep, gi, name,
                                   img_dir / name))

    # keep only the first n_prompts prompts; require the full 10 images each
    sel = [s for s in rows if s.prompt_index < n_prompts]
    sel.sort(key=lambda s: s.global_index)

    # sanity: prompt text must match the official prompt list
    for s in sel:
        if s.prompt_index < len(prompts) and s.prompt != prompts[s.prompt_index]:
            raise SystemExit(
                f'manifest/filename prompt mismatch at index {s.prompt_index}: '
                f'{s.prompt!r} != official {prompts[s.prompt_index]!r}')

    missing = [s for s in sel if not s.path.exists()]
    if missing:
        log(f'WARNING: {len(missing)} selected images do not exist yet '
            f'(e.g. {missing[0].image}) -- they will be reported as failures')
    return sel, n_prompts, source


# --------------------------------------------------------------------------
# Run directory preparation
# --------------------------------------------------------------------------

# Output directories written by the official evaluators.  They MUST be cleared
# before each run: the evaluators never truncate their own vqa_result.json, so a
# rerun with fewer images would silently re-parse the previous run's scores.
STALE_OUTPUT_DIRS = ['annotation_blip', 'annotation_clip', 'annotation_num',
                     'annotation_3_in_1', 'labels']


def prepare_run_dir(run_dir, samples, category, n_prompts, logfh):
    """Create run_dir/samples (symlinks) + run_dir/dataset (verbatim subsets)."""
    for name in STALE_OUTPUT_DIRS:
        d = run_dir / name
        if d.exists():
            shutil.rmtree(d)
            log(f'  cleared stale evaluator output: {d}', logfh)

    samples_dir = run_dir / 'samples'
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    for s in samples:
        link = samples_dir / s.image
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            os.symlink(s.path, link)
            linked += 1
        except OSError as e:
            log(f'  symlink failed for {s.image}: {e}', logfh)
    src_note = samples[0].path.parent if samples else 'n/a'
    log(f'  samples/: {linked} symlinks -> {src_note}', logfh)

    # dataset/ : verbatim line subsets of the official prompt files.
    ds = run_dir / 'dataset'
    ds.mkdir(parents=True, exist_ok=True)

    def subset_copy(src, dst, keep):
        with open(src) as fh:
            lines = [ln.rstrip('\n') for ln in fh if ln.strip()]
        if keep is None:
            kept = lines
        else:
            kept = [ln for ln in lines if ln in keep]
        with open(dst, 'w') as fh:
            for ln in kept:
                fh.write(ln + '\n')
        return len(lines), len(kept)

    prompts = official_prompts(category)
    sel_prompts = prompts[:n_prompts]
    keep = set(sel_prompts)

    total, kept = subset_copy(REPO / f'examples/dataset/{category}_val.txt',
                              ds / f'{category}_val.txt',
                              keep if n_prompts < len(prompts) else None)
    log(f'  dataset/{category}_val.txt: {kept}/{total} official lines '
        f'(verbatim{" subset" if n_prompts < len(prompts) else " = full file"})',
        logfh)

    # extra files needed by the complex / 3-in-1 path
    for extra in ('complex_val_spatial.txt', 'complex_val_action.txt'):
        src = REPO / f'examples/dataset/{extra}'
        if src.is_file() and category == 'complex':
            t, k = subset_copy(src, ds / extra, keep)
            log(f'  dataset/{extra}: {k}/{t} official lines (verbatim subset)', logfh)

    # numeracy needs relative dataset/new_objects.txt from anywhere
    nobj = REPO / 'examples/dataset/new_objects.txt'
    if nobj.is_file():
        shutil.copyfile(nobj, ds / 'new_objects.txt')

    return samples_dir


# --------------------------------------------------------------------------
# Evaluator steps
# --------------------------------------------------------------------------

def step_blip(run_dir, logfh, timeout):
    """BLIPvqa_eval/BLIP_vqa.py --out_dir=<run_dir>, cwd=BLIPvqa_eval."""
    cwd = REPO / 'BLIPvqa_eval'
    # BLIP's measured peak is only 1753 MB, so its batch_size_test=32 is left
    # alone unless the caller explicitly asks for a cap.
    saved = _SHIM['cap']
    _SHIM['cap'] = _SHIM['blip_cap']
    try:
        rc, secs = run_cmd(wrap_with_shim(['BLIP_vqa.py', f'--out_dir={run_dir}']),
                           cwd, run_dir / 'logs' / 'blip.log', timeout)
    finally:
        _SHIM['cap'] = saved
    return rc, secs


def step_unidet2d(run_dir, logfh, timeout, complex_mode=False):
    """UniDet_eval/2D_spatial_eval.py --outpath=<run_dir>, cwd=UniDet_eval.

    ``--complex`` uses argparse type=bool, so ANY non-empty string is True;
    we therefore pass the flag only when complex mode is wanted and never pass
    the literal 'False'.
    """
    cwd = REPO / 'UniDet_eval'
    cmd = wrap_with_shim(['2D_spatial_eval.py', f'--outpath={run_dir}'])
    if complex_mode:
        cmd += ['--complex', 'True']
    rc, secs = run_cmd(cmd, cwd, run_dir / 'logs' / 'unidet2d.log', timeout)
    return rc, secs


def step_unidet3d(run_dir, logfh, timeout):
    """UniDet_eval/3D_spatial_eval.py --outpath=<run_dir>, cwd=UniDet_eval.

    NOTE: --outpath must NOT have a trailing slash or the depth-map paths
    computed inside the evaluator do not match where they were written.
    """
    cwd = REPO / 'UniDet_eval'
    # The DPT-Hybrid depth phase upsamples to a large feature map and OOMs even
    # at the global cap, so it gets its own (smaller) cap.
    saved = _SHIM['cap']
    _SHIM['cap'] = min(saved, 2) if saved else 0
    try:
        rc, secs = run_cmd(wrap_with_shim(['3D_spatial_eval.py',
                                           f'--outpath={run_dir}']),
                           cwd, run_dir / 'logs' / 'unidet3d.log', timeout)
    finally:
        _SHIM['cap'] = saved
    return rc, secs


def step_unidet_numeracy(run_dir, logfh, timeout):
    """UniDet_eval/numeracy_eval.py --outpath=<run_dir>, cwd=UniDet_eval."""
    cwd = REPO / 'UniDet_eval'
    rc, secs = run_cmd(wrap_with_shim(['numeracy_eval.py',
                                       f'--outpath={run_dir}']),
                       cwd, run_dir / 'logs' / 'numeracy.log', timeout)
    return rc, secs


def step_clip(run_dir, logfh, timeout, complex_mode=False):
    """CLIPScore_eval/CLIP_similarity.py --outpath=<run_dir>, cwd=repo root."""
    # CLIPScore_eval has no hardcoded batch size (it scores one image at a
    # time) and is run from the repo root, so it needs no shim.
    cmd = [sys.executable, 'CLIPScore_eval/CLIP_similarity.py',
           f'--outpath={run_dir}']
    if complex_mode:
        cmd += ['--complex', 'True']
    rc, secs = run_cmd(cmd, REPO, run_dir / 'logs' / 'clip.log', timeout)
    return rc, secs


def step_three_in_one(run_dir, logfh, timeout):
    """3_in_1_eval/3_in_1.py --outpath=<run_dir> --data_path=<run_dir>/dataset."""
    cwd = REPO / '3_in_1_eval'
    rc, secs = run_cmd([sys.executable, '3_in_1.py',
                        f'--outpath={run_dir}',
                        f'--data_path={run_dir / "dataset"}'],
                       cwd, run_dir / 'logs' / '3in1.log', timeout)
    return rc, secs


STEP_FUNCS = {
    'blip': step_blip,
    'unidet2d': step_unidet2d,
    'unidet2d_complex': lambda d, l, t: step_unidet2d(d, l, t, complex_mode=True),
    'unidet3d': step_unidet3d,
    'unidet_num': step_unidet_numeracy,
    'clip': step_clip,
    'clip_complex': lambda d, l, t: step_clip(d, l, t, complex_mode=True),
    'three_in_one': step_three_in_one,
}

# where each evaluator writes its result, relative to the run dir, and how its
# question_id relates to the manifest
RESULT_FILES = {
    'blip': ('annotation_blip/vqa_result.json', 'positional'),
    'unidet2d': ('labels/annotation_obj_detection_2d/vqa_result.json', 'global_index'),
    'unidet3d': ('labels/annotation_obj_detection_3d/vqa_result.json', 'global_index'),
    'unidet_num': ('annotation_num/vqa_result.json', 'global_index'),
    'clip': ('annotation_clip/vqa_result.json', 'positional'),
    # 3_in_1 builds its array prompt-major over --data_path/complex_val.txt,
    # so question_id = dataset_order_index * 10 + repeat.
    'three_in_one': ('annotation_3_in_1/vqa_result.json', 'prompt_major'),
}


def parse_result(run_dir, step, samples, dataset_order):
    """Return (dict sample_key -> score, error_message_or_None)."""
    # 'unidet2d_complex' and 'clip_complex' write to the same locations as
    # their non-complex counterparts
    base_step = step[:-len('_complex')] if step.endswith('_complex') else step
    rel, mode = RESULT_FILES[base_step]
    path = run_dir / rel
    if not path.is_file():
        return {}, f'{step}: expected result file not found: {path}'
    try:
        with open(path) as fh:
            rows = json.load(fh)
    except Exception as e:  # noqa: BLE001
        return {}, f'{step}: could not parse {path}: {e}'

    if mode == 'positional':
        key = lambda qid: samples[qid].global_index if 0 <= qid < len(samples) else None  # noqa: E731
    elif mode == 'global_index':
        key = lambda qid: qid  # noqa: E731
    else:  # prompt_major
        def key(qid):
            order, rep = divmod(qid, IMAGES_PER_PROMPT)
            if order >= len(dataset_order):
                return None
            entry = dataset_order[order]
            return entry * IMAGES_PER_PROMPT + rep

    out = {}
    for r in rows:
        try:
            gi = key(int(r['question_id']))
        except (KeyError, TypeError, ValueError):
            continue
        if gi is None:
            continue
        try:
            out[gi] = float(r['answer'])
        except (TypeError, ValueError) as e:
            out[gi] = e
    return out, None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Official T2I-CompBench++ evaluation driver.')
    ap.add_argument('--method', required=True,
                    choices=['tpscda', 'frozen', 'pooled', 'fullspan', 'tokenpair',
                             'gateopt', 'b1', 'b2', 'b2_neutral', 'b2_frozengate',
                             'probe_rolegate', 'probe_early', 'probe_late',
                             'probe_random', 'tpscda_nodistill', 'probe_ditpair', 'pair_replace',
                             # pair-replacement edit rules, scored at inference
                             # on the paper's TP-SCDA checkpoint
                             'probe_raise', 'probe_equalize', 'probe_gated', 'probe_outside',
                             'probe_embedbind', 'frozen_embedbind',
                             'frozen_eb010', 'frozen_eb020', 'frozen_eb030',
                             'frozen_ebn015', 'frozen_ebn030',
                             'frozen_ebc015', 'frozen_ebc030',
                             'frozen_ebp2', 'frozen_ebp3', 'frozen_eb040', 'frozen_eb050',
                             'frozen_ebcol020', 'frozen_ebs020', 'frozen_ebs030', 'frozen_outside',
                             'frozen_sink50', 'frozen_sink00', 'frozen_current',
                             # seed replication of the colour result
                             'frozen_current_s44', 'frozen_current_s45',
                             'frozen_eb020_s44', 'frozen_eb020_s45',
                             # replacement scheme (reweight) at 25/50/80%
                             'tpscda_nd_rw25', 'tpscda_nd_rw50', 'tpscda_nd_rw80'])
    ap.add_argument('--category', required=True, choices=CATEGORIES)
    ap.add_argument('--limit', type=int, default=None,
                    help='use only the FIRST N prompts of the category '
                         '(N*10 images). Default: all 300 prompts.')
    ap.add_argument('--images-dir', default=None,
                    help='override the image root for this method; default '
                         '/root/compbench_work/images/<method>. Used to point '
                         'at fixture directories for plumbing validation.')
    ap.add_argument('--run-root', default=str(RUN_ROOT))
    ap.add_argument('--results-root', default=str(RESULTS_ROOT))
    ap.add_argument('--target-dir', default=None,
                    help='results subdir; default <method>/<category>')
    ap.add_argument('--timeout', type=int, default=7200,
                    help='per-evaluator-step timeout in seconds')
    ap.add_argument('--max-batch', type=int, default=8,
                    help='clamp the evaluators\' DataLoader batch size to this '
                         'value (official code hardcodes 64 for UniDet and 32 '
                         'for BLIP, which OOMs next to a running training job). '
                         '0 disables the shim.')
    ap.add_argument('--blip-max-batch', type=int, default=0,
                    help='separate batch cap for BLIP (0 = leave the official '
                         'batch_size_test=32 alone; BLIP peaks at only 1753 MB).')
    ap.add_argument('--min-free-gb', type=float, default=6.0,
                    help='abort (exit 2) before starting an evaluator if the '
                         'GPU has less than this much free memory. Applied as a '
                         'floor on top of the per-step requirements in '
                         'STEP_MIN_FREE_GB. 0 disables.')
    ap.add_argument('--skip-vram-check', action='store_true',
                    help='do not check free VRAM before each step')
    args = ap.parse_args()

    category = args.category
    steps, metric, metric_long = CATEGORY_SPEC[category]
    run_dir = Path(args.run_root) / args.method / category
    target_dir = args.target_dir or f'{args.method}/{category}'
    res_dir = Path(args.results_root) / target_dir
    res_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)

    logfh = open(run_dir / 'run.log', 'a')
    log(f'=== run_eval.py method={args.method} category={category} '
        f'limit={args.limit} max_batch={args.max_batch} ===', logfh)

    if (args.max_batch and args.max_batch > 0) or (args.blip_max_batch or 0) > 0:
        _SHIM['path'] = write_shim(run_dir)
    _SHIM['cap'] = args.max_batch or 0
    _SHIM['blip_cap'] = args.blip_max_batch or 0
    if _SHIM['cap']:
        log(f'UniDet batch-size cap active: DataLoader batch_size clamped to '
            f'{_SHIM["cap"]} via {_SHIM["path"]}', logfh)
    if _SHIM['blip_cap']:
        log(f'BLIP batch-size cap active: clamped to {_SHIM["blip_cap"]}', logfh)
    if not _SHIM['cap'] and not _SHIM['blip_cap']:
        log('batch-size caps disabled (official batch sizes in effect)', logfh)

    samples, n_prompts, manifest_source = load_samples(
        args.method, category, args.limit, args.images_dir)
    n_images = len(samples)
    log(f'selected {n_images} images over {n_prompts} prompts '
        f'(manifest source: {manifest_source})', logfh)

    prepare_run_dir(run_dir, samples, category, n_prompts, logfh)

    # dataset_order[i] = global_index of the i-th prompt in the run's
    # complex_val.txt.  Derived from the file we just wrote, so it is correct
    # regardless of how the subset was chosen.
    ds_file = run_dir / 'dataset' / f'{category}_val.txt'
    prompts = official_prompts(category)
    with open(ds_file) as fh:
        ds_prompts = [ln.rstrip('\n') for ln in fh if ln.strip()]
    dataset_order = []
    for p in ds_prompts:
        if p not in prompts:
            raise SystemExit(f'prompt in run dataset not in official list: {p!r}')
        # prompt_index of the i-th line of the run's complex_val.txt; parse_result
        # turns this into the global sequential image index
        dataset_order.append(prompts.index(p))

    step_times = {}
    step_rcs = {}
    step_vram = {}
    scores = {}
    parse_errors = {}
    abort_reason = None

    for step in steps:
        log(f'--- step: {step} ---', logfh)
        need_gb = max(args.min_free_gb, STEP_MIN_FREE_GB.get(step, 0.0))
        if not args.skip_vram_check and need_gb > 0:
            used, free = gpu_mem_mb()
            if free is None:
                log('    VRAM preflight: nvidia-smi unavailable, skipping check',
                    logfh)
            elif free < need_gb * 1024:
                abort_reason = (
                    f'insufficient free VRAM before step "{step}": '
                    f'{free} MiB free ({used} MiB used) but {need_gb:.1f} GiB '
                    f'({int(need_gb * 1024)} MiB) required. Another job is '
                    f'using the GPU; serialize the runs or lower --min-free-gb.')
                log('    ABORT: ' + abort_reason, logfh)
                step_rcs[step] = None
                break
            else:
                log(f'    VRAM preflight ok: {free} MiB free / {used} MiB used '
                    f'(need {int(need_gb * 1024)} MiB)', logfh)

        sampler = VramSampler()
        sampler.start()
        rc, secs = STEP_FUNCS[step](run_dir, logfh, args.timeout)
        sampler.stop()

        step_rcs[step] = rc
        step_times[step] = secs
        step_vram[step] = {
            'max_gpu_used_mb': sampler.max_used_mb,
            'min_gpu_free_mb': sampler.min_free_mb,
            'samples': sampler.samples,
            'note': ('whole-GPU counters sampled via nvidia-smi every 2 s; '
                     'includes any co-running job'),
        }
        log(f'    rc={rc} wall={secs:.1f}s  '
            f'vram peak_used={sampler.max_used_mb} MiB '
            f'min_free={sampler.min_free_mb} MiB', logfh)
        got, err = parse_result(run_dir, step, samples, dataset_order)
        if err:
            log(f'    {err}', logfh)
            parse_errors[step] = err
        else:
            log(f'    parsed {len(got)} scores', logfh)
        scores[step] = got
        if rc != 0:
            log(f'    NOTE: {step} exited {rc}; continuing so failures are recorded',
                logfh)

    # the final metric is the LAST step of the chain
    final_step = steps[-1]
    final_scores = scores.get(final_step, {})

    # ---- per_prompt.csv -------------------------------------------------
    per_prompt = res_dir / 'per_prompt.csv'
    fields = ['prompt_id', 'prompt', 'seed', 'generated_image', 'metric',
              'prediction', 'success', 'failure', 'error_message',
              'category', 'prompt_index', 'repeat', 'global_index', 'status']
    n_ok = n_fail = 0
    failure_rows = []
    with open(per_prompt, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in samples:
            val = final_scores.get(s.global_index)
            row = dict(prompt_id=s.prompt_id, prompt=s.prompt, seed=s.repeat,
                       generated_image=s.image, metric=metric,
                       category=s.category, prompt_index=s.prompt_index,
                       repeat=s.repeat, global_index=s.global_index)
            if isinstance(val, float):
                row.update(prediction=f'{val:.6f}', success=1, failure=0,
                           error_message='', status='ok')
                n_ok += 1
            else:
                # figure out why
                if not s.path.exists():
                    msg = f'image file missing: {s.path}'
                elif val is not None and not isinstance(val, float):
                    msg = f'evaluator returned non-numeric answer: {val!r}'
                elif abort_reason:
                    msg = f'RUN ABORTED before scoring: {abort_reason}'
                else:
                    msg = (f'no score produced for global_index={s.global_index} '
                           f'by {final_step}'
                           + (f' ({parse_errors[final_step]})'
                              if final_step in parse_errors else ''))
                row.update(prediction='', success=0, failure=1,
                           error_message=msg, status='failed')
                failure_rows.append(dict(
                    prompt_id=s.prompt_id, prompt=s.prompt, seed=s.repeat,
                    generated_image=s.image, metric=metric, status='failed',
                    category=s.category, prompt_index=s.prompt_index,
                    repeat=s.repeat, global_index=s.global_index,
                    error_message=msg))
                n_fail += 1
            w.writerow(row)

    with open(res_dir / 'failures.csv', 'w', newline='') as fh:
        ffields = ['prompt_id', 'prompt', 'seed', 'generated_image', 'metric',
                   'status', 'category', 'prompt_index', 'repeat',
                   'global_index', 'error_message']
        w = csv.DictWriter(fh, fieldnames=ffields)
        w.writeheader()
        w.writerows(failure_rows)

    # ---- summary --------------------------------------------------------
    vals = [v for v in final_scores.values() if isinstance(v, float)]
    mean = sum(vals) / len(vals) if vals else None
    summary = {
        'method': args.method,
        'category': category,
        'metric': metric,
        'metric_long_name': metric_long,
        'official_mean_score': mean,
        'num_prompts': n_prompts,
        'num_images_selected': n_images,
        'num_images_scored': len(vals),
        'num_images_failed': n_fail,
        'is_full_official_run': (n_prompts == len(official_prompts(category))),
        'evaluator_chain': steps,
        'evaluator_exit_codes': step_rcs,
        'evaluator_wall_seconds': {k: round(v, 2) for k, v in step_times.items()},
        'evaluator_vram': step_vram,
        'evaluator_scores_parsed': {s: len(v) for s, v in scores.items()},
        'all_steps_scored_every_image': all(
            len(scores.get(s, {})) == n_images for s in steps
            if s != 'three_in_one') if n_images else None,
        'status': 'aborted_vram' if abort_reason else (
            'ok' if n_fail == 0 else 'completed_with_failures'),
        'abort_reason': abort_reason,
        'manifest_source': manifest_source,
        'seconds_per_image': (
            round(sum(step_times.values()) / n_images, 3) if n_images else None),
    }
    with open(res_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    with open(res_dir / 'summary.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['method', 'category', 'metric', 'official_mean_score',
                    'num_prompts', 'num_images_selected', 'num_images_scored',
                    'num_images_failed', 'is_full_official_run'])
        w.writerow([args.method, category, metric,
                    '' if mean is None else f'{mean:.6f}', n_prompts, n_images,
                    len(vals), n_fail, summary['is_full_official_run']])

    # ---- config.json ----------------------------------------------------
    code_hash = {}
    for rel in EVALUATOR_FILES:
        p = REPO / rel
        code_hash[rel] = sha256_file(p)
    git_commit = None
    try:
        git_commit = subprocess.run(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=20).stdout.decode().strip() or None
    except Exception:  # noqa: BLE001
        git_commit = None

    ckpts = {}
    if 'blip' in steps:
        ckpts['blip_vqa'] = CHECKPOINTS['blip_vqa']
        ckpts['bert_tokenizer'] = CHECKPOINTS['bert_tokenizer']
    if 'unidet2d' in steps or 'unidet2d_complex' in steps:
        ckpts['unidet_rs200'] = CHECKPOINTS['unidet_rs200']
    if 'unidet3d' in steps:
        ckpts['unidet_r50'] = CHECKPOINTS['unidet_r50']
        ckpts['depth_dpt'] = CHECKPOINTS['depth_dpt']
    if 'unidet_num' in steps:
        ckpts['unidet_r50'] = CHECKPOINTS['unidet_r50']
    if 'clip' in steps or 'clip_complex' in steps:
        ckpts['clip_vitb32'] = CHECKPOINTS['clip_vitb32']

    config = {
        'driver': 'run_eval.py',
        'driver_version': '1.0',
        'method': args.method,
        'category': category,
        'limit_prompts': args.limit,
        'images_per_prompt': IMAGES_PER_PROMPT,
        'num_prompts': n_prompts,
        'num_images': n_images,
        'run_dir': str(run_dir),
        'results_dir': str(res_dir),
        'evaluator': {
            'name': metric_long,
            'metric_slug': metric,
            'steps': steps,
            'source_repo': 'https://github.com/Karine-Huang/T2I-CompBench',
            'repo_dir': str(REPO),
            'git_commit': git_commit,
            'git_commit_note': (
                'repo copy is not a git checkout (no .git); code_hash below is '
                'the authoritative version identifier'),
            'code_sha256': code_hash,
            'official_evaluator_files_modified': False,
        },
        'vram': {
            'min_free_gb_floor': args.min_free_gb,
            'per_step_min_free_gb': {
                s: max(args.min_free_gb, STEP_MIN_FREE_GB.get(s, 0.0))
                for s in steps},
            'per_step_peak': step_vram,
            'preflight_enabled': not args.skip_vram_check,
            'note': ('peak/min counters are whole-GPU readings sampled every '
                     '2 s; subtract any co-running job. Only the blip minimum '
                     '(3.0 GiB) is backed by a measured peak (1753 MB); every '
                     'other value is the --min-free-gb floor or a lower bound '
                     'from an observed OOM, so refine them from per_step_peak '
                     'after the first run in a dedicated GPU window.'),
        },
        'gpu_adaptation': {
            'unidet_max_batch': args.max_batch,
            'blip_max_batch': args.blip_max_batch,
            'blip_note': ('0 means BLIP keeps its official batch_size_test=32; '
                          'its measured peak is 1753 MB so capping it only '
                          'wastes time'),
            'mechanism': ('subprocess shim that clamps '
                          'torch.utils.data.DataLoader batch_size; the official '
                          'evaluator .py files and configs are untouched'),
            'reason': ('official code hardcodes batch_size=64 (UniDet 2D/3D/'
                       'numeracy) and batch_size_test=32 (BLIP) which OOMs when '
                       'the GPU is shared with a training job'),
            'numerically_equivalent': ('every evaluator emits one independent '
                                       'score per image and preserves loader '
                                       'order; nothing is batch-normalised '
                                       'across images at inference'),
        },
        'checkpoints': ckpts,
        'dataset': {
            'source': 'examples/dataset/*.txt (official, unmodified)',
            'files_used': [f'{category}_val.txt'] + (
                ['complex_val_spatial.txt', 'complex_val_action.txt']
                if category == 'complex' else []),
            'run_dir_dataset_dir': str(run_dir / 'dataset'),
            'note': ('run_dir/dataset holds VERBATIM line subsets of the '
                     'official files (full files when --limit is not given), '
                     'because 3_in_1.py needs a prompt list matching the images'),
        },
        'manifest_source': manifest_source,
        'images_dir': (str(Path(args.images_dir)) if args.images_dir
                       else str(IMAGES_ROOT / args.method)),
        'status': summary['status'],
        'abort_reason': abort_reason,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(res_dir / 'config.json', 'w') as fh:
        json.dump(config, fh, indent=2)

    log('', logfh)
    log(f'RESULT {metric} = '
        f'{"n/a" if mean is None else f"{mean:.6f}"} '
        f'({len(vals)} scored, {n_fail} failed)', logfh)
    log(f'wrote {per_prompt}', logfh)
    log(f'wrote {res_dir / "summary.csv"} , summary.json, failures.csv, config.json',
        logfh)
    if abort_reason:
        log(f'ABORTED (exit 2): {abort_reason}', logfh)
    logfh.close()
    return 2 if abort_reason else 0


if __name__ == '__main__':
    sys.exit(main())
