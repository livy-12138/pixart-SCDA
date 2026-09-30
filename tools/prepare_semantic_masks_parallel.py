#!/usr/bin/env python3
"""Parallel semantic-mask preparation; skips archives already containing masks."""
import argparse, json, os, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from tqdm import tqdm
from tools.prepare_semantic_masks import build_semantic_masks
from tools.test_text_pseudolabels import load_nlp
from transformers import AutoTokenizer

_NLP = None
_TOK = None

def init_worker(model, tokenizer_path):
    global _NLP, _TOK
    _NLP = load_nlp(model)
    _TOK = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)

def stem(path):
    return '_'.join(path.rsplit('/', 1)).rsplit('.', 1)[0]

def process(item):
    record, feature_root, overwrite = item
    path = Path(feature_root) / f'{stem(record["path"])}.npz'
    if not path.exists(): return 'missing'
    try:
        with np.load(path, allow_pickle=False) as archive:
            if 'semantic_token_masks' in archive and not overwrite: return 'exists'
            values = {k: archive[k] for k in archive.files}
            length = values['caption_feature'].shape[1]
        values['semantic_token_masks'] = build_semantic_masks(record['prompt'], _NLP, _TOK, length)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as h:
            tmp = Path(h.name)
        np.savez_compressed(tmp, **values)
        os.replace(tmp, path)
        return 'updated'
    except Exception as exc:
        return f'error:{exc}'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--json-path',required=True); p.add_argument('--feature-root',required=True); p.add_argument('--tokenizer',required=True); p.add_argument('--workers',type=int,default=8); p.add_argument('--overwrite',action='store_true'); args=p.parse_args()
    records=json.loads(Path(args.json_path).read_text())
    jobs=[(r,args.feature_root,args.overwrite) for r in records]
    counts={}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=('en_core_web_sm',args.tokenizer)) as pool:
        for result in tqdm(pool.map(process,jobs), total=len(jobs), desc='Semantic masks'):
            key=result.split(':',1)[0]; counts[key]=counts.get(key,0)+1
    print(counts)

if __name__=='__main__': main()
