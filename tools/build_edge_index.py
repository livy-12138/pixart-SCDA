#!/usr/bin/env python3
"""Build a compact object->attribute edge index for the training captions.

Why this exists
---------------
The cached feature archives (`caption_feature_wmask/*.npz`) contain only three
role masks (object / attribute / relation) and no edge structure, and no prompt
text either.  Both planned improvements need the *parsed* object-attribute
edges at training time:

  B1  make the pair-aware bias follow the parsed dependency edges instead of a
      purely learned content similarity;
  B2  add an explicit attention-alignment loss that rewards routing an object's
      patch attention onto its bound attribute token.

Rather than regenerating 118 294 archives (79 GB), this script derives the edges
once from `data_info.json` (which holds the raw captions) and stores them as a
sparse triple list keyed by the same stem the dataset uses.

Output (one file, a few tens of MB):
    edges: (n_edges, 3) int32  -> (sample_index, object_subtoken, attribute_subtoken)
    stems: (n_samples,) str    -> dataset sample key, i.e. feature_stem(path)

Usage:
    python tools/build_edge_index.py \
        --data-info /root/private_data/data/COCO2017Prepared/partition/data_info.json \
        --output    /root/private_data/data/COCO2017Prepared/partition/edge_index.npz
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from tools.prepare_semantic_masks import clean_caption, overlapping_indices
from tools.test_text_pseudolabels import build_labels, load_nlp, non_space_tokens


def feature_stem(image_path):
    """Must match tools/prepare_semantic_masks.feature_stem exactly."""
    return '_'.join(image_path.rsplit('/', 1)).rsplit('.', 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-info', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--tokenizer', default=str(ROOT / 'output/pretrained_models/t5_ckpts/t5-v1_1-xxl'))
    ap.add_argument('--sequence-length', type=int, default=120)
    ap.add_argument('--spacy-model', default='en_core_web_sm')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    import json
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    nlp = load_nlp(args.spacy_model)

    records = json.loads(args.data_info.read_text(encoding='utf-8'))
    if args.limit:
        records = records[:args.limit]
    print(f'{len(records)} captions')

    stems, edges = [], []
    n_no_edge = n_no_object = 0
    for index, record in enumerate(tqdm(records, desc='Building edge index')):
        prompt = clean_caption(record['prompt'])
        doc = nlp(prompt)
        labels = build_labels(doc)
        words = non_space_tokens(doc)
        encoded = tokenizer(
            prompt, max_length=args.sequence_length, padding='max_length',
            truncation=True, add_special_tokens=True, return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
        if hasattr(offsets, 'tolist'):
            offsets = offsets.tolist()

        stems.append(feature_stem(record['path']))
        objects = labels['objects']
        if not objects:
            n_no_object += 1
            continue
        found = False
        for obj in objects:
            attributes = obj.get('attributes') or []
            if not attributes:
                continue
            # Expand both sides from spaCy word indices to T5 sub-token indices.
            object_tokens = overlapping_indices(obj['token_indices'], words, offsets)
            attribute_tokens = overlapping_indices(attributes, words, offsets)
            for obj_token in sorted(object_tokens):
                for attr_token in sorted(attribute_tokens):
                    if obj_token == attr_token:
                        continue
                    edges.append((index, obj_token, attr_token))
                    found = True
        if not found:
            n_no_edge += 1

    edge_array = np.asarray(edges, dtype=np.int32).reshape(-1, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,
                        edges=edge_array,
                        stems=np.asarray(stems, dtype=object).astype('U'))
    print(f'\nwrote {args.output}')
    print(f'  samples           : {len(stems)}')
    print(f'  edges             : {len(edge_array)}')
    print(f'  samples with edges: {len(stems) - n_no_object - n_no_edge}')
    print(f'  no object parsed  : {n_no_object}')
    print(f'  object but no edge: {n_no_edge}')
    if len(edge_array):
        print(f'  object tokens/edge: mean {edge_array[:,1].size/len(stems):.2f} per sample')


if __name__ == '__main__':
    main()
