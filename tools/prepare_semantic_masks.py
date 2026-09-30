#!/usr/bin/env python3
"""Add dependency/rule-derived T5 semantic masks to PixArt feature archives.

The script mirrors PixArt's T5 caption cleaning before both spaCy parsing and
T5 tokenization. It stores ``semantic_token_masks`` with shape ``[3, T]`` in
each archive: object-root, attribute, and relation token masks respectively.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from diffusion.model.t5 import T5Embedder
try:
    from .test_text_pseudolabels import build_labels, load_nlp, non_space_tokens
except ImportError:  # also support ``python tools/prepare_semantic_masks.py``
    from test_text_pseudolabels import build_labels, load_nlp, non_space_tokens


def feature_stem(image_path):
    return '_'.join(image_path.rsplit('/', 1)).rsplit('.', 1)[0]


def clean_caption(text):
    # Feature extraction calls this method twice through text_preprocessing.
    cleaner = T5Embedder.__new__(T5Embedder)
    return cleaner.clean_caption(cleaner.clean_caption(text))


def overlapping_indices(word_indices, word_tokens, offsets):
    """Expand compact spaCy word indices to overlapping T5 sub-token indices."""
    result = set()
    for word_index in word_indices:
        token = word_tokens[word_index]
        start, end = token.idx, token.idx + len(token.text)
        for t5_index, (t5_start, t5_end) in enumerate(offsets):
            if t5_end > t5_start and min(end, t5_end) > max(start, t5_start):
                result.add(t5_index)
    return result


def build_semantic_masks(prompt, nlp, tokenizer, sequence_length):
    prompt = clean_caption(prompt)
    doc = nlp(prompt)
    labels = build_labels(doc)
    word_tokens = non_space_tokens(doc)
    encoded = tokenizer(
        prompt,
        max_length=sequence_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoded['offset_mapping']
    if hasattr(offsets, 'tolist'):
        offsets = offsets.tolist()

    # Preserve the complete noun phrase. Using only the syntactic root drops
    # modifiers such as "red", "small", and compound nouns such as "fire
    # truck", leaving the object branch without the information it is meant
    # to represent.
    object_words = [
        index
        for obj in labels['objects']
        for index in obj['token_indices']
    ]
    attribute_words = labels['attribute_token_indices']
    relation_words = [index for relation in labels['relations'] for index in relation['token_indices']]
    groups = (object_words, attribute_words, relation_words)
    masks = np.zeros((3, sequence_length), dtype=np.float32)
    for group_index, word_indices in enumerate(groups):
        for token_index in overlapping_indices(word_indices, word_tokens, offsets):
            masks[group_index, token_index] = 1.
    return masks


def build_semantic_edges(prompt, nlp, tokenizer, sequence_length, max_edges=24):
    """Parsed object->attribute edges for one caption, as the model consumes them.

    Mirrors ``build_semantic_masks`` exactly: same caption cleaning, same spaCy
    document, same T5 sub-token expansion.  Only the mapping is different -- an
    object token is paired with every attribute token attached to that object in
    the dependency parse, instead of all attributes being pooled into one mask.

    Returns ``(edges, count)`` where ``edges`` is an int64 ``(max_edges, 2)``
    array padded with -1 and ``count`` is the number of real rows.  The order of
    the pairs -- and therefore the truncation to ``max_edges`` -- matches
    ``tools/build_edge_index.py``, which produced the training-time index.
    """
    prompt = clean_caption(prompt)
    doc = nlp(prompt)
    labels = build_labels(doc)
    word_tokens = non_space_tokens(doc)
    encoded = tokenizer(
        prompt,
        max_length=sequence_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoded['offset_mapping']
    if hasattr(offsets, 'tolist'):
        offsets = offsets.tolist()

    pairs = []
    for obj in labels['objects']:
        attributes = obj.get('attributes') or []
        if not attributes:
            continue
        object_tokens = overlapping_indices(obj['token_indices'], word_tokens, offsets)
        attribute_tokens = overlapping_indices(attributes, word_tokens, offsets)
        for obj_token in sorted(object_tokens):
            for attr_token in sorted(attribute_tokens):
                if obj_token == attr_token:
                    continue
                pairs.append((obj_token, attr_token))

    edges = np.full((max_edges, 2), -1, dtype=np.int64)
    count = min(len(pairs), max_edges)
    for row, (obj_token, attr_token) in enumerate(pairs[:count]):
        edges[row, 0] = obj_token
        edges[row, 1] = attr_token
    return edges, count


def update_archive(path, masks, overwrite):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if 'semantic_token_masks' in values and not overwrite:
        return False
    values['semantic_token_masks'] = masks
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **values)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json-path', required=True, type=Path, help='InternalData partition/data_info.json')
    parser.add_argument('--feature-root', required=True, type=Path, help='caption_feature_wmask directory')
    parser.add_argument('--tokenizer', required=True, help='Same T5 tokenizer path used for feature extraction')
    parser.add_argument('--spacy-model', default='en_core_web_sm')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='0 processes every record')
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit('transformers is required in the feature-preparation environment') from exc
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not getattr(tokenizer, 'is_fast', False):
        raise SystemExit('A fast tokenizer is required because offset_mapping is needed for alignment.')
    nlp = load_nlp(args.spacy_model)
    records = json.loads(args.json_path.read_text(encoding='utf-8'))
    if args.limit:
        records = records[:args.limit]

    updated = missing = 0
    for record in tqdm(records, desc='Preparing semantic masks'):
        archive_path = args.feature_root / f'{feature_stem(record["path"])}.npz'
        if not archive_path.exists():
            missing += 1
            continue
        with np.load(archive_path, allow_pickle=False) as archive:
            sequence_length = archive['caption_feature'].shape[1]
        masks = build_semantic_masks(record['prompt'], nlp, tokenizer, sequence_length)
        updated += update_archive(archive_path, masks, args.overwrite)
    print(f'Updated {updated} feature archives; skipped {missing} missing feature archives.')


if __name__ == '__main__':
    main()

# Colour vocabulary of the CompBench colour category (extracted from its own
# prompts) plus the usual near-synonyms, so a tag can be restricted to colour
# attributes and leave shape / texture / spatial structure alone.
COLOR_WORDS = frozenset("""
blue red brown green black yellow white gold silver pink gray grey purple beige
orange teal cyan magenta maroon tan violet navy olive turquoise crimson scarlet
amber ivory cream bronze copper""".split())


def color_attribute_indices(prompt, nlp, tokenizer, sequence_length):
    """Sub-token indices of attributes whose WORD is a colour word."""
    prompt = clean_caption(prompt)
    doc = nlp(prompt)
    labels = build_labels(doc)
    word_tokens = non_space_tokens(doc)
    encoded = tokenizer(prompt, max_length=sequence_length, padding='max_length',
                        truncation=True, add_special_tokens=True,
                        return_offsets_mapping=True)
    offsets = encoded['offset_mapping']
    if hasattr(offsets, 'tolist'):
        offsets = offsets.tolist()
    out = set()
    for wi in labels['attribute_token_indices']:
        if wi < len(word_tokens) and word_tokens[wi].text.lower() in COLOR_WORDS:
            out |= overlapping_indices([wi], word_tokens, offsets)
    return out
