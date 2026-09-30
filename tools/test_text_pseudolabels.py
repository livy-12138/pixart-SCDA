#!/usr/bin/env python3
"""Inspect English dependency/rule-based pseudo labels for T2I prompts.

This is an analysis tool only. It does not train a token-role classifier and
does not modify the PixArt model. Install spaCy and an English pipeline first:
``pip install spacy && python -m spacy download en_core_web_sm``.
"""

import argparse
import json
from pathlib import Path


def load_nlp(model_name):
    try:
        import spacy
    except ImportError as exc:
        raise SystemExit("spaCy is required: pip install spacy") from exc
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{model_name}' is unavailable. Install it with: "
            f"python -m spacy download {model_name}"
        ) from exc


def non_space_tokens(doc):
    return [token for token in doc if not token.is_space]


def noun_chunks(doc):
    """Return noun phrases and their root tokens, with a parser fallback."""
    try:
        chunks = list(doc.noun_chunks)
    except ValueError:
        # Blank/tokenizer-only pipelines have no dependency parser; fall back
        # to individual POS-tagged nouns instead of crashing the mask pass.
        chunks = []
    if chunks:
        return chunks
    return [token.doc[token.i : token.i + 1] for token in doc
            if token.pos_ in {"NOUN", "PROPN"}]


def build_labels(doc):
    tokens = non_space_tokens(doc)
    token_ids = {token.i: index for index, token in enumerate(tokens)}
    objects = []
    object_by_root = {}

    for chunk in noun_chunks(doc):
        root = chunk.root
        if root.pos_ not in {"NOUN", "PROPN"}:
            continue
        indices = [token_ids[token.i] for token in chunk if not token.is_punct]
        if not indices:
            continue
        # Deduplicate nested chunks while retaining the longer phrase.
        if root.i in object_by_root:
            current = objects[object_by_root[root.i]]
            current["token_indices"] = sorted(set(current["token_indices"] + indices))
            current["text"] = " ".join(tokens[i].text for i in current["token_indices"])
            continue
        object_by_root[root.i] = len(objects)
        objects.append({
            "text": chunk.text,
            "root": root.text,
            "root_index": token_ids[root.i],
            "token_indices": indices,
            "attributes": [],
        })

    def object_index(token):
        return object_by_root.get(token.i)

    def first_object_index(*candidates):
        for candidate in candidates:
            index = object_index(candidate)
            if index is not None:
                return index
        return None

    attribute_indices = set()
    for obj in objects:
        # ``root_index`` is in the compact non-space token list; convert back
        # to spaCy's document index before traversing dependency children.
        root_token = tokens[obj["root_index"]]
        for child in root_token.children:
            if child.dep_ in {"amod", "compound", "nummod", "poss"}:
                index = token_ids.get(child.i)
                if index is not None:
                    obj["attributes"].append(index)
                    attribute_indices.add(index)

    relations = []
    relation_deps = {"prep", "dobj", "pobj", "attr", "xcomp", "advmod"}
    relation_keys = set()
    for verb in doc:
        if verb.pos_ not in {"VERB", "AUX"}:
            continue
        subjects = [child for child in verb.children if child.dep_ in {"nsubj", "nsubjpass"}]
        # In prompts such as "a person riding a bicycle", spaCy parses the
        # participle as acl with the governing noun as its implicit subject.
        if not subjects and verb.dep_ in {"acl", "relcl"}:
            subjects = [verb.head]
        targets = [child for child in verb.children if child.dep_ in relation_deps]
        for subject in subjects:
            subject_id = first_object_index(subject, subject.head)
            for target in targets:
                candidate = target
                if target.dep_ == "prep":
                    candidate = next((child for child in target.children if child.dep_ == "pobj"), target)
                target_id = first_object_index(candidate, candidate.head)
                if subject_id is None or target_id is None or subject_id == target_id:
                    continue
                predicate = verb.text
                if target.dep_ == "prep" and target.text.lower() != predicate.lower():
                    predicate = f"{predicate} {target.text}"
                relation_token_indices = [token_ids[verb.i]]
                if target.dep_ == "prep" and target.i in token_ids:
                    relation_token_indices.append(token_ids[target.i])
                relations.append({
                    "subject": subject_id,
                    "predicate": predicate,
                    "object": target_id,
                    "token_indices": relation_token_indices,
                })
                relation_keys.add((subject_id, predicate, target_id))

    # Nominal spatial relations, e.g. "a car beside a bicycle", have no verb.
    for prep in doc:
        if prep.dep_ != "prep" or prep.head.pos_ not in {"NOUN", "PROPN"}:
            continue
        target = next((child for child in prep.children if child.dep_ == "pobj"), None)
        if target is None:
            continue
        subject_id = first_object_index(prep.head, prep.head.head)
        target_id = first_object_index(target, target.head)
        if subject_id is None or target_id is None or subject_id == target_id:
            continue
        predicate = prep.text
        key = (subject_id, predicate, target_id)
        if key not in relation_keys:
            relations.append({
                "subject": subject_id,
                "predicate": predicate,
                "object": target_id,
                "token_indices": [token_ids[prep.i]],
            })
            relation_keys.add(key)

    return {
        "tokens": [{"index": i, "text": token.text, "pos": token.pos_, "dep": token.dep_}
                   for i, token in enumerate(tokens)],
        "global_token_indices": list(range(len(tokens))),
        "objects": objects,
        "attribute_token_indices": sorted(attribute_indices),
        "relations": relations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", help="English prompt; repeat for multiple prompts")
    parser.add_argument("--file", type=Path, help="UTF-8 file with one prompt per line")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy English pipeline")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON")
    args = parser.parse_args()
    prompts = list(args.text or [])
    if args.file:
        prompts.extend(line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not prompts:
        parser.error("provide --text or --file")

    nlp = load_nlp(args.model)
    results = [{"prompt": prompt, "labels": build_labels(nlp(prompt))} for prompt in prompts]
    if args.as_json:
        print(json.dumps(results, indent=2))
        return
    for result in results:
        labels = result["labels"]
        print(f"\nPrompt: {result['prompt']}")
        print("Tokens:", " ".join(f"{t['index']}:{t['text']}" for t in labels["tokens"]))
        print("Objects:")
        for i, obj in enumerate(labels["objects"]):
            print(f"  [{i}] {obj['text']} | attrs={obj['attributes']} | tokens={obj['token_indices']}")
        print("Relations:")
        for relation in labels["relations"]:
            print(f"  {relation['subject']} -{relation['predicate']}-> {relation['object']}")


if __name__ == "__main__":
    main()
