#!/usr/bin/env python3
"""Generate fixed prompt groups with a trained SCDA PixArt checkpoint."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from diffusion import DPMS
from diffusion.model.nets import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from tools.prepare_semantic_masks import (build_semantic_edges, build_semantic_masks,
                                          color_attribute_indices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--groups-dir', default='prompt_groups')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--cfg-scale', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--semantic-conditioning', action='store_true',
                   help='Enable SCDA semantic residual injection (default: baseline).')
    p.add_argument('--semantic-adapter-dim', type=int, default=64,
                   help='SCDA adapter bottleneck dimension used by the checkpoint.')
    p.add_argument('--semantic-residual-scale', type=float, default=1.0)
    p.add_argument('--semantic-token-attention', action='store_true',
                   help='Use token-level role/pair-aware SCDA attention instead of pooled residuals.')
    p.add_argument('--semantic-token-gate-activation', type=str, default='tanh',
                   choices=['tanh', 'sigmoid'],
                   help='Must match the value used at training time; it is config, not weights.')
    p.add_argument('--semantic-token-gate-max', type=float, default=1.0,
                   help='Maximum multiplier for the learned token-attention gate.')
    p.add_argument('--semantic-pair-edge-gate', action='store_true',
                   help='B1: restrict the pair-aware bias to parsed object->attribute '
                        'edges. Must match the training config, and the edges must be '
                        'built with the same parser/order/limit as the training index.')
    p.add_argument('--max-semantic-edges', type=int, default=24,
                   help='must match max_semantic_edges in the training config')
    p.add_argument('--cross-attn-pair-replace', action='store_true',
                   help='Rewrite the DiT\'s own cross-attention weights at the parsed '
                        'object->attribute pairs. Needs role masks and edges, which this '
                        'script builds automatically (they are the mechanism\'s input, '
                        'not an optional extra).')
    p.add_argument('--pair-replace-mode', type=str, default='replace',
                   choices=['replace', 'raise', 'equalize', 'gated', 'outside', 'reweight', 'sink'],
                   help="replace = attribute overwritten by its nouns' weight, "
                        "whole row renormalised; raise = attribute lifted only "
                        "where it sits below its nouns, row left un-normalised "
                        "(row sum grows by the raise); equalize = attribute "
                        "lifted to its nouns' mean AND that amount debited from "
                        "those same nouns, so the row still sums to 1 while "
                        "tokens outside the pair never move.")
    p.add_argument('--pair-replace-gate-gamma', type=float, default=1.0,
                   help='gated mode: sharpness of the patch-to-object preference. '
                        '1.0 = proportional (soft); larger approaches raising only '
                        "the patch's single most-attended object.")
    p.add_argument('--pair-replace-fund', type=str, default='proportional',
                   choices=['proportional', 'tail', 'uniform'],
                   help="outside mode: which tokens pay for the lift -- "
                        "'proportional' (by weight), 'tail' (smallest first, "
                        "spares the largest), 'uniform' (equal share).")
    p.add_argument('--embed-bind-filter', type=str, default='all',
                   choices=['all', 'color'],
                   help="color = tag ONLY attributes whose word is a colour, so "
                        "shape/texture/spatial structure is left alone.")
    p.add_argument('--embed-bind-position-scale', type=float, default=1.0,
                   help='multiplicative boost for the 2nd, 3rd... pair: the model '
                        'binds the first-mentioned object better (measured), so a '
                        'later pair gets alpha * scale**rank. 1.0 = uniform.')
    p.add_argument('--embed-bind-mode', type=str, default='add',
                   choices=['add', 'normalize', 'contrast', 'scale'],
                   help='add = raw addition; normalize = mix then restore the '
                        'original norm (removes the key-scale confound); contrast = '
                        'add the noun and subtract the other pair objects.')
    p.add_argument('--embed-bind-alpha', type=float, default=0.0,
                   help='value-space probe: add alpha * (noun embedding) to each '
                        'bound attribute token before the DiT. Leaves attention '
                        'weights untouched. 0 = off.')
    p.add_argument('--sink-factor', type=float, default=0.0,
                   help='sink mode: surviving fraction of each row argmax (0 = removed)')
    p.add_argument('--content-target', type=float, default=0.5,
                   help='reweight mode: share of each attention row the parsed '
                        'elements (object/attribute/relation) should hold. All of '
                        'them are scaled by one factor, so their internal ratios '
                        'are preserved; the rest of the row is diluted.')
    p.add_argument('--pair-replace-renorm', action='store_true',
                   help='raise mode only: rescale the whole row back to sum 1 after '
                        'the raise, which shrinks every other token proportionally.')
    p.add_argument('--pair-replace-strength', type=float, default=1.0,
                   help='1.0 = attribute takes the noun\'s weight outright; below 1.0 '
                        'blends towards the DiT\'s own weights.')
    p.add_argument('--no-semantic-masks', action='store_true',
                   help='A/B escape hatch: omit role masks so the pair-replace path is '
                        'skipped entirely. Used to demonstrate that the mechanism is '
                        'inert without them.')
    p.add_argument('--t5-cache', default='output/pretrained_models/t5_ckpts')
    p.add_argument('--vae-path', default='output/pretrained_models/sd-vae-ft-ema')
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    torch.manual_seed(args.seed)

    latent_size = args.image_size // 8
    model = PixArt_XL_2(input_size=latent_size, lewei_scale=0.5 if args.image_size == 256 else 1,
                        semantic_conditioning=args.semantic_conditioning,
                        semantic_adapter_dim=args.semantic_adapter_dim,
                        semantic_dropout=0,
                        semantic_residual_scale=args.semantic_residual_scale,
                        semantic_token_attention=args.semantic_token_attention,
                        semantic_token_gate_max=args.semantic_token_gate_max,
                        semantic_token_gate_activation=args.semantic_token_gate_activation,
                        semantic_pair_edge_gate=args.semantic_pair_edge_gate,
                        cross_attn_pair_replace=args.cross_attn_pair_replace,
                        cross_attn_pair_replace_strength=args.pair_replace_strength,
                        cross_attn_pair_replace_mode=args.pair_replace_mode,
                        cross_attn_pair_replace_renorm=args.pair_replace_renorm,
                        cross_attn_pair_replace_fund=args.pair_replace_fund,
                        cross_attn_pair_replace_content_target=args.content_target,
                        cross_attn_pair_replace_sink_factor=args.sink_factor,
                        cross_attn_pair_replace_gate_gamma=args.pair_replace_gate_gamma).to(device, dtype=dtype).eval()
    state = torch.load(args.checkpoint, map_location='cpu')
    state = state.get('state_dict', state)
    state.pop('pos_embed', None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'checkpoint loaded; missing={len(missing)} unexpected={len(unexpected)}')

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, dtype=dtype).eval()
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_cache,
                    torch_dtype=torch.float, model_max_length=120)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.t5_cache + '/t5-v1_1-xxl', use_fast=True, local_files_only=True)
    import spacy
    nlp = spacy.load('en_core_web_sm')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for group_file in sorted(Path(args.groups_dir).glob('group_*.txt')):
        group_out = output / group_file.stem
        group_out.mkdir(exist_ok=True)
        prompts = [line.strip() for line in group_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        with torch.inference_mode():
            emb, text_mask = t5.get_text_embeddings(prompts)
            emb = emb[:, None].to(device=device, dtype=dtype)
            if args.embed_bind_alpha > 0:
                # Value-space binding tag: give every attribute sub-token a
                # small share of the embedding of the noun it is bound to, so
                # the colour token itself carries "which object I belong to".
                # The attention WEIGHTS are left completely untouched -- the
                # sink keeps its 96% and the model's own spatial assignment is
                # undisturbed, which is what every weight-editing rule broke.
                import numpy as np
                pairs = [build_semantic_edges(p, nlp, tokenizer, emb.shape[2],
                                              args.max_semantic_edges)
                         for p in prompts]
                n_tag = 0
                _color_idx = None
                if args.embed_bind_filter == 'color':
                    _color_idx = [color_attribute_indices(pp, nlp, tokenizer, emb.shape[2])
                                  for pp in prompts]
                for b, (edges, count) in enumerate(pairs):
                    obj_idx = {int(o) for o, _ in edges[:count]}
                    # Positional weighting: the model binds the FIRST-mentioned
                    # object markedly better (measured on the base model, within
                    # every colour: +7..+22 points, prompts are pair-swapped so
                    # it is a pure position effect).  Later pairs therefore get
                    # a stronger tag.
                    _order = sorted({int(o) for o, _ in edges[:count]})
                    _rank = {o: k for k, o in enumerate(_order)}
                    for o, a in edges[:count]:
                        o, a = int(o), int(a)
                        if _color_idx is not None and a not in _color_idx[b]:
                            continue
                        _a_eff = args.embed_bind_alpha * (args.embed_bind_position_scale ** _rank.get(o, 0))
                        vec = emb[b, 0, a].clone()
                        if args.embed_bind_mode == 'normalize':
                            # mix, then restore the original NORM: raw addition
                            # inflates |emb[a]|, which rescales its key vector
                            # and silently changes its attention weight for every
                            # patch -- a confound we want out of the comparison.
                            mixed = (1 - _a_eff) * vec + _a_eff * emb[b, 0, o]
                            emb[b, 0, a] = mixed * (vec.norm() / mixed.norm().clamp_min(1e-6))
                        elif args.embed_bind_mode == 'contrast':
                            # push AWAY from the other pair's object as well: the
                            # observed failure is cross-pair bleeding ("green"
                            # from the vase pair landing on the apple), so
                            # separating the two attributes is the targeted fix.
                            others = [j for j in obj_idx if j != o]
                            push = emb[b, 0, o].clone()
                            for j in others:
                                push = push - emb[b, 0, j]
                            emb[b, 0, a] = vec + _a_eff * push
                        elif args.embed_bind_mode == 'scale':
                            # pure norm inflation, NO content mixing: tests
                            # whether the gain comes from the attribute token's
                            # magnitude (its key strength) rather than from the
                            # noun's content being copied in.
                            emb[b, 0, a] = vec * (1.0 + _a_eff)
                        else:
                            emb[b, 0, a] = vec + _a_eff * emb[b, 0, o]
                        n_tag += 1
                print(f'embed-bind: tagged {n_tag} attribute tokens '
                      f'(alpha={args.embed_bind_alpha})')
            null = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None].to(dtype)
            # The pair-replace path is gated on role masks inside PixArt.forward,
            # so a run that enables the mechanism MUST supply them -- building
            # them only for --semantic-conditioning silently disables it (this is
            # how the 2026-09-20 pair_replace evaluation scored a model whose
            # mechanism never ran).
            need_masks = ((args.semantic_conditioning or args.cross_attn_pair_replace)
                          and not args.no_semantic_masks)
            semantic_masks = torch.stack([
                torch.from_numpy(build_semantic_masks(prompt, nlp, tokenizer, emb.shape[2]))
                for prompt in prompts
            ]).to(device=device, dtype=dtype) if need_masks else None
            hw = torch.full((len(prompts), 2), args.image_size, device=device, dtype=dtype)
            ar = torch.ones((len(prompts), 1), device=device, dtype=dtype)
            data_info = {'img_hw': hw, 'aspect_ratio': ar}
            if semantic_masks is not None:
                data_info['semantic_token_masks'] = semantic_masks
            if args.semantic_pair_edge_gate or args.cross_attn_pair_replace:
                # Same parser, same pair order, same truncation as
                # tools/build_edge_index.py -- verified row-for-row.
                import numpy as np
                parsed = [build_semantic_edges(prompt, nlp, tokenizer, emb.shape[2],
                                               args.max_semantic_edges)
                          for prompt in prompts]
                data_info['semantic_edges'] = torch.from_numpy(
                    np.stack([e for e, _ in parsed])).to(device=device)
                data_info['semantic_edge_count'] = torch.from_numpy(
                    np.asarray([c for _, c in parsed])).to(device=device)
            kwargs = dict(data_info=data_info, mask=text_mask)
            noise = torch.randn(len(prompts), 4, latent_size, latent_size, device=device, dtype=dtype)
            solver = DPMS(model.forward_with_dpmsolver, condition=emb, uncondition=null,
                          cfg_scale=args.cfg_scale, model_kwargs=kwargs)
            latents = solver.sample(noise, steps=args.steps, order=2,
                                    skip_type='time_uniform', method='multistep')
            images = vae.decode((latents / 0.18215).to(dtype)).sample
            for prompt, image in zip(prompts, images):
                filename = prompt[:100].replace('/', '_') + '.png'
                save_image(image.float(), group_out / filename, normalize=True, value_range=(-1, 1))
        print(f'{group_file.stem}: {len(prompts)} images -> {group_out}')


if __name__ == '__main__':
    main()
