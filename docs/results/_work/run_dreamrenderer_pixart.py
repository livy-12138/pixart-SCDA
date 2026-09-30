#!/usr/bin/env python3
"""在 PixArt-α（T5 + DiT）上跑 DreamRenderer（适配版）—— 独立脚本，不改主代码。

完整迁移说明见 `dreamrenderer.py` 的模块 docstring。要点：
  * **Bridge Image Tokens 未迁移** —— PixArt 无 Joint Attention，无宿主结构
  * **Hard/Soft Attribute Binding 已适配** —— 从联合注意力掩码改为交叉注意力掩码
  * **实例区域自导出** —— 不用外部检测器/LLM，从模型自身的交叉注意力估计每个
    patch 归属哪个对象，再对其施加硬掩码
  * hard binding 只用于**中间层**（沿用原论文结论）

用法：
    python run_dreamrenderer_pixart.py --category color --limit-prompts 100 \
        --out-root /root/compbench_work/images_dr
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'results/_work'))

import torch
import spacy
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from diffusion import DPMS
from diffusion.model.nets import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from tools.prepare_semantic_masks import (build_semantic_edges, build_semantic_masks,
                                          clean_caption)
import dreamrenderer as DR

CKPT = str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth')
T5_CACHE = str(ROOT / 'output/pretrained_models/t5_ckpts')
VAE_PATH = str(ROOT / 'output/pretrained_models/sd-vae-ft-ema')
MAX_EDGES = 24


def image_seed(base, cat_idx, prompt_idx, repeat):
    return (base * 1_000_003 + cat_idx * 10_007 + prompt_idx * 101 + repeat) % (2 ** 31 - 1)


def parse_prompt(prompt, nlp, tokenizer, L):
    """给出 (对象 token, 每个对象的属性 token, 全局 token)。"""
    edges, n_e = build_semantic_edges(prompt, nlp, tokenizer, L, MAX_EDGES)
    obj_idx, attr_of = [], {}
    for o, a in edges[:n_e]:
        o, a = int(o), int(a)
        if o not in attr_of:
            attr_of[o] = []
            obj_idx.append(o)
        if a not in attr_of[o]:
            attr_of[o].append(a)
    roles = build_semantic_masks(prompt, nlp, tokenizer, L)      # [3, L]
    role_any = roles.sum(0) > 0
    # 这一路的 tokenize 必须与 build_semantic_masks / build_semantic_edges 同源：
    #   1) 同样先 clean_caption —— 它做小写化等改写，原样 tokenize 会让子词数与
    #      位置和上面两张 mask 对不上；
    #   2) 同样 padding='max_length' —— 否则 attention_mask 只有实际 token 那么长，
    #      下面的 range(L)（L=120）取 valid[t] 直接越界，就是之前 IndexError 的来源。
    tok = tokenizer(clean_caption(prompt), max_length=L, truncation=True,
                    padding='max_length', add_special_tokens=True)
    valid = [bool(v) for v in tok['attention_mask']]
    global_idx = [t for t in range(L) if valid[t] and not role_any[t]]
    return obj_idx, attr_of, global_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', required=True)
    ap.add_argument('--dataset-dir', default='/root/compbench_work/eval/T2I-CompBench/examples/dataset')
    ap.add_argument('--out-root', default='/root/compbench_work/images_dr')
    ap.add_argument('--limit-prompts', type=int, default=0)
    ap.add_argument('--images-per-prompt', type=int, default=10)
    ap.add_argument('--end-repeat', type=int, default=1)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--cfg-scale', type=float, default=4.0)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--mid-lo', type=float, default=0.25, help='hard binding 起始层比例')
    ap.add_argument('--mid-hi', type=float, default=0.75, help='hard binding 结束层比例')
    args = ap.parse_args()

    device, dtype = 'cuda', torch.float16
    nlp = spacy.load('en_core_web_sm')
    out_root = Path(args.out_root) / 'dreamrenderer'
    cat_out = out_root / args.category
    cat_out.mkdir(parents=True, exist_ok=True)

    print('加载模型 ...', flush=True)
    model = PixArt_XL_2(input_size=args.image_size // 8, lewei_scale=1,
                        semantic_conditioning=True, semantic_token_attention=True,
                        semantic_token_gate_max=0.08, semantic_residual_scale=0.0
                        ).to(device, dtype=dtype).eval()
    state = torch.load(CKPT, map_location='cpu')
    state = state.get('state_dict', state)
    state.pop('pos_embed', None)
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f'  checkpoint missing={len(miss)} unexpected={len(unexp)}', flush=True)
    del state
    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device, dtype=dtype).eval()
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=T5_CACHE,
                    torch_dtype=torch.float, model_max_length=120)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(T5_CACHE + '/t5-v1_1-xxl',
                                              use_fast=True, local_files_only=True)

    lo, hi = DR.apply(model, args.mid_lo, args.mid_hi)

    prompts = [l.strip() for l in
               (Path(args.dataset_dir) / f'{args.category}_val.txt').read_text().splitlines()
               if l.strip()]
    if args.limit_prompts:
        prompts = prompts[:args.limit_prompts]

    rows = []
    t_start = time.time()
    n_no_obj = 0
    for rep in range(args.end_repeat):
        for p_i, prompt in enumerate(prompts):
            name = f'{prompt}_{p_i * args.images_per_prompt + rep:06d}.png'
            if (cat_out / name).exists():
                continue
            seed = image_seed(args.seed, 0, p_i, rep)
            gen = torch.Generator(device=device).manual_seed(seed)

            emb, text_mask = t5.get_text_embeddings([prompt])
            emb = emb[:, None].to(device=device, dtype=dtype)
            null = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None].to(dtype)
            L = emb.shape[2]

            obj_idx, attr_of, global_idx = parse_prompt(prompt, nlp, tokenizer, L)
            if not obj_idx:
                n_no_obj += 1                  # 无对象—属性对 → 机制为空操作
            DR.set_prompt_state(obj_idx, attr_of, global_idx)

            hw = torch.full((1, 2), args.image_size, device=device, dtype=dtype)
            ar = torch.ones((1, 1), device=device, dtype=dtype)
            masks = torch.stack([torch.from_numpy(
                build_semantic_masks(prompt, nlp, tokenizer, L))]).to(device, dtype)
            data_info = {'img_hw': hw, 'aspect_ratio': ar, 'semantic_token_masks': masks}

            noise = torch.randn(1, 4, args.image_size // 8, args.image_size // 8,
                                generator=gen, device=device, dtype=dtype)
            solver = DPMS(model.forward_with_dpmsolver, emb, null, args.cfg_scale,
                          model_kwargs=dict(data_info=data_info, mask=text_mask))
            latents = solver.sample(noise, steps=args.steps, order=2,
                                    skip_type='time_uniform', method='multistep')
            image = vae.decode((latents / 0.18215).to(dtype)).sample
            save_image(image.float(), cat_out / name, normalize=True, value_range=(-1, 1))
            rows.append(dict(category=args.category, prompt=prompt, prompt_index=p_i,
                             repeat=rep, global_index=p_i * args.images_per_prompt + rep,
                             image=name, n_obj=len(obj_idx)))
            if (p_i + 1) % 20 == 0 or p_i == len(prompts) - 1:
                el = time.time() - t_start
                print(f'  {args.category} {p_i+1}/{len(prompts)}  {el/(p_i+1):.1f} s/img', flush=True)

    DR.revert()

    man = out_root / f'manifest_{args.category}.csv'
    write_header = not man.exists()
    with open(man, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['category', 'prompt', 'prompt_index',
                                           'repeat', 'global_index', 'image', 'n_obj'])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    (out_root / f'config_{args.category}.json').write_text(json.dumps(dict(
        method='dreamrenderer_pixart', category=args.category, checkpoint=CKPT,
        seed=args.seed, steps=args.steps, cfg_scale=args.cfg_scale,
        mid_layers=[lo, hi], bridge_tokens='not_migrated (PixArt 无 Joint Attention)',
        region_source='self-derived from cross-attention ownership',
        n_prompts=len(prompts), n_without_obj_pairs=n_no_obj,
    ), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'完成 {len(rows)} 张 → {cat_out}（其中 {n_no_obj} 条无对象—属性对，机制为空操作）', flush=True)


if __name__ == '__main__':
    main()
