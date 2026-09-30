#!/usr/bin/env python3
"""两轴注意力干预的运行入口（LCAR 系列，**产物独立存放，不进 TP-SCDA 论文**）。

机制见 `additive_sharpen.py`：
  * token 轴（加性抬升）：只抬角色词，不压其他 token —— 修"零和再分配压扁背景"
  * 空间轴（对象竞争锐化）：对每个 patch 在对象 token 间做幂次锐化，
    让各对象注意力图在空间上互相分离 —— 目标是救计数

本脚本用**运行时 monkey-patch**，不改动主代码库。

用法：
    python run_lcar_twaxis.py --category numeracy --limit-prompts 100 \
        --delta 0.4 --gamma 1.0 --tag lcar_add_d04 --out-root /root/compbench_work/images_lcar
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
from tools.prepare_semantic_masks import build_semantic_edges, build_semantic_masks
import additive_sharpen as AS

CKPT = str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth')
T5_CACHE = str(ROOT / 'output/pretrained_models/t5_ckpts')
VAE_PATH = str(ROOT / 'output/pretrained_models/sd-vae-ft-ema')
MAX_EDGES = 24


def image_seed(base, cat_idx, prompt_idx, repeat):
    return (base * 1_000_003 + cat_idx * 10_007 + prompt_idx * 101 + repeat) % (2 ** 31 - 1)


def role_and_obj(prompt, nlp, tokenizer, L):
    """角色 token（对象+属性）与对象 token 下标。"""
    edges, n_e = build_semantic_edges(prompt, nlp, tokenizer, L, MAX_EDGES)
    obj, attr = set(), set()
    for o, a in edges[:n_e]:
        obj.add(int(o)); attr.add(int(a))
    return sorted(obj | attr), sorted(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', required=True)
    ap.add_argument('--dataset-dir', default='/root/compbench_work/eval/T2I-CompBench/examples/dataset')
    ap.add_argument('--out-root', default='/root/compbench_work/images_lcar')
    ap.add_argument('--limit-prompts', type=int, default=0)
    ap.add_argument('--images-per-prompt', type=int, default=10)
    ap.add_argument('--end-repeat', type=int, default=1)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--cfg-scale', type=float, default=4.0)
    ap.add_argument('--image-size', type=int, default=512)
    # 两轴参数
    ap.add_argument('--delta', type=float, default=0.0, help='token 轴：加性抬升的整行增量上限（0=关）')
    ap.add_argument('--gamma', type=float, default=1.0, help='空间轴：对象竞争锐化指数（1.0=关）')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--batch-size', type=int, default=1, help='逐样本设置解析状态，建议 1')
    args = ap.parse_args()

    device, dtype = 'cuda', torch.float16
    nlp = spacy.load('en_core_web_sm')
    out_root = Path(args.out_root) / args.tag
    cat_out = out_root / args.category
    cat_out.mkdir(parents=True, exist_ok=True)

    print(f'加载模型 ... (tag={args.tag} delta={args.delta} gamma={args.gamma})', flush=True)
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

    AS.apply(model)

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

            role, obj = role_and_obj(prompt, nlp, tokenizer, L)
            if not obj:
                n_no_obj += 1
            AS.set_role_state(role, obj, delta=args.delta, gamma=args.gamma)

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
                             image=name, n_role=len(role), n_obj=len(obj)))
            if (p_i + 1) % 20 == 0 or p_i == len(prompts) - 1:
                el = time.time() - t_start
                print(f'  {args.category} {p_i+1}/{len(prompts)}  {el/(p_i+1):.1f} s/img', flush=True)

    AS.revert()

    man = out_root / f'manifest_{args.category}.csv'
    wh = not man.exists()
    with open(man, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['category', 'prompt', 'prompt_index', 'repeat',
                                           'global_index', 'image', 'n_role', 'n_obj'])
        if wh:
            w.writeheader()
        w.writerows(rows)
    (out_root / f'config_{args.category}.json').write_text(json.dumps(dict(
        method='lcar_twaxis', tag=args.tag, category=args.category, checkpoint=CKPT,
        delta=args.delta, gamma=args.gamma, seed=args.seed, steps=args.steps,
        cfg_scale=args.cfg_scale, n_prompts=len(prompts), n_without_obj=n_no_obj,
    ), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'完成 {len(rows)} 张 → {cat_out}（{n_no_obj} 条无对象—属性对，机制为空操作）', flush=True)


if __name__ == '__main__':
    main()
