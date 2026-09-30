#!/usr/bin/env python3
"""在 PixArt-α（T5 + DiT）上跑 SynGen —— 独立脚本，不改动 gen_compbench.py。

与原方法的差异（全部记录，供论文标注）：

| 方面 | SynGen 官方（SD 1.4/1.5） | 本实现（PixArt-α） |
|---|---|---|
| 主干 | UNet cross-attn | DiT `MultiHeadCrossAttention`（全部 block） |
| 文本编码 | CLIP（77 token） | T5（120 token，含 padding） |
| (名词,修饰词) 抽取 | spaCy 依存分析 | **复用 `build_semantic_edges`**，直接给 (对象,属性) 对 |
| 注意力图 | 16×16 = 256 | 32×32 = 1024 |
| 采样器 | 官方用 DDIM | **仍是 DPM-Solver 20 步**（用 `correcting_xt_fn` 钩子注入引导） |

用法：
    python run_syngen_pixart.py --category color --limit-prompts 100 \
        --out-root /root/compbench_work/images_syngen --step-size 20 --num-steps 5
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

import numpy as np
import torch
import spacy
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from diffusion.model.nets import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from tools.prepare_semantic_masks import build_semantic_edges, build_semantic_masks
from syngen_pixart import guided_DPMS, SynGenCorrector

CKPT = str(ROOT / 'output/coco2017_token_pair_learnable_layers/checkpoints/epoch_1_step_14786.pth')
T5_CACHE = str(ROOT / 'output/pretrained_models/t5_ckpts')
VAE_PATH = str(ROOT / 'output/pretrained_models/sd-vae-ft-ema')
MAX_EDGES = 24


def image_seed(base, cat_idx, prompt_idx, repeat):
    """与 gen_compbench.py 完全一致的种子规则（保证同提示词同种子）。"""
    return (base * 1_000_003 + cat_idx * 10_007 + prompt_idx * 101 + repeat) % (2 ** 31 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', required=True)
    ap.add_argument('--dataset-dir', default='/root/compbench_work/eval/T2I-CompBench/examples/dataset')
    ap.add_argument('--out-root', default='/root/compbench_work/images_syngen')
    ap.add_argument('--limit-prompts', type=int, default=0)
    ap.add_argument('--images-per-prompt', type=int, default=10)
    ap.add_argument('--end-repeat', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=1, help='SynGen 需要逐样本梯度，建议 1')
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--cfg-scale', type=float, default=4.0)
    ap.add_argument('--image-size', type=int, default=512)
    # SynGen 超参（官方默认：step_size=20，前 num_steps 步引导）
    ap.add_argument('--step-size', type=float, default=20.0)
    ap.add_argument('--num-steps', type=int, default=5, help='前多少步做梯度引导')
    ap.add_argument('--lam-neg', type=float, default=1.0)
    ap.add_argument('--capture-layers', default='', help='逗号分隔的 block 下标；空=全部')
    args = ap.parse_args()

    # ⚠ DiT 必须整网 fp32，不能用 fp16。
    # 实测（diag_backward_dtype.py）：这个 checkpoint 在 fp16 下**前向正常但反向全 NaN**
    #   fp16: ∂(out.sum())/∂x → 16384/16384 个 NaN（x 共 16384 个元素）
    #   fp32: 同一份权重 → 0 个 NaN，absmax 2.217
    # SynGen 的本质是对潜变量求梯度，fp16 反向直接产出 NaN 梯度，第一步就把 x 变成 NaN。
    # 因此这里整网 fp32（其余方法仍 fp16；这是本方法的实现约束，需在论文里注明）。
    device, dtype = 'cuda', torch.float32
    nlp = spacy.load('en_core_web_sm')
    out_root = Path(args.out_root) / 'syngen'
    cat_out = out_root / args.category
    cat_out.mkdir(parents=True, exist_ok=True)

    # ── 模型（与 gen_compbench 完全相同的配置与检查点）────────────────
    print('加载模型 ...', flush=True)
    model = PixArt_XL_2(input_size=args.image_size // 8, lewei_scale=1,
                        semantic_conditioning=True, semantic_token_attention=True,
                        semantic_token_gate_max=0.08, semantic_residual_scale=0.0
                        ).to(device, dtype=dtype).eval()
    state = torch.load(CKPT, map_location='cpu')
    state = state.get('state_dict', state)
    state.pop('pos_embed', None)
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f'  checkpoint 载入 missing={len(miss)} unexpected={len(unexp)}', flush=True)
    del state
    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device, dtype=dtype).eval()
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=T5_CACHE,
                    torch_dtype=torch.float, model_max_length=120)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(T5_CACHE + '/t5-v1_1-xxl',
                                              use_fast=True, local_files_only=True)

    prompts = [l.strip() for l in
               (Path(args.dataset_dir) / f'{args.category}_val.txt').read_text().splitlines()
               if l.strip()]
    if args.limit_prompts:
        prompts = prompts[:args.limit_prompts]

    layers = [int(x) for x in args.capture_layers.split(',') if x.strip()] or None
    cat_idx = 0                      # 逐类单跑时类别位置恒为 0
    end_repeat = args.end_repeat
    rows = []
    t_start = time.time()

    for rep in range(end_repeat):
        for p_i, prompt in enumerate(prompts):
            name = f'{prompt}_{p_i * args.images_per_prompt + rep:06d}.png'
            if (cat_out / name).exists():
                continue
            seed = image_seed(args.seed, cat_idx, p_i, rep)
            gen = torch.Generator(device=device).manual_seed(seed)

            emb, text_mask = t5.get_text_embeddings([prompt])
            # 与 gen_compbench.py:794 一致：加一个维度 → [B,1,L,D]
            emb = emb[:, None].to(device=device, dtype=dtype)
            null = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None].to(dtype)
            hw = torch.full((1, 2), args.image_size, device=device, dtype=dtype)
            ar = torch.ones((1, 1), device=device, dtype=dtype)
            masks = torch.stack([torch.from_numpy(
                build_semantic_masks(prompt, nlp, tokenizer, emb.shape[2]))]).to(device, dtype)
            data_info = {'img_hw': hw, 'aspect_ratio': ar, 'semantic_token_masks': masks}

            # (对象, 属性) 对 —— SynGen 的 (noun, modifier)
            edges, n_e = build_semantic_edges(prompt, nlp, tokenizer, emb.shape[2], MAX_EDGES)
            pairs = [(int(a), int(o)) for o, a in edges[:n_e]]     # (modifier, noun)
            pairs_by_batch = [pairs]

            corrector = SynGenCorrector(
                model.forward_with_dpmsolver, emb, null,
                dict(data_info=data_info, mask=text_mask), pairs_by_batch,
                num_guided_steps=args.num_steps, step_size=args.step_size,
                lam_neg=args.lam_neg, capture_layers=layers)

            noise = torch.randn(1, 4, args.image_size // 8, args.image_size // 8,
                                generator=gen, device=device, dtype=dtype)
            solver = guided_DPMS(model.forward_with_dpmsolver, emb, null, args.cfg_scale,
                                 dict(data_info=data_info, mask=text_mask), corrector)
            latents = solver.sample(noise, steps=args.steps, order=2,
                                    skip_type='time_uniform', method='multistep')
            image = vae.decode((latents / 0.18215).to(dtype)).sample
            save_image(image.float(), cat_out / name, normalize=True, value_range=(-1, 1))
            rows.append(dict(category=args.category, prompt=prompt, prompt_index=p_i,
                             repeat=rep, global_index=p_i * args.images_per_prompt + rep,
                             image=name, n_pairs=len(pairs), guided=corrector.calls))
            if (p_i + 1) % 20 == 0 or p_i == len(prompts) - 1:
                el = time.time() - t_start
                print(f'  {args.category} {p_i+1}/{len(prompts)}  {el/(p_i+1):.1f} s/img', flush=True)

    man = out_root / f'manifest_{args.category}.csv'
    write_header = not man.exists()
    with open(man, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['category', 'prompt', 'prompt_index',
                                           'repeat', 'global_index', 'image',
                                           'n_pairs', 'guided'])
        if write_header:
            w.writeheader()
        w.writerows(rows)

    cfg_out = dict(method='syngen_pixart', category=args.category, checkpoint=CKPT,
                   seed=args.seed, steps=args.steps, cfg_scale=args.cfg_scale,
                   step_size=args.step_size, num_guided_steps=args.num_steps,
                   lam_neg=args.lam_neg, capture_layers=layers,
                   n_prompts=len(prompts))
    (out_root / f'config_{args.category}.json').write_text(
        json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'完成 {len(rows)} 张 → {cat_out}', flush=True)


if __name__ == '__main__':
    main()
