#!/usr/bin/env python3
"""Produce the training-side deliverables required by the paper's section 4.1
and section 4.3.6 (layer / timestep gate statistics)."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path('/root/private_data/PixArt-alpha-attentiongate')
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

FONT = ROOT / 'results/_work/fonts/NotoSansCJKsc-Regular.otf'
if FONT.exists():
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(str(FONT))
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

RUN = ROOT / 'output/coco2017_token_pair_learnable_layers'
CKPT = RUN / 'checkpoints/epoch_1_step_14786.pth'
OUT = ROOT / 'results/training'
OUT.mkdir(parents=True, exist_ok=True)
ROLES = ['global', 'object', 'attribute', 'relation']


def load_metrics():
    p = RUN / 'experiment_tables/training_metrics.csv'
    rows = list(csv.DictReader(open(p)))
    return rows


def write_loss_csv(rows):
    keep = ['global_step', 'elapsed_seconds', 'loss', 'diffusion_loss',
            'semantic_reg_loss', 'distill_loss', 'lr', 'grad_norm',
            'gpu_memory_gb', 'gpu_peak_memory_gb', 'semantic_residual_norm',
            'semantic_token_gate', 'semantic_pair_strength']
    keep += [f'semantic_layer_gate_{r}' for r in ROLES]
    with open(OUT / 'loss.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keep, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_loss(rows):
    step = np.array([float(r['global_step']) for r in rows])
    loss = np.array([float(r['loss']) for r in rows])
    diff = np.array([float(r['diffusion_loss']) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(step, loss, lw=1.2, color='#2b6cb0', label='总损失')
    ax[0].plot(step, diff, lw=1.2, color='#dd6b20', label='扩散损失')
    ax[0].set_xlabel('训练步数'); ax[0].set_ylabel('损失'); ax[0].legend()
    ax[0].set_title('TP-SCDA 训练损失曲线（可学习层门控，14 786 步）')
    ax[0].grid(alpha=0.3)
    # smoothed view
    k = max(1, len(loss) // 40)
    sm = np.convolve(loss, np.ones(k) / k, mode='valid')
    ax[1].plot(step[k - 1:], sm, lw=1.6, color='#2f855a')
    ax[1].set_xlabel('训练步数'); ax[1].set_ylabel(f'总损失（{k} 点滑动平均）')
    ax[1].set_title('平滑后的损失趋势'); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'loss_curve.png', dpi=200)
    print('wrote loss_curve.png')


def param_counts():
    from diffusion.model.nets import PixArt_XL_2
    model = PixArt_XL_2(input_size=64, lewei_scale=1, semantic_conditioning=True,
                        semantic_adapter_dim=64, semantic_dropout=0,
                        semantic_residual_scale=0.0, semantic_token_attention=True,
                        semantic_token_gate_max=0.08)
    state = torch.load(CKPT, map_location='cpu')
    state = state.get('state_dict', state)
    state.pop('pos_embed', None)
    model.load_state_dict(state, strict=False)
    # Exact predicate used by train_scripts/train.py::configure_trainable_parameters
    TRAINABLE_PREFIXES = ('semantic_adapters.', 'semantic_layer_scale',
                          'semantic_time_gate.', 'semantic_token_cross_attention.',
                          'semantic_token_gate', 'semantic_token_layer_gate')
    total = sum(p.numel() for p in model.parameters())
    trainable = 0
    t_names, frozen_prefixes = [], {}
    for n, p in model.named_parameters():
        if n.startswith(TRAINABLE_PREFIXES):
            trainable += p.numel(); t_names.append((n, p.numel()))
        else:
            frozen_prefixes[n.split('.')[0]] = frozen_prefixes.get(n.split('.')[0], 0) + p.numel()
    lines = [
        'TP-SCDA 参数统计（判据与 train_scripts/train.py 的 '
        'configure_trainable_parameters 完全一致）',
        '',
        f'总参数量 (PixArt-XL-2-512 + TP-SCDA 模块): {total:,}',
        f'可训练参数量: {trainable:,}',
        f'可训练占比: {trainable/total*100:.4f}%',
        f'冻结参数量: {total-trainable:,} ({(total-trainable)/total*100:.4f}%)',
        '',
        '可训练参数张量明细:',
    ]
    for n, c in sorted(t_names):
        lines.append(f'  {n}: {c:,}')
    lines += ['', '冻结模块汇总（按顶层模块名）:']
    for k, c in sorted(frozen_prefixes.items(), key=lambda kv: -kv[1]):
        lines.append(f'  {k}: {c:,}')
    (OUT / 'trainable_params.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[2:7]))
    return dict(total=total, trainable=trainable,
                trainable_fraction=trainable / total,
                trainable_names=[n for n, _ in t_names])


def gate_from_ckpt():
    state = torch.load(CKPT, map_location='cpu')
    state = state.get('state_dict', state)
    lg = state['semantic_token_layer_gate'].float()          # (depth, 4)
    tg_W = state.get('semantic_time_gate.weight')
    tg_b = state.get('semantic_time_gate.bias')
    gate = torch.sigmoid(lg).numpy()
    with open(OUT / 'layer_gate_statistics.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['layer'] + [f'gate_{r}' for r in ROLES])
        for l in range(gate.shape[0]):
            w.writerow([l] + [f'{gate[l, i]:.6f}' for i in range(4)])
        w.writerow(['mean'] + [f'{gate[:, i].mean():.6f}' for i in range(4)])
        w.writerow(['min'] + [f'{gate[:, i].min():.6f}' for i in range(4)])
        w.writerow(['max'] + [f'{gate[:, i].max():.6f}' for i in range(4)])
    return gate, tg_W, tg_b


def plot_layer_gate(gate):
    """Show both the absolute gate values and the deviation from the
    initialisation, because in this checkpoint the gates barely moved."""
    init = np.full((gate.shape[0], 4), 1 / (1 + np.exp(4.0)))
    init[:, 0] = 1 / (1 + np.exp(-4.0))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    colors = ['#2b6cb0', '#dd6b20', '#2f855a', '#b83280']
    for i, r in enumerate(ROLES):
        ax[0].plot(range(gate.shape[0]), gate[:, i], marker='o', ms=3, lw=1.3,
                   color=colors[i], label=r)
        ax[2].plot(range(gate.shape[0]), gate[:, i] - init[:, i], marker='o',
                   ms=3, lw=1.3, color=colors[i], label=r)
    ax[0].axhline(init[0, 0], ls='--', c='#2b6cb0', lw=1, alpha=.6)
    ax[0].axhline(init[0, 1], ls='--', c='#dd6b20', lw=1, alpha=.6)
    ax[0].set_yscale('log'); ax[0].legend(title='角色分支', fontsize=8)
    ax[0].set_title('门控绝对值 λ_l = σ(w_l)\n(虚线=初始化值)')
    ax[1].bar(range(4), gate.mean(0), color=colors)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(ROLES)
    ax[1].set_ylim(0, 1.05)
    ax[1].axhline(init[0, 0], ls='--', c='k', lw=1, alpha=.5)
    ax[1].set_title('各角色门控均值')
    ax[2].axhline(0, color='k', lw=.8)
    ax[2].legend(fontsize=8)
    ax[2].set_title('相对初始化值的偏移\n（最大偏移仅 '
                    f'{np.abs(gate - init).max():.1e}）')
    for a in ax:
        a.grid(alpha=.3)
    ax[0].set_xlabel('DiT 层号'); ax[1].set_ylabel('λ'); ax[2].set_xlabel('DiT 层号')
    fig.suptitle('TP-SCDA 逐层角色门控（epoch_1_step_14786）：各层几乎与初始化值重合，'
                 '未学到层间差异', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / 'layer_gate.png', dpi=200)
    print('wrote layer_gate.png')


def plot_timestep_gate(tg_W, tg_b):
    """The time gate exists in the class but is NOT used by the TP-SCDA
    (token-attention) path -- it is only reached by the pooled residual path.
    We plot it for reference and say so explicitly on the figure."""
    from diffusion.model.nets import PixArt_XL_2
    m = PixArt_XL_2(input_size=64, lewei_scale=1, semantic_conditioning=True,
                    semantic_adapter_dim=64, semantic_dropout=0,
                    semantic_residual_scale=0.0, semantic_token_attention=True,
                    semantic_token_gate_max=0.08)
    # replicate the model's own timestep embedding
    ts = torch.arange(0, 1000, 10)
    t_emb = m.t_embedder(ts) if hasattr(m, 't_embedder') else None
    fig, ax = plt.subplots(figsize=(8, 3.4))
    if tg_W is not None and t_emb is not None:
        with torch.no_grad():
            g = torch.sigmoid(torch.nn.functional.linear(t_emb.float(), tg_W.float(), tg_b.float())).numpy()
        for i, r in enumerate(ROLES):
            ax.plot(ts.numpy(), g[:, i], lw=1.4, label=r)
        ax.legend(title='角色分支')
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, '时间步门控在 TP-SCDA 路径中未参与计算',
                ha='center', va='center', fontsize=12)
    ax.set_xlabel('扩散时间步 t'); ax.set_ylabel('σ(W_g f(t) + b_g)')
    ax.set_title('时间步门控：该模块仅被 pooled 残差路径使用，TP-SCDA 的 token-pair 路径不依赖 t')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / 'timestep_gate.png', dpi=200)
    print('wrote timestep_gate.png')


def main():
    rows = load_metrics()
    write_loss_csv(rows)
    plot_loss(rows)
    pc = param_counts()

    # training config
    cfg = {}
    for line in (RUN / 'config.py').read_text().splitlines():
        if ' = ' in line:
            k, v = line.split(' = ', 1)
            cfg[k] = v
    meta = json.loads((RUN / 'experiment_tables/'
                       'coco2017_token_pair_learnable_layers_metadata.json').read_text())
    (OUT / 'config.json').write_text(json.dumps(dict(
        run_name='coco2017_token_pair_learnable_layers',
        note='Training configuration actually used for the TP-SCDA checkpoint '
             'evaluated in the paper.',
        optimizer='AdamW', learning_rate='1e-5 before sqrt auto-scaling; '
                                        '3.536e-6 effective (see auto_lr rule=sqrt)',
        weight_decay=0.01, eps=1e-10, gradient_clip=1.0,
        lr_schedule='constant with 500 warmup steps',
        train_batch_size=8, gradient_accumulation_steps=4, effective_batch_size=32,
        num_epochs=1, total_steps=14786,
        image_size=512, mixed_precision='fp16', ema_rate=0.9999,
        semantic_conditioning=True, semantic_token_attention=True,
        semantic_token_gate_max=0.08, semantic_residual_scale=0.0,
        semantic_adapter_dim=64, semantic_reg_coef=0.02,
        distill_coef=0.5, distill_interval=8,
        trainable_params=pc['trainable'], total_params=pc['total'],
        trainable_fraction=pc['trainable_fraction'],
        trainable_param_names=pc['trainable_names'],
        config_snapshot=cfg,
        metadata=meta.get('_cfg_dict', {}),
    ), indent=2, ensure_ascii=False)[:200000])

    # training wall-clock from the log
    ts = [l.split(' - ')[0] for l in (RUN / 'train_log.log').read_text(
        errors='replace').splitlines() if 'Step/Epoch' in l]
    wall = None
    if len(ts) > 1:
        from datetime import datetime
        a = datetime.strptime(ts[0], '%Y-%m-%d %H:%M:%S,%f')
        b = datetime.strptime(ts[-1], '%Y-%m-%d %H:%M:%S,%f')
        wall = (b - a).total_seconds()

    peak = max(float(r['gpu_peak_memory_gb']) for r in rows if r['gpu_peak_memory_gb'])
    (OUT / 'checkpoint_info.json').write_text(json.dumps(dict(
        checkpoint=str(CKPT),
        checkpoint_bytes=CKPT.stat().st_size,
        step=14786, epoch=1,
        training_wall_clock_seconds=wall,
        training_wall_clock_hours=(wall / 3600 if wall else None),
        gpu_peak_memory_during_training_gb=peak,
        gpu_peak_memory_note='measured by the training loop (torch.cuda.max_memory_allocated)',
        seed=43,
    ), indent=2, ensure_ascii=False))
    print(f'training wall clock: {wall/3600:.2f} h, peak GPU {peak:.2f} GB')

    gate, tg_W, tg_b = gate_from_ckpt()
    plot_layer_gate(gate)
    plot_timestep_gate(tg_W, tg_b)
    print('layer gate means:', {r: float(gate[:, i].mean()) for i, r in enumerate(ROLES)})


if __name__ == '__main__':
    main()
