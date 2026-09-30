"""LCAR —— 局部竞争注意重分配（Local Competition Attention Reallocation）。

作用点：**只改写 token-pair 支路自身的 logit**（``SemanticTokenCrossAttention``
里那一处 pair bias）。主干 cross-attention 的 QKV / 权重全程不动，也不新增任何
可训练参数 —— 全部量都从既有支路激活里导出。

数学记号与论文《LCAR 服务器执行指南_v2》§4.2 一一对应：

    支路原式      B       = softplus(s) * bmm(object_attention, pair)      [B,M,L]
    LCAR          B_lcar  = B + boost + supp

其中 boost 只写进「锚点所属对象的配对属性 token」，supp 只压「**他**对象的对象
token 与其属性 token」—— 背景 / 关系 / 场景 token 一律不动（铁律 9）。

``enable=False`` 时本模块完全不介入，调用方走原式，逐位等价。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
class LCARConfig:
    """一个 run 的完整 LCAR 配置。所有字段都进 ``lcar_configs/*.json``。"""

    __slots__ = ('enable', 'mu', 'rho', 'tau_rel', 'lam', 'kernel', 'assign',
                 'suppress_set', 'anchor_src', 'anchor_update', 'mass_conserv',
                 'eta', 'layers', 'calc_ratio', 't_max', 'grid', 'use_parsed_edges',
                 'replace_graded', 'replace_t_center', 'replace_t_edge',
                 'replace_window', 'replace_additive', 'replace_delta_max',
                 'replace_obj_sharpen', 'verbose')

    def __init__(self, enable=False, mu=0.5, rho=1.0, tau_rel=0.1, lam=1.0,
                 kernel='gauss', assign='soft', suppress_set='obj+attr',
                 anchor_src='attr', anchor_update='once', mass_conserv=False,
                 eta=1.0, layers='all', calc_ratio=0.25, t_max=1000.0,
                 grid=None, use_parsed_edges=True,
                 replace_graded=False, replace_t_center=0.8, replace_t_edge=0.1,
                 replace_window=None, replace_additive=False, replace_delta_max=0.2,
                 replace_obj_sharpen=1.0,
                 verbose=False):
        self.enable = bool(enable)
        self.mu = float(mu)                       # 竞争抑制系数 ∈ [0,1]
        self.rho = int(rho)                       # 半径倍数 ∈ {4,8,12}
        self.tau_rel = float(tau_rel)             # 支撑域相对阈值
        self.lam = float(lam)                     # 锚点打分里 h 的权重
        self.kernel = kernel                      # gauss | hard
        self.assign = assign                      # soft | hard
        self.suppress_set = suppress_set          # obj | obj+attr | none
        self.anchor_src = anchor_src              # attr | obj | random
        self.anchor_update = anchor_update        # once | every5 | ema
        self.mass_conserv = bool(mass_conserv)
        self.eta = float(eta)                     # 搬运比例
        self.layers = layers                      # all | mid
        self.calc_ratio = float(calc_ratio)       # 前多少比例的去噪步算锚点
        self.t_max = float(t_max)                 # 时间轴量程（PixArt 约定 1000）
        self.grid = grid                          # 显式指定 patch 网格边长；None=自动
        self.use_parsed_edges = bool(use_parsed_edges)
        # ---- 空间渐变替换式（主干注意力重加权，不是加性偏置）----------------
        # 把 reweight 模式里全图统一的 content_target 换成按锚点渐变的场：
        #   锚点处 t_center，沿半径线性降到 r 处的 t_edge，r 之外为 0（不替换）。
        self.replace_graded = bool(replace_graded)
        self.replace_t_center = float(replace_t_center)
        self.replace_t_edge = float(replace_t_edge)
        # 时间窗 (lo, hi)：只在这个去噪区间内施加替换，其余步完全不替换。
        # 用**归一化进度**表示：p = 1 − t/t_max，0 = 最早期（噪声最大），
        # 1 = 最末期（接近成图）。默认 None 表示全程施加。
        self.replace_window = tuple(replace_window) if replace_window else None
        # 加性替换：只抬角色 token、不压其他 token（行和 = 1 + rise）
        self.replace_additive = bool(replace_additive)
        self.replace_delta_max = float(replace_delta_max)
        # 空间轴：对象 token 间的竞争锐化指数（>1 生效，1.0 = 关闭）
        self.replace_obj_sharpen = float(replace_obj_sharpen)
        self.verbose = bool(verbose)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


def in_replace_window(timestep, cfg: LCARConfig):
    """当前去噪步是否落在替换时间窗内（窗为 None 时恒为 True）。

    时间窗用归一化进度 p = 1 − t/t_max 表示：0 = 最早期（噪声最大），
    1 = 最末期（接近成图）。
    """
    if cfg.replace_window is None:
        return True
    lo, hi = cfg.replace_window
    t = float(timestep.flatten()[0]) if torch.is_tensor(timestep) else float(timestep)
    p = 1.0 - (t / cfg.t_max if cfg.t_max > 0 else 0.0)      # 0=最早，1=最末
    return (p >= lo) and (p <= hi)


# --------------------------------------------------------------------------
# 跨步缓存
# --------------------------------------------------------------------------
class LCARState:
    """锚点在前 ``calc_ratio`` 比例的去噪步算一次并缓存（按 block 分键）。"""

    def __init__(self):
        self.delta = {}        # block -> [B, M, L] 已算好的重分配量
        self.target_field = {} # block -> [B, M] 空间渐变的 content_target 场
        self.anchors = {}      # block -> list[dict] 锚点明细（诊断用）
        self.computed_at = {}  # block -> 该缓存是在第几个去噪步算的
        self.diag = []         # 每个 prompt 的锚点诊断行
        self.step_id = -1      # 去噪步序号（timestep 每变化一次 +1）
        self.last_t = None

    def tick(self, timestep):
        """timestep 每变化一次就把步序号 +1；同一去噪步内的各 block 共享序号。"""
        t = float(timestep.flatten()[0]) if torch.is_tensor(timestep) else float(timestep)
        if self.last_t is None or t != self.last_t:
            self.step_id += 1
            self.last_t = t
        return self.step_id

    def reset(self):
        self.delta.clear()
        self.target_field.clear()
        self.anchors.clear()
        self.computed_at.clear()
        self.diag.clear()
        self.step_id = -1
        self.last_t = None


# --------------------------------------------------------------------------
# patch 网格距离
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 替换目标 token 的排除表
# --------------------------------------------------------------------------
# **最小干预**：只排除数词。
#
# 为什么只有数词：
#   * numeracy 崩溃的元凶定位得很干净 —— "six airplanes" 里解析器把 six 同时标成
#     **对象**和**属性**，替换把注意力质量抬到 six 上，模型就数不出数。把 six 移出
#     替换目标即可，不需要动别的词。
#   * 其余 7 类的提示词里**不出现数词**，所以这张表对它们是**完全的空操作** ——
#     color / shape / texture / spatial / 3d_spatial / non_spatial 的生成与不含
#     排除表的版本逐位相同，已知的 color 增益不受任何影响。
#
# 为什么不再放冠词 / 方位词 / 关系虚词：
#   实测一并剔除会**损害生成质量**（A/B 对照里 color 明显过饱和、部分图退化）——
#   冠词本来无害，剔除它只是打破了注意力重分配的平衡。曾一次排除 73 个词，结果
#   把已经 work 的 color 弄坏了。方位词 / 关系词是否有害尚未被单独验证，在没有
#   证据之前不引入。
LCAR_EXCLUDE_WORDS = {
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
    'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
    'hundred', 'thousand',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
}


def build_exclude_mask(tokens, length):
    """把 T5 子词序列映射成 [L] 的排除掩码（True = 不参与替换）。

    ``tokens`` 是 tokenizer 给出的子词字符串（可能带 '▁' 前缀）。只要子词
    本身或它剥离前缀后的形式落在排除表里，就整段排除。
    """
    mask = torch.zeros(length, dtype=torch.bool)
    for i, t in enumerate(tokens[:length]):
        w = t.replace('▁', '').replace('</w>', '').strip().lower()
        if w in LCAR_EXCLUDE_WORDS:
            mask[i] = True
    return mask


def support_radius(coords, omega, anchor, rho=1.0):
    """支撑域半径：锚点到 Ω_o 边界的距离。

    原规格的 ``r = ρ·√|Ω_o|`` 在本分辨率下量纲不对 —— 32×32 网格里最大距离
    只有 44 个 patch，而 ρ=8、|Ω_o|=414 会算出 r=163，导致**没有任何 patch 落在
    半径之外**，"局部竞争"退化成全局。改用 Ω_o 自身的边界做半径后：

      * r 随物体大小自适应（小物体小半径，大物体大半径）；
      * 半径外 == 对象区域外，"半径外不替换" 才有实际含义；
      * ``rho`` 仍保留为倍率（默认 1.0 即恰好取到边界）。

    返回一个 Python float；Ω_o 只有一个 patch 或被裁剪时下限为 2。
    """
    idx = torch.nonzero(omega, as_tuple=False).flatten()
    if idx.numel() == 0:
        return 2.0
    d = torch.linalg.norm(coords[idx] - coords[anchor], dim=-1)
    return max(float(d.max()) * float(rho), 2.0)


def _grid_coords(n_patches, grid=None, device=None, dtype=torch.float32):
    """返回 [n_patches, 2] 的 (row, col) 坐标。n_patches 必须是完全平方数。"""
    if grid is None:
        grid = int(round(math.sqrt(n_patches)))
        if grid * grid != n_patches:
            raise ValueError(f'patch 数 {n_patches} 不是完全平方，无法还原网格')
    rows = torch.arange(grid, device=device, dtype=dtype)
    cols = torch.arange(grid, device=device, dtype=dtype)
    rr, cc = torch.meshgrid(rows, cols, indexing='ij')
    return torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=-1)


# --------------------------------------------------------------------------
# 核心：计算 B_lcar - B
# --------------------------------------------------------------------------
def lcar_delta(S_pre, object_attention, pair, object_mask, attribute_mask,
               beta, cfg: LCARConfig, state: LCARState, block_index,
               timestep, edge_pairs=None, edge_count=None, dist=None):
    """算出要加在支路 bias 上的 ``boost + supp``，形状 [B, M, L]。

    参数
    ----
    S_pre            [B, M, L]  加 bias 之前的支路 logit（按 head 平均），锚点打分用
    object_attention [B, M, L]  每个 patch 在对象 token 上的注意力（已是 softmax 结果）
    pair             [B, L, L]  对象 token × 属性 token 的软亲和度（已按 role 掩码）
    object_mask      [B, L]     对象 token 位置
    attribute_mask   [B, L]     属性 token 位置
    beta             scalar     softplus(pair_strength)，复用现有强度，不新增参数
    dist             [M, M]     可选的预计算 patch 距离矩阵

    返回 [B, M, L]；无对象—属性对时返回全零（空操作）。
    """
    B, M, L = S_pre.shape
    device, dtype = S_pre.device, S_pre.dtype

    out = torch.zeros(B, M, L, device=device, dtype=torch.float32)

    # ---- 缓存策略（对应指南 §4.2 的 anchor_update 消融）
    # once   : 在前 calc_ratio 比例的早期步里算一次，之后各步直接用缓存
    # every5 : 每 5 个去噪步重算一次
    # ema    : 每次重算后与旧缓存做指数滑动平均
    step_id = state.tick(timestep)
    cached = state.delta.get(block_index)
    if cached is not None:
        if cfg.anchor_update == 'once':
            return cached.to(dtype)
        if cfg.anchor_update == 'every5' and (step_id - state.computed_at[block_index]) < 5:
            return cached.to(dtype)
        if cfg.anchor_update == 'ema' and (step_id - state.computed_at[block_index]) < 1:
            return cached.to(dtype)
    if not _in_anchor_window(timestep, cfg):
        if cached is not None:
            return cached.to(dtype)
        return torch.zeros(B, M, L, device=device, dtype=dtype)

    coords = _grid_coords(M, grid=cfg.grid, device=device, dtype=torch.float32)
    beta_f = float(beta.detach().float()) if torch.is_tensor(beta) else float(beta)

    anchors_per_batch = []
    any_anchor = False

    for b in range(B):
        anchors = _build_anchors(
            b, S_pre, object_attention, pair, object_mask, attribute_mask,
            coords, cfg, edge_pairs, edge_count)
        anchors_per_batch.append(anchors)
        if anchors:
            any_anchor = True

    state.anchors[block_index] = anchors_per_batch

    if not any_anchor:
        # 没有任何 (对象, 属性) 对 —— 逐位返回 0（空操作）
        state.delta[block_index] = out
        state.computed_at[block_index] = step_id
        return out.to(dtype)

    for b in range(B):
        anchors = anchors_per_batch[b]
        if not anchors:
            continue
        K = len(anchors)
        # ---- 径向核 κ [K, M] 与软归属 w [K, M]
        kappa = torch.zeros(K, M, device=device, dtype=torch.float32)
        for k, a in enumerate(anchors):
            c = a['anchor']
            d = torch.linalg.norm(coords - coords[c], dim=-1)     # [M]
            r = support_radius(coords, a['omega'], c, cfg.rho)
            sig = max(r / 2.0, 1.0)
            if cfg.kernel == 'gauss':
                kappa[k] = torch.exp(-(d * d) / (2.0 * sig * sig))
            else:
                kappa[k] = (d < r).float()

        if cfg.assign == 'hard':
            w = F.one_hot(kappa.argmax(dim=0), num_classes=K).t().float()
        else:
            w = kappa / (kappa.sum(dim=0, keepdim=True) + 1e-6)

        # 支撑域内归一：只在 Ω_o 上有归属
        supp_mask = torch.stack([a['omega'] for a in anchors], dim=0).float()  # [K,M]
        w = w * supp_mask

        # ---- 3) 增强：本对象及其配对属性 token
        h = object_attention[b]            # [M, L]
        for k, a in enumerate(anchors):
            o, at = a['o'], a['a']
            out[b, :, at] += beta_f * h[:, o] * float(pair[b, o, at]) * w[k]

        # ---- 4) 竞争抑制：只压「他对象」的对象 token 与其属性 token
        if cfg.suppress_set != 'none' and cfg.mu > 0:
            owner = w.argmax(dim=0)                     # [M] 每个 patch 归属的锚点
            for k, a in enumerate(anchors):
                o_k = a['o']
                own = (owner == k).float() * a['omega'].float()   # [M]
                own = (owner == k).float()
                for o2 in a['objects']:
                    if o2 == o_k:
                        continue
                    targets = [o2]
                    if cfg.suppress_set == 'obj+attr':
                        targets = targets + list(a['attrs_of'].get(o2, []))
                    for t in targets:
                        out[b, :, t] -= (cfg.mu * beta_f * h[:, o2] * w[k] * own)

    if cfg.anchor_update == 'ema' and cached is not None:
        out = 0.5 * cached.float() + 0.5 * out
    state.delta[block_index] = out
    state.computed_at[block_index] = step_id
    return out.to(dtype)


def spatial_content_target(S_pre, object_attention, pair, object_mask, attribute_mask,
                           cfg: LCARConfig, state: LCARState, block_index,
                           timestep, edge_pairs=None, edge_count=None):
    """按**锚点**渐变的 ``content_target`` 空间场，形状 [B, M]。

    这是**已被验证有效**的那一版：8 类完整评测显示 color 相对冻结 +0.0722、
    并把 spatial / 3d_spatial / shape 从无差别替换造成的破坏中救回。

        t(p) = t_center − (t_center − t_edge) · (d(p, c_owner) / r_owner)   d ≤ r_owner
             = 0                                                            d > r_owner

    半径 ``r`` = 锚点到支撑域 Ω_o 边界的距离（由数据自定，随物体大小自适应）。
    半径外 == Ω_o 之外 —— **完全不替换**。这条"区域外置 0"是它能救回其他类的
    关键：若把强度铺满全图，每个 patch 都拿到中等强度，生成会整体崩坏。

    每个 Patch 取**最近的锚点**作为 owner，因此不同对象的支撑域互不串扰。
    """
    B, M, L = S_pre.shape
    device, dtype = S_pre.device, S_pre.dtype
    step_id = state.tick(timestep)
    cached = state.target_field.get(block_index)
    if cached is not None:
        return cached.to(dtype)
    if not _in_anchor_window(timestep, cfg):
        return torch.zeros(B, M, device=device, dtype=dtype)

    coords = _grid_coords(M, grid=cfg.grid, device=device, dtype=torch.float32)
    field = torch.zeros(B, M, device=device, dtype=torch.float32)
    span = cfg.replace_t_center - cfg.replace_t_edge

    for b in range(B):
        anchors = _build_anchors(b, S_pre, object_attention, pair, object_mask,
                                 attribute_mask, coords, cfg, edge_pairs, edge_count)
        if not anchors:
            continue
        best_d = torch.full((M,), float('inf'), device=device)
        best_t = torch.zeros(M, device=device)
        for a in anchors:
            c = a['anchor']
            d = torch.linalg.norm(coords - coords[c], dim=-1)          # [M]
            r = support_radius(coords, a['omega'], c, cfg.rho)
            frac = (d / r).clamp(0.0, 1.0)
            t = cfg.replace_t_center - span * frac
            inside = a['omega'] & (d <= r)
            t = torch.where(inside, t, torch.zeros_like(t))
            take = d < best_d
            best_d = torch.where(take, d, best_d)
            best_t = torch.where(take, t, best_t)
        field[b] = best_t

    state.target_field[block_index] = field
    state.anchors[block_index] = [anchors]
    state.computed_at[block_index] = step_id

    import os as _os
    if _os.environ.get('LCAR_DEBUG_FIELD'):
        _c = B // 2
        for _lbl, _sl in (('无条件路', slice(0, _c)), ('条件路', slice(_c, B))):
            _f = field[_sl]
            print(f'[lcar-field] {_lbl} 支撑域占比={float((_f>0).float().mean())*100:5.1f}%  '
                  f't均值={float(_f[_f>0].mean()) if (_f>0).any() else 0:.3f}  '
                  f'锚点数={sum(len(x) for x in state.anchors.get(block_index, [[]])[_sl]) if state.anchors.get(block_index) else 0}',
                  flush=True)
    return field.to(dtype)


def _in_anchor_window(timestep, cfg: LCARConfig):
    """判断当前步是否落在「前 calc_ratio 比例的去噪步」内。

    PixArt 的 timestep 从 t_max 递减到 0，因此「前 25% 步」= timestep 较大的一段。
    """
    if cfg.calc_ratio >= 1.0:
        return True
    if timestep is None:
        return True
    t = float(timestep.flatten()[0]) if torch.is_tensor(timestep) else float(timestep)
    # 归一化到 [0,1]，1 表示最早期
    frac = t / cfg.t_max if cfg.t_max > 0 else 0.0
    return frac >= (1.0 - cfg.calc_ratio)


def _build_anchors(b, S_pre, object_attention, pair, object_mask, attribute_mask,
                   coords, cfg: LCARConfig, edge_pairs, edge_count):
    """为一个 batch 元素找出全部 (对象, 属性) 锚点。"""
    L = object_mask.shape[1]
    obj_idx = torch.nonzero(object_mask[b] > 0, as_tuple=False).flatten().tolist()
    attr_idx = torch.nonzero(attribute_mask[b] > 0, as_tuple=False).flatten().tolist()
    if not obj_idx or not attr_idx:
        return []

    h = object_attention[b]                       # [M, L]
    S = S_pre[b]                                  # [M, L]

    # 对象 -> 其配对属性。优先用解析器给出的边；没有就从软亲和度里取
    attrs_of = {}
    if cfg.use_parsed_edges and edge_pairs is not None:
        n = int(edge_count[b]) if edge_count is not None else edge_pairs.shape[1]
        for e in range(min(n, edge_pairs.shape[1])):
            o, a = int(edge_pairs[b, e, 0]), int(edge_pairs[b, e, 1])
            if o < 0 or a < 0:
                continue
            attrs_of.setdefault(o, set()).add(a)
    if not attrs_of:
        for o in obj_idx:
            row = pair[b, o]
            for a in attr_idx:
                if float(row[a]) > 0.5:
                    attrs_of.setdefault(o, set()).add(a)

    anchors = []
    for o in obj_idx:
        col = h[:, o]                             # [M] 对象 o 的逐 patch 响应
        mx = float(col.max())
        if mx <= 0:
            continue
        omega = col > (cfg.tau_rel * mx)          # [M] 支撑域 Ω_o
        if int(omega.sum()) == 0:
            continue
        for a in sorted(attrs_of.get(o, ())):
            if cfg.anchor_src == 'random':
                # 复刻 probe_random 的随机锚点写法（只影响消融，结果不利须如实报告）
                c = int(torch.randint(0, col.shape[0], (1,)).item())
            elif cfg.anchor_src == 'obj':
                masked = col.masked_fill(~omega, -1e4)
                c = int(masked.argmax())
            else:  # 'attr'（默认）
                score = S[:, a].masked_fill(~omega, -1e4) + cfg.lam * col
                c = int(score.argmax())
            anchors.append(dict(o=o, a=a, anchor=c, omega=omega,
                                objects=obj_idx, attrs_of=attrs_of))
    return anchors


# --------------------------------------------------------------------------
# 挂载 / 卸载
# --------------------------------------------------------------------------
def attach(model, cfg: LCARConfig, state: LCARState = None):
    """把 LCAR 配置挂到主干上所有支路模块。

    ``cfg.enable=False`` 时也照样挂载 —— 但支路里那个 if 分支不会执行，
    因此与完全不打补丁的行为逐位相同，便于做 C2 的等价性自测。
    ``layers='mid'`` 时只挂中间层（消融用）。
    """
    state = state or LCARState()
    targets = [m for m in model.modules()
               if type(m).__name__ == 'SemanticTokenCrossAttention']
    if cfg.layers == 'mid' and targets:
        lo = len(targets) // 4
        hi = len(targets) - lo
        keep = set(range(lo, hi))
        targets = [m for i, m in enumerate(targets) if i in keep]
    for m in targets:
        m.lcar = cfg
        m.lcar_state = state
    return len(targets), state


def detach(model):
    """摘掉 LCAR，回到原始支路行为。"""
    n = 0
    for m in model.modules():
        if type(m).__name__ == 'SemanticTokenCrossAttention':
            m.lcar = None
            m.lcar_state = None
            n += 1
    return n


# --------------------------------------------------------------------------
# 图4 需要的中间量导出
# --------------------------------------------------------------------------
def export_mechanism(state: LCARState, block_index):
    """取某个 block 缓存的锚点/核，供 capture_lcar.py 写 npz。"""
    return {'anchors': state.anchors.get(block_index),
            'delta': state.delta.get(block_index)}
