from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / '_figures'
OUT.mkdir(exist_ok=True)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def box(ax, x, y, w, h, text, color='#E8EEF5', edge='#2E74B5', fs=9):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.02',fc=color,ec=edge,lw=1.4)
    ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,wrap=True)

def arrow(ax, x1,y1,x2,y2, text=''):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,lw=1.2,color='#555'))
    if text: ax.text((x1+x2)/2,(y1+y2)/2+0.03,text,ha='center',fontsize=8,color='#444')

def structure():
    fig,ax=plt.subplots(figsize=(12,6)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis('off')
    box(ax,.3,2.3,1.5,1.1,'文本提示词\np', '#FDEBD0','#C97A10'); box(ax,2.2,2.3,1.5,1.1,'T5-XXL\n编码器','#E8F6E8','#3A8D40')
    box(ax,4.1,2.3,1.5,1.1,'CaptionEmbedder\nD=1152','#E8EEF5'); box(ax,6.0,2.3,1.5,1.1,'文本 token\nY[L,120,D]','#E8EEF5')
    box(ax,2.2,.55,1.5,1.0,'VAE\n压缩','#FDEBD0','#C97A10'); box(ax,4.1,.55,1.5,1.0,'加噪 latent\nz_t','#FDEBD0','#C97A10'); box(ax,6.0,.55,1.5,1.0,'Patchify + 2D位置\nX_t[Q,D]','#FDEBD0','#C97A10')
    box(ax,8.0,1.45,2.0,2.0,'PixArt DiT\n28 blocks\nAdaLN-single\nSelf-Attn\nCross-Attn\nMLP','#EAF3F8','#1F4D78',10)
    box(ax,10.5,2.3,1.2,1.1,'预测噪声\nεθ','#E8F6E8','#3A8D40')
    arrow(ax,1.8,2.85,2.2,2.85); arrow(ax,3.7,2.85,4.1,2.85); arrow(ax,5.6,2.85,6,2.85); arrow(ax,7.5,2.85,8,2.85,'cross-attn')
    arrow(ax,3.7,1.05,4.1,1.05); arrow(ax,5.6,1.05,6,1.05); arrow(ax,7.5,1.05,8,1.75,'image tokens')
    arrow(ax,10,2.45,10.5,2.85)
    box(ax,6.0,4.55,3.0,.8,'MS-SCDA 语义残差：全局/对象/属性/关系','#FFF2CC','#B7950B',9)
    arrow(ax,7.5,4.55,8.8,3.45,'按层与时间步注入')
    ax.set_title('图1  PixArt-α 与 MS-SCDA 总体结构',fontsize=13,fontweight='bold'); fig.tight_layout(); fig.savefig(OUT/'pixart_overall.png',dpi=220); plt.close(fig)

def split():
    fig,ax=plt.subplots(figsize=(12,5)); ax.set_xlim(0,12); ax.set_ylim(0,5); ax.axis('off')
    box(ax,.3,3.3,2.2,1.0,'原始 prompt\n"A red car beside a blue bike"','#FDEBD0','#C97A10')
    box(ax,3.0,3.3,1.8,1.0,'spaCy 依存分析\n词性/依存边','#E8F6E8','#3A8D40')
    box(ax,5.3,3.55,1.3,.55,'对象\ncar, bike','#D6EAF8','#2874A6'); box(ax,5.3,2.65,1.3,.55,'属性\nred, blue','#FADBD8','#A93226'); box(ax,7.0,3.55,1.3,.55,'关系\nbeside','#E8DAEF','#7D3C98'); box(ax,7.0,2.65,1.3,.55,'全局\n全部 token','#E5E7E9','#566573')
    box(ax,9.0,3.3,2.3,1.0,'T5 fast tokenizer\nSentencePiece offsets','#E8EEF5','#2E74B5')
    box(ax,9.0,1.35,2.3,1.0,'四类 mask [120]\noverlap 对齐 + 池化','#FFF2CC','#B7950B')
    for a,b in [(2.5,3.8),(4.8,5.3),(6.6,7),(8.3,9),(10.15,10.15)]: pass
    arrow(ax,2.5,3.8,3,3.8); arrow(ax,4.8,3.8,5.3,3.8); arrow(ax,4.8,3.8,5.3,2.95); arrow(ax,4.8,3.8,7,3.8); arrow(ax,4.8,3.8,7,2.95); arrow(ax,8.3,3.8,9,3.8); arrow(ax,10.15,3.3,10.15,2.35)
    ax.text(9.0,.75,'子词区间 (a_j,b_j) 与词区间 (s_i,e_i) 存在正长度交叠 => mask=1',fontsize=9)
    ax.set_title('图2  文本拆分、子词对齐与四类语义条件构造',fontsize=13,fontweight='bold'); fig.tight_layout(); fig.savefig(OUT/'text_split.png',dpi=220); plt.close(fig)

def injection():
    fig,ax=plt.subplots(figsize=(12,5)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis('off')
    ax.add_patch(plt.Rectangle((1,1),9,3.8,fc='#F8F9F9',ec='#566573'))
    ax.text(.4,4.7,'DiT层',fontsize=10,fontweight='bold'); ax.text(1.2,5.25,'前半层',fontsize=9); ax.text(4.8,5.25,'中间层',fontsize=9); ax.text(8.4,5.25,'后半层',fontsize=9)
    for i in range(14):
        x=1.2+i*.62; c='#D6EAF8' if i<7 else ('#E8DAEF' if i<10 else '#FADBD8')
        ax.add_patch(plt.Rectangle((x,2),.45,1.4,fc=c,ec='#7B7D7D')); ax.text(x+.225,1.75,str(i+1),ha='center',fontsize=7)
    ax.text(1.2,.9,'对象分支',color='#2874A6',fontsize=9); ax.text(5.2,.9,'关系分支',color='#7D3C98',fontsize=9); ax.text(8.4,.9,'属性分支',color='#A93226',fontsize=9)
    for y,label,col in [(4.25,'全局 residual','#B7950B'),(3.75,'对象 residual','#2874A6'),(3.25,'关系 residual','#7D3C98'),(2.75,'属性 residual','#A93226')]:
        ax.annotate(label,xy=(6.0,y),xytext=(10.3,y),arrowprops=dict(arrowstyle='->',color=col),color=col,fontsize=9)
    ax.text(1.2,1.25,'t: 高噪声 -> 低噪声，gamma_k(t) 动态调节各语义分支强度',fontsize=9)
    ax.set_title('图3  MS-SCDA 在 PixArt DiT 层级和扩散时间步中的插入位置',fontsize=13,fontweight='bold'); fig.tight_layout(); fig.savefig(OUT/'injection.png',dpi=220); plt.close(fig)

def detailed_structure():
    """High-detail publication figure: data paths, tensor shapes, block internals and injection sites."""
    fig, ax = plt.subplots(figsize=(18, 10)); ax.set_xlim(0, 18); ax.set_ylim(0, 10); ax.axis('off')
    # Main horizontal data flow
    box(ax, .3, 7.7, 2.0, 1.0, 'Prompt p\ntext string', '#FDEBD0', '#C97A10', 10)
    box(ax, 2.8, 7.7, 2.0, 1.0, 'T5-XXL\nL=120, d=4096', '#E8F6E8', '#3A8D40', 10)
    box(ax, 5.3, 7.7, 2.0, 1.0, 'CaptionEmbedder\n4096 -> D=1152', '#E8EEF5', '#2E74B5', 9)
    box(ax, 7.8, 7.7, 2.0, 1.0, 'Text tokens Y\n[ B,120,1152 ]', '#E8EEF5', '#2E74B5', 9)
    arrow(ax, 2.3, 8.2, 2.8, 8.2); arrow(ax, 4.8, 8.2, 5.3, 8.2); arrow(ax, 7.3, 8.2, 7.8, 8.2)
    box(ax, .3, 5.9, 2.0, 1.0, 'Image x0\nH x W x 3', '#FDEBD0', '#C97A10', 10)
    box(ax, 2.8, 5.9, 2.0, 1.0, 'VAE encoder\nlatent [B,4,h,w]', '#FDEBD0', '#C97A10', 9)
    box(ax, 5.3, 5.9, 2.0, 1.0, 'Noise scheduler\nzt = a_t z0 + s_t eps', '#FDEBD0', '#C97A10', 8.5)
    box(ax, 7.8, 5.9, 2.0, 1.0, 'Patchify + 2D pos\nQ=h/2 x w/2', '#FDEBD0', '#C97A10', 8.5)
    arrow(ax, 2.3, 6.4, 2.8, 6.4); arrow(ax, 4.8, 6.4, 5.3, 6.4); arrow(ax, 7.3, 6.4, 7.8, 6.4)
    # Central stack
    box(ax, 10.4, 5.75, 3.0, 3.1, 'PixArt-XL/2\nDiT backbone\n28 repeated blocks\n\nX: [B,Q,1152]\nT: timestep embedding', '#D6EAF8', '#1F4D78', 11)
    arrow(ax, 9.8, 6.4, 10.4, 6.9, 'image tokens'); arrow(ax, 9.8, 8.2, 10.4, 7.8, 'cross-attn')
    box(ax, 14.3, 6.45, 1.9, 1.1, 'Noise head\nlinear D -> 4p2', '#E8F6E8', '#3A8D40', 9)
    arrow(ax, 13.4, 7.3, 14.3, 7.0); box(ax, 16.3, 6.45, 1.4, 1.1, 'eps_theta\n[B,4,h,w]', '#E8F6E8', '#3A8D40', 8.5); arrow(ax, 16.2, 7.0, 16.3, 7.0)
    # Semantic branch panel
    ax.add_patch(plt.Rectangle((.25, .35), 9.55, 4.8, fc='#FFFDF5', ec='#B7950B', lw=1.5))
    ax.text(.5, 4.82, 'MS-SCDA semantic branch (added, trainable; PixArt/T5 frozen)', fontsize=11, fontweight='bold', color='#7D6608')
    box(ax, .55, 3.65, 2.1, .75, 'spaCy dependency parse\nobject / attribute / relation', '#E8F6E8', '#3A8D40', 8.5)
    box(ax, 3.0, 3.65, 2.1, .75, 'T5 offset alignment\nmask_obj, mask_attr, mask_rel', '#E8F6E8', '#3A8D40', 8.3)
    arrow(ax, 2.65, 4.02, 3.0, 4.02)
    labels = [('global', '#E5E7E9', '#566573'), ('object', '#D6EAF8', '#2874A6'), ('attribute', '#FADBD8', '#A93226'), ('relation', '#E8DAEF', '#7D3C98')]
    for i,(lab,c,e) in enumerate(labels):
        x = .65 + (i%2)*2.45; y = 2.35 - (i//2)*1.0
        box(ax, x, y, 1.95, .65, lab+' pool c^k\nmean over masked Y', c, e, 8.2)
    box(ax, 5.9, 2.0, 1.95, 1.0, '4 bottleneck adapters\n1152 -> 64 -> 1152\nzero-init W_up', '#FFF2CC', '#B7950B', 8.5)
    arrow(ax, 5.1, 3.95, 5.9, 2.75); arrow(ax, 4.6, 2.7, 5.9, 2.7); arrow(ax, 4.6, 1.7, 5.9, 2.3)
    box(ax, 8.25, 2.0, 1.2, 1.0, 'R_l(t)\nlayer scale\n+ time gate', '#FFF2CC', '#B7950B', 8.5); arrow(ax, 7.85, 2.5, 8.25, 2.5)
    arrow(ax, 9.45, 2.5, 11.5, 5.75, 'add to block hidden states')
    # Expanded DiT block
    ax.add_patch(plt.Rectangle((10.4, .35), 7.25, 4.9, fc='#F8F9F9', ec='#1F4D78', lw=1.5))
    ax.text(10.65, 4.82, 'Expanded DiT block l', fontsize=11, fontweight='bold', color='#1F4D78')
    box(ax, 10.75, 3.85, 1.65, .62, 'AdaLN-single\n(X, t)', '#D6EAF8', '#2874A6', 8.5)
    box(ax, 12.75, 3.85, 1.65, .62, 'Self-Attention\nQ x Q', '#D6EAF8', '#2874A6', 8.5)
    box(ax, 14.75, 3.85, 1.65, .62, 'Residual +\nAdaLN', '#D6EAF8', '#2874A6', 8.5)
    arrow(ax, 12.4, 4.16, 12.75, 4.16); arrow(ax, 14.4, 4.16, 14.75, 4.16)
    box(ax, 10.75, 2.65, 1.65, .62, 'AdaLN\n(X, t)', '#EAF3F8', '#2E74B5', 8.5)
    box(ax, 12.75, 2.65, 1.65, .62, 'Cross-Attention\nK,V from Y', '#EAF3F8', '#2E74B5', 8.5)
    box(ax, 14.75, 2.65, 1.65, .62, 'Add semantic\nresidual R_l(t)', '#FFF2CC', '#B7950B', 8.5)
    arrow(ax, 12.4, 2.96, 12.75, 2.96); arrow(ax, 14.4, 2.96, 14.75, 2.96)
    box(ax, 10.75, 1.45, 1.65, .62, 'AdaLN\n(X, t)', '#EAF3F8', '#2E74B5', 8.5)
    box(ax, 12.75, 1.45, 1.65, .62, 'MLP\nGELU / linear', '#EAF3F8', '#2E74B5', 8.5)
    box(ax, 14.75, 1.45, 1.65, .62, 'Block output\nX_{l+1}', '#EAF3F8', '#2E74B5', 8.5)
    arrow(ax, 12.4, 1.76, 12.75, 1.76); arrow(ax, 14.4, 1.76, 14.75, 1.76)
    ax.text(16.65, 3.0, 'u_lk: layer windows\nbeta_lk: learned scale\ngamma_k(t): sigmoid gate\ns=0.25', fontsize=8.5, va='center', color='#7D6608')
    ax.set_title('Detailed PixArt-α + MS-SCDA Architecture', fontsize=16, fontweight='bold'); fig.tight_layout(); fig.savefig(OUT/'pixart_detailed_structure.png', dpi=300, bbox_inches='tight'); plt.close(fig)

if __name__=='__main__': structure(); split(); injection(); detailed_structure()
