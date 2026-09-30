#!/usr/bin/env python3
"""Build the consolidated SCDA experiment report as a Word document."""
from pathlib import Path
import csv
import statistics
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'SCDA_PixArt_完整实验对比报告_2026-08-30.docx'
NAVY, BLUE, PALE = '1F4D78', '2E74B5', 'E8EEF5'

def font(run, size=10, bold=False, color=None):
    run.font.name = 'Microsoft YaHei'; run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    run.font.size = Pt(size); run.bold = bold
    if color: run.font.color.rgb = RGBColor.from_string(color)

def para(doc, text, size=10, bold=False, align=None):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(5)
    if align is not None: p.alignment = align
    font(p.add_run(text), size, bold); return p

def head(doc, text, level=1):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(5)
    font(p.add_run(text), 13 if level == 1 else 11, True, NAVY if level == 1 else BLUE)

def shade(cell, value=PALE):
    el = OxmlElement('w:shd'); el.set(qn('w:fill'), value); cell._tc.get_or_add_tcPr().append(el)

def table(doc, headers, rows, size=8.3):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        shade(t.rows[0].cells[i]); p=t.rows[0].cells[i].paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; font(p.add_run(str(h)), size, True, NAVY)
    for row in rows:
        cells=t.add_row().cells
        for i, value in enumerate(row):
            p=cells[i].paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER if len(str(value)) < 28 else WD_ALIGN_PARAGRAPH.LEFT; font(p.add_run(str(value)), size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def mean_sd(rows, model):
    vals=[float(r['clip_score']) for r in rows if r['model']==model]
    return statistics.mean(vals), statistics.pstdev(vals)

def main():
    with (ROOT/'output/final_multiseed_clip_scores.csv').open() as f: scores=list(csv.DictReader(f))
    with (ROOT/'output/loss_reg_clip_scores.csv').open() as f: scores += list(csv.DictReader(f))
    with (ROOT/'output/ablate_object_clip_scores.csv').open() as f: scores += list(csv.DictReader(f))
    stats={m:mean_sd(scores,m) for m in ('baseline','trial_02','trial_09','improved','loss_reg','ablate_object')}
    baseline=stats['baseline'][0]
    doc=Document(); sec=doc.sections[0]; sec.top_margin=sec.bottom_margin=Cm(1.8); sec.left_margin=sec.right_margin=Cm(1.7)
    title=doc.add_paragraph(); title.alignment=WD_ALIGN_PARAGRAPH.CENTER; font(title.add_run('PixArt SCDA 训练与统一评测完整报告'),18,True,NAVY)
    para(doc,'版本：2026-08-30  |  范围：可追溯训练、baseline 对比、问题分析与后续方案',9,align=WD_ALIGN_PARAGRAPH.CENTER)
    head(doc,'1. 结论摘要')
    para(doc,f'当前证据不支持 SCDA 已优于 baseline。统一 CLIP 重评中，baseline 为 {baseline:.6f}，本轮改进 SCDA 为 {stats["improved"][0]:.6f}，相对差异 {(stats["improved"][0]/baseline-1)*100:.2f}%。改进版训练稳定、无 NaN/Inf，但稳定训练损失下降不等价于生成一致性提升。')
    para(doc,'本报告只陈述工作区可核验的实验。早期口头提及的“十次训练”中，未在目录、配置、日志或 checkpoint 中留下可复核证据的轮次不纳入定量比较，避免把缺失数据伪装成结论。')
    head(doc,'2. 统一评测协议')
    table(doc,['项目','统一设置'],[['文本','asset/samples.txt，64 条'],['种子','43 / 44 / 45，每模型 192 张'],['分辨率','512 x 512'],['采样','DPM-Solver，20 steps，CFG=4.0'],['一致性指标','CLIP ViT-B/32 图文余弦相似度；长文本按 77 token 截断'],['质量指标','本轮无真实 COCO 参考图像，未计算 FID/KID；未以代理指标替代质量结论']])
    head(doc,'3. 训练参数与结果总表')
    table(doc,['实验/方法','数据与步数','可训练模块/关键参数','训练结果','生成评测/结论'],[
        ['Baseline','预训练 checkpoint；未训练','PixArt-XL-2 native-gate-init；语义条件关闭','不适用','CLIP 0.273209；本报告参照'],
        ['Gate v2 / Trial 02','历史日志 70 点；完整训练配置未留存','Gate 方案；其余参数不可完全核验','loss 0.1907 -> 0.1513，min 0.1334；曾记录 grad_norm=NaN','CLIP 0.259399，较 baseline -5.06%'],
        ['Gate v3 / Trial 09','历史日志 70 点；完整训练配置未留存','neutrality + consistency 正则；其余参数不可完全核验','loss 0.1908 -> 0.1519，min 0.1337；日志无 NaN','CLIP 0.256700，较 baseline -6.05%'],
        ['SCDA pilot','512 COCO；10 epoch / 160 step','19 个 adapter/gate/scale 张量；PixArt/T5 冻结','loss 0.2076 -> 0.1516；residual 0 -> 0.0577；无 NaN','仅验证数据流/梯度；未做公平生成对比'],
        ['SCDA improved','8,617 COCO；10 epoch / 2,700 step','adapter dim=64；lr=5e-5；wd=0.03；dropout=0.1；residual=0.25；四分支','final loss=0.132721；grad=0.009002；residual=0.407930；无 NaN/Inf','CLIP 0.264862，较 baseline -3.06%；优于 Trial 02/09，但仍未超 baseline'],
        ['SCDA loss_reg','8,617 COCO；10 epoch / 2,700 step','adapter dim=64；lr=5e-5；wd=0.03；dropout=0.1；residual=0.10；四分支；正则系数=0.01','total=0.133204；diffusion=0.133001；semantic reg=0.020297；grad=0.003847；residual=0.136511；无 NaN/Inf','CLIP 0.268161，较 baseline -1.85%；较 SCDA improved +1.25%'],
        ['SCDA ablate global+object','8,617 COCO；10 epoch / 2,700 step','adapter dim=64；lr=5e-5；wd=0.03；dropout=0.1；residual=0.10；仅 global/object；正则系数=0.01','total=0.133272；diffusion=0.133074；semantic reg=0.019842；grad=0.003749；residual=0.132236；无 NaN/Inf','CLIP 0.267053，较 baseline -2.25%；低于 loss_reg 0.41%']])
    head(doc,'4. 多种子 CLIP 结果')
    result=[]
    for model,label in [('baseline','Baseline'),('trial_02','Trial 02'),('trial_09','Trial 09'),('improved','SCDA improved'),('loss_reg','SCDA loss_reg'),('ablate_object','SCDA global+object')]:
        per=[statistics.mean(float(x['clip_score']) for x in scores if x['model']==model and int(x['seed'])==s) for s in (43,44,45)]
        m,sd=stats[model]; result.append([label,*[f'{x:.6f}' for x in per],f'{m:.6f}',f'{sd:.6f}',f'{(m/baseline-1)*100:+.2f}%'])
    table(doc,['模型','seed43','seed44','seed45','192 图均值','图级标准差','相对 baseline'],result)
    para(doc,'说明：图级标准差反映 prompt 难度差异，不等同于 seed 间不确定性。三个 seed 的模型均值均显示 baseline 最好；因此不能依据单一 prompt 或单一 seed 宣称提升。',9)
    head(doc,'5. 问题分析')
    for item in ['语义残差已从零初始化增长到 0.408，但其学习目标仍是扩散重建损失，未直接优化对象、属性和关系遵循，可能出现“收敛但偏离文本”的情形。','当前语义 mask 由文本解析伪标签构造，属性与关系覆盖率低于对象；解析错误会把不可靠条件注入所有层，削弱语义信号。','四个语义分支同时训练且 adapter dim=64，缺少 object-only、object+attribute 与各 residual scale 的消融，尚不能定位退化来自哪个分支或强度。','CLIP 是整体语义代理，对组合关系不敏感；同时缺失 COCO 真实参考集，当前不能对图像保真度作 FID/KID 结论。']:
        para(doc,'- '+item)
    head(doc,'6. 后续实验与决策规则')
    table(doc,['优先级','实验','固定设置','通过标准/决策'],[
        ['P0','残差强度','scale=0.10/0.25/0.50/1.00；其余同 improved','先用 64x3；均值超过 baseline 且三 seed 中至少 2 个不低于 baseline，才扩大评测'],
        ['P1','分支消融','object；object+attribute；all；scale 取 P0 最优','定位退化分支；关系分支只有在关系子集增益明确时保留'],
        ['P2','评测集扩展','新增 COCO 风格对象/属性/关系 prompt；每类 >=100；seeds >=5','报告总体和分组均值/置信区间，禁止只报总体 CLIP'],
        ['P3','结构遵循','GenEval、T2I-CompBench 或 VQA 人工盲评','对象计数、属性绑定、空间关系分别给分；与 CLIP 联合判断'],
        ['P4','图像质量','准备真实 COCO reference 及特征','计算 FID/KID；同一预处理和样本数；避免仅靠清晰度/熵代理'],
        ['P5','训练目标','筛选后的最佳结构加入低权重语义一致性/偏好损失','先小规模验证不损害 FID，再做完整训练；保存每步各分支范数']])
    head(doc,'7. 可复现性与产物')
    para(doc,'训练 checkpoint：output/scda_improved_full/checkpoints/epoch_10_step_2700.pth、output/scda_loss_reg/checkpoints/epoch_10_step_2700.pth；训练 epoch 表：各 output/*/experiment_tables/epoch_summary.csv；统一 CLIP 原始分数：output/final_multiseed_clip_scores.csv、output/loss_reg_clip_scores.csv；按 seed 汇总：对应 *_summary.csv。')
    para(doc,'本次新增了 tools/evaluate_multiseed.py 的 --semantic-residual-scale 参数，确保改进模型按训练时 0.25 推理；新增 tools/score_clip_multiseed.py，以本地 CLIP 模型在所有候选上统一重评分。',9)
    doc.save(OUT); print(OUT)

if __name__ == '__main__': main()
