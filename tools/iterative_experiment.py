#!/usr/bin/env python3
"""Create a reproducible experiment record from PixArt training runs.

This is intentionally conservative: it never deletes checkpoints and records
missing runtime assets as a blocking issue instead of silently starting over.
"""
import csv, json, re, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
TRASH_OUTPUT = Path('/root/private_data/.Trash-0/files/output')
ARCHIVE = ROOT / 'experiments'
PROMPTS = ROOT / 'asset/samples.txt'

def parse_log(path):
    rows=[]
    rx=re.compile(r'loss:([0-9.]+).*?(?:diffusion_loss:([0-9.]+))?.*?(?:gate_neutrality_loss:([0-9.]+))?.*?(?:gate_consistency_loss:([0-9.]+))?.*?(?:gate_mean:([0-9.]+))?.*?(?:gate_std:([0-9.]+))?.*?(?:grad_norm:([0-9A-Za-z.+-]+))?')
    for line in path.read_text(errors='ignore').splitlines():
        m=rx.search(line)
        if m:
            vals=m.groups(); rows.append({'loss':float(vals[0]),'diffusion_loss':float(vals[1]) if vals[1] else None,
                'neutrality_loss':float(vals[2]) if vals[2] else None,'consistency_loss':float(vals[3]) if vals[3] else None,
                'gate_mean':float(vals[4]) if vals[4] else None,'gate_std':float(vals[5]) if vals[5] else None,
                'grad_norm':vals[6]})
    return rows

def summarize(name, source):
    run=ARCHIVE/name; (run/'images').mkdir(parents=True,exist_ok=True)
    log=source/'train_log.log'; rows=parse_log(log) if log.exists() else []
    metrics={}
    if rows:
        metrics={'steps_logged':len(rows),'loss_first':rows[0]['loss'],'loss_last':rows[-1]['loss'],
                 'loss_min':min(r['loss'] for r in rows),'loss_mean_last100':sum(r['loss'] for r in rows[-100:])/min(100,len(rows)),
                 'nan_grad_steps':sum(str(r['grad_norm']).lower()=='nan' for r in rows),
                 'gate_mean_last':rows[-1]['gate_mean'],'gate_std_last':rows[-1]['gate_std']}
    for src in [source/'eval_256/baseline_native_gate_neutral.png',source/'eval_256/gate_v2_epoch10_step3130.png']:
        if src.exists(): shutil.copy2(src,run/'images'/src.name)
    for src in [source/'config.py', source/'gate_metrics.csv']:
        if src.exists(): shutil.copy2(src,run/src.name)
    (run/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return metrics

def main():
    ARCHIVE.mkdir(exist_ok=True)
    v2=summarize('run_002_gate_v2',TRASH_OUTPUT/'coco2014_gate_v2')
    v3=summarize('run_003_gate_v3',TRASH_OUTPUT/'coco2014_gate_v3')
    prompts=PROMPTS.read_text().splitlines() if PROMPTS.exists() else []
    record={'generated_at':datetime.now().isoformat(),'fixed_prompt_count':len(prompts),
            'fixed_prompt_file':str(PROMPTS),'runs':{'run_002_gate_v2':v2,'run_003_gate_v3':v3},
            'runtime_blockers':['正常工作区缺少 COCO2014Prepared10K/data_info.json','PixArt gate-init checkpoint 仅存在回收站路径；VAE 可用但 T5 权重目录为空'],
            'next_actions':['恢复或重新挂载数据与 T5 权重后，从 v3 epoch_10_step_3130 续训','每轮使用 asset/samples.txt 全部固定提示词，记录 CLIP/FID/KID 与人工结构评分','针对 v2 NaN 梯度保留 v3 的正则，并加入梯度异常自动降学习率']}
    (ARCHIVE/'EXPERIMENT_LOG.md').write_text('# PixArt 迭代训练记录\n\n'+json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(record,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
