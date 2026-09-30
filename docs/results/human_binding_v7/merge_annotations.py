"""把两名标注者各自导出的 CSV 合成一张表，交给 score_annotation.py。

用法：
  python3 merge_annotations.py 标注者1.csv 标注者2.csv [输出.csv]

两份输入都可以是「标注页.html」点导出得到的文件（annotator1 列里是 `A:x B:y`）。
"""
import csv
import sys
from pathlib import Path

D = Path('/root/private_data/PixArt-alpha-attentiongate/results/human_binding_v7')


def read_cells(path):
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8-sig')):
        # 导出文件把结果写在 annotator1 列；也兼容已分列的表
        cell = (r.get('annotator1') or '').strip()
        if cell:
            out[int(r['item_id'])] = cell
    return out


def main():
    a = read_cells(sys.argv[1])
    b = read_cells(sys.argv[2])
    out_p = Path(sys.argv[3]) if len(sys.argv) > 3 else D / 'annotation_sheet_filled.csv'
    base = list(csv.DictReader((D / 'annotation_sheet.csv').open()))
    n1 = n2 = both = 0
    with out_p.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['item_id', 'category', 'prompt', 'image_A', 'image_B',
                    'annotator1', 'annotator2', 'notes'])
        for r in base:
            iid = int(r['item_id'])
            c1, c2 = a.get(iid, ''), b.get(iid, '')
            n1 += bool(c1); n2 += bool(c2); both += bool(c1 and c2)
            w.writerow([iid, r['category'], r['prompt'], r['image_A'], r['image_B'],
                        c1, c2, ''])
    print(f'标注者1 填了 {n1} 行，标注者2 填了 {n2} 行，两边都有 {both} 行')
    print(f'-> {out_p}')
    print(f'下一步：python3 score_annotation.py {out_p}')


if __name__ == '__main__':
    main()
