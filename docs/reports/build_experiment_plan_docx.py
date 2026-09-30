from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


OUTPUT = Path.home() / "Desktop" / "SCDA_实验计划与数据表.docx"
NAVY, BLUE, LIGHT_BLUE, LIGHT_GRAY = "1F4D78", "2E74B5", "E8EEF5", "F2F4F7"


def set_font(run, size=10.5, bold=None, color=None, name="Times New Roman", italic=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if italic is not None:
        run.italic = italic


def shade(cell, color):
    tc_pr = cell._tc.get_or_add_tcPr()
    node = tc_pr.find(qn("w:shd"))
    if node is None:
        node = OxmlElement("w:shd")
        tc_pr.append(node)
    node.set(qn("w:fill"), color)


def set_cell(cell, width):
    tc_pr = cell._tc.get_or_add_tcPr()
    width_el = tc_pr.find(qn("w:tcW"))
    if width_el is None:
        width_el = OxmlElement("w:tcW")
        tc_pr.append(width_el)
    width_el.set(qn("w:w"), str(width))
    width_el.set(qn("w:type"), "dxa")
    mar = tc_pr.first_child_found_in("w:tcMar")
    if mar is None:
        mar = OxmlElement("w:tcMar")
        tc_pr.append(mar)
    for side, value in (("top", 80), ("bottom", 80), ("start", 120), ("end", 120)):
        el = mar.find(qn(f"w:{side}"))
        if el is None:
            el = OxmlElement(f"w:{side}")
            mar.append(el)
        el.set(qn("w:w"), str(value))
        el.set(qn("w:type"), "dxa")
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    layout = tbl_pr.first_child_found_in("w:tblLayout")
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    indent = OxmlElement("w:tblInd")
    indent.set(qn("w:w"), "120")
    indent.set(qn("w:type"), "dxa")
    tbl_pr.append(indent)
    for grid_col, width in zip(table._tbl.tblGrid.gridCol_lst, widths):
        grid_col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell(cell, width)


def mark_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def add_table(doc, headers, rows, widths, font_size=8.7):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    geometry(table, widths)
    mark_header(table.rows[0])
    for i, value in enumerate(headers):
        cell = table.rows[0].cells[i]
        shade(cell, LIGHT_BLUE)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(value), font_size, bold=True, color=NAVY, name="Microsoft YaHei")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(value) < 18 else WD_ALIGN_PARAGRAPH.LEFT
            set_font(p.add_run(value), font_size, name="Microsoft YaHei")
    geometry(table, widths)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    return table


def heading(doc, text, level=1):
    styles = {1: (13.5, NAVY, 13, 6), 2: (11.5, BLUE, 9, 4)}
    size, color, before, after = styles[level]
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(text), size, bold=True, color=color, name="Microsoft YaHei")


def body(doc, text, bold=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.2
    set_font(p.add_run(text), 10.2, bold=bold, name="Microsoft YaHei")
    return p


def footer_page(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(paragraph.add_run("Page "), 8.5, color="666666")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)


def build():
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2.0)
    section.left_margin = section.right_margin = Cm(2.0)
    section.header_distance = section.footer_distance = Inches(0.45)
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_font(header.add_run("SCDA 实验计划与数据表 | Draft"), 8.5, color="777777")
    footer_page(section.footer.paragraphs[0])

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(5)
    set_font(title.add_run("SCDA 实验计划与数据表"), 18, bold=True, name="Microsoft YaHei")
    sub = document.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(14)
    set_font(sub.add_run("Text-Semantic Adapter for PixArt DiT"), 10.5, color="555555", italic=True)
    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(meta.add_run("负责人：______________    起始日期：______________    实验目录：______________"), 9.5, name="Microsoft YaHei", color="555555")

    heading(document, "1. 实验目标与统一约束", 1)
    body(document, "本计划用于验证 SCDA 是否在不引入 bbox 或人工关系标注的前提下，提升 PixArt 的对象组合、属性绑定和关系一致性。除对比方法本身外，训练样本、分辨率、采样器、CFG、采样步数与随机种子必须保持一致。")
    add_table(document, ["项目", "统一设置 / 填写栏"], [
        ["基线 checkpoint", "PixArt-XL-2-256x256 / 路径：________________"],
        ["训练数据", "COCO 2017 / 实际训练样本数：________________"],
        ["语义条件", "object + attribute + relation；spaCy 模型：________________"],
        ["训练精度", "fp16 / bf16 / 其他：________________"],
        ["采样协议", "sampler：________ CFG：________ steps：________"],
        ["固定随机种子", "________________________________________"],
        ["评测 prompt 集", "GenEval / T2I-CompBench / 自建集：________________"],
    ], [2600, 6760], 9.2)

    heading(document, "2. 执行前检查清单", 1)
    add_table(document, ["编号", "检查项", "通过标准", "状态", "记录"], [
        ["P0-1", "T5 特征完整", "caption_feature 与 attention_mask 均可读取", "□", ""],
        ["P0-2", "语义 mask 完整", "semantic_token_masks 形状为 [3,120]", "□", ""],
        ["P0-3", "伪标签抽检", "随机 50 条 prompt 对象/属性/关系可接受", "□", ""],
        ["P0-4", "checkpoint 可加载", "仅 semantic adapter 报 missing keys", "□", ""],
        ["P0-5", "单 batch 前向", "无 NaN；loss 有限；梯度非零", "□", ""],
        ["P0-6", "初始等价性", "零初始化 SCDA 与 baseline 输出差异接近 0", "□", ""],
        ["P0-7", "推理条件接口", "同一 prompt 可在线生成 semantic mask", "□", ""],
    ], [850, 2200, 3000, 650, 2660])

    heading(document, "3. 分阶段实验计划", 1)
    add_table(document, ["阶段", "实验 ID", "目的", "运行内容", "停止/通过条件", "结果目录"], [
        ["0", "P0", "数据与链路验证", "准备 mask；单 batch；10–100 step smoke run", "无 NaN；adapter 有梯度；输出不恒为零", "________________"],
        ["1", "B0", "基线复现", "Frozen PixArt 推理与评测", "保存固定 seeds 的图像与全部指标", "________________"],
        ["2", "B1", "容量对照", "全量 PixArt 微调（可选）", "与 B0 同协议", "________________"],
        ["3", "B2", "全局 adapter 对照", "仅启用 global adapter", "验证增益非仅来自参数量", "________________"],
        ["4", "M1", "对象条件", "global + object", "对象/计数指标变化", "________________"],
        ["5", "M2", "属性条件", "M1 + attribute", "属性绑定变化", "________________"],
        ["6", "M3", "关系条件", "M2 + relation", "关系正确率变化", "________________"],
        ["7", "M4", "层级消融", "移除可学习 layer scale 或固定全层注入", "验证层级窗口贡献", "________________"],
        ["8", "M5", "完整方法", "完整 SCDA + time gate + dropout", "主结果与多随机种子复现", "________________"],
    ], [600, 780, 1550, 2500, 2400, 1530], 8.2)

    heading(document, "4. 推荐运行顺序", 1)
    for text in [
        "先只处理 100–500 个样本，检查语义 mask 中 object / attribute / relation 三行的非零比例。",
        "使用 P0 smoke run 观察 diffusion loss、adapter 输出范数、adapter 梯度范数与显存。",
        "固定 prompt 集和随机种子，先产出 B0、B2、M5 的图像，快速判断方法是否值得继续。",
        "确认 M5 不退化后再依次运行 M1、M2、M3、M4，避免大规模无效消融。",
        "主实验完成后，至少重复 3 个随机种子；对评分差异报告均值和标准差。",
    ]:
        p = document.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(3)
        set_font(p.add_run(text), 10.2, name="Microsoft YaHei")

    heading(document, "5. 每次训练运行登记表", 1)
    add_table(document, ["字段", "填写内容", "字段", "填写内容"], [
        ["实验 ID", "________________", "日期 / 运行人", "________________"],
        ["Git commit / diff", "________________", "配置文件", "________________"],
        ["训练样本数", "________________", "GPU / 显存", "________________"],
        ["batch / accumulate", "________________", "学习率 / wd", "________________"],
        ["epochs / steps", "________________", "总训练时长", "________________"],
        ["峰值显存", "________________", "最终 checkpoint", "________________"],
        ["是否恢复训练", "________________", "异常记录", "________________"],
    ], [1600, 3080, 1600, 3080], 9.0)

    heading(document, "6. 训练过程数据表", 1)
    add_table(document, ["实验 ID", "step", "diffusion loss", "adapter grad norm", "adapter output norm", "GPU memory", "备注"], [
        ["____", "____", "____", "____", "____", "____", ""], ["____", "____", "____", "____", "____", "____", ""],
        ["____", "____", "____", "____", "____", "____", ""], ["____", "____", "____", "____", "____", "____", ""],
        ["____", "____", "____", "____", "____", "____", ""], ["____", "____", "____", "____", "____", "____", ""],
    ], [1100, 900, 1500, 1600, 1700, 1200, 1360])
    body(document, "建议每隔固定 step 写入一次。若 adapter output norm 始终为 0 或 grad norm 长期接近 0，应暂停并检查零初始化、requires_grad、语义 mask 和 optimizer 参数组。", bold=True)

    heading(document, "7. 主结果数据表", 1)
    add_table(document, ["方法", "Seed", "GenEval", "T2I-CompBench", "TIFA/VQAScore", "CLIPScore", "FID", "备注"], [
        ["B0", "____", "____", "____", "____", "____", "____", ""], ["B0", "____", "____", "____", "____", "____", "____", ""],
        ["B0", "____", "____", "____", "____", "____", "____", ""], ["B2", "____", "____", "____", "____", "____", "____", ""],
        ["M1", "____", "____", "____", "____", "____", "____", ""], ["M2", "____", "____", "____", "____", "____", "____", ""],
        ["M3", "____", "____", "____", "____", "____", "____", ""], ["M4", "____", "____", "____", "____", "____", "____", ""],
        ["M5", "____", "____", "____", "____", "____", "____", ""], ["M5", "____", "____", "____", "____", "____", "____", ""],
        ["M5", "____", "____", "____", "____", "____", "____", ""],
    ], [850, 750, 1200, 1550, 1500, 1100, 900, 1510])

    heading(document, "8. 分项与消融数据表", 1)
    add_table(document, ["方法", "对象存在/计数", "属性绑定", "关系正确", "图像质量", "主要发现"], [
        ["B0 Baseline", "____", "____", "____", "____", ""], ["B2 Global", "____", "____", "____", "____", ""],
        ["M1 +Object", "____", "____", "____", "____", ""], ["M2 +Attribute", "____", "____", "____", "____", ""],
        ["M3 +Relation", "____", "____", "____", "____", ""], ["M4 – time gate", "____", "____", "____", "____", ""],
        ["M5 Full SCDA", "____", "____", "____", "____", ""],
    ], [1700, 1600, 1500, 1500, 1100, 1960])
    add_table(document, ["设置", "取值", "对象指标", "属性指标", "关系指标", "FID", "结论"], [
        ["adapter dim r", "16", "____", "____", "____", "____", ""], ["adapter dim r", "32", "____", "____", "____", "____", ""],
        ["adapter dim r", "64", "____", "____", "____", "____", ""], ["semantic dropout", "0.00", "____", "____", "____", "____", ""],
        ["semantic dropout", "0.10", "____", "____", "____", "____", ""], ["semantic dropout", "0.20", "____", "____", "____", "____", ""],
    ], [1600, 1000, 1500, 1500, 1500, 800, 1460])

    heading(document, "9. 伪标签质量与对齐质量", 1)
    add_table(document, ["类别", "人工标注数", "Precision", "Recall", "F1", "T5 对齐覆盖率", "错误模式"], [
        ["对象", "____", "____", "____", "____", "____", ""], ["属性", "____", "____", "____", "____", "____", ""],
        ["关系", "____", "____", "____", "____", "____", ""],
    ], [1000, 1200, 1100, 1100, 900, 1500, 2560])
    add_table(document, ["Prompt", "模型/seed", "观察到的错误", "可能原因", "是否纳入论文失败案例"], [
        ["", "", "", "", "□ 是  □ 否"], ["", "", "", "", "□ 是  □ 否"], ["", "", "", "", "□ 是  □ 否"],
        ["", "", "", "", "□ 是  □ 否"], ["", "", "", "", "□ 是  □ 否"],
    ], [2400, 1350, 2100, 2100, 1410])

    heading(document, "10. 最终论文结果核对", 1)
    add_table(document, ["核对项", "完成", "说明/文件位置"], [
        ["所有对比方法使用同一评测协议", "□", ""], ["主结果包含至少 3 个 seed 的均值和标准差", "□", ""],
        ["包含伪标签 F1 与 T5 对齐覆盖率", "□", ""], ["包含参数量、显存、训练/推理时间", "□", ""],
        ["包含对象/属性/关系分支消融", "□", ""], ["包含正例与失败案例的可视化", "□", ""],
        ["未把设计假设写成实验结论", "□", ""],
    ], [4300, 1000, 4060], 9.2)

    document.core_properties.title = "SCDA 实验计划与数据表"
    document.core_properties.author = ""
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
