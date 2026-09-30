from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


PATH = Path.home() / "Desktop" / "SCDA_论文格式报告.docx"


def configure_style(style, size, color, before, after):
    style.font.name = "Times New Roman"
    style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = RGBColor.from_string(color)
    style.paragraph_format.space_before = Pt(before)
    style.paragraph_format.space_after = Pt(after)
    style.paragraph_format.line_spacing = 1.15
    style.paragraph_format.keep_with_next = True
    style.paragraph_format.keep_together = True
    p_pr = style._element.get_or_add_pPr()
    outline = p_pr.find(qn("w:outlineLvl"))
    if outline is None:
        outline = OxmlElement("w:outlineLvl")
        p_pr.append(outline)
    outline.set(qn("w:val"), "9")


def main():
    document = Document(PATH)
    styles = document.styles
    style_defs = {
        "Paper Section": (14, "1F4D78", 14, 7),
        "Paper Subsection": (12, "2E74B5", 10, 5),
        "Paper Subsubsection": (11, "1F4D78", 8, 4),
    }
    for name, values in style_defs.items():
        if name not in styles:
            style = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        else:
            style = styles[name]
        configure_style(style, *values)

    replacements = {"Heading 1": "Paper Section", "Heading 2": "Paper Subsection", "Heading 3": "Paper Subsubsection"}
    changed = 0
    for paragraph in document.paragraphs:
        style_name = paragraph.style.name
        if style_name in replacements:
            paragraph.style = styles[replacements[style_name]]
            changed += 1
    document.save(PATH)
    print(f"Updated {changed} heading paragraphs in {PATH}")


if __name__ == "__main__":
    main()
