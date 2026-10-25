"""Shared layout helpers for the two report builders.

`build_report.py` writes the standalone 15-page report; `build_part_e_report.py`
writes the same content into the unit's Part E template.  Both need the same
paragraph, table, figure and caption styling, so it lives here rather than in
each builder.
"""
from __future__ import annotations

from pathlib import Path

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

NAVY = RGBColor(0x1F, 0x38, 0x64)
GREY = RGBColor(0x40, 0x40, 0x40)
FILL_HEAD = "1F3864"
FILL_TOTAL = "DCE6F2"
FILL_BAND = "F2F5FA"


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
THEME_ATTRS = ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme")


def force_font(element, name: str) -> None:
    """Set the four rFonts slots and drop any theme reference that would win."""
    rPr = element.find(f"{W}rPr")
    if rPr is None:
        return
    rFonts = rPr.find(f"{W}rFonts")
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    for attr in THEME_ATTRS:
        key = f"{W}{attr}"
        if key in rFonts.attrib:
            del rFonts.attrib[key]
    for slot in ("ascii", "hAnsi", "eastAsia", "cs"):
        rFonts.set(f"{W}{slot}", name)


def style_document(doc, heading_colour=NAVY, body_size=11,
                   margins=(2.2, 2.3), line_spacing=1.15) -> None:
    """Calibri body text, navy headings, A4 page with the given margins."""
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin = sec.bottom_margin = Cm(margins[0])
    sec.left_margin = sec.right_margin = Cm(margins[1])

    normal = doc.styles["Normal"]
    force_font(normal.element, "Calibri")
    normal.font.name = "Calibri"
    normal.font.size = Pt(body_size)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = line_spacing
    for name, size, before in (("Heading 1", 15, 18), ("Heading 2", 12.5, 14)):
        st = doc.styles[name]
        force_font(st.element, "Calibri")
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = heading_colour
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(8)
        st.paragraph_format.keep_with_next = True


def shade(cell, fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


class Layout:
    """Paragraph/table/figure emitters bound to one Document."""

    def __init__(self, doc, fig_dir: Path):
        self.doc = doc
        self.fig_dir = Path(fig_dir)

    def body(self, text: str):
        p = self.doc.add_paragraph(text)
        p.paragraph_format.space_after = Pt(6)
        return p

    def bullet(self, text: str):
        if "List Bullet" in {s.name for s in self.doc.styles}:
            p = self.doc.add_paragraph(text, style="List Bullet")
        else:
            # the Part E template ships no list style: draw the bullet ourselves
            p = self.doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.65)
            p.paragraph_format.first_line_indent = Cm(-0.4)
            p.add_run("\u2022  " + text)
        p.paragraph_format.space_after = Pt(3)
        return p

    def caption(self, text: str):
        p = self.doc.add_paragraph()
        r = p.add_run(text)
        r.italic = True
        r.font.size = Pt(9)
        r.font.color.rgb = GREY
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(12)
        return p

    def figure(self, name: str, width_cm: float, text: str):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(str(self.fig_dir / name), width=Cm(width_cm))
        self.caption(text)
        return p

    def table(self, rows, widths, text: str):
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for row in t.rows:
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Cm(widths[j])
                cell.text = ""
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                r = p.add_run(str(value))
                r.font.size = Pt(9)
                if i == 0:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    shade(cell, FILL_HEAD)
                elif str(rows[i][0]) == "Total":
                    r.bold = True
                    shade(cell, FILL_TOTAL)
                elif i % 2 == 0:
                    shade(cell, FILL_BAND)
        self.caption(text)
        return t

    def page_break(self) -> None:
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def add_page_footer(doc, left_text: str) -> None:
    """Centre footer: '<left_text>    |    page <n>' with a live PAGE field."""
    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run(f"{left_text}    |    page ")
    run.font.size = Pt(8.5)
    run.font.color.rgb = GREY
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    rel = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "17")
    rPr.append(size)
    rel.append(rPr)
    t = OxmlElement("w:t")
    t.text = "1"
    rel.append(t)
    fld.append(rel)
    footer._p.append(fld)
