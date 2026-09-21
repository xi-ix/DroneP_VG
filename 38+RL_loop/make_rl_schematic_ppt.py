#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "38+RL_loop/RL_结构示意图.pptx"

COLORS = {
    "bg": RGBColor(248, 249, 251),
    "ink": RGBColor(31, 41, 55),
    "muted": RGBColor(91, 107, 124),
    "blue": RGBColor(219, 234, 254),
    "blue_dark": RGBColor(37, 99, 235),
    "green": RGBColor(220, 252, 231),
    "green_dark": RGBColor(22, 163, 74),
    "yellow": RGBColor(254, 243, 199),
    "yellow_dark": RGBColor(180, 83, 9),
    "purple": RGBColor(237, 233, 254),
    "purple_dark": RGBColor(109, 40, 217),
    "gray": RGBColor(229, 231, 235),
    "gray_dark": RGBColor(107, 114, 128),
    "red": RGBColor(254, 226, 226),
    "red_dark": RGBColor(220, 38, 38),
    "white": RGBColor(255, 255, 255),
}


def set_text(shape, text: str, size=12, bold=False, color="ink", align=PP_ALIGN.CENTER):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    for idx, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = "Arial"
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = COLORS[color]


def rect(slide, x, y, w, h, text="", fill="white", line="gray_dark", size=12, bold=False, color="ink", radius=True):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLORS[fill]
    shape.line.color.rgb = COLORS[line]
    shape.line.width = Pt(1.1)
    if text:
        set_text(shape, text, size=size, bold=bold, color=color)
    return shape


def label(slide, x, y, w, h, text, size=12, color="muted", bold=False, align=PP_ALIGN.CENTER):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    set_text(shape, text, size=size, bold=bold, color=color, align=align)
    return shape


def arrow(slide, x1, y1, x2, y2, color="gray_dark", width=1.8):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = COLORS[color]
    conn.line.width = Pt(width)
    conn.line.end_arrowhead = True
    return conn


def vector(slide, x, y, dim_label, prefix, n_cells, fill, line, w=0.16, h=0.28, gap=0.018):
    label(slide, x, y - 0.38, n_cells * (w + gap), 0.25, dim_label, size=13, color=line, bold=True)
    for i in range(n_cells):
        cell = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + i * (w + gap)), Inches(y), Inches(w), Inches(h))
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLORS[fill]
        cell.line.color.rgb = COLORS[line]
        cell.line.width = Pt(0.55)
        if i in {0, 1, n_cells - 2, n_cells - 1}:
            token = f"{prefix}{i + 1}" if i < 2 else ("..." if i == n_cells - 2 else f"{prefix}{38 if prefix == 's' else 20}")
            set_text(cell, token, size=5.6, bold=False, color="ink")
    return n_cells * w + max(0, n_cells - 1) * gap


def linear_stack(slide, x, y, text, fill, dark):
    # Layer shown as a compressed stack of thin rectangles.
    for i, scale in enumerate([1.0, 0.82, 0.64]):
        rect(slide, x + i * 0.10, y + i * 0.06, 0.78 * scale, 0.70 - i * 0.05, "", fill=fill, line=dark, radius=False)
    rect(slide, x - 0.06, y + 0.86, 1.08, 0.38, text, fill="white", line=dark, size=10.5, bold=True, color=dark)


def small_node(slide, x, y, text, fill="gray", line="gray_dark"):
    return rect(slide, x, y, 0.64, 0.46, text, fill=fill, line=line, size=11, bold=True, color="ink")


def score_bars(slide, x, y):
    heights = [0.42, 0.70, 0.55, 0.95, 0.48]
    labels = ["A0", "A1", "A2", "A3", "Ak"]
    for i, (h, t) in enumerate(zip(heights, labels)):
        fill = "red" if t == "A3" else "purple"
        line = "red_dark" if t == "A3" else "purple_dark"
        rect(slide, x + i * 0.32, y + (1.0 - h), 0.20, h, "", fill=fill, line=line, radius=False)
        label(slide, x + i * 0.25 - 0.02, y + 1.08, 0.35, 0.20, t, size=8.5, color="ink", bold=t == "A3")
    label(slide, x - 0.05, y - 0.33, 1.70, 0.25, "scores", size=12, color="purple_dark", bold=True)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = COLORS["bg"]

    label(slide, 0.45, 0.22, 12.4, 0.42, "RL State-Action Scorer：模块符号", size=23, color="ink", bold=True)
    label(slide, 0.45, 0.68, 12.4, 0.28, "每类结构只展示一个代表符号，具体层数与维度写在符号内部", size=11.5, color="muted")

    # This slide is intentionally a component sheet rather than a full data-flow chart.
    tile_w, tile_h = 3.86, 2.12
    tile_x = [0.55, 4.74, 8.93]
    tile_y = [1.18, 3.75]
    for y in tile_y:
        for x in tile_x:
            rect(slide, x, y, tile_w, tile_h, fill="white", line="gray", radius=True)

    # 1. Shared vector symbol for both state and action inputs.
    label(slide, 0.78, 1.40, 3.40, 0.25, "输入向量（state / action）", size=14, color="blue_dark", bold=True)
    vector(slide, 1.02, 1.94, "38-d state   |   20-d action", "x", 16, "blue", "blue_dark", w=0.17, h=0.36, gap=0.025)
    label(slide, 0.90, 2.55, 3.15, 0.42, "x = [x₁, x₂, …, xₙ]ᵀ", size=15, color="ink", bold=True)
    label(slide, 0.82, 2.94, 3.30, 0.20, "状态：38-d    动作：20-d", size=10.5, color="muted")

    # 2. One representative Linear symbol.
    label(slide, 4.98, 1.40, 3.35, 0.25, "Linear 层", size=14, color="green_dark", bold=True)
    for i, (w, h) in enumerate([(1.44, 0.86), (1.13, 0.68), (0.84, 0.51)]):
        rect(slide, 5.30 + i * 0.18, 1.88 + i * 0.15, w, h, fill="green", line="green_dark", radius=False)
    label(slide, 5.18, 2.84, 3.05, 0.31, "Linear(d_in → d_out)", size=14, color="ink", bold=True)
    label(slide, 5.15, 3.18, 3.10, 0.20, "权重 W + 偏置 b", size=10.5, color="muted")

    # 3. Normalization and activation as compact operation symbols.
    label(slide, 9.17, 1.40, 3.35, 0.25, "归一化 / 非线性", size=14, color="yellow_dark", bold=True)
    small_node(slide, 9.55, 2.03, "LN", fill="yellow", line="yellow_dark")
    label(slide, 9.37, 2.57, 1.02, 0.22, "LayerNorm", size=10, color="muted")
    small_node(slide, 11.00, 2.03, "ReLU", fill="gray", line="gray_dark")
    label(slide, 10.91, 2.57, 1.18, 0.22, "max(0,x)", size=10, color="muted")
    label(slide, 9.27, 2.94, 3.15, 0.20, "逐层变换中的两个操作", size=10.5, color="muted")

    # 4. Embedding symbol.
    label(slide, 0.78, 3.98, 3.40, 0.25, "Embedding 输出", size=14, color="blue_dark", bold=True)
    vector(slide, 1.02, 4.53, "64-d embedding", "e", 12, "blue", "blue_dark", w=0.19, h=0.38, gap=0.025)
    label(slide, 0.90, 5.15, 3.15, 0.42, "e = f(x) ∈ ℝ⁶⁴", size=15, color="ink", bold=True)
    label(slide, 0.82, 5.56, 3.30, 0.20, "state 与 action 各得到一个向量", size=10.5, color="muted")

    # 5. Fusion symbol: concatenation and element-wise interaction.
    label(slide, 4.98, 3.98, 3.35, 0.25, "融合", size=14, color="yellow_dark", bold=True)
    rect(slide, 5.18, 4.51, 0.72, 0.44, "eₛ", fill="blue", line="blue_dark", size=13, bold=True)
    rect(slide, 6.04, 4.51, 0.72, 0.44, "eₐ", fill="green", line="green_dark", size=13, bold=True)
    rect(slide, 6.90, 4.51, 0.72, 0.44, "⊙", fill="yellow", line="yellow_dark", size=17, bold=True)
    label(slide, 5.03, 5.11, 2.85, 0.36, "[eₛ ; eₐ ; eₛ ⊙ eₐ]", size=15, color="ink", bold=True)
    label(slide, 5.12, 5.56, 2.70, 0.20, "64 + 64 + 64 = 192-d", size=10.5, color="muted")

    # 6. Scorer symbol.
    label(slide, 9.17, 3.98, 3.35, 0.25, "Action Scorer", size=14, color="purple_dark", bold=True)
    for i, (w, h) in enumerate([(1.55, 0.80), (1.18, 0.59), (0.78, 0.40)]):
        rect(slide, 9.48 + i * 0.18, 4.47 + i * 0.14, w, h, fill="purple", line="purple_dark", radius=False)
    label(slide, 9.28, 5.28, 3.05, 0.31, "192 → 96 → 1", size=15, color="ink", bold=True)
    label(slide, 9.24, 5.65, 3.12, 0.20, "Q(s, a) / score(a)", size=10.5, color="muted")

    # Bottom selection symbol spans the width and keeps the visual vocabulary compact.
    label(slide, 0.78, 6.38, 1.38, 0.24, "动作选择", size=14, color="red_dark", bold=True)
    score_bars(slide, 2.18, 6.00)
    arrow(slide, 4.22, 6.57, 5.00, 6.57, color="red_dark", width=1.5)
    rect(slide, 5.13, 6.30, 1.18, 0.54, "argmax", fill="red", line="red_dark", size=14, bold=True)
    arrow(slide, 6.42, 6.57, 7.12, 6.57, color="red_dark", width=1.5)
    rect(slide, 7.25, 6.30, 1.18, 0.54, "a*", fill="red", line="red_dark", size=16, bold=True)
    label(slide, 8.70, 6.38, 3.50, 0.32, "在 [STOP, a₁, …, aₖ] 中选分数最高者", size=12, color="ink", bold=True, align=PP_ALIGN.LEFT)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
