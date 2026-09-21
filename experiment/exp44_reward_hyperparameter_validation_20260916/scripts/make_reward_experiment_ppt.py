#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path("/home/wangzhe/.codex/attachments/d90a7a0a-27a7-4d7f-815d-5a7a7f82b0ec/结题答辩.pptx")
OUT = ROOT / "PPT/结题答辩_Reward参数实验补充版.pptx"

STYLE = runpy.run_path(str(ROOT / "38+RL_loop/make_experiment_results_ppt.py"))
C = STYLE["C"]
W, H = STYLE["W"], STYLE["H"]
rect = STYLE["rect"]
text = STYLE["text"]
rich_text = STYLE["rich_text"]
section_label = STYLE["section_label"]
metric_chip = STYLE["metric_chip"]
table = STYLE["table"]
draw_horizontal_bars = STYLE["draw_horizontal_bars"]


def base_slide(prs: Presentation, number: int, footer: int, title: str, subtitle: str):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = C["bg"]
    rect(slide, 0, 0, W, 0.11, "navy", "navy", 0)
    text(slide, 0.52, 0.28, 0.55, 0.34, f"{number:02d}", 12, "blue", True)
    text(slide, 1.02, 0.22, 11.75, 0.52, title, 24, "ink", True)
    text(slide, 1.03, 0.72, 11.7, 0.30, subtitle, 10.5, "muted")
    text(slide, 12.46, 7.12, 0.38, 0.20, str(footer), 9, "muted", align=PP_ALIGN.RIGHT)
    return slide


def add_notes(slide, page_function: str, script: str, qa: str) -> None:
    slide.notes_slide.notes_text_frame.text = (
        f"【页面功能】\n{page_function}\n\n"
        f"【主讲稿】\n{script}\n\n"
        f"【可能追问】\n{qa}"
    )


def conclusion_band(slide, y: float, main_runs, detail: str, h: float = 1.05) -> None:
    rect(slide, 0.55, y, 12.23, h, "paper", "line")
    section_label(slide, 0.82, y + 0.14, 1.35, "结论")
    rich_text(slide, 2.05, y + 0.09, 10.35, 0.42, main_runs, 13.0)
    text(slide, 2.05, y + 0.51, 10.35, 0.30, detail, 11.2, "muted")


def draw_search_flow(slide, x: float, y: float, w: float, h: float) -> None:
    rect(slide, x, y, w, h, "paper", "line")
    text(slide, x + 0.25, y + 0.14, w - 0.5, 0.32, "只用 validation 选参，test 最后一次确认", 12, "ink", True)
    stages = [
        ("01", "分组五折", "150 张 val\n同序列不跨折", "blue"),
        ("02", "宽范围粗筛", "136 组 · seed 42\n180 epochs", "teal"),
        ("03", "稳定性复核", "12 组 · 3 seeds\n300 epochs", "amber"),
        ("04", "冻结后测试", "Current + Challenger\ntest 不再调参", "blue"),
    ]
    gap = 0.18
    box_w = (w - 0.52 - gap * 3) / 4
    for idx, (num, title, detail, color) in enumerate(stages):
        bx = x + 0.26 + idx * (box_w + gap)
        rect(slide, bx, y + 0.72, box_w, 1.42, "gray_light" if idx == 0 else "paper", "line")
        text(slide, bx + 0.10, y + 0.82, 0.40, 0.24, num, 10, color, True)
        text(slide, bx + 0.10, y + 1.08, box_w - 0.20, 0.28, title, 11, "ink", True)
        text(slide, bx + 0.10, y + 1.39, box_w - 0.20, 0.52, detail, 9.2, "muted")
        if idx < len(stages) - 1:
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.CHEVRON,
                Inches(bx + box_w + 0.035), Inches(y + 1.22), Inches(0.11), Inches(0.31),
            )
            arrow.fill.solid(); arrow.fill.fore_color.rgb = C["line"]
            arrow.line.color.rgb = C["line"]
    text(slide, x + 0.30, y + 2.38, w - 0.60, 0.52,
         "筛选规则：动作数与候选框 ≤ Current 的 105%；优先 Acc@0.5，差异 < 0.002 时比较面积小目标 Recall、mAP 和动作数。",
         10.2, "muted")


def draw_weight_sensitivity(slide, x: float, y: float, w: float, h: float) -> None:
    rect(slide, x, y, w, h, "paper", "line")
    text(slide, x + 0.25, y + 0.14, w - 0.5, 0.32, "权重变化：性能平台与动作成本", 12, "ink", True)
    weights = [0.0000, 0.2392, 0.3268, 0.4000, 0.4621, 0.5155, 0.6025]
    acc = [0.5057, 0.5061, 0.5058, 0.5051, 0.5065, 0.5066, 0.5063]
    actions = [1.220, 1.153, 1.053, 1.020, 1.107, 1.127, 1.080]
    px, py, pw, ph = x + 0.50, y + 0.72, w - 0.82, h - 1.22
    for i in range(4):
        yy = py + ph * i / 3
        line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(px), Inches(yy), Inches(px + pw), Inches(yy))
        line.line.color.rgb = C["line"]
        line.line.width = Pt(0.8)
    series = [(acc, 0.5045, 0.5070, "blue"), (actions, 0.98, 1.24, "amber")]
    for values, lo, hi, color in series:
        points = []
        for idx, value in enumerate(values):
            xx = px + pw * idx / (len(values) - 1)
            yy = py + ph * (1 - (value - lo) / (hi - lo))
            points.append((xx, yy))
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
            line.line.color.rgb = C[color]
            line.line.width = Pt(2.1)
        for idx, (xx, yy) in enumerate(points):
            dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(xx - 0.045), Inches(yy - 0.045), Inches(0.09), Inches(0.09))
            dot.fill.solid(); dot.fill.fore_color.rgb = C[color]; dot.line.color.rgb = C[color]
            if idx == 3:
                ring = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(xx - 0.085), Inches(yy - 0.085), Inches(0.17), Inches(0.17))
                ring.fill.background(); ring.line.color.rgb = C["ink"]; ring.line.width = Pt(1.4)
    for idx, value in enumerate(weights):
        xx = px + pw * idx / (len(weights) - 1)
        text(slide, xx - 0.25, py + ph + 0.10, 0.50, 0.24, f"{value:.2f}", 8.4, "muted", align=PP_ALIGN.CENTER)
    rich_text(slide, x + w - 2.38, y + 0.16, 2.05, 0.24,
              [("● ", "blue", True), ("Acc", "muted", False), ("    ● ", "amber", True), ("动作/图", "muted", False)], 9.2, PP_ALIGN.CENTER)
    text(slide, px + pw * 3 / 6 - 0.42, py - 0.27, 0.84, 0.22, "Current 0.40", 8.8, "ink", True, PP_ALIGN.CENTER)


def draw_bootstrap_ci(slide, x: float, y: float, w: float, h: float) -> None:
    rect(slide, x, y, w, h, "paper", "line")
    text(slide, x + 0.25, y + 0.14, w - 0.5, 0.32, "配对 Bootstrap：Challenger − Current", 12, "ink", True)
    rows = [
        ("Acc@0.5", -0.00482, 0.00000, -0.00227, "amber"),
        ("Acc@0.75", -0.00160, 0.00053, -0.00050, "gray"),
        ("弱类 Recall", -0.01042, 0.00000, -0.00495, "teal"),
        ("面积小目标", -0.00719, -0.00078, -0.00364, "blue"),
    ]
    lo_all, hi_all = -0.011, 0.002
    px, pw = x + 1.58, w - 1.95
    zero_x = px + pw * (0 - lo_all) / (hi_all - lo_all)
    zero = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(zero_x), Inches(y + 0.68), Inches(zero_x), Inches(y + h - 0.47))
    zero.line.color.rgb = C["coral"]; zero.line.width = Pt(1.2); zero.line.dash_style = 2
    text(slide, zero_x - 0.20, y + 0.45, 0.40, 0.20, "0", 8.5, "coral", True, PP_ALIGN.CENTER)
    for idx, (label, low, high, center, color) in enumerate(rows):
        yy = y + 0.94 + idx * 0.55
        text(slide, x + 0.16, yy - 0.16, 1.28, 0.31, label, 9.3, "muted", align=PP_ALIGN.RIGHT)
        low_x = px + pw * (low - lo_all) / (hi_all - lo_all)
        high_x = px + pw * (high - lo_all) / (hi_all - lo_all)
        center_x = px + pw * (center - lo_all) / (hi_all - lo_all)
        line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(low_x), Inches(yy), Inches(high_x), Inches(yy))
        line.line.color.rgb = C[color]; line.line.width = Pt(3.0)
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(center_x - 0.055), Inches(yy - 0.055), Inches(0.11), Inches(0.11))
        dot.fill.solid(); dot.fill.fore_color.rgb = C[color]; dot.line.color.rgb = C[color]
        text(slide, center_x - 0.75, yy + 0.10, 1.50, 0.20,
             f"[{low:+.4f}, {high:+.4f}]", 7.3, color, True, PP_ALIGN.CENTER)
    text(slide, x + 0.25, y + h - 0.27, w - 0.50, 0.20, "负值位于左侧 = Current 更好；区间不跨 0 = 差异显著", 8.7, "muted")


def slide_formula(prs: Presentation):
    slide = base_slide(prs, 6, 4, "Reward 公式与系数设计", "正奖励衡量检测收益 · 成本项抑制无效扩展 · 系数表达相对优先级")
    metric_chip(slide, 0.55, 1.18, 2.35, "主召回项", "0.40 · Recall@0.5", "blue", 15)
    metric_chip(slide, 3.03, 1.18, 2.35, "弱类强化项", "0.45 · 最高权重", "teal", 15)
    metric_chip(slide, 5.51, 1.18, 2.35, "扩展成本", "0.010 / 0.012", "amber", 15)
    section_label(slide, 0.55, 2.08, 7.25, "公式与各项作用")
    rect(slide, 0.55, 2.48, 7.30, 0.82, "paper", "line")
    text(slide, 0.77, 2.60, 6.86, 0.56,
         "R = 0.40 ΔR@0.5 + 0.22 ΔR@0.75 + 0.45 ΔRweak + 0.15 ΔP\n"
         "     − 0.010 Cbox − 0.012 Naction − 0.012 Istop",
         15.0, "ink", True, PP_ALIGN.CENTER)
    table(slide, 0.55, 3.48, [1.42, 0.82, 2.18, 2.88], 0.34,
          ["分量", "系数", "优化目标", "设计原因"],
          [["ΔR@0.5", "+0.40", "基础目标命中", "召回优先，但不独占奖励"],
           ["ΔR@0.75", "+0.22", "高质量定位", "避免只追求低质量命中"],
           ["ΔRweak", "+0.45", "弱类别召回", "航拍小目标/弱类是研究重点"],
           ["ΔPrecision", "+0.15", "控制误检", "扩展候选时保持排序质量"],
           ["Cbox", "−0.010", "候选规模", "抑制候选框无效膨胀"],
           ["Naction", "−0.012", "动作预算", "约束推理成本"],
           ["Istop", "−0.012", "过早停止", "保留有效反馈机会"]],
          {2: "teal_light"}, 8.7)
    draw_horizontal_bars(slide, 8.09, 2.08, 4.69, 3.46,
                         ["Recall@0.5", "Recall@0.75", "Weak recall", "Precision"],
                         [0.40, 0.22, 0.45, 0.15],
                         ["blue", "gray", "teal", "amber"],
                         "正奖励相对权重 · 总和 1.22", "{:.2f}")
    rect(slide, 8.09, 5.70, 4.69, 0.72, "paper", "line")
    rich_text(slide, 8.32, 5.84, 4.20, 0.40,
              [("权重不是概率", "blue", True), ("，不要求和为 1；\n", "ink", False),
               ("比较时固定总尺度", "teal", True), ("，只检验相对比例。", "ink", False)], 10.5)
    text(slide, 0.58, 6.76, 11.8, 0.22, "注：Current 的 small 项按弱类别集合 {1,2,3,7,8,10} 统计；实验另测面积 small 与 hybrid 定义。", 9.0, "muted")
    add_notes(
        slide,
        "解释 reward 为什么由这些检测收益与成本项组成，先给出设计逻辑，再为后续系数实验建立口径。",
        "这一页先回答公式层面的合理性。Recall@0.5 是主要命中目标，Recall@0.75 保证定位质量，弱类召回权重最高是因为航拍任务的核心困难是小目标和弱类别。Precision 用来约束误检，三个成本项则防止策略通过不断增加候选和动作获得表面收益。四个正权重之和是 1.22，它们不是概率；真正重要的是相对比例以及它们相对于成本项的尺度。",
        "如果老师问 small 为什么不是面积定义：当前公式采用弱类别先验，但 Exp44 同时测试了面积 small 和二者混合，二者没有稳定超过 Current。",
    )
    return slide


def slide_protocol(prs: Presentation):
    slide = base_slide(prs, 7, 5, "Reward 超参数验证协议", "组件消融 + 宽范围敏感性 + 二因素交互 + 全局权重采样 · test 不参与选参")
    metric_chip(slide, 0.55, 1.18, 2.35, "宽范围粗筛", "136 组配置", "blue")
    metric_chip(slide, 3.03, 1.18, 2.35, "满足成本约束", "44 组可行", "teal")
    metric_chip(slide, 5.51, 1.18, 2.35, "完整稳定性复核", "12 组 · 3 seeds", "amber", 15)
    draw_search_flow(slide, 0.55, 2.08, 7.28, 3.34)
    section_label(slide, 8.09, 2.08, 4.69, "搜索空间覆盖")
    table(slide, 8.09, 2.48, [2.72, 1.97], 0.49,
          ["实验类型", "配置数"],
          [["Current + 消融 + small 定义", "13"],
           ["7 类单因素趋势", "35"],
           ["3 组二因素响应面", "48"],
           ["四维单纯形全局采样", "40"],
           ["合计", "136"]],
          {4: "blue_light"}, 9.5)
    conclusion_band(slide, 5.66,
                    [("选参证据来自 validation", "blue", True), ("；test 只在配置冻结后用于一次独立泛化确认。", "ink", False)],
                    "避免用 test 反复试权重；所有入围配置均重新训练策略，而不是只换评分公式。")
    text(slide, 0.58, 6.86, 11.8, 0.18, "成本约束：平均动作数与预测框数不超过 Current 的 105%。", 8.8, "muted")
    add_notes(
        slide,
        "证明参数不是凭经验挑选后直接报告，而是经过无 test 泄漏的系统搜索和多种子复核。",
        "这一页讲实验可信度。我们先把 150 张 validation 按序列分成五折，避免相邻帧跨折。粗筛覆盖 136 组，包括逐项消融、七类单因素曲线、三个交互面和四十个全局样本。满足动作数和候选框不超过 Current 105% 的配置才算可行。随后选 12 组做五折三种子、300 epoch 复核。最后冻结 Current 和最强 Challenger，test 只评测一次，不再根据 test 改参数。",
        "如果老师问为什么不是全网格：四维权重加三个成本项的完整笛卡尔积会产生大量冗余组合，因此采用单因素、关键交互面和空间填充采样结合，既看趋势也覆盖全局。",
    )
    return slide


def slide_recall_weight(prs: Presentation):
    slide = base_slide(prs, 8, 6, "为什么 Recall@0.5 的系数是 0.40", "保持正奖励总和 1.22 · 扫描 0 至 0.60 · 同时比较检测收益与动作成本")
    metric_chip(slide, 0.55, 1.18, 2.35, "正奖励占比", "0.40 / 1.22 = 32.8%", "blue", 14.5)
    metric_chip(slide, 3.03, 1.18, 2.35, "扫描范围", "7 点 · 0 → 0.60", "teal", 15)
    metric_chip(slide, 5.51, 1.18, 2.35, "确认阶段动作数", "1.447 < 1.513", "amber", 15)
    section_label(slide, 0.55, 2.13, 6.75, "Recall@0.5 单因素宽范围扫描")
    table(slide, 0.55, 2.55, [1.10, 1.20, 1.32, 1.12, 1.48], 0.37,
          ["w_R@0.5", "Acc@0.5", "Small R", "动作/图", "成本约束"],
          [["0.0000", "0.5057", "0.3841", "1.220", "不满足"],
           ["0.2392", "0.5061", "0.3846", "1.153", "不满足"],
           ["0.3268", "0.5058", "0.3841", "1.053", "满足"],
           ["0.4000", "0.5051", "0.3839", "1.020", "满足"],
           ["0.4621", "0.5065", "0.3848", "1.107", "不满足"],
           ["0.5155", "0.5066", "0.3848", "1.127", "不满足"],
           ["0.6025", "0.5063", "0.3848", "1.080", "不满足"]],
          {3: "blue_light"}, 8.8)
    draw_weight_sensitivity(slide, 7.66, 2.13, 5.12, 3.37)
    conclusion_band(slide, 5.76,
                    [("0.40 位于稳定平台", "blue", True), ("：三种子确认中 Acc 与 0.3268 几乎相同，但小目标召回、mAP 和动作效率更好。", "ink", False)],
                    "结论是“经验稳定工作点”，不是声称 0.40 在任意小数精度上唯一最优。")
    text(slide, 0.58, 6.91, 11.8, 0.16, "确认：Current vs w=0.3268 → Acc 0.5095 vs 0.5095；Small R 0.3884 vs 0.3880；OOF mAP 0.1295 vs 0.1293。", 8.8, "muted")
    add_notes(
        slide,
        "直接回答老师最可能提出的问题：为什么第一个系数恰好写成 0.40。",
        "0.40 不是理论解析得到的唯一常数，而是正奖励总权重中的 32.8%。我们保持总尺度 1.22 不变，把该权重从 0 扫描到约 0.60。较大的权重在单次粗筛中偶尔提高约千分之一的 Acc，但动作数超过 105% 成本约束。满足预算且最接近的竞争点是 0.3268；它进入五折三种子复核后，Acc 与 Current 几乎相同，但 Current 的小目标召回、OOF mAP 更高，动作数也更少。因此保留 0.40。",
        "如果老师追问为什么不是 0.39 或 0.41：实验支持的是 0.40 所在的稳定区间，不支持小数点后任意精度的唯一最优性；取 0.40 是为了使用简单、可解释的代表值。",
    )
    return slide


def slide_final(prs: Presentation):
    slide = base_slide(prs, 9, 7, "独立 Test 验证与最终选择", "验证集冻结配置后评测 · 3 seeds · 逐图配对 Bootstrap 10,000 次")
    metric_chip(slide, 0.55, 1.18, 2.35, "Current · Acc@0.5", "0.5297 ± 0.0003", "blue", 15)
    metric_chip(slide, 3.03, 1.18, 2.35, "面积小目标 Recall", "+0.0030", "teal")
    metric_chip(slide, 5.51, 1.18, 2.35, "最终选择", "保留 Current", "amber", 15.5)
    section_label(slide, 0.55, 2.13, 6.65, "Current 与最强 Challenger")
    table(slide, 0.55, 2.55, [1.62, 1.02, 1.02, 1.02, 1.25, 1.02], 0.63,
          ["配置", "Acc@0.5", "Acc@0.75", "mAP@0.5", "面积Small R", "动作/图"],
          [["Current", "0.5297", "0.3307", "0.1398", "0.4168", "1.362"],
           ["Challenger", "0.5279", "0.3301", "0.1396", "0.4138", "1.358"]],
          {0: "blue_light"}, 8.7)
    rect(slide, 0.55, 4.23, 6.95, 1.17, "paper", "line")
    section_label(slide, 0.80, 4.38, 2.15, "对照说明")
    text(slide, 2.17, 4.32, 5.06, 0.34, "Challenger：提高 small 与 precision 相对权重", 10.6, "ink", True)
    text(slide, 2.17, 4.72, 5.06, 0.45, "两者网络、候选源、推理轮数和重评分流程完全相同；仅 reward 权重不同。", 9.8, "muted")
    draw_bootstrap_ci(slide, 7.72, 2.13, 5.06, 3.27)
    conclusion_band(slide, 5.64,
                    [("最终保留 Current", "blue", True), ("：三种子下 Acc、mAP 和小目标召回均领先；面积小目标 Recall 的 95% CI 不跨 0。", "ink", False)],
                    "Acc 区间上界触及 0，因此表述为稳定领先，不夸大为所有指标均显著。")
    text(slide, 0.58, 6.82, 11.8, 0.22, "结论边界：证明预注册搜索空间内的经验稳健性与局部优势，不声称数学上的全局最优。", 9.0, "muted")
    add_notes(
        slide,
        "用完全独立的 test 结果收束参数实验，说明最终为什么没有替换当前公式，并给出统计边界。",
        "验证集按照预注册规则保留 Current，同时冻结最强的非当前配置作为 Challenger。独立 test 上，Current 的三种子平均 Acc@0.5、mAP 和面积小目标召回都更高，动作数基本相同。逐图 Bootstrap 中，面积小目标 Recall 的差异区间完全位于零左侧，说明 Current 在这一核心指标上具有显著优势。Acc 的区间上界触及零，所以这里只称稳定领先，不称所有指标都显著。最终保留原 reward。",
        "如果老师问是否用 test 反向选参：没有。Current 和 Challenger 在 test 前已经冻结，test 后没有继续搜索；本页只报告泛化验证。",
    )
    return slide


def move_new_slides_before(prs: Presentation, insertion_index: int, count: int) -> None:
    slide_ids = prs.slides._sldIdLst
    new_ids = list(slide_ids)[-count:]
    for slide_id in new_ids:
        slide_ids.remove(slide_id)
    for offset, slide_id in enumerate(new_ids):
        slide_ids.insert(insertion_index + offset, slide_id)


def build() -> Path:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    prs = Presentation(SOURCE)
    original_count = len(prs.slides)
    slide_formula(prs)
    slide_protocol(prs)
    slide_recall_weight(prs)
    slide_final(prs)
    # Insert after the five experiment pages (original slides 9-13), before Conclusion.
    move_new_slides_before(prs, insertion_index=13, count=4)
    prs.core_properties.title = "结题答辩：Reward 参数实验补充版"
    prs.core_properties.subject = "增加 Reward 公式设计、超参数搜索与独立验证"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    check = Presentation(OUT)
    assert original_count == 15 and len(check.slides) == 19
    assert "Reward 公式与系数设计" in " ".join(shape.text for shape in check.slides[13].shapes if hasattr(shape, "text"))
    assert "【页面功能】" in check.slides[13].notes_slide.notes_text_frame.text
    return OUT


if __name__ == "__main__":
    print(build())
