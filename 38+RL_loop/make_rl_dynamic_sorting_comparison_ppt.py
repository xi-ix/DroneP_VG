#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches


ROOT = Path(__file__).resolve().parents[1]
STYLE = runpy.run_path(str(ROOT / "38+RL_loop/make_experiment_results_ppt.py"))
OUT = ROOT / "PPT/RL动态选择与排序优化对比_统一风格.pptx"

C = STYLE["C"]
W = STYLE["W"]
H = STYLE["H"]
base_slide = STYLE["base_slide"]
metric_chip = STYLE["metric_chip"]
section_label = STYLE["section_label"]
table = STYLE["table"]
rect = STYLE["rect"]
text = STYLE["text"]
rich_text = STYLE["rich_text"]
draw_horizontal_bars = STYLE["draw_horizontal_bars"]


def slide_rl_comparison(prs):
    slide = base_slide(
        prs,
        1,
        "RL 动态选择与固定策略对比",
        "同一评测协议 · Query base、固定动作三轮 RL 与 Qwen 动态动作三轮对比",
    )

    metric_chip(slide, 0.55, 1.18, 2.35, "固定 RL · Acc@0.5", "0.5293", "blue")
    metric_chip(slide, 3.03, 1.18, 2.35, "动态 RL · 候选框", "35,869 · −12.3%", "teal", 14.5)
    metric_chip(slide, 5.51, 1.18, 2.35, "动态 RL · mAP@0.5", "0.1424 · 最高", "amber", 15)

    section_label(slide, 0.55, 2.13, 7.30, "统一指标对比")
    table(
        slide,
        0.55,
        2.55,
        [1.82, 1.25, 1.17, 1.17, 1.17],
        0.55,
        ["方法", "预测框数", "Acc@0.5", "Acc@0.75", "mAP@0.5"],
        [
            ["Query base", "22,298", "0.4651", "0.3131", "0.1343"],
            ["固定动作三轮 RL", "40,904", "0.5293", "0.3310", "0.1397"],
            ["Qwen 动态动作三轮", "35,869", "0.5004", "0.3248", "0.1424"],
        ],
        {1: "blue_light", 2: "teal_light"},
        9.7,
    )

    draw_horizontal_bars(
        slide,
        8.09,
        1.18,
        4.69,
        3.56,
        ["Query base", "固定动作 RL", "Qwen 动态 RL"],
        [0.4651, 0.5293, 0.5004],
        ["gray", "blue", "teal"],
        "Acc@0.5",
    )

    rect(slide, 0.55, 4.98, 12.23, 1.42, "paper", "line")
    section_label(slide, 0.82, 5.15, 2.00, "结论")
    rich_text(
        slide,
        2.05,
        5.08,
        10.25,
        0.43,
        [
            ("固定动作 RL", "blue", True),
            (" 召回更强；", "ink", False),
            ("Qwen 动态 RL", "teal", True),
            (" 候选减少约 12.3%，并取得最高 mAP。", "ink", False),
        ],
        13.3,
    )
    text(
        slide,
        2.05,
        5.55,
        10.25,
        0.50,
        "动态 prompt pool 使 RL 从“固定类别 ID 选择”扩展为“状态—动作匹配”，但当前不存在所有指标均占优的单一策略。",
        11.3,
        "muted",
    )
    text(slide, 0.58, 6.76, 11.8, 0.22, "数据来自原始两页 PPT，数值与口径保持不变。", 9.2, "muted")


def slide_sorting(prs):
    slide = base_slide(
        prs,
        2,
        "固定候选集上的排序优化",
        "三种方法读取同一份 52,872 个 metadata 候选框 · 规则组仅筛选 · 排序器仅重标定分数",
    )

    metric_chip(slide, 0.55, 1.18, 2.35, "排序器 · mAP@0.5", "0.1690", "blue")
    metric_chip(slide, 3.03, 1.18, 2.35, "相对融合前提升", "+0.0226", "teal")
    metric_chip(slide, 5.51, 1.18, 2.35, "候选数 / Acc@0.5", "保持不变", "amber", 15)

    section_label(slide, 0.55, 2.13, 7.30, "固定候选集控制变量")
    table(
        slide,
        0.55,
        2.55,
        [1.82, 1.25, 1.17, 1.17, 1.17],
        0.55,
        ["方法", "预测框数", "Acc@0.5", "Acc@0.75", "mAP@0.5"],
        [
            ["富元信息融合前", "52,872", "0.5374", "0.3463", "0.1464"],
            ["规则筛选对照", "9,586", "0.3193", "0.2491", "0.1187"],
            ["排序器校准", "52,872", "0.5374", "0.3491", "0.1690"],
        ],
        {2: "blue_light"},
        9.7,
    )

    draw_horizontal_bars(
        slide,
        8.09,
        1.18,
        4.69,
        3.56,
        ["融合前", "规则筛选", "排序器校准"],
        [0.1464, 0.1187, 0.1690],
        ["gray", "amber", "blue"],
        "mAP@0.5",
    )

    rect(slide, 0.55, 4.98, 12.23, 1.42, "paper", "line")
    section_label(slide, 0.82, 5.15, 2.00, "结论")
    rich_text(
        slide,
        2.05,
        5.08,
        10.25,
        0.43,
        [
            ("候选框数量和 Acc@0.5 不变", "ink", True),
            ("，mAP@0.5 从 ", "ink", False),
            ("0.1464", "muted", True),
            (" 提升至 ", "ink", False),
            ("0.1690", "blue", True),
            ("。", "ink", False),
        ],
        13.3,
    )
    text(
        slide,
        2.05,
        5.55,
        10.25,
        0.50,
        "提升来自候选置信度校准与排序，而不是增加候选框；规则硬筛选会同时损失候选覆盖与排序质量。",
        11.3,
        "muted",
    )
    text(slide, 0.58, 6.76, 11.8, 0.22, "Exp38 将候选来源、query 匹配和小目标先验纳入最终排序。", 9.2, "muted")


def build():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "RL 动态选择与排序优化对比"
    prs.core_properties.subject = "统一视觉风格版"
    prs.core_properties.author = "DroneP_VG"
    slide_rl_comparison(prs)
    slide_sorting(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
