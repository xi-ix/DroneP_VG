#!/usr/bin/env python3
from __future__ import annotations

import json
import runpy
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STYLE = runpy.run_path(str(ROOT / "38+RL_loop/make_experiment_results_ppt.py"))
DATA = json.loads((HERE / "log_performance_cost/summary.json").read_text(encoding="utf-8"))
OUT = ROOT / "PPT/Qwen动态RL性能成本对比.pptx"

W, H = STYLE["W"], STYLE["H"]
base_slide = STYLE["base_slide"]
metric_chip = STYLE["metric_chip"]
section_label = STYLE["section_label"]
table = STYLE["table"]
rect = STYLE["rect"]
text = STYLE["text"]
rich_text = STYLE["rich_text"]
draw_horizontal_bars = STYLE["draw_horizontal_bars"]


def build() -> None:
    results = DATA["results"]
    derived = DATA["derived"]
    query = results["query_only"]
    full = results["all_qwen_mapped_actions"]
    rl = results["qwen_dynamic_rl"]

    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    slide = base_slide(
        prs,
        1,
        "Qwen 动态 RL：以更低成本逼近全展开性能",
        "VisDrone test：150 张 / 7,973 GT · 同一候选融合、富元信息重评分与 class-aware 评测流程",
    )

    metric_chip(
        slide, 0.55, 1.18, 2.25,
        "RL 相对原始 Query", f"Acc +{derived['rl_acc_gain_over_query_pp']:.2f} pp", "blue", 15,
    )
    metric_chip(
        slide, 2.93, 1.18, 2.25,
        "保留全展开增益", f"{derived['rl_retained_acc_gain_pct']:.1f}%", "teal", 17,
    )
    metric_chip(
        slide, 5.31, 1.18, 2.25,
        "相对全展开动作", f"-{derived['rl_action_reduction_vs_full_pct']:.1f}%", "amber", 17,
    )

    draw_horizontal_bars(
        slide, 7.79, 1.18, 4.99, 3.63,
        ["原始 Query", "动态 RL", "全 Qwen 动作"],
        [query["acc_05"], rl["acc_05"], full["acc_05"]],
        ["gray", "blue", "teal"],
        "Acc@0.5：性能逐级提升",
        "{:.4f}",
    )

    section_label(slide, 0.55, 2.13, 6.95, "三组策略的性能—成本对照")
    table(
        slide, 0.55, 2.55,
        [1.62, 1.10, 1.03, 1.03, 1.03, 1.20], 0.54,
        ["策略", "预测框", "Acc@0.5", "Acc@0.75", "mAP@0.5", "动作/图"],
        [
            ["原始 Query", f"{query['prediction_boxes']:,.0f}", f"{query['acc_05']:.4f}", f"{query['acc_075']:.4f}", f"{query['map_05']:.4f}", f"{query['actions_per_image']:.3f}"],
            ["全 Qwen 动作", f"{full['prediction_boxes']:,.0f}", f"{full['acc_05']:.4f}", f"{full['acc_075']:.4f}", f"{full['map_05']:.4f}", f"{full['actions_per_image']:.3f}"],
            ["Qwen 动态 RL", f"{rl['prediction_boxes']:,.0f}", f"{rl['acc_05']:.4f}", f"{rl['acc_075']:.4f}", f"{rl['map_05']:.4f}", f"{rl['actions_per_image']:.3f}"],
        ],
        {2: "blue_light"}, 9.3,
    )

    rect(slide, 0.55, 4.98, 12.23, 1.42, "paper", "line")
    section_label(slide, 0.82, 5.14, 2.10, "核心结论")
    rich_text(
        slide, 2.05, 5.06, 10.35, 0.48,
        [
            ("动态 RL", "blue", True),
            (" 将 Acc@0.5 提升至 ", "ink", False),
            (f"{rl['acc_05'] * 100:.2f}%", "blue", True),
            (f"，与全展开仅差 {(full['acc_05'] - rl['acc_05']) * 100:.2f} pp；", "ink", False),
            ("但动作开销降低 ", "ink", False),
            (f"{derived['rl_action_reduction_vs_full_pct']:.1f}%", "amber", True),
            ("。", "ink", False),
        ], 12.8,
    )
    rich_text(
        slide, 2.05, 5.54, 10.35, 0.48,
        [
            ("选择优势：", "teal", True),
            (f"仅使用全展开 {rl['actions_per_image'] / full['actions_per_image'] * 100:.1f}% 的动作预算，", "ink", False),
            (f"保留 {derived['rl_retained_acc_gain_pct']:.1f}% 的 Acc 增益", "teal", True),
            (f"，并减少 {derived['rl_box_reduction_vs_full_pct']:.1f}% 最终候选框。", "ink", False),
        ], 11.7,
    )
    text(
        slide, 0.58, 6.73, 12.0, 0.27,
        "注：全 Qwen 动作 = 执行每张图缓存 Qwen 输出中的全部唯一映射动作；动作数不包含 STOP。",
        9.0, "muted",
    )

    prs.core_properties.title = "Qwen 动态 RL 性能成本对比"
    prs.core_properties.subject = "原始 Query、全 Qwen 映射动作与 Qwen 动态 RL 对照"
    prs.core_properties.author = "DroneP_VG"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
