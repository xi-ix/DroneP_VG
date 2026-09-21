#!/usr/bin/env python3
from __future__ import annotations

import json
import runpy
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches


ROOT = Path(__file__).resolve().parents[2]
STYLE = runpy.run_path(str(ROOT / "38+RL_loop/make_experiment_results_ppt.py"))
DATA = json.loads((Path(__file__).resolve().parent / "log/summary.json").read_text(encoding="utf-8"))
OUT = ROOT / "PPT/Qwen动态RL等预算对照实验.pptx"

W, H = STYLE["W"], STYLE["H"]
base_slide = STYLE["base_slide"]
metric_chip = STYLE["metric_chip"]
section_label = STYLE["section_label"]
table = STYLE["table"]
rect = STYLE["rect"]
text = STYLE["text"]
rich_text = STYLE["rich_text"]
draw_horizontal_bars = STYLE["draw_horizontal_bars"]


def build():
    r = DATA["results"]
    dyn = r["qwen_dynamic_rl"]
    rank = r["prompt_rank_equal_budget"]
    rnd = r["random_equal_budget"]

    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    slide = base_slide(
        prs,
        1,
        "Qwen 动态 RL：同动作池、同预算公平对照",
        "VisDrone test：150 张 / 7,973 GT · 每图使用相同 Qwen 映射动作池 · 非 STOP 预算严格一致",
    )

    metric_chip(slide, 0.55, 1.18, 2.25, "统一动作预算", "0.953 次/图", "amber", 15)
    metric_chip(slide, 2.93, 1.18, 2.25, "动态 RL vs 随机", "Acc +0.65 pp", "blue", 15)
    metric_chip(slide, 5.31, 1.18, 2.25, "小目标 Recall", "+0.99 pp", "teal", 15)

    draw_horizontal_bars(
        slide, 7.79, 1.18, 4.99, 3.63,
        ["随机选择", "Qwen 动态 RL", "Prompt-rank"],
        [rnd["acc_05"], dyn["acc_05"], rank["acc_05"]],
        ["gray", "blue", "teal"], "Acc@0.5（相同逐图预算）"
    )

    section_label(slide, 0.55, 2.13, 6.95, "等预算测试结果")
    table(
        slide, 0.55, 2.55,
        [1.86, 1.10, 1.04, 1.04, 1.04, 1.28], 0.54,
        ["方法", "预测框", "Acc@0.5", "Acc@0.75", "mAP@0.5", "小目标Recall"],
        [
            ["随机选择（20 seeds）", f"{rnd['prediction_boxes']:.0f}±{rnd['prediction_boxes_std']:.0f}", f"{rnd['acc_05']:.4f}", f"{rnd['acc_075']:.4f}", f"{rnd['map_05']:.4f}", f"{rnd['small_final_recall_05']:.4f}"],
            ["Qwen 动态 RL", f"{dyn['prediction_boxes']:.0f}", f"{dyn['acc_05']:.4f}", f"{dyn['acc_075']:.4f}", f"{dyn['map_05']:.4f}", f"{dyn['small_final_recall_05']:.4f}"],
            ["Prompt-rank", f"{rank['prediction_boxes']:.0f}", f"{rank['acc_05']:.4f}", f"{rank['acc_075']:.4f}", f"{rank['map_05']:.4f}", f"{rank['small_final_recall_05']:.4f}"],
        ],
        {1: "blue_light", 2: "teal_light"}, 9.0,
    )

    rect(slide, 0.55, 4.98, 12.23, 1.42, "paper", "line")
    section_label(slide, 0.82, 5.14, 2.10, "实验结论")
    rich_text(
        slide, 2.05, 5.07, 10.35, 0.44,
        [
            ("已证明：", "ink", True),
            ("动态 RL 在相同动作池和逐图预算下优于随机选择，说明状态反馈带来有效动作选择。", "blue", True),
        ], 12.8,
    )
    rich_text(
        slide, 2.05, 5.53, 10.35, 0.52,
        [
            ("尚未证明：", "ink", True),
            ("Prompt-rank 的 Acc 与 mAP 仍略高（+0.51 pp / +0.0003），当前不能声称动态 RL 整体最优。", "teal", False),
        ], 11.5,
    )
    text(
        slide, 0.58, 6.73, 11.9, 0.25,
        "协议：相同初始候选、候选合并去重、富元信息重评分与 class-aware 评测；随机结果为 seeds 42–61。",
        9.0, "muted",
    )

    prs.core_properties.title = "Qwen 动态 RL 等预算对照实验"
    prs.core_properties.subject = "动态 RL、Prompt-rank 与随机策略公平比较"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
