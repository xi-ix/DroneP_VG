#!/usr/bin/env python3
"""Generate real-image before/after examples for the Exp38 final reranker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EXP38 = ROOT / "experiment/exp38_online_rich_metadata_fusion_20260712/log"
IMAGE_ROOT = ROOT / "dataset/VisDroneSplit1000Guarded/VisDrone2019-DET-test/images"
GT_ROOT = ROOT / "experiment/exp30_real_query_focal_rerank_20260711/log/gt_xyxy/test"
OUTPUT_DIR = ROOT / "report_assets/ranker_examples"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

CANVAS_BG = "#F4F6F8"
PANEL_BG = "#FFFFFF"
INK = "#17202A"
MUTED = "#667085"
GOOD = "#119C73"
BAD = "#E05252"
GT_COLOR = "#F0B429"
BEFORE_ACCENT = "#4A6572"
AFTER_ACCENT = "#1677B8"

CLASS_NAMES = {
    1: "行人",
    2: "人群",
    3: "自行车",
    4: "汽车",
    5: "面包车",
    6: "卡车",
    7: "三轮车",
    8: "棚式三轮车",
    9: "公交车",
    10: "摩托车",
}

CASES = [
    ("0000087_01580_d_0000005", "误检 Top-1 被抑制，高质量行人框整体前移"),
    ("0000293_03601_d_0000940", "错误行人框退出榜首，汽车真框占据 Top-5"),
    ("9999972_00000_d_0000104", "靠后的高质量面包车框进入 Top-5"),
    ("9999955_00000_d_0000034", "低质量人群框被替换，行人框排序更集中"),
]


@dataclass(frozen=True)
class GroundTruth:
    class_id: int
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Candidate:
    source_index: int
    class_id: int
    box: tuple[float, float, float, float]
    before_score: float
    after_score: float
    best_iou: float
    nearest_gt: tuple[float, float, float, float] | None
    query: str
    source_prompt: str
    source_type: str


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def intersection_over_union(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-9)


def read_gt(stem: str) -> list[GroundTruth]:
    rows = []
    for line in (GT_ROOT / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append(GroundTruth(int(float(parts[0])), tuple(map(float, parts[1:5]))))
    return rows


def read_candidates(stem: str) -> list[Candidate]:
    metadata = [
        json.loads(line)
        for line in (EXP38 / "metadata/test" / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    after_rows = [
        line.split()
        for line in (EXP38 / "fusion_predictions/test" / f"{stem}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(metadata) != len(after_rows):
        raise ValueError(f"Candidate count mismatch for {stem}: {len(metadata)} != {len(after_rows)}")

    ground_truth = read_gt(stem)
    candidates = []
    for index, (row, after) in enumerate(zip(metadata, after_rows)):
        class_id = int(row["class_id"])
        box = tuple(map(float, row["box_xyxy"]))
        after_box = tuple(map(float, after[1:5]))
        if max(abs(a - b) for a, b in zip(box, after_box)) > 0.02:
            raise ValueError(f"Candidate order mismatch for {stem}, row {index}")
        matching_gt = [gt for gt in ground_truth if gt.class_id == class_id]
        nearest = max(matching_gt, key=lambda gt: intersection_over_union(box, gt.box), default=None)
        best_iou = intersection_over_union(box, nearest.box) if nearest else 0.0
        candidates.append(
            Candidate(
                source_index=index,
                class_id=class_id,
                box=box,
                before_score=float(row["final_score"]),
                after_score=float(after[5]),
                best_iou=best_iou,
                nearest_gt=nearest.box if nearest else None,
                query=str(row.get("query", "")),
                source_prompt=str(row.get("source_prompt", "")),
                source_type=str(row.get("source_type", "")),
            )
        )
    return candidates


def rounded_rectangle(draw: ImageDraw.ImageDraw, xy, radius=8, fill=PANEL_BG, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def fit_image(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, float, float]:
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    offset_x = (target_w - resized.width) / 2
    offset_y = (target_h - resized.height) / 2
    return resized, scale, offset_x, offset_y


def dashed_rectangle(draw: ImageDraw.ImageDraw, xy, color: str, width: int = 3, dash: int = 10):
    x1, y1, x2, y2 = xy
    for start in range(round(x1), round(x2), dash * 2):
        draw.line((start, y1, min(start + dash, x2), y1), fill=color, width=width)
        draw.line((start, y2, min(start + dash, x2), y2), fill=color, width=width)
    for start in range(round(y1), round(y2), dash * 2):
        draw.line((x1, start, x1, min(start + dash, y2)), fill=color, width=width)
        draw.line((x2, start, x2, min(start + dash, y2)), fill=color, width=width)


def expanded_box(xy, amount: int, bounds: tuple[int, int]):
    x1, y1, x2, y2 = xy
    width, height = bounds
    return (
        max(0, x1 - amount),
        max(0, y1 - amount),
        min(width - 1, x2 + amount),
        min(height - 1, y2 + amount),
    )


def draw_gt_box(draw: ImageDraw.ImageDraw, xy, bounds: tuple[int, int], label: bool = True):
    if xy[2] < 0 or xy[3] < 0 or xy[0] >= bounds[0] or xy[1] >= bounds[1]:
        return
    gt_box = expanded_box(xy, 4, bounds)
    if gt_box[2] <= gt_box[0] or gt_box[3] <= gt_box[1]:
        return
    draw.rectangle(gt_box, outline="#1B1B1B", width=8)
    draw.rectangle(gt_box, outline=GT_COLOR, width=5)
    if label:
        label_x = max(1, gt_box[2] - 39)
        label_y = min(bounds[1] - 25, gt_box[3] + 2)
        draw.rectangle((label_x, label_y, label_x + 38, label_y + 23), fill="#1B1B1B")
        draw.text((label_x + 7, label_y), "GT", font=font(15, True), fill=GT_COLOR)


def ranked(candidates: list[Candidate], mode: str, count: int = 5) -> list[Candidate]:
    score_key = (lambda item: item.before_score) if mode == "before" else (lambda item: item.after_score)
    return sorted(candidates, key=score_key, reverse=True)[:count]


def render_full_image(source: Image.Image, items: list[Candidate], size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "#DDE2E7")
    fitted, scale, offset_x, offset_y = fit_image(source, size)
    canvas.paste(fitted, (round(offset_x), round(offset_y)))
    draw = ImageDraw.Draw(canvas)
    seen_gt = set()
    for item in items:
        if item.nearest_gt is None:
            continue
        gt_key = tuple(round(value, 1) for value in item.nearest_gt)
        if gt_key in seen_gt:
            continue
        seen_gt.add(gt_key)
        gx1, gy1, gx2, gy2 = item.nearest_gt
        mapped_gt = (
            offset_x + gx1 * scale,
            offset_y + gy1 * scale,
            offset_x + gx2 * scale,
            offset_y + gy2 * scale,
        )
        draw_gt_box(draw, mapped_gt, size, label=True)
    for rank_index, item in reversed(list(enumerate(items, start=1))):
        color = GOOD if item.best_iou >= 0.5 else BAD
        x1, y1, x2, y2 = item.box
        mapped = (
            offset_x + x1 * scale,
            offset_y + y1 * scale,
            offset_x + x2 * scale,
            offset_y + y2 * scale,
        )
        draw.rectangle(mapped, outline=color, width=4)
        badge_x, badge_y = mapped[0], max(0, mapped[1] - 27)
        draw.rounded_rectangle((badge_x, badge_y, badge_x + 32, badge_y + 27), radius=4, fill=color)
        draw.text((badge_x + 9, badge_y + 1), str(rank_index), font=font(18, True), fill="white")
    return canvas


def crop_around(source: Image.Image, item: Candidate, size: int = 142) -> tuple[Image.Image, tuple[float, float, float, float]]:
    x1, y1, x2, y2 = item.box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(56.0, (x2 - x1) * 3.2, (y2 - y1) * 3.2)
    crop_box = (
        max(0.0, center_x - side / 2),
        max(0.0, center_y - side / 2),
        min(float(source.width), center_x + side / 2),
        min(float(source.height), center_y + side / 2),
    )
    crop = source.crop(tuple(round(value) for value in crop_box)).resize((size, size), Image.Resampling.LANCZOS)
    return crop, crop_box


def render_crop(
    source: Image.Image,
    item: Candidate,
    rank_index: int,
    score: float,
    previous_rank: int | None = None,
) -> Image.Image:
    tile = Image.new("RGB", (150, 228), PANEL_BG)
    crop, crop_box = crop_around(source, item)
    tile.paste(crop, (4, 4))
    draw = ImageDraw.Draw(tile)
    color = GOOD if item.best_iou >= 0.5 else BAD
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    sx, sy = 142 / max(1, crop_x2 - crop_x1), 142 / max(1, crop_y2 - crop_y1)

    def map_box(box):
        x1, y1, x2, y2 = box
        return (4 + (x1 - crop_x1) * sx, 4 + (y1 - crop_y1) * sy, 4 + (x2 - crop_x1) * sx, 4 + (y2 - crop_y1) * sy)

    if item.nearest_gt is not None:
        draw_gt_box(draw, map_box(item.nearest_gt), (150, 142), label=True)
    draw.rectangle(map_box(item.box), outline=color, width=4)
    draw.rounded_rectangle((8, 8, 40, 38), radius=4, fill=color)
    draw.text((17, 8), str(rank_index), font=font(19, True), fill="white")
    if previous_rank is not None:
        previous_text = f"原#{previous_rank}"
        previous_box = draw.textbbox((0, 0), previous_text, font=font(14, True))
        previous_width = previous_box[2] - previous_box[0] + 12
        draw.rounded_rectangle((146 - previous_width, 8, 146, 35), radius=4, fill="#FFFFFFE6")
        draw.text((140 - previous_width, 8), previous_text, font=font(14, True), fill=INK)
    draw.text((5, 150), f"{CLASS_NAMES[item.class_id]}  {score:.3f}", font=font(17, True), fill=INK)
    draw.text((5, 177), f"IoU {item.best_iou:.3f}", font=font(16), fill=color)
    source_label = "别名" if item.source_type == "alias" else "基础"
    prompt = item.source_prompt if item.source_type == "alias" else item.query
    draw.text((5, 201), f"提示: {prompt} [{source_label}]", font=font(13), fill=MUTED)
    return tile


def summary(items: list[Candidate]) -> tuple[int, float]:
    return sum(item.best_iou >= 0.5 for item in items), sum(item.best_iou for item in items) / len(items)


def render_column(source: Image.Image, candidates: list[Candidate], mode: str) -> Image.Image:
    width, height = 820, 850
    canvas = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(canvas)
    accent = BEFORE_ACCENT if mode == "before" else AFTER_ACCENT
    title = "重排序前 · 原始分数" if mode == "before" else "重排序后 · 融合分数"
    items = ranked(candidates, mode)
    tp_count, mean_iou = summary(items)
    draw.rectangle((0, 0, width, 8), fill=accent)
    draw.text((24, 22), title, font=font(27, True), fill=INK)
    metric = f"Top-5 高质量框 {tp_count}/5    平均 IoU {mean_iou:.3f}"
    draw.text((24, 62), metric, font=font(18), fill=MUTED)
    full = render_full_image(source, items, (772, 470))
    canvas.paste(full, (24, 103))
    score_attr = "before_score" if mode == "before" else "after_score"
    before_rank = {
        item.source_index: rank_index
        for rank_index, item in enumerate(sorted(candidates, key=lambda item: item.before_score, reverse=True), start=1)
    }
    for index, item in enumerate(items, start=1):
        old_rank = before_rank[item.source_index] if mode == "after" else None
        crop = render_crop(source, item, index, getattr(item, score_attr), old_rank)
        canvas.paste(crop, (24 + (index - 1) * 154, 592))
    return canvas


def render_case(stem: str, description: str, case_index: int) -> tuple[Image.Image, dict]:
    source = Image.open(IMAGE_ROOT / f"{stem}.jpg").convert("RGB")
    candidates = read_candidates(stem)
    before_items = ranked(candidates, "before")
    after_items = ranked(candidates, "after")
    before_tp, before_iou = summary(before_items)
    after_tp, after_iou = summary(after_items)
    before_rank_by_index = {
        item.source_index: rank_index
        for rank_index, item in enumerate(
            sorted(candidates, key=lambda item: item.before_score, reverse=True), start=1
        )
    }

    canvas = Image.new("RGB", (1740, 1090), CANVAS_BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 30), f"案例 {case_index}  {description}", font=font(34, True), fill=INK)
    draw.text((50, 80), f"测试图像 {stem}  ·  候选框数量 {len(candidates)}  ·  绿/红=预测框  ·  黄黑双线且标有 GT=真实框", font=font(18), fill=MUTED)
    before_panel = render_column(source, candidates, "before")
    after_panel = render_column(source, candidates, "after")
    canvas.paste(before_panel, (40, 140))
    canvas.paste(after_panel, (880, 140))
    draw.rounded_rectangle((620, 1010, 1120, 1068), radius=8, fill="#E8F4FB")
    delta_text = f"Top-5：{before_tp} → {after_tp} 个高质量框    平均 IoU：{before_iou:.3f} → {after_iou:.3f}"
    text_box = draw.textbbox((0, 0), delta_text, font=font(20, True))
    draw.text(((1740 - (text_box[2] - text_box[0])) / 2, 1025), delta_text, font=font(20, True), fill=AFTER_ACCENT)

    after_records = rank_records(after_items, "after")
    for record in after_records:
        record["before_rank"] = before_rank_by_index[record["source_index"]]
    details = {
        "case": case_index,
        "stem": stem,
        "description": description,
        "candidate_count": len(candidates),
        "before": rank_records(before_items, "before"),
        "after": after_records,
        "before_top5_quality_count": before_tp,
        "after_top5_quality_count": after_tp,
        "before_top5_mean_iou": round(before_iou, 6),
        "after_top5_mean_iou": round(after_iou, 6),
    }
    return canvas, details


def rank_records(items: list[Candidate], mode: str) -> list[dict]:
    records = []
    for rank_index, item in enumerate(items, start=1):
        records.append(
            {
                "rank": rank_index,
                "source_index": item.source_index,
                "class_id": item.class_id,
                "class_name": CLASS_NAMES[item.class_id],
                "score": round(item.before_score if mode == "before" else item.after_score, 6),
                "iou": round(item.best_iou, 6),
                "high_quality": item.best_iou >= 0.5,
                "box_xyxy": [round(value, 2) for value in item.box],
                "query": item.query,
                "source_prompt": item.source_prompt,
                "source_type": item.source_type,
            }
        )
    return records


def make_contact_sheet(images: list[Image.Image]) -> Image.Image:
    thumb_w = 1160
    thumb_h = round(images[0].height * thumb_w / images[0].width)
    sheet = Image.new("RGB", (1240, 120 + len(images) * (thumb_h + 30)), CANVAS_BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((40, 28), "最终排序器：真实测试集重排序案例", font=font(38, True), fill=INK)
    draw.text((40, 78), "同一候选集，仅改变置信度排序；局部放大展示 Top-5 候选框。", font=font(20), fill=MUTED)
    y = 120
    for image in images:
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (40, y))
        y += thumb_h + 30
    return sheet


def write_readme(details: list[dict]) -> None:
    lines = [
        "# 最终排序器真实图片案例",
        "",
        "## 推荐主图",
        "",
        "答辩优先使用 `clear_query_ranker_example.png`。该图来自 RefDrone test，原始提示词为 `The pedestrians on the right bottom corner of the image.`，正确框在无 Exp36 source 排序器中由第 6 名升至第 1 名。对应审计数据见 `clear_query_case.json`，生成命令为 `python report_assets/figures/generate_clear_query_ranker_example.py`。",
        "",
        "## 补充案例",
        "",
        "这些图使用 VisDroneSplit1000Guarded 测试集真实图片与标注生成。重排序前采用 metadata 中的 `final_score`，重排序后采用 Exp38 `fusion_predictions/test` 中的融合分数；候选框集合和坐标完全不变。",
        "",
        "基础 GroundingDINO 使用联合提示词：`pedestrian, people, bicycle, car, van, truck, tricycle, awning tricycle, bus, motor`。小目标召回还会使用 `person, people, group of people, tricycle, covered tricycle, awning tricycle, motorcycle, motorbike, scooter, bicycle` 等别名提示。每个局部框下方显示该候选的实际来源提示；`[基础]` 表示来自联合提示词，`[别名]` 表示来自单独的别名提示。",
        "",
        "判定口径：候选框与同类别 GT 的最大 IoU，`IoU >= 0.5` 记为高质量框。图中绿色或红色为预测框，黄黑双线且带 `GT` 标签的框为同类别真实标注。",
        "",
        "| 案例 | 图像 | 变化 | Top-5 高质量框 | Top-5 平均 IoU |",
        "|---|---|---|---:|---:|",
    ]
    for item in details:
        lines.append(
            f"| {item['case']} | `{item['stem']}` | {item['description']} | "
            f"{item['before_top5_quality_count']} → {item['after_top5_quality_count']} | "
            f"{item['before_top5_mean_iou']:.3f} → {item['after_top5_mean_iou']:.3f} |"
        )
    lines += [
        "",
        "## 输出文件",
        "",
        "- `ranker_examples_overview.png`：四个案例汇总长图。",
        "- `case_01.png` 至 `case_04.png`：适合论文或答辩单独使用的高清图。",
        "- `case_details.json`：每个 Top-5 框的原排名、类别、查询词、来源提示词、分数、IoU 和坐标。",
        "",
        "重新生成：",
        "",
        "```bash",
        "python report_assets/figures/generate_ranker_examples.py",
        "```",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = []
    details = []
    for index, (stem, description) in enumerate(CASES, start=1):
        image, case_details = render_case(stem, description, index)
        image.save(OUTPUT_DIR / f"case_{index:02d}.png", optimize=True)
        images.append(image)
        details.append(case_details)
    make_contact_sheet(images).save(OUTPUT_DIR / "ranker_examples_overview.png", optimize=True)
    (OUTPUT_DIR / "case_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(details)
    print(f"Generated {len(images)} cases in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
