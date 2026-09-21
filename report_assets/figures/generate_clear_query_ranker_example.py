#!/usr/bin/env python3
"""Render one immediately understandable RefDrone reranking example."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "report_assets/ranker_examples/clear_query_case.json"
OUTPUT_PATH = ROOT / "report_assets/ranker_examples/clear_query_ranker_example.png"
BEFORE_OUTPUT_PATH = ROOT / "report_assets/ranker_examples/clear_query_before.png"
AFTER_OUTPUT_PATH = ROOT / "report_assets/ranker_examples/clear_query_after.png"
IMAGE_ROOT = ROOT / "dataset/RefDrone/VisDrone2019-DET-test/images"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

BG = "#F3F5F7"
PANEL = "#FFFFFF"
INK = "#15202B"
MUTED = "#667085"
RED = "#E24A4A"
GREEN = "#10966D"
YELLOW = "#FFC928"
BLUE = "#1479B8"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def map_box(box, source_box, target_box):
    sx1, sy1, sx2, sy2 = source_box
    tx1, ty1, tx2, ty2 = target_box
    scale_x = (tx2 - tx1) / (sx2 - sx1)
    scale_y = (ty2 - ty1) / (sy2 - sy1)
    x1, y1, x2, y2 = box
    return (
        tx1 + (x1 - sx1) * scale_x,
        ty1 + (y1 - sy1) * scale_y,
        tx1 + (x2 - sx1) * scale_x,
        ty1 + (y2 - sy1) * scale_y,
    )


def fit_image(image: Image.Image, target_size: tuple[int, int]):
    tw, th = target_size
    scale = min(tw / image.width, th / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return resized, scale, (tw - resized.width) / 2, (th - resized.height) / 2


def draw_gt(draw: ImageDraw.ImageDraw, box, label: bool = False):
    x1, y1, x2, y2 = box
    expanded = (x1 - 4, y1 - 4, x2 + 4, y2 + 4)
    draw.rectangle(expanded, outline="#151515", width=9)
    draw.rectangle(expanded, outline=YELLOW, width=5)
    if label:
        draw.rounded_rectangle((x1 - 4, y2 + 6, x1 + 48, y2 + 34), radius=4, fill="#151515")
        draw.text((x1 + 5, y2 + 6), "GT", font=font(17, True), fill=YELLOW)


def draw_prediction(
    draw: ImageDraw.ImageDraw,
    box,
    rank: int,
    color: str,
    label_prefix: str = "",
    label_offset: tuple[int, int] = (0, 0),
    line_width: int = 3,
    label_size: int = 15,
):
    x1, y1, x2, y2 = box
    draw.rectangle(box, outline=color, width=line_width)
    label = f"{label_prefix}#{rank}"
    label_font = font(label_size, True)
    bbox = draw.textbbox((0, 0), label, font=label_font)
    width = bbox[2] - bbox[0] + 14
    label_height = label_size + 9
    label_x = x1 + label_offset[0]
    label_y = max(2, y1 - label_height - 3 + label_offset[1])
    draw.rounded_rectangle((label_x, label_y, label_x + width, label_y + label_height), radius=3, fill=color)
    draw.text((label_x + 7, label_y), label, font=label_font, fill="white")


def render_full_panel(source: Image.Image, data: dict, mode: str) -> Image.Image:
    panel = Image.new("RGB", (820, 610), PANEL)
    draw = ImageDraw.Draw(panel)
    title = "重排序前：正确框未进入 Top-3" if mode == "before" else "重排序后：正确框升至 Top-1"
    accent = "#526A78" if mode == "before" else BLUE
    draw.rectangle((0, 0, 820, 8), fill=accent)
    draw.text((24, 22), title, font=font(27, True), fill=INK)
    draw.text((24, 64), "红色=低质量预测   绿色=高质量预测   黄黑双线=GT", font=font(16), fill=MUTED)

    display = (24, 105, 796, 584)
    fitted, scale, ox, oy = fit_image(source, (display[2] - display[0], display[3] - display[1]))
    image_x, image_y = round(display[0] + ox), round(display[1] + oy)
    panel.paste(fitted, (image_x, image_y))

    def full_map(box):
        return tuple(
            value
            for pair in zip(
                (image_x, image_y, image_x, image_y),
                (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale),
            )
            for value in [pair[0] + pair[1]]
        )

    for gt_box in data["gt_boxes"]:
        draw_gt(draw, full_map(gt_box), label=False)

    rank_key = "before_rank" if mode == "before" else "after_rank"
    top_items = sorted(data["candidates"], key=lambda item: item[rank_key])[:3]
    for item in reversed(top_items):
        color = GREEN if item["best_iou"] >= 0.5 else RED
        draw_prediction(draw, full_map(item["box"]), item[rank_key], color)

    roi = full_map((1038, 880, 1400, 1050))
    draw.rectangle(roi, outline=YELLOW, width=4)
    draw.rounded_rectangle((roi[0], roi[1] - 32, roi[0] + 126, roi[1] - 3), radius=4, fill="#151515")
    draw.text((roi[0] + 8, roi[1] - 31), "查询目标区域", font=font(16, True), fill=YELLOW)
    return panel


def render_zoom(source: Image.Image, data: dict, mode: str) -> Image.Image:
    zoom = Image.new("RGB", (820, 285), PANEL)
    draw = ImageDraw.Draw(zoom)
    crop_box = (1020, 855, 1400, 1050)
    crop = source.crop(crop_box).resize((772, 205), Image.Resampling.LANCZOS)
    zoom.paste(crop, (24, 52))
    target_box = (24, 52, 796, 257)
    draw.text((24, 12), "右下角局部放大", font=font(21, True), fill=INK)
    for gt_box in data["gt_boxes"]:
        if gt_box[2] < crop_box[0]:
            continue
        draw_gt(draw, map_box(gt_box, crop_box, target_box), label=False)

    correct = next(item for item in data["candidates"] if item["best_iou"] >= 0.5 and item["after_rank"] == 1)
    mapped = map_box(correct["box"], crop_box, target_box)
    if mode == "before":
        draw_prediction(draw, mapped, correct["before_rank"], "#717B85", "原")
        caption = "正确框原排名第 6，未进入 Top-3"
        color = "#526A78"
    else:
        draw_prediction(draw, mapped, correct["after_rank"], GREEN, "新")
        caption = f"正确框升至第 1，IoU={correct['best_iou']:.3f}"
        color = GREEN
    box = draw.textbbox((0, 0), caption, font=font(19, True))
    draw.rounded_rectangle((796 - (box[2] - box[0]) - 18, 9, 796, 42), radius=5, fill=color)
    draw.text((787 - (box[2] - box[0]), 10), caption, font=font(19, True), fill="white")
    return zoom


def render_clean_single(source: Image.Image, data: dict, mode: str) -> Image.Image:
    """Render one presentation-ready view with the current Top-3 boxes."""
    canvas = Image.new("RGB", (1400, 1050), BG)
    draw = ImageDraw.Draw(canvas)
    before = mode == "before"
    title = "重排序前 · Top-3" if before else "重排序后 · Top-3"
    accent = "#526A78" if before else BLUE
    candidates = sorted(data["candidates"], key=lambda item: item["before_rank" if before else "after_rank"])
    top_items = candidates[:3]

    draw.rounded_rectangle((35, 28, 1365, 153), radius=8, fill="#17232D")
    draw.text((60, 43), title, font=font(25, True), fill=YELLOW)
    draw.text((60, 79), f'“{data["query"]}”', font=font(29, True), fill="white")
    draw.text((60, 119), f'中文：{data["query_zh"]}', font=font(19), fill="#D5DEE5")

    panel = (35, 178, 1365, 808)
    draw.rounded_rectangle(panel, radius=6, fill=PANEL)
    fitted, scale, ox, oy = fit_image(source, (1280, 590))
    image_x, image_y = round(60 + ox), round(198 + oy)
    canvas.paste(fitted, (image_x, image_y))

    def full_map(box):
        return (
            image_x + box[0] * scale,
            image_y + box[1] * scale,
            image_x + box[2] * scale,
            image_y + box[3] * scale,
        )

    rank_color = accent if before else GREEN
    rank_colors = [rank_color, rank_color, rank_color]
    main_label_offsets = [(0, -12), (-18, 34), (22, -8)]
    for rank_index, item in reversed(list(enumerate(top_items, start=1))):
        draw_prediction(
            draw,
            full_map(item["box"]),
            rank_index,
            rank_colors[rank_index - 1],
            label_offset=main_label_offsets[rank_index - 1],
        )

    zoom_panel = (35, 833, 1365, 1020)
    draw.rounded_rectangle(zoom_panel, radius=6, fill=PANEL)
    tile_w, tile_h, gap = 410, 142, 24
    for rank_index, item in enumerate(top_items, start=1):
        x1, y1, x2, y2 = item["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        crop_w, crop_h = 300.0, 160.0
        crop_x = max(0.0, min(source.width - crop_w, cx - crop_w / 2))
        crop_y = max(0.0, min(source.height - crop_h, cy - crop_h / 2))
        crop_box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
        tile_x = 60 + (rank_index - 1) * (tile_w + gap)
        tile_y = 858
        crop = source.crop(tuple(round(value) for value in crop_box)).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas.paste(crop, (tile_x, tile_y))
        mapped = map_box(item["box"], crop_box, (tile_x, tile_y, tile_x + tile_w, tile_y + tile_h))
        draw_prediction(draw, mapped, rank_index, rank_colors[rank_index - 1])
        draw.rounded_rectangle((tile_x + 8, tile_y + 8, tile_x + 104, tile_y + 39), radius=4, fill=rank_colors[rank_index - 1])
        draw.text((tile_x + 19, tile_y + 8), f"Top-{rank_index}", font=font(17, True), fill="white")

    footer = "原始排序 Top-3" if before else "重排序 Top-3 · 目标候选由原第 6 名升至第 1 名"
    draw.text((60, 1000), footer, font=font(17, True), fill=accent)
    note_box = draw.textbbox((0, 0), "图中不叠加 GT", font=font(16))
    draw.text((1340 - (note_box[2] - note_box[0]), 1001), "图中不叠加 GT", font=font(16), fill=MUTED)
    return canvas


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    source = Image.open(IMAGE_ROOT / f"{data['stem']}.jpg").convert("RGB")
    canvas = Image.new("RGB", (1740, 1110), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rounded_rectangle((40, 28, 1700, 150), radius=8, fill="#17232D")
    draw.text((66, 43), "原始提示词", font=font(19, True), fill=YELLOW)
    draw.text((66, 73), f'“{data["query"]}”', font=font(31, True), fill="white")
    draw.text((66, 116), f'中文：{data["query_zh"]}', font=font(20), fill="#D5DEE5")

    canvas.paste(render_full_panel(source, data, "before"), (40, 180))
    canvas.paste(render_full_panel(source, data, "after"), (880, 180))
    canvas.paste(render_zoom(source, data, "before"), (40, 810))
    canvas.paste(render_zoom(source, data, "after"), (880, 810))

    draw.rounded_rectangle((690, 748, 1050, 798), radius=8, fill="#E7F4FB")
    draw.text((731, 756), "正确框：原 #6  →  新 #1", font=font(22, True), fill=BLUE)
    draw.text(
        (40, 1080),
        "RefDrone test · 无 Exp36 source 排序器（seed 42）· GT 仅用于图示与 IoU 核验，不参与推理",
        font=font(16),
        fill=MUTED,
    )
    canvas.save(OUTPUT_PATH, optimize=True)
    render_clean_single(source, data, "before").save(BEFORE_OUTPUT_PATH, optimize=True)
    render_clean_single(source, data, "after").save(AFTER_OUTPUT_PATH, optimize=True)
    print(f"Generated {OUTPUT_PATH}")
    print(f"Generated {BEFORE_OUTPUT_PATH}")
    print(f"Generated {AFTER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
