#!/usr/bin/env python3
"""Self-contained test entry for the latest DroneP_VG scheme.

The script does not import or call earlier experiment scripts. It rebuilds final
scores from fixed metadata plus trained checkpoints, evaluates baseline and the
model, and saves random visual comparisons with GT/baseline/model boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs/latest_test"
PRED_ROOT = OUT_ROOT / "predictions"
SAMPLE_ROOT = OUT_ROOT / "samples"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"

DATA_ROOT = ROOT / "dataset/VisDroneSplit1000Guarded"
SPLIT_MANIFEST = ROOT / "experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv"
GT_ROOT = ROOT / "experiment/exp30_real_query_focal_rerank_20260711/log/gt_xyxy/test"
META_ROOT = ROOT / "experiment/exp38_online_rich_metadata_fusion_20260712/log/metadata/test"
BASELINE_PRED_ROOT = ROOT / "experiment/baseline/groundingdino_base_visdrone_split1000guarded_test_only_20260429/log/predictions"

TRAIN_ALIAS_CKPT = ROOT / "outputs/latest_train/checkpoints/alias_reranker_best.pt"
TRAIN_FUSION_CKPT = ROOT / "outputs/latest_train/checkpoints/rich_fusion_best.pt"
FALLBACK_ALIAS_CKPT = ROOT / "experiment/exp36_small_target_alias_rerank_20260711/log/exp36_small_target_alias_rerank_20260711_best.pt"
FALLBACK_FUSION_CKPT = ROOT / "experiment/exp38_online_rich_metadata_fusion_20260712/log/exp38_rich_metadata_fusion_20260712_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_ALPHA = 0.15

SMALL_CLASSES = {1, 2, 3, 7, 8, 10}
WEAK_SMALL_CLASSES = {2, 7, 8, 10}
PERSON_CLASSES = {1, 2}
SMALL_VEHICLE_CLASSES = {3, 7, 8, 10}
LARGE_VEHICLE_CLASSES = {4, 5, 6, 9}
VEHICLE_CLASSES = SMALL_VEHICLE_CLASSES | LARGE_VEHICLE_CLASSES
CONFUSION_GROUPS = {1: {2}, 2: {1}, 3: {7, 10}, 7: {3, 8, 10}, 8: {7, 10}, 10: {3, 7, 8}}
AREA_PRIOR = {1: 0.0019, 2: 0.0011, 3: 0.0020, 4: 0.0200, 5: 0.0180, 6: 0.0250, 7: 0.0062, 8: 0.0037, 9: 0.0300, 10: 0.0018}
PROMPT_VOCAB = [
    "gdino_base",
    "person",
    "people",
    "group of people",
    "tricycle",
    "covered tricycle",
    "awning tricycle",
    "motorcycle",
    "motorbike",
    "scooter",
    "bicycle",
]
CLASS_NAMES = {1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van", 6: "truck", 7: "tricycle", 8: "awning tricycle", 9: "bus", 10: "motor"}


@dataclass
class Det:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


class AliasReranker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class RichFusionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Linear(80, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def read_test_stems() -> list[str]:
    stems: list[str] = []
    with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["split"] == "test":
                stems.append(row["stem"])
    return sorted(stems)


def read_metadata(stem: str) -> list[dict[str, object]]:
    path = META_ROOT / f"{stem}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_dets(path: Path, is_gt: bool = False) -> list[Det]:
    dets: list[Det] = []
    if not path.exists():
        return dets
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if is_gt and len(parts) != 5:
            continue
        if not is_gt and len(parts) != 6:
            continue
        cls = int(float(parts[0]))
        if cls not in CLASS_NAMES:
            continue
        x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
        score = 1.0 if is_gt else float(parts[5])
        if x2 > x1 and y2 > y1:
            dets.append(Det(cls, x1, y1, x2, y2, score))
    return dets


def det_from_meta(row: dict[str, object], score: float) -> Det:
    x1, y1, x2, y2 = [float(v) for v in row["box_xyxy"]]
    return Det(int(row["class_id"]), x1, y1, x2, y2, float(score))


def iou_det(a: Det, b: Det) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / max(area_a + area_b - inter, 1e-9)


def hierarchy_features(class_id: int) -> list[float]:
    return [
        1.0 if class_id in SMALL_CLASSES else 0.0,
        1.0 if class_id in WEAK_SMALL_CLASSES else 0.0,
        1.0 if class_id in PERSON_CLASSES else 0.0,
        1.0 if class_id in VEHICLE_CLASSES else 0.0,
        1.0 if class_id in SMALL_VEHICLE_CLASSES else 0.0,
        1.0 if class_id in LARGE_VEHICLE_CLASSES else 0.0,
        1.0 if class_id == 10 else 0.0,
        1.0 if class_id in {7, 8} else 0.0,
        1.0 if class_id == 8 else 0.0,
        1.0 if class_id in {3, 7, 8, 10} else 0.0,
    ]


def alias_box_features(det: Det, width: int, height: int) -> list[float]:
    bw = max(1.0, det.x2 - det.x1)
    bh = max(1.0, det.y2 - det.y1)
    area = (bw * bh) / max(1.0, width * height)
    prior = AREA_PRIOR.get(det.class_id, 0.01)
    scale_fit = math.exp(-abs(math.log(max(area, 1e-8) / max(prior, 1e-8))))
    score = min(max(det.score, 1e-6), 1.0 - 1e-6)
    base = [
        score,
        math.log(score / (1.0 - score)),
        math.sqrt(area),
        math.log(max(area, 1e-8)),
        bw / width,
        bh / height,
        math.log(max(bw / bh, 1e-6)),
        ((det.x1 + det.x2) / 2.0) / width,
        ((det.y1 + det.y2) / 2.0) / height,
        det.class_id / 10.0,
        scale_fit,
        min(1.0, area / max(prior, 1e-8)),
    ]
    one_hot = [1.0 if det.class_id == class_id else 0.0 for class_id in range(1, 11)]
    return base + hierarchy_features(det.class_id) + one_hot + [0.0, 0.0]


def safe_logit(score: float) -> float:
    score = min(max(float(score), 1e-6), 1.0 - 1e-6)
    return math.log(score / (1.0 - score))


def rich_features(row: dict[str, object], alias_score: float) -> list[float]:
    class_id = int(row["class_id"])
    source_class_id = int(row.get("source_class_id", class_id))
    final_score = float(row["final_score"])
    gdino_score = float(row.get("gdino_score", 0.0))
    query_match_score = float(row.get("query_match_score", final_score))
    small_bonus = float(row.get("small_area_bonus", 0.0))
    area_ratio = float(row.get("area_ratio", 0.0))
    rank = float(row.get("rank", 1.0))
    width = float(row.get("image_width", 1.0))
    height = float(row.get("image_height", 1.0))
    x1, y1, x2, y2 = [float(value) for value in row["box_xyxy"]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    prior = AREA_PRIOR.get(class_id, 0.01)
    source_prompt = str(row.get("source_prompt", ""))
    prompt_one_hot = [1.0 if source_prompt == prompt else 0.0 for prompt in PROMPT_VOCAB]
    score_features = [
        final_score,
        gdino_score,
        query_match_score,
        alias_score,
        small_bonus,
        final_score - gdino_score,
        final_score - query_match_score,
        query_match_score - gdino_score,
        alias_score - final_score,
        final_score * alias_score,
        safe_logit(final_score),
        safe_logit(alias_score),
    ]
    geom = [
        math.sqrt(max(area_ratio, 0.0)),
        math.log(max(area_ratio, 1e-8)),
        bw / max(1.0, width),
        bh / max(1.0, height),
        math.log(max(bw / bh, 1e-6)),
        ((x1 + x2) / 2.0) / max(1.0, width),
        ((y1 + y2) / 2.0) / max(1.0, height),
        math.exp(-abs(math.log(max(area_ratio, 1e-8) / max(prior, 1e-8)))),
        min(1.0, area_ratio / max(prior, 1e-8)),
        math.log1p(rank) / 8.0,
    ]
    source_features = [
        1.0 if row.get("source_type") == "alias" else 0.0,
        1.0 if row.get("source_type") == "base" else 0.0,
        1.0 if source_class_id != class_id else 0.0,
        source_class_id / 10.0,
        class_id / 10.0,
    ]
    class_one_hot = [1.0 if class_id == idx else 0.0 for idx in range(1, 11)]
    source_class_one_hot = [1.0 if source_class_id == idx else 0.0 for idx in range(1, 11)]
    return score_features + geom + source_features + hierarchy_features(class_id) + class_one_hot + source_class_one_hot + prompt_one_hot


def load_models(alias_ckpt_path: Path, fusion_ckpt_path: Path) -> tuple[nn.Module, torch.Tensor, torch.Tensor, nn.Module, torch.Tensor, torch.Tensor, float]:
    alias_ckpt = torch.load(alias_ckpt_path, map_location="cpu")
    alias_model = AliasReranker(input_dim=alias_ckpt["mu"].shape[1]).to(DEVICE)
    alias_model.load_state_dict(alias_ckpt["model"])
    alias_model.eval()

    fusion_ckpt = torch.load(fusion_ckpt_path, map_location="cpu")
    fusion_model = RichFusionModel(input_dim=fusion_ckpt["mu"].shape[1]).to(DEVICE)
    fusion_model.load_state_dict(fusion_ckpt["model"])
    fusion_model.eval()

    alpha = DEFAULT_ALPHA
    train_summary = ROOT / "outputs/latest_train/train_summary.json"
    if train_summary.exists() and fusion_ckpt_path == TRAIN_FUSION_CKPT:
        data = json.loads(train_summary.read_text(encoding="utf-8"))
        alpha = float(data.get("config", {}).get("best_alpha", DEFAULT_ALPHA))
    return (
        alias_model,
        alias_ckpt["mu"].float(),
        alias_ckpt["sigma"].float(),
        fusion_model,
        fusion_ckpt["mu"].float(),
        fusion_ckpt["sigma"].float(),
        alpha,
    )


def alias_scores(rows: Sequence[dict[str, object]], model: nn.Module, mu: torch.Tensor, sigma: torch.Tensor) -> list[float]:
    dets = [det_from_meta(row, float(row["final_score"])) for row in rows]
    features = [alias_box_features(det, int(row["image_width"]), int(row["image_height"])) for det, row in zip(dets, rows)]
    if not features:
        return []
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        x = ((x - mu.cpu()) / sigma.cpu()).to(DEVICE)
        return torch.sigmoid(model(x)).cpu().tolist()


def write_model_predictions(alias_ckpt: Path, fusion_ckpt: Path, alpha_override: float | None) -> tuple[Path, float]:
    alias_model, alias_mu, alias_sigma, fusion_model, fusion_mu, fusion_sigma, alpha = load_models(alias_ckpt, fusion_ckpt)
    if alpha_override is not None:
        alpha = alpha_override
    if PRED_ROOT.exists():
        shutil.rmtree(PRED_ROOT)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    for stem in read_test_stems():
        rows = read_metadata(stem)
        a_scores = alias_scores(rows, alias_model, alias_mu, alias_sigma)
        features = [rich_features(row, score) for row, score in zip(rows, a_scores)]
        if features:
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32)
                x = ((x - fusion_mu.cpu()) / fusion_sigma.cpu()).to(DEVICE)
                f_scores = torch.sigmoid(fusion_model(x)).cpu().tolist()
        else:
            f_scores = []
        with (PRED_ROOT / f"{stem}.txt").open("w", encoding="utf-8") as f:
            for row, a_score, f_score in zip(rows, a_scores, f_scores):
                base_blend = 0.5 * float(row["final_score"]) + 0.5 * float(a_score)
                final_score = alpha * base_blend + (1.0 - alpha) * float(f_score)
                x1, y1, x2, y2 = [float(v) for v in row["box_xyxy"]]
                f.write(f"{int(row['class_id'])} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {final_score:.6f}\n")
    return PRED_ROOT, alpha


def parse_line(line: str, is_prediction: bool) -> tuple[int, tuple[float, float, float, float], float] | None:
    parts = line.strip().split()
    if len(parts) not in (5, 6):
        return None
    class_id = int(float(parts[0]))
    x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
    score = float(parts[5]) if is_prediction and len(parts) == 6 else 1.0
    if class_id not in CLASS_NAMES or x2 <= x1 or y2 <= y1:
        return None
    return class_id, (x1, y1, x2, y2), score


def iou_tuple(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def evaluate_detections(gt_dir: Path, pred_dir: Path, iou_thr: float = 0.5) -> dict[str, float]:
    gt_files = sorted(gt_dir.glob("*.txt"))
    total_gt = total_pred = tp_05 = tp_075 = 0
    cls_gt_count = {c: 0 for c in range(1, 11)}
    for gt_path in gt_files:
        gt_objs = []
        for line in gt_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_line(line, False)
            if parsed is not None:
                gt_objs.append(parsed)
        pred_path = pred_dir / f"{gt_path.stem}.txt"
        pred_objs = []
        if pred_path.exists():
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                parsed = parse_line(line, True)
                if parsed is not None:
                    pred_objs.append(parsed)
        total_gt += len(gt_objs)
        total_pred += len(pred_objs)
        for cls, _, _ in gt_objs:
            cls_gt_count[cls] += 1
        used = [False] * len(gt_objs)
        for cls, box, _ in sorted(pred_objs, key=lambda item: item[2], reverse=True):
            best_idx, best_val = -1, 0.0
            for idx, (gt_cls, gt_box, _) in enumerate(gt_objs):
                if used[idx] or gt_cls != cls:
                    continue
                cur = iou_tuple(box, gt_box)
                if cur > best_val:
                    best_idx, best_val = idx, cur
            if best_idx >= 0 and best_val >= 0.5:
                used[best_idx] = True
                tp_05 += 1
                if best_val >= 0.75:
                    tp_075 += 1
    aps: list[float] = []
    for cls in range(1, 11):
        gt_count = cls_gt_count[cls]
        if gt_count == 0:
            continue
        ranked: list[tuple[float, int]] = []
        for gt_path in gt_files:
            gt_boxes = []
            for line in gt_path.read_text(encoding="utf-8").splitlines():
                parsed = parse_line(line, False)
                if parsed is not None and parsed[0] == cls:
                    gt_boxes.append(parsed[1])
            pred_path = pred_dir / f"{gt_path.stem}.txt"
            pred_objs = []
            if pred_path.exists():
                for line in pred_path.read_text(encoding="utf-8").splitlines():
                    parsed = parse_line(line, True)
                    if parsed is not None:
                        pred_objs.append(parsed)
            used = [False] * len(gt_boxes)
            for _, box, score in sorted([p for p in pred_objs if p[0] == cls], key=lambda item: item[2], reverse=True):
                best_idx, best_val = -1, 0.0
                for idx, gt_box in enumerate(gt_boxes):
                    if used[idx]:
                        continue
                    cur = iou_tuple(box, gt_box)
                    if cur > best_val:
                        best_idx, best_val = idx, cur
                if best_idx >= 0 and best_val >= iou_thr:
                    used[best_idx] = True
                    ranked.append((score, 1))
                else:
                    ranked.append((score, 0))
        if not ranked:
            aps.append(0.0)
            continue
        tp_cum = fp_cum = 0.0
        recalls = [0.0]
        precisions = [0.0]
        for _, label in sorted(ranked, key=lambda item: item[0], reverse=True):
            tp_cum += float(label == 1)
            fp_cum += float(label == 0)
            recalls.append(tp_cum / max(1, gt_count))
            precisions.append(tp_cum / max(tp_cum + fp_cum, 1e-9))
        recalls.append(1.0)
        precisions.append(0.0)
        for idx in range(len(precisions) - 2, -1, -1):
            precisions[idx] = max(precisions[idx], precisions[idx + 1])
        ap = 0.0
        for idx in range(1, len(recalls)):
            if recalls[idx] != recalls[idx - 1]:
                ap += (recalls[idx] - recalls[idx - 1]) * precisions[idx]
        aps.append(ap)
    return {
        "file_count": float(len(gt_files)),
        "gt_box_count": float(total_gt),
        "prediction_box_count": float(total_pred),
        "tp_05": float(tp_05),
        "tp_075": float(tp_075),
        "acc_05": tp_05 / total_gt if total_gt else 0.0,
        "acc_075": tp_075 / total_gt if total_gt else 0.0,
        "map_05": sum(aps) / len(aps) if aps else 0.0,
    }


def font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_text_bg(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int], fnt) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=fnt)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=fill)
    draw.text((x, y), text, fill=(255, 255, 255), font=fnt)


def draw_panel(image: Image.Image, title: str, gt: Sequence[Det], pred: Sequence[Det], pred_color: tuple[int, int, int], gt_only: bool = False) -> Image.Image:
    scale = max(1.0, 760.0 / max(image.width, image.height))
    canvas = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(canvas)
    label_font = font(17)
    title_font = font(23)
    for idx, det in enumerate(gt, start=1):
        box = [det.x1 * scale, det.y1 * scale, det.x2 * scale, det.y2 * scale]
        draw.rectangle(box, outline=(0, 170, 80), width=3)
        draw_text_bg(draw, (int(box[0]), max(0, int(box[1]) - 25)), f"G{idx} {CLASS_NAMES[det.class_id]}", (0, 135, 65), label_font)
    if not gt_only:
        for idx, det in enumerate(sorted(pred, key=lambda item: item.score, reverse=True)[:45], start=1):
            box = [det.x1 * scale, det.y1 * scale, det.x2 * scale, det.y2 * scale]
            draw.rectangle(box, outline=pred_color, width=3)
            draw_text_bg(draw, (int(box[0]), min(canvas.height - 25, max(0, int(box[1]) - 25))), f"P{idx} {CLASS_NAMES[det.class_id]} {det.score:.2f}", pred_color, label_font)
    draw.rectangle((0, 0, canvas.width, 40), fill=(25, 25, 25))
    draw.text((10, 7), title, fill=(255, 255, 255), font=title_font)
    return canvas


def save_sample(stem: str, out_path: Path) -> None:
    image = Image.open(DATA_ROOT / "VisDrone2019-DET-test/images" / f"{stem}.jpg").convert("RGB")
    gt = read_dets(GT_ROOT / f"{stem}.txt", is_gt=True)
    baseline = read_dets(BASELINE_PRED_ROOT / f"{stem}.txt")
    model = read_dets(PRED_ROOT / f"{stem}.txt")
    panels = [
        draw_panel(image, "(a) GT", gt, [], (0, 170, 80), gt_only=True),
        draw_panel(image, "(b) Baseline + GT", gt, baseline, (230, 130, 20)),
        draw_panel(image, "(c) My model + GT", gt, model, (40, 105, 230)),
    ]
    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, height + 42), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    draw = ImageDraw.Draw(canvas)
    draw.text((14, height + 10), "Green G#: ground truth   Orange P#: baseline prediction   Blue P#: my model prediction", fill=(20, 20, 20), font=font(21))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def generate_samples(count: int, seed: int) -> list[str]:
    stems = [
        stem
        for stem in read_test_stems()
        if (DATA_ROOT / "VisDrone2019-DET-test/images" / f"{stem}.jpg").exists()
        and (GT_ROOT / f"{stem}.txt").exists()
        and (BASELINE_PRED_ROOT / f"{stem}.txt").exists()
        and (PRED_ROOT / f"{stem}.txt").exists()
    ]
    chosen = random.Random(seed).sample(stems, k=min(count, len(stems)))
    if SAMPLE_ROOT.exists():
        shutil.rmtree(SAMPLE_ROOT)
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    for idx, stem in enumerate(chosen, start=1):
        save_sample(stem, SAMPLE_ROOT / f"sample_{idx:02d}_{stem}.png")
    return chosen


def latest_train_is_full() -> bool:
    summary_path = ROOT / "outputs/latest_train/train_summary.json"
    if not summary_path.exists():
        return False
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    config = data.get("config", {})
    return int(config.get("alias_epochs", 0)) >= 40 and int(config.get("fusion_epochs", 0)) >= 60


def choose_checkpoint(primary: Path, fallback: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if primary.exists() and latest_train_is_full():
        return primary
    return fallback


def require_inputs(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("Missing required data/artifacts:\n  " + "\n  ".join(missing))


def write_summary(baseline_metrics: dict[str, float], model_metrics: dict[str, float], sample_stems: Sequence[str], alias_ckpt: Path, fusion_ckpt: Path, alpha: float) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "device": str(DEVICE),
        "alpha": alpha,
        "alias_checkpoint": str(alias_ckpt.relative_to(ROOT)),
        "fusion_checkpoint": str(fusion_ckpt.relative_to(ROOT)),
        "gt_dir": str(GT_ROOT.relative_to(ROOT)),
        "baseline_pred_dir": str(BASELINE_PRED_ROOT.relative_to(ROOT)),
        "model_pred_dir": str(PRED_ROOT.relative_to(ROOT)),
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "sample_stems": list(sample_stems),
        "sample_dir": str(SAMPLE_ROOT.relative_to(ROOT)),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Latest Test Summary",
        "",
        "| Method | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 | TP@0.5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Baseline | {int(baseline_metrics['prediction_box_count'])} | {baseline_metrics['acc_05']:.4f} | {baseline_metrics['acc_075']:.4f} | {baseline_metrics['map_05']:.4f} | {int(baseline_metrics['tp_05'])} |",
        f"| My model | {int(model_metrics['prediction_box_count'])} | {model_metrics['acc_05']:.4f} | {model_metrics['acc_075']:.4f} | {model_metrics['map_05']:.4f} | {int(model_metrics['tp_05'])} |",
        "",
        f"- alpha: `{alpha:.2f}`",
        f"- alias checkpoint: `{alias_ckpt.relative_to(ROOT)}`",
        f"- fusion checkpoint: `{fusion_ckpt.relative_to(ROOT)}`",
        f"- model predictions: `{PRED_ROOT.relative_to(ROOT)}`",
        f"- random samples: `{SAMPLE_ROOT.relative_to(ROOT)}`",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-contained latest DroneP_VG test/evaluation.")
    parser.add_argument("--alias-ckpt", type=Path, default=None, help="Optional alias checkpoint path.")
    parser.add_argument("--fusion-ckpt", type=Path, default=None, help="Optional rich-fusion checkpoint path.")
    parser.add_argument("--alpha", type=float, default=None, help="Override score blend alpha.")
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to draw.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alias_ckpt = choose_checkpoint(TRAIN_ALIAS_CKPT, FALLBACK_ALIAS_CKPT, args.alias_ckpt)
    fusion_ckpt = choose_checkpoint(TRAIN_FUSION_CKPT, FALLBACK_FUSION_CKPT, args.fusion_ckpt)
    require_inputs([SPLIT_MANIFEST, GT_ROOT, META_ROOT, BASELINE_PRED_ROOT, alias_ckpt, fusion_ckpt])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[test.py] device={DEVICE}")
    print(f"[test.py] alias checkpoint={alias_ckpt}")
    print(f"[test.py] fusion checkpoint={fusion_ckpt}")
    model_pred_dir, alpha = write_model_predictions(alias_ckpt, fusion_ckpt, args.alpha)
    baseline_metrics = evaluate_detections(GT_ROOT, BASELINE_PRED_ROOT)
    model_metrics = evaluate_detections(GT_ROOT, model_pred_dir)
    sample_stems = generate_samples(args.samples, args.seed)
    write_summary(baseline_metrics, model_metrics, sample_stems, alias_ckpt, fusion_ckpt, alpha)

    print("\n[test.py] Final metrics")
    print(f"Baseline: Acc@0.5={baseline_metrics['acc_05']:.4f}, Acc@0.75={baseline_metrics['acc_075']:.4f}, mAP@0.5={baseline_metrics['map_05']:.4f}, pred={int(baseline_metrics['prediction_box_count'])}")
    print(f"My model: Acc@0.5={model_metrics['acc_05']:.4f}, Acc@0.75={model_metrics['acc_075']:.4f}, mAP@0.5={model_metrics['map_05']:.4f}, pred={int(model_metrics['prediction_box_count'])}")
    print(f"[test.py] Summary: {SUMMARY_MD}")
    print(f"[test.py] Samples: {SAMPLE_ROOT}")


if __name__ == "__main__":
    main()
