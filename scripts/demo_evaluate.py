#!/usr/bin/env python3
"""
Lightweight demo for code inspection.

It evaluates the sample detection results under scripts/data and writes
visualized images plus a compact metric report. The script is intentionally
self-contained so reviewers can run it without the training environment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def parse_gt(path: Path) -> list[Box]:
    """Parse GT rows: x, y, w, h, conf, class_id, ..."""
    boxes: list[Box] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = [float(item) for item in line.split(",")]
        if len(parts) < 6:
            raise ValueError(f"{path}:{line_no} has fewer than 6 columns")
        x, y, w, h, conf, cls = parts[:6]
        boxes.append(Box(int(cls), x, y, x + w, y + h, conf))
    return boxes


def parse_pred(path: Path, conf_threshold: float) -> list[Box]:
    """Parse predictions: class_id x1 y1 x2 y2 confidence."""
    boxes: list[Box] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = [float(item) for item in line.split()]
        if len(parts) < 6:
            raise ValueError(f"{path}:{line_no} has fewer than 6 columns")
        cls, x1, y1, x2, y2, score = parts[:6]
        if score >= conf_threshold:
            boxes.append(Box(int(cls), x1, y1, x2, y2, score))
    return boxes


def iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union_area = a.area + b.area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def match_predictions(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(pred_boxes):
        for gt_idx, gt in enumerate(gt_boxes):
            if pred.cls != gt.cls:
                continue
            overlap = iou(pred, gt)
            if overlap >= iou_threshold:
                candidates.append((overlap, pred_idx, gt_idx))

    for overlap, pred_idx, gt_idx in sorted(candidates, reverse=True):
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, overlap))

    false_positives = [idx for idx in range(len(pred_boxes)) if idx not in used_pred]
    false_negatives = [idx for idx in range(len(gt_boxes)) if idx not in used_gt]
    return matches, false_positives, false_negatives


def summarize(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def draw_title(draw: ImageDraw.ImageDraw, image_width: int, title: str, subtitle: str) -> None:
    font = ImageFont.load_default()
    draw.rectangle([0, 0, image_width, 38], fill="#111827")
    draw.text((10, 6), title, fill="#ffffff", font=font)
    draw.text((10, 22), subtitle, fill="#d1d5db", font=font)


def draw_box(draw: ImageDraw.ImageDraw, box: Box, color: str, label: str) -> None:
    y_offset = 38
    xy = [box.x1, box.y1 + y_offset, box.x2, box.y2 + y_offset]
    draw.rectangle(xy, outline=color, width=3)
    text_xy = (box.x1 + 2, max(y_offset, box.y1 + y_offset - 14))
    draw.text(text_xy, label, fill=color, font=ImageFont.load_default())


def open_canvas(image_path: Path, title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.open(image_path).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 38), "#111827")
    canvas.paste(image, (0, 38))
    draw = ImageDraw.Draw(canvas)
    draw_title(draw, canvas.width, title, subtitle)
    return canvas, draw


def save_gt_visualization(image_path: Path, output_path: Path, gt_boxes: list[Box]) -> None:
    image, draw = open_canvas(
        image_path,
        "GT annotations generated from scripts/data/test/annotations",
        "Green boxes are manually prepared ground-truth targets.",
    )
    for gt in gt_boxes:
        draw_box(draw, gt, "#2ecc71", f"GT c{gt.cls}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_prediction_visualization(
    image_path: Path,
    output_path: Path,
    pred_boxes: list[Box],
    conf_threshold: float,
) -> None:
    image, draw = open_canvas(
        image_path,
        "Predictions generated from scripts/data/pred",
        f"Blue boxes are model predictions after confidence threshold >= {conf_threshold:.2f}.",
    )
    for pred in pred_boxes:
        draw_box(draw, pred, "#3498db", f"P c{pred.cls} {pred.score:.2f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_match_visualization(
    image_path: Path,
    output_path: Path,
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    matches: Iterable[tuple[int, int, float]],
    false_positives: Iterable[int],
    false_negatives: Iterable[int],
    iou_threshold: float,
) -> None:
    image, draw = open_canvas(
        image_path,
        "Evaluation result generated by scripts/demo_evaluate.py",
        f"Green/blue=matched GT/prediction, red=false positive, yellow=false negative, IoU >= {iou_threshold:.2f}.",
    )

    matched_gt = set()
    matched_pred = set()
    for pred_idx, gt_idx, overlap in matches:
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        draw_box(draw, gt_boxes[gt_idx], "#2ecc71", f"GT c{gt_boxes[gt_idx].cls}")
        draw_box(draw, pred_boxes[pred_idx], "#3498db", f"TP {overlap:.2f}")

    for pred_idx in false_positives:
        if pred_idx not in matched_pred:
            pred = pred_boxes[pred_idx]
            draw_box(draw, pred, "#e74c3c", f"FP c{pred.cls} {pred.score:.2f}")

    for gt_idx in false_negatives:
        if gt_idx not in matched_gt:
            gt = gt_boxes[gt_idx]
            draw_box(draw, gt, "#f1c40f", f"FN c{gt.cls}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def evaluate_sample(
    image_path: Path,
    annotation_dir: Path,
    prediction_dir: Path,
    output_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
) -> dict[str, object]:
    stem = image_path.stem
    gt_path = annotation_dir / f"{stem}.txt"
    pred_path = prediction_dir / f"{stem}.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing annotation: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_path}")

    gt_boxes = parse_gt(gt_path)
    pred_boxes = parse_pred(pred_path, conf_threshold)
    matches, false_positives, false_negatives = match_predictions(
        gt_boxes, pred_boxes, iou_threshold
    )
    metrics = summarize(len(matches), len(false_positives), len(false_negatives))
    gt_vis_path = output_dir / f"{stem}_01_gt_annotations.jpg"
    pred_vis_path = output_dir / f"{stem}_02_model_predictions.jpg"
    match_vis_path = output_dir / f"{stem}_03_evaluation_matches.jpg"
    save_gt_visualization(image_path, gt_vis_path, gt_boxes)
    save_prediction_visualization(image_path, pred_vis_path, pred_boxes, conf_threshold)
    save_match_visualization(
        image_path,
        match_vis_path,
        gt_boxes,
        pred_boxes,
        matches,
        false_positives,
        false_negatives,
        iou_threshold,
    )
    return {
        "image": image_path.name,
        "gt": len(gt_boxes),
        "pred": len(pred_boxes),
        "visualizations": {
            "gt_annotations": str(gt_vis_path),
            "model_predictions": str(pred_vis_path),
            "evaluation_matches": str(match_vis_path),
        },
        **metrics,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the lightweight DroneP_VG demo.")
    parser.add_argument("--data-root", type=Path, default=repo_root / "scripts" / "data")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "scripts" / "demo_output")
    parser.add_argument("--conf-threshold", type=float, default=0.30)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    args = parser.parse_args()

    image_dir = args.data_root / "test" / "image"
    annotation_dir = args.data_root / "test" / "annotations"
    prediction_dir = args.data_root / "pred"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No demo images found in {image_dir}")

    sample_results = [
        evaluate_sample(
            image_path,
            annotation_dir,
            prediction_dir,
            args.output_dir,
            args.conf_threshold,
            args.iou_threshold,
        )
        for image_path in image_paths
    ]

    total_tp = sum(int(item["tp"]) for item in sample_results)
    total_fp = sum(int(item["fp"]) for item in sample_results)
    total_fn = sum(int(item["fn"]) for item in sample_results)
    overall = summarize(total_tp, total_fp, total_fn)
    report = {
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "samples": sample_results,
        "overall": overall,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("DroneP_VG demo evaluation")
    print(f"data_root      : {args.data_root}")
    print(f"output_dir     : {args.output_dir}")
    print(f"conf/iou       : {args.conf_threshold:.2f} / {args.iou_threshold:.2f}")
    print()
    print("image                              gt  pred  tp  fp  fn  precision  recall   f1")
    print("-" * 86)
    for item in sample_results:
        print(
            f"{item['image']:<34} "
            f"{item['gt']:>3} {item['pred']:>5} {item['tp']:>3} "
            f"{item['fp']:>3} {item['fn']:>3} "
            f"{item['precision']:>9.3f} {item['recall']:>7.3f} {item['f1']:>6.3f}"
        )
    print("-" * 86)
    print(
        f"{'overall':<34} "
        f"{'':>3} {'':>5} {overall['tp']:>3} {overall['fp']:>3} {overall['fn']:>3} "
        f"{overall['precision']:>9.3f} {overall['recall']:>7.3f} {overall['f1']:>6.3f}"
    )
    print()
    print(f"metrics saved  : {report_path}")
    print("visualizations :")
    for item in sample_results:
        visualizations = item["visualizations"]
        print(f"  {item['image']}")
        print(f"    GT annotations    : {visualizations['gt_annotations']}")
        print(f"    model predictions : {visualizations['model_predictions']}")
        print(f"    evaluation result : {visualizations['evaluation_matches']}")


if __name__ == "__main__":
    main()
