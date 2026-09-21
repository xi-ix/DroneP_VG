#!/usr/bin/env python3
"""
Build the final experiment-result structure from GT and prediction txt files.

Default input uses scripts/data so the script is runnable during inspection.
For formal experiments, pass --gt-dir, --pred-dir, and --output-dir explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DetBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0


def parse_xyxy_line(line: str, is_prediction: bool) -> DetBox | None:
    parts = line.replace(",", " ").split()
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    x1, y1, x2, y2 = (float(v) for v in parts[1:5])
    score = float(parts[5]) if is_prediction and len(parts) >= 6 else 1.0
    if x2 <= x1 or y2 <= y1:
        return None
    return DetBox(class_id, x1, y1, x2, y2, score)


def parse_visdrone_gt_line(line: str) -> DetBox | None:
    parts = [float(v) for v in line.split(",")]
    if len(parts) < 6:
        return None
    x, y, w, h, _, class_id = parts[:6]
    if w <= 0 or h <= 0:
        return None
    return DetBox(int(class_id), x, y, x + w, y + h, 1.0)


def read_boxes(path: Path, is_prediction: bool, gt_format: str) -> list[DetBox]:
    if not path.exists():
        return []
    boxes: list[DetBox] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        box = (
            parse_xyxy_line(line, is_prediction=True)
            if is_prediction
            else parse_visdrone_gt_line(line)
            if gt_format == "visdrone"
            else parse_xyxy_line(line, is_prediction=False)
        )
        if box is not None:
            boxes.append(box)
    return boxes


def iou(a: DetBox, b: DetBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def count_matches(gt_boxes: list[DetBox], pred_boxes: list[DetBox], threshold: float, class_aware: bool) -> int:
    used_gt: set[int] = set()
    tp = 0
    for pred in sorted(pred_boxes, key=lambda b: b.score, reverse=True):
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in used_gt:
                continue
            if class_aware and pred.class_id != gt.class_id:
                continue
            overlap = iou(pred, gt)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx >= 0 and best_iou >= threshold:
            used_gt.add(best_idx)
            tp += 1
    return tp


def count_acc_matches(gt_boxes: list[DetBox], pred_boxes: list[DetBox], class_aware: bool) -> tuple[int, int]:
    used_gt: set[int] = set()
    tp_05 = 0
    tp_075 = 0
    for pred in sorted(pred_boxes, key=lambda b: b.score, reverse=True):
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in used_gt:
                continue
            if class_aware and pred.class_id != gt.class_id:
                continue
            overlap = iou(pred, gt)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx >= 0 and best_iou >= 0.5:
            used_gt.add(best_idx)
            tp_05 += 1
            if best_iou >= 0.75:
                tp_075 += 1
    return tp_05, tp_075


def continuous_ap(labels: list[int], gt_count: int) -> float:
    if gt_count <= 0 or not labels:
        return 0.0
    tp_cum = 0
    fp_cum = 0
    recalls = [0.0]
    precisions = [0.0]
    for label in labels:
        if label:
            tp_cum += 1
        else:
            fp_cum += 1
        recalls.append(tp_cum / gt_count)
        precisions.append(tp_cum / max(tp_cum + fp_cum, 1))
    recalls.append(1.0)
    precisions.append(0.0)
    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])
    ap = 0.0
    for idx in range(len(recalls) - 1):
        if recalls[idx + 1] != recalls[idx]:
            ap += (recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]
    return ap


def evaluate_ap_for_class(
    class_id: int,
    stems: Iterable[str],
    gt_by_stem: dict[str, list[DetBox]],
    pred_by_stem: dict[str, list[DetBox]],
    iou_threshold: float,
) -> tuple[float, int, int]:
    gt_count = sum(1 for stem in stems for box in gt_by_stem[stem] if box.class_id == class_id)
    ranked: list[tuple[float, str, DetBox]] = []
    for stem in stems:
        ranked.extend((box.score, stem, box) for box in pred_by_stem[stem] if box.class_id == class_id)
    ranked.sort(key=lambda item: item[0], reverse=True)

    used: dict[str, set[int]] = {stem: set() for stem in stems}
    labels: list[int] = []
    for _, stem, pred in ranked:
        gt_boxes = [box for box in gt_by_stem[stem] if box.class_id == class_id]
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in used[stem]:
                continue
            overlap = iou(pred, gt)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            used[stem].add(best_idx)
            labels.append(1)
        else:
            labels.append(0)
    return continuous_ap(labels, gt_count), gt_count, len(ranked)


def evaluate(gt_dir: Path, pred_dir: Path, gt_format: str, class_aware: bool) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
    stems = sorted(path.stem for path in gt_dir.glob("*.txt"))
    gt_by_stem = {stem: read_boxes(gt_dir / f"{stem}.txt", False, gt_format) for stem in stems}
    pred_by_stem = {stem: read_boxes(pred_dir / f"{stem}.txt", True, gt_format) for stem in stems}

    per_image: list[dict[str, object]] = []
    total_gt = total_pred = total_tp_05 = total_tp_075 = 0
    for stem in stems:
        gt_boxes = gt_by_stem[stem]
        pred_boxes = pred_by_stem[stem]
        tp_05, tp_075 = count_acc_matches(gt_boxes, pred_boxes, class_aware)
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        total_tp_05 += tp_05
        total_tp_075 += tp_075
        per_image.append({
            "stem": stem,
            "gt_box_count": len(gt_boxes),
            "prediction_box_count": len(pred_boxes),
            "tp_05": tp_05,
            "tp_075": tp_075,
            "acc_05": tp_05 / len(gt_boxes) if gt_boxes else 0.0,
            "acc_075": tp_075 / len(gt_boxes) if gt_boxes else 0.0,
        })

    class_ids = sorted({box.class_id for boxes in gt_by_stem.values() for box in boxes})
    per_class: list[dict[str, object]] = []
    ap_values: list[float] = []
    for class_id in class_ids:
        ap, gt_count, pred_count = evaluate_ap_for_class(class_id, stems, gt_by_stem, pred_by_stem, 0.5)
        tp_05 = sum(
            count_matches(
                [box for box in gt_by_stem[stem] if box.class_id == class_id],
                [box for box in pred_by_stem[stem] if box.class_id == class_id],
                0.5,
                class_aware=True,
            )
            for stem in stems
        )
        ap_values.append(ap)
        per_class.append({
            "class_id": class_id,
            "gt_box_count": gt_count,
            "prediction_box_count": pred_count,
            "tp_05": tp_05,
            "recall_05": tp_05 / gt_count if gt_count else 0.0,
            "precision_05": tp_05 / pred_count if pred_count else 0.0,
            "ap_05": ap,
        })

    metrics = {
        "file_count": float(len(stems)),
        "gt_box_count": float(total_gt),
        "prediction_box_count": float(total_pred),
        "acc_05": total_tp_05 / total_gt if total_gt else 0.0,
        "acc_075": total_tp_075 / total_gt if total_gt else 0.0,
        "map_05": sum(ap_values) / len(ap_values) if ap_values else 0.0,
        "tp_05": float(total_tp_05),
        "tp_075": float(total_tp_075),
    }
    return metrics, per_class, per_image


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, experiment_name: str, metrics: dict[str, float], config: dict[str, object]) -> None:
    lines = [
        f"# {experiment_name} Final Evaluation Summary",
        "",
        "## Config",
        f"- gt_dir: {config['gt_dir']}",
        f"- pred_dir: {config['pred_dir']}",
        f"- eval_mode: {'class-aware' if config['class_aware'] else 'class-agnostic'}",
        f"- gt_format: {config['gt_format']}",
        "",
        "## Metrics",
        f"- Label files considered: {int(metrics['file_count'])}",
        f"- GT boxes: {int(metrics['gt_box_count'])}",
        f"- Prediction boxes: {int(metrics['prediction_box_count'])}",
        f"- Acc@0.5: {metrics['acc_05']:.4f} ({int(metrics['tp_05'])}/{int(metrics['gt_box_count'])})",
        f"- Acc@0.75: {metrics['acc_075']:.4f} ({int(metrics['tp_075'])}/{int(metrics['gt_box_count'])})",
        f"- mAP@0.5: {metrics['map_05']:.4f}",
        "",
        "## Output Structure",
        "- summary.json: machine-readable final result",
        "- evaluation_summary_final_class_aware.md: human-readable final report",
        "- metrics_table.csv: one-row key metric table for paper/report tables",
        "- per_class_metrics.csv: per-class AP/recall/precision",
        "- per_image_metrics.csv: per-image match statistics",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate final detection results and build result files.")
    parser.add_argument("--gt-dir", type=Path, default=repo_root / "scripts/data/test/annotations")
    parser.add_argument("--pred-dir", type=Path, default=repo_root / "scripts/data/pred")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "scripts/final_results")
    parser.add_argument("--experiment-name", default="inspection_demo")
    parser.add_argument("--gt-format", choices=["xyxy", "visdrone"], default="visdrone")
    parser.add_argument("--class-agnostic", action="store_true")
    args = parser.parse_args()

    class_aware = not args.class_agnostic
    metrics, per_class, per_image = evaluate(args.gt_dir, args.pred_dir, args.gt_format, class_aware)
    config = {
        "experiment_name": args.experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gt_dir": str(args.gt_dir),
        "pred_dir": str(args.pred_dir),
        "output_dir": str(args.output_dir),
        "gt_format": args.gt_format,
        "class_aware": class_aware,
        "iou_thresholds": [0.5, 0.75],
        "ap_iou_threshold": 0.5,
    }
    summary = {
        "config": config,
        "final_metrics": metrics,
        "full_metrics": metrics,
        "per_class_metrics": per_class,
        "per_image_metrics_path": str(args.output_dir / "per_image_metrics.csv"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "evaluation_summary_final_class_aware.md", args.experiment_name, metrics, config)
    write_csv(args.output_dir / "per_class_metrics.csv", per_class)
    write_csv(args.output_dir / "per_image_metrics.csv", per_image)
    write_csv(args.output_dir / "metrics_table.csv", [{
        "experiment": args.experiment_name,
        "file_count": int(metrics["file_count"]),
        "gt_box_count": int(metrics["gt_box_count"]),
        "prediction_box_count": int(metrics["prediction_box_count"]),
        "acc_05": f"{metrics['acc_05']:.6f}",
        "acc_075": f"{metrics['acc_075']:.6f}",
        "map_05": f"{metrics['map_05']:.6f}",
    }])

    print(f"[Done] final results written to {args.output_dir}")
    print(f"Acc@0.5={metrics['acc_05']:.4f} Acc@0.75={metrics['acc_075']:.4f} mAP@0.5={metrics['map_05']:.4f}")


if __name__ == "__main__":
    main()
