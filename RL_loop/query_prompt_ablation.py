#!/usr/bin/env python3
"""Query-aware prompt expansion ablation.

This script is self-contained: it does not import or call earlier experiment
scripts. It directly uses GroundingDINO's Python API when generating candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "RL_loop/outputs/query_prompt_ablation"
PRED_ROOT = OUT_ROOT / "predictions"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"

DATA_ROOT = ROOT / "dataset/VisDroneSplit1000Guarded"
SPLIT_MANIFEST = ROOT / "experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv"
GROUNDINGDINO_ROOT = Path("/home/wangzhe/GroundingDINO")
GDINO_CONFIG = GROUNDINGDINO_ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CHECKPOINT = GROUNDINGDINO_ROOT / "weights/groundingdino_swint_ogc.pth"

BASE_BOX_THRESHOLD = 0.16
BASE_TEXT_THRESHOLD = 0.18
QUERY_BOX_THRESHOLD = 0.03
QUERY_TEXT_THRESHOLD = 0.12
ALIAS_BOX_THRESHOLD = 0.015
ALIAS_TEXT_THRESHOLD = 0.10

VARIANTS = ("fixed10", "query_only", "query_alias", "query_alias_fallback")
SMALL_CLASSES = {1, 2, 3, 7, 8, 10}

CLASS_NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning tricycle",
    9: "bus",
    10: "motor",
}

CANONICAL_QUERY = {
    1: "pedestrian",
    2: "group of people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "small tricycle",
    8: "covered tricycle",
    9: "bus",
    10: "small motorcycle",
}

BASE_PROMPTS = [
    (1, "pedestrian"),
    (2, "people"),
    (3, "bicycle"),
    (4, "car"),
    (5, "van"),
    (6, "truck"),
    (7, "tricycle"),
    (8, "awning tricycle"),
    (9, "bus"),
    (10, "motor"),
]

CLASS_KEYWORDS = {
    1: ["pedestrian", "person", "man", "woman", "walker"],
    2: ["people", "persons", "crowd", "pedestrians", "group of people"],
    3: ["bicycle", "bike", "cyclist"],
    4: ["car", "sedan", "suv", "automobile", "vehicle"],
    5: ["van", "minivan"],
    6: ["truck", "pickup", "lorry"],
    7: ["tricycle", "three wheel", "three wheeled"],
    8: ["awning tricycle", "covered tricycle", "canopy tricycle"],
    9: ["bus", "passenger vehicle"],
    10: ["motor", "motorcycle", "motorbike", "scooter"],
}

ALIAS_PROMPTS = {
    1: ["pedestrian", "person", "walker"],
    2: ["people", "group of people", "crowd"],
    3: ["bicycle", "bike", "cyclist"],
    4: ["car", "sedan", "suv", "automobile"],
    5: ["van", "minivan"],
    6: ["truck", "pickup", "lorry"],
    7: ["tricycle", "three wheel vehicle", "small tricycle"],
    8: ["covered tricycle", "awning tricycle", "canopy tricycle"],
    9: ["bus", "passenger vehicle"],
    10: ["motor", "motorcycle", "motorbike", "scooter", "bicycle", "tricycle"],
}

CONFUSION_EXPANSION = {
    1: [2],
    2: [1],
    3: [7, 10],
    7: [3, 8, 10],
    8: [7, 10],
    10: [3, 7, 8],
}


@dataclass
class Det:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source_prompt: str = ""


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("-", " ")).split()


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def infer_query_classes(query: str) -> list[int]:
    normalized = f" {normalize(query)} "
    hits: list[int] = []
    ordered = sorted(CLASS_KEYWORDS.items(), key=lambda item: max(len(k) for k in item[1]), reverse=True)
    for class_id, keywords in ordered:
        for keyword in keywords:
            if f" {keyword} " in normalized:
                hits.append(class_id)
                break
    return hits


def unique_prompts(items: Sequence[tuple[int, str]]) -> list[tuple[int, str]]:
    seen: set[tuple[int, str]] = set()
    out: list[tuple[int, str]] = []
    for class_id, prompt in items:
        key = (class_id, normalize(prompt))
        if key not in seen:
            seen.add(key)
            out.append((class_id, prompt))
    return out


def prompt_plan(query: str, target_class: int, variant: str) -> list[tuple[int, str, float, float]]:
    parsed = infer_query_classes(query) or [target_class]
    expanded_classes = list(parsed)
    for class_id in parsed:
        expanded_classes.extend(CONFUSION_EXPANSION.get(class_id, []))

    if variant == "fixed10":
        return [(class_id, prompt, BASE_BOX_THRESHOLD, BASE_TEXT_THRESHOLD) for class_id, prompt in BASE_PROMPTS]
    if variant == "query_only":
        return [(target_class, query, QUERY_BOX_THRESHOLD, QUERY_TEXT_THRESHOLD)]

    alias_items: list[tuple[int, str]] = [(target_class, query)]
    for class_id in expanded_classes:
        for prompt in ALIAS_PROMPTS.get(class_id, [CLASS_NAMES[class_id]]):
            alias_items.append((class_id, prompt))

    if variant == "query_alias":
        return [(class_id, prompt, ALIAS_BOX_THRESHOLD, ALIAS_TEXT_THRESHOLD) for class_id, prompt in unique_prompts(alias_items)]
    if variant == "query_alias_fallback":
        merged = unique_prompts(alias_items + BASE_PROMPTS)
        return [
            (
                class_id,
                prompt,
                BASE_BOX_THRESHOLD if (class_id, prompt) in BASE_PROMPTS else ALIAS_BOX_THRESHOLD,
                BASE_TEXT_THRESHOLD if (class_id, prompt) in BASE_PROMPTS else ALIAS_TEXT_THRESHOLD,
            )
            for class_id, prompt in merged
        ]
    raise ValueError(f"Unknown variant: {variant}")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def load_groundingdino_model(device: str):
    if str(GROUNDINGDINO_ROOT) not in sys.path:
        sys.path.insert(0, str(GROUNDINGDINO_ROOT))
    from groundingdino.util.inference import Model

    return Model(
        model_config_path=str(GDINO_CONFIG),
        model_checkpoint_path=str(GDINO_CHECKPOINT),
        device=resolve_device(device),
    )


def gdino_predict(model, image_bgr: np.ndarray, prompts: Sequence[str], box_thr: float, text_thr: float):
    detections = model.predict_with_classes(
        image=image_bgr,
        classes=list(prompts),
        box_threshold=box_thr,
        text_threshold=text_thr,
    )
    return np.asarray(detections.xyxy), np.asarray(detections.confidence), np.asarray(detections.class_id, dtype=object)


def generate_candidates(model, image_path: Path, query: str, target_class: int, variant: str) -> list[Det]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    candidates: list[Det] = []
    plan = prompt_plan(query, target_class, variant)
    grouped: dict[tuple[float, float], list[tuple[int, str]]] = {}
    for class_id, prompt, box_thr, text_thr in plan:
        grouped.setdefault((box_thr, text_thr), []).append((class_id, prompt))

    for (box_thr, text_thr), items in grouped.items():
        prompts = [prompt for _, prompt in items]
        boxes, scores, raw_ids = gdino_predict(model, image_bgr, prompts, box_thr, text_thr)
        for box, score, raw_id in zip(boxes, scores, raw_ids):
            if raw_id is None:
                continue
            prompt_idx = int(raw_id)
            if prompt_idx < 0 or prompt_idx >= len(items):
                continue
            class_id, prompt = items[prompt_idx]
            candidates.append(Det(class_id, float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score), prompt))
    return candidates


def merged_prompt_plan(variant: str) -> list[tuple[int, str, float, float]]:
    items: list[tuple[int, str, float, float]] = []
    for class_id, query in CANONICAL_QUERY.items():
        items.extend(prompt_plan(query, class_id, variant))
    seen: set[tuple[int, str, float, float]] = set()
    merged: list[tuple[int, str, float, float]] = []
    for class_id, prompt, box_thr, text_thr in items:
        key = (class_id, normalize(prompt), box_thr, text_thr)
        if key in seen:
            continue
        seen.add(key)
        merged.append((class_id, prompt, box_thr, text_thr))
    return merged


def generate_candidates_from_plan(model, image_path: Path, plan: Sequence[tuple[int, str, float, float]]) -> list[Det]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    candidates: list[Det] = []
    grouped: dict[tuple[float, float], list[tuple[int, str]]] = {}
    for class_id, prompt, box_thr, text_thr in plan:
        grouped.setdefault((box_thr, text_thr), []).append((class_id, prompt))

    for (box_thr, text_thr), items in grouped.items():
        prompts = [prompt for _, prompt in items]
        boxes, scores, raw_ids = gdino_predict(model, image_bgr, prompts, box_thr, text_thr)
        for box, score, raw_id in zip(boxes, scores, raw_ids):
            if raw_id is None:
                continue
            prompt_idx = int(raw_id)
            if prompt_idx < 0 or prompt_idx >= len(items):
                continue
            class_id, prompt = items[prompt_idx]
            candidates.append(Det(class_id, float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score), prompt))
    return candidates


def read_split_stems(split: str) -> list[str]:
    stems: list[str] = []
    with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["split"] == split:
                stems.append(row["stem"])
    return sorted(stems)


def image_path(split: str, stem: str) -> Path:
    return DATA_ROOT / f"VisDrone2019-DET-{split}" / "images" / f"{stem}.jpg"


def read_gt(split: str, stem: str) -> list[Det]:
    path = DATA_ROOT / f"VisDrone2019-DET-{split}" / "annotations" / f"{stem}.txt"
    img_path = image_path(split, stem)
    if not path.exists() or not img_path.exists():
        return []
    with Image.open(img_path) as image:
        width, height = image.size
    boxes: list[Det] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 6:
            continue
        x, y, w, h = [float(v) for v in parts[:4]]
        class_id = int(float(parts[5]))
        if class_id not in CLASS_NAMES:
            continue
        x1 = max(0.0, min(width - 1.0, x))
        y1 = max(0.0, min(height - 1.0, y))
        x2 = max(x1 + 1e-6, min(float(width), x + w))
        y2 = max(y1 + 1e-6, min(float(height), y + h))
        boxes.append(Det(class_id, x1, y1, x2, y2, 1.0))
    return boxes


def write_predictions(path: Path, boxes: Sequence[Det]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for box in sorted(boxes, key=lambda item: item.score, reverse=True):
            f.write(f"{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {box.score:.6f}\n")


def read_pred(path: Path) -> list[Det]:
    boxes: list[Det] = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) != 6:
            continue
        class_id = int(float(parts[0]))
        x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
        score = float(parts[5])
        if class_id in CLASS_NAMES and x2 > x1 and y2 > y1:
            boxes.append(Det(class_id, x1, y1, x2, y2, score))
    return boxes


def iou(a: Det, b: Det) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / max(area_a + area_b - inter, 1e-9)


def evaluate_detection(gt_by_stem: dict[str, list[Det]], pred_dir: Path) -> dict[str, float]:
    total_gt = sum(len(items) for items in gt_by_stem.values())
    total_pred = 0
    tp05 = 0
    tp075 = 0
    cls_gt_count = {class_id: 0 for class_id in CLASS_NAMES}
    for stem, gt_boxes in gt_by_stem.items():
        pred_boxes = read_pred(pred_dir / f"{stem}.txt")
        total_pred += len(pred_boxes)
        for gt in gt_boxes:
            cls_gt_count[gt.class_id] += 1
        used = [False] * len(gt_boxes)
        for pred in sorted(pred_boxes, key=lambda item: item.score, reverse=True):
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(gt_boxes):
                if used[idx] or gt.class_id != pred.class_id:
                    continue
                cur = iou(pred, gt)
                if cur > best_iou:
                    best_iou = cur
                    best_idx = idx
            if best_idx >= 0 and best_iou >= 0.5:
                used[best_idx] = True
                tp05 += 1
                if best_iou >= 0.75:
                    tp075 += 1

    aps: list[float] = []
    for class_id, gt_count in cls_gt_count.items():
        if gt_count == 0:
            continue
        ranked: list[tuple[float, int]] = []
        for stem, gt_boxes_all in gt_by_stem.items():
            gt_boxes = [gt for gt in gt_boxes_all if gt.class_id == class_id]
            pred_boxes = [pred for pred in read_pred(pred_dir / f"{stem}.txt") if pred.class_id == class_id]
            used = [False] * len(gt_boxes)
            for pred in sorted(pred_boxes, key=lambda item: item.score, reverse=True):
                best_idx = -1
                best_iou = 0.0
                for idx, gt in enumerate(gt_boxes):
                    if used[idx]:
                        continue
                    cur = iou(pred, gt)
                    if cur > best_iou:
                        best_iou = cur
                        best_idx = idx
                if best_idx >= 0 and best_iou >= 0.5:
                    used[best_idx] = True
                    ranked.append((pred.score, 1))
                else:
                    ranked.append((pred.score, 0))
        if not ranked:
            aps.append(0.0)
            continue
        tp_cum = 0.0
        fp_cum = 0.0
        recalls = [0.0]
        precisions = [0.0]
        for _, label in sorted(ranked, key=lambda item: item[0], reverse=True):
            tp_cum += float(label == 1)
            fp_cum += float(label == 0)
            recalls.append(tp_cum / max(gt_count, 1))
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
        "gt_box_count": float(total_gt),
        "candidate_count": float(total_pred),
        "tp_05": float(tp05),
        "tp_075": float(tp075),
        "acc_05": tp05 / total_gt if total_gt else 0.0,
        "acc_075": tp075 / total_gt if total_gt else 0.0,
        "map_05": sum(aps) / len(aps) if aps else 0.0,
    }


def candidate_recall(gt_by_stem: dict[str, list[Det]], pred_dir: Path, small_only: bool = False) -> float:
    total = 0
    hit = 0
    for stem, gt_boxes in gt_by_stem.items():
        pred_boxes = read_pred(pred_dir / f"{stem}.txt")
        for gt in gt_boxes:
            if small_only and gt.class_id not in SMALL_CLASSES:
                continue
            total += 1
            if any(pred.class_id == gt.class_id and iou(pred, gt) >= 0.5 for pred in pred_boxes):
                hit += 1
    return hit / total if total else 0.0


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    stems = read_split_stems(args.split)
    if args.limit and args.limit > 0:
        stems = stems[: args.limit]
    gt_by_stem = {stem: read_gt(args.split, stem) for stem in stems}
    model = load_groundingdino_model(args.device)

    for variant in VARIANTS:
        out_dir = PRED_ROOT / args.split / variant
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plan = merged_prompt_plan(variant)
        print(f"[{variant}] prompt_count={len(plan)} grouped_calls={len({(p[2], p[3]) for p in plan})}", flush=True)
        for idx, stem in enumerate(stems, start=1):
            image = image_path(args.split, stem)
            boxes = generate_candidates_from_plan(model, image, plan)
            write_predictions(out_dir / f"{stem}.txt", boxes)
            if idx == 1 or idx % 10 == 0 or idx == len(stems):
                print(f"[{variant}] {idx}/{len(stems)} stem={stem} boxes={len(boxes)}", flush=True)

    metrics: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        pred_dir = PRED_ROOT / args.split / variant
        item = evaluate_detection(gt_by_stem, pred_dir)
        item["candidate_recall_05"] = candidate_recall(gt_by_stem, pred_dir, small_only=False)
        item["small_candidate_recall_05"] = candidate_recall(gt_by_stem, pred_dir, small_only=True)
        metrics[variant] = item
    return {"split": args.split, "image_count": len(stems), "metrics": metrics}


def write_summary(summary: dict[str, object]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = summary["metrics"]
    lines = [
        "# Query-aware Prompt Ablation",
        "",
        f"- split: `{summary['split']}`",
        f"- image_count: `{summary['image_count']}`",
        "",
        "| Variant | Candidates | Acc@0.5 | Acc@0.75 | mAP@0.5 | Cand Recall@0.5 | Small Cand Recall@0.5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        item = metrics[variant]
        lines.append(
            f"| {variant} | {int(item['candidate_count'])} | {item['acc_05']:.4f} | {item['acc_075']:.4f} | "
            f"{item['map_05']:.4f} | {item['candidate_recall_05']:.4f} | {item['small_candidate_recall_05']:.4f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run() -> None:
    for class_id, query in CANONICAL_QUERY.items():
        print(f"\n[{class_id}] {CLASS_NAMES[class_id]} | query={query}")
        for variant in VARIANTS:
            prompts = [prompt for _, prompt, _, _ in prompt_plan(query, class_id, variant)]
            print(f"  {variant}: {prompts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run query-aware prompt expansion ablation.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=5, help="Image limit. Use 0 for the full split.")
    parser.add_argument("--device", default="auto", help="GroundingDINO device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt plans without running GroundingDINO.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run()
        return
    for path in [DATA_ROOT, SPLIT_MANIFEST, GROUNDINGDINO_ROOT, GDINO_CONFIG, GDINO_CHECKPOINT]:
        if not path.exists():
            raise RuntimeError(f"Missing required path: {path}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = run_experiment(args)
    write_summary(summary)
    print(f"[Done] summary={SUMMARY_MD}")


if __name__ == "__main__":
    main()
