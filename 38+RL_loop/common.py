#!/usr/bin/env python3
from __future__ import annotations

import csv
import functools
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "38+RL_loop/outputs"
CKPT_PATH = OUT_ROOT / "rl_policy_exp38_reward.pt"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"

DATA_ROOT = ROOT / "dataset/VisDroneSplit1000Guarded"
SPLIT_MANIFEST = ROOT / "experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv"
GT_ROOT = ROOT / "experiment/exp30_real_query_focal_rerank_20260711/log/gt_xyxy"
RL_CAND_ROOT = ROOT / "RL_loop/outputs/query_prompt_ablation/predictions"
RL_PLAIN_ROOT = ROOT / "RL_loop/outputs/closed_loop_rl/predictions"
EXP38_META_ROOT = ROOT / "experiment/exp38_online_rich_metadata_fusion_20260712/log/metadata"
EXP38_PRED_ROOT = ROOT / "experiment/exp38_online_rich_metadata_fusion_20260712/log/predictions/test"

SEED = 42
CLASS_IDS = list(range(1, 11))
SMALL_CLASSES = {1, 2, 3, 7, 8, 10}
BASE_VARIANT = "query_alias_fallback"
ACTION_NAMES = [
    "STOP",
    "ADD_GDINO_BASE",
    "ADD_PERSON",
    "ADD_PEOPLE",
    "ADD_GROUP_OF_PEOPLE",
    "ADD_TRICYCLE",
    "ADD_COVERED_TRICYCLE",
    "ADD_AWNING_TRICYCLE",
    "ADD_MOTORCYCLE",
    "ADD_MOTORBIKE",
    "ADD_SCOOTER",
    "ADD_BICYCLE",
]
ACTION_TO_PROMPTS = {
    1: ("gdino_base",),
    2: ("person",),
    3: ("people",),
    4: ("group of people",),
    5: ("tricycle",),
    6: ("covered tricycle",),
    7: ("awning tricycle",),
    8: ("motorcycle",),
    9: ("motorbike",),
    10: ("scooter",),
    11: ("bicycle",),
}
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


@dataclass(frozen=True)
class Det:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


def require_inputs() -> None:
    required = [
        SPLIT_MANIFEST,
        GT_ROOT / "val",
        GT_ROOT / "test",
        EXP38_META_ROOT / "val",
        EXP38_META_ROOT / "test",
    ]
    for split in ["val", "test"]:
        required.append(RL_CAND_ROOT / split / BASE_VARIANT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing required artifacts:\n  " + "\n  ".join(missing))


def read_split_stems(split: str) -> list[str]:
    stems: list[str] = []
    with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["split"] == split:
                stems.append(row["stem"])
    return sorted(stems)


def image_size(split: str, stem: str) -> tuple[int, int]:
    path = DATA_ROOT / f"VisDrone2019-DET-{split}" / "images" / f"{stem}.jpg"
    with Image.open(path) as image:
        return image.size


def read_gt(split: str, stem: str) -> list[Det]:
    path = GT_ROOT / split / f"{stem}.txt"
    boxes: list[Det] = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
        if class_id in CLASS_NAMES and x2 > x1 and y2 > y1:
            boxes.append(Det(class_id, x1, y1, x2, y2, 1.0))
    return boxes


@functools.lru_cache(maxsize=None)
def read_pred_file(path: Path) -> list[Det]:
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


def read_candidate(split: str, variant: str, stem: str) -> list[Det]:
    return read_pred_file(RL_CAND_ROOT / split / variant / f"{stem}.txt")


@functools.lru_cache(maxsize=None)
def read_exp38_meta(split: str, stem: str) -> list[dict[str, object]]:
    path = EXP38_META_ROOT / split / f"{stem}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def meta_det(row: dict[str, object]) -> Det:
    x1, y1, x2, y2 = [float(v) for v in row["box_xyxy"]]
    return Det(int(row["class_id"]), x1, y1, x2, y2, float(row["final_score"]))


def exp38_prompt_boxes(split: str, stem: str, prompts: Sequence[str]) -> list[Det]:
    prompt_set = set(prompts)
    boxes: list[Det] = []
    for row in read_exp38_meta(split, stem):
        if str(row.get("source_prompt", "")) in prompt_set:
            boxes.append(meta_det(row))
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


def deduplicate(boxes: Sequence[Det], iou_thr: float = 0.92) -> list[Det]:
    selected: list[Det] = []
    selected_by_class: dict[int, list[Det]] = {}
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        same_class = selected_by_class.setdefault(box.class_id, [])
        if all(iou(box, kept) < iou_thr for kept in same_class):
            selected.append(box)
            same_class.append(box)
    return selected


def merge_action_pool(split: str, stem: str, action: int) -> list[Det]:
    base = read_candidate(split, BASE_VARIANT, stem)
    if action == 0:
        return deduplicate(base)
    extra = exp38_prompt_boxes(split, stem, ACTION_TO_PROMPTS[action])
    return deduplicate(base + extra)


def apply_action_to_pool(split: str, stem: str, current: Sequence[Det], action: int) -> list[Det]:
    if action == 0:
        return deduplicate(current)
    extra = exp38_prompt_boxes(split, stem, ACTION_TO_PROMPTS[action])
    return deduplicate(list(current) + extra)


def exp38_rescore(split: str, stem: str, boxes: Sequence[Det], match_iou: float = 0.90) -> tuple[list[Det], int]:
    meta_boxes = [meta_det(row) for row in read_exp38_meta(split, stem)]
    rescored: list[Det] = []
    matched = 0
    for box in boxes:
        best: Det | None = None
        best_iou = 0.0
        for cand in meta_boxes:
            if cand.class_id != box.class_id:
                continue
            cur = iou(box, cand)
            if cur > best_iou:
                best_iou = cur
                best = cand
        if best is not None and best_iou >= match_iou:
            matched += 1
            rescored.append(best)
        else:
            rescored.append(box)
    return deduplicate(rescored), matched


def state_from_boxes(boxes: Sequence[Det], width: int, height: int) -> list[float]:
    total = len(boxes)
    scores = [box.score for box in boxes]
    areas = [max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1) / max(1.0, width * height) for box in boxes]
    per_class_count: list[float] = []
    per_class_top: list[float] = []
    per_class_area: list[float] = []
    for class_id in CLASS_IDS:
        cls_indices = [idx for idx, box in enumerate(boxes) if box.class_id == class_id]
        per_class_count.append(min(len(cls_indices) / 80.0, 1.0))
        per_class_top.append(max([boxes[idx].score for idx in cls_indices] or [0.0]))
        per_class_area.append(sum(areas[idx] for idx in cls_indices) / max(1, len(cls_indices)))
    small_count = sum(1 for box in boxes if box.class_id in SMALL_CLASSES)
    mean_score = sum(scores) / max(1, total)
    score_std = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / max(1, total))
    return [
        min(total / 300.0, 1.0),
        mean_score,
        max(scores or [0.0]),
        score_std,
        sum(areas) / max(1, total),
        small_count / max(1, total),
        *per_class_count,
        *per_class_top,
        *per_class_area,
        0.0,
        1.0,
    ]


def match_stats(pred_boxes: Sequence[Det], gt_boxes: Sequence[Det]) -> dict[str, float]:
    used = [False] * len(gt_boxes)
    tp = 0
    tp75 = 0
    small_tp = 0
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
            tp += 1
            if best_iou >= 0.75:
                tp75 += 1
            if gt_boxes[best_idx].class_id in SMALL_CLASSES:
                small_tp += 1
    small_gt = sum(1 for gt in gt_boxes if gt.class_id in SMALL_CLASSES)
    return {
        "recall": tp / len(gt_boxes) if gt_boxes else 0.0,
        "recall75": tp75 / len(gt_boxes) if gt_boxes else 0.0,
        "precision": tp / len(pred_boxes) if pred_boxes else 0.0,
        "small_recall": small_tp / small_gt if small_gt else 0.0,
        "pred_count": float(len(pred_boxes)),
    }


def reward_for_action(split: str, stem: str, action: int, base_stats: dict[str, float], gt_boxes: Sequence[Det]) -> float:
    scored, _ = exp38_rescore(split, stem, merge_action_pool(split, stem, action))
    stats = match_stats(scored, gt_boxes)
    added_boxes = max(0.0, stats["pred_count"] - base_stats["pred_count"])
    box_penalty = added_boxes / max(1.0, base_stats["pred_count"])
    stop_penalty = 0.012 if action == 0 else 0.0
    action_cost = 0.015 if action != 0 else 0.0
    return (
        0.40 * (stats["recall"] - base_stats["recall"])
        + 0.22 * (stats["recall75"] - base_stats["recall75"])
        + 0.45 * (stats["small_recall"] - base_stats["small_recall"])
        + 0.15 * (stats["precision"] - base_stats["precision"])
        - 0.012 * box_penalty
        - action_cost
        - stop_penalty
    )


def reward_from_stats(
    stats: dict[str, float],
    base_stats: dict[str, float],
    action_count: int,
    used_stop: bool,
) -> float:
    added_boxes = max(0.0, stats["pred_count"] - base_stats["pred_count"])
    box_penalty = added_boxes / max(1.0, base_stats["pred_count"])
    action_cost = 0.012 * action_count
    stop_penalty = 0.012 if used_stop else 0.0
    return (
        0.40 * (stats["recall"] - base_stats["recall"])
        + 0.22 * (stats["recall75"] - base_stats["recall75"])
        + 0.45 * (stats["small_recall"] - base_stats["small_recall"])
        + 0.15 * (stats["precision"] - base_stats["precision"])
        - 0.010 * box_penalty
        - action_cost
        - stop_penalty
    )


def continuous_ap(ranked: Sequence[tuple[float, int]], gt_count: int) -> float:
    if not ranked:
        return 0.0
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
    return ap


def evaluate_split(split: str, pred_dir: Path) -> dict[str, float]:
    stems = read_split_stems(split)
    total_gt = total_pred = tp05 = tp75 = 0
    cls_gt_count = {class_id: 0 for class_id in CLASS_IDS}
    gt_cache = {stem: read_gt(split, stem) for stem in stems}
    pred_cache = {stem: read_pred_file(pred_dir / f"{stem}.txt") for stem in stems}
    for stem in stems:
        gt_boxes = gt_cache[stem]
        pred_boxes = pred_cache[stem]
        total_gt += len(gt_boxes)
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
                    tp75 += 1
    aps: list[float] = []
    for class_id in CLASS_IDS:
        gt_count = cls_gt_count[class_id]
        if gt_count == 0:
            continue
        ranked: list[tuple[float, int]] = []
        for stem in stems:
            gt_boxes = [gt for gt in gt_cache[stem] if gt.class_id == class_id]
            pred_boxes = [pred for pred in pred_cache[stem] if pred.class_id == class_id]
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
        aps.append(continuous_ap(ranked, gt_count))
    return {
        "file_count": float(len(stems)),
        "gt_box_count": float(total_gt),
        "prediction_box_count": float(total_pred),
        "tp_05": float(tp05),
        "tp_075": float(tp75),
        "acc_05": tp05 / total_gt if total_gt else 0.0,
        "acc_075": tp75 / total_gt if total_gt else 0.0,
        "map_05": sum(aps) / len(aps) if aps else 0.0,
    }


def write_predictions(out_dir: Path, predictions: dict[str, Sequence[Det]]) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, boxes in predictions.items():
        with (out_dir / f"{stem}.txt").open("w", encoding="utf-8") as f:
            for box in sorted(boxes, key=lambda item: item.score, reverse=True):
                f.write(f"{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {box.score:.6f}\n")
