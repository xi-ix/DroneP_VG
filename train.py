#!/usr/bin/env python3
"""Self-contained training entry for the latest DroneP_VG scheme.

This script does not import or call any earlier experiment script. It trains the
latest feedback-calibration heads from fixed candidate files:

- Stage 1: alias hard-negative reranker, trained from Exp35 predictions.
- Stage 2: rich metadata fusion calibrator, trained from Exp38 online metadata
  and the stage-1 score.
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
from typing import Iterable, Sequence

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs/latest_train"
CKPT_DIR = OUT_ROOT / "checkpoints"
PRED_DIR = OUT_ROOT / "predictions"
SUMMARY_JSON = OUT_ROOT / "train_summary.json"

DATA_ROOT = ROOT / "dataset/VisDroneSplit1000Guarded"
SPLIT_MANIFEST = ROOT / "experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv"
GT_ROOT = ROOT / "experiment/exp30_real_query_focal_rerank_20260711/log/gt_xyxy"
EXP35_PRED_ROOT = ROOT / "experiment/exp35_online_candidate_compression_20260711/log/predictions"
META_ROOT = ROOT / "experiment/exp38_online_rich_metadata_fusion_20260712/log/metadata"
BASE_PRED_ROOT = ROOT / "experiment/exp38_online_rich_metadata_fusion_20260712/log/predictions"

ALIAS_CKPT = CKPT_DIR / "alias_reranker_best.pt"
FUSION_CKPT = CKPT_DIR / "rich_fusion_best.pt"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALIAS_EPOCHS = 40
FUSION_EPOCHS = 60
BATCH_SIZE = 4096
ALIAS_LR = 0.001
FUSION_LR = 0.0007

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
class Box:
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


def read_split_map() -> dict[str, list[str]]:
    split_map = {"train": [], "val": [], "test": []}
    with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["split"] in split_map:
                split_map[row["split"]].append(row["stem"])
    return {split: sorted(stems) for split, stems in split_map.items()}


def image_path(split: str, stem: str) -> Path:
    return DATA_ROOT / f"VisDrone2019-DET-{split}" / "images" / f"{stem}.jpg"


def image_size(split: str, stem: str) -> tuple[int, int]:
    with Image.open(image_path(split, stem)) as image:
        return image.size


def read_boxes(path: Path, is_gt: bool = False) -> list[Box]:
    boxes: list[Box] = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if is_gt and len(parts) != 5:
            continue
        if not is_gt and len(parts) != 6:
            continue
        class_id = int(float(parts[0]))
        x1, y1, x2, y2 = [float(value) for value in parts[1:5]]
        score = 1.0 if is_gt else float(parts[5])
        if x2 > x1 and y2 > y1 and class_id in CLASS_NAMES:
            boxes.append(Box(class_id, x1, y1, x2, y2, score))
    return boxes


def read_metadata(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / max(area_a + area_b - inter, 1e-9)


def best_iou(box: Box, gt_boxes: Sequence[Box], same_class_only: bool) -> float:
    best = 0.0
    for gt in gt_boxes:
        if same_class_only and gt.class_id != box.class_id:
            continue
        best = max(best, iou(box, gt))
    return best


def confusion_iou(box: Box, gt_boxes: Sequence[Box]) -> float:
    confusable = CONFUSION_GROUPS.get(box.class_id, set())
    return max([iou(box, gt) for gt in gt_boxes if gt.class_id in confusable] or [0.0])


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


def alias_box_features(box: Box, width: int, height: int, gt_boxes: Sequence[Box] = ()) -> list[float]:
    bw = max(1.0, box.x2 - box.x1)
    bh = max(1.0, box.y2 - box.y1)
    area = (bw * bh) / max(1.0, width * height)
    prior = AREA_PRIOR.get(box.class_id, 0.01)
    scale_fit = math.exp(-abs(math.log(max(area, 1e-8) / max(prior, 1e-8))))
    score = min(max(box.score, 1e-6), 1.0 - 1e-6)
    base = [
        score,
        math.log(score / (1.0 - score)),
        math.sqrt(area),
        math.log(max(area, 1e-8)),
        bw / width,
        bh / height,
        math.log(max(bw / bh, 1e-6)),
        ((box.x1 + box.x2) / 2.0) / width,
        ((box.y1 + box.y2) / 2.0) / height,
        box.class_id / 10.0,
        scale_fit,
        min(1.0, area / max(prior, 1e-8)),
    ]
    one_hot = [1.0 if box.class_id == class_id else 0.0 for class_id in range(1, 11)]
    overlap = [best_iou(box, gt_boxes, same_class_only=False) if gt_boxes else 0.0, confusion_iou(box, gt_boxes) if gt_boxes else 0.0]
    return base + hierarchy_features(box.class_id) + one_hot + overlap


def alias_label_and_weight(box: Box, gt_boxes: Sequence[Box]) -> tuple[float, float]:
    same_iou = best_iou(box, gt_boxes, same_class_only=True)
    conf_iou = confusion_iou(box, gt_boxes)
    label = 1.0 if same_iou >= 0.5 else 0.0
    weight = 1.0
    if box.class_id in SMALL_CLASSES:
        weight *= 1.8
    if box.class_id in WEAK_SMALL_CLASSES:
        weight *= 2.2
    if label > 0.5:
        weight *= 2.0
    elif box.score >= 0.30 or conf_iou >= 0.30:
        weight *= 2.5
    return label, weight


def build_alias_dataset(split: str, stems: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    labels: list[float] = []
    weights: list[float] = []
    for stem in stems:
        width, height = image_size(split, stem)
        pred_boxes = read_boxes(EXP35_PRED_ROOT / split / f"{stem}.txt")
        gt_boxes = read_boxes(GT_ROOT / split / f"{stem}.txt", is_gt=True)
        for box in pred_boxes:
            label, weight = alias_label_and_weight(box, gt_boxes)
            features.append(alias_box_features(box, width, height, ()))
            labels.append(label)
            weights.append(weight)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)


def train_binary_model(
    model: nn.Module,
    ckpt_path: Path,
    x_train_raw: torch.Tensor,
    y_train: torch.Tensor,
    w_train: torch.Tensor,
    x_hold_raw: torch.Tensor,
    y_hold: torch.Tensor,
    epochs: int,
    lr: float,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict[str, float]]:
    mu = x_train_raw.mean(dim=0, keepdim=True)
    sigma = x_train_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train = (x_train_raw - mu) / sigma
    x_hold = (x_hold_raw - mu) / sigma
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_train, y_train, w_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
    best: dict[str, float] = {"epoch": 0, "hold_loss": float("inf")}
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb, wb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss_raw = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (loss_raw * wb).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            hold_loss = float(nn.functional.binary_cross_entropy_with_logits(model(x_hold.to(DEVICE)).cpu(), y_hold).item())
        if hold_loss < best["hold_loss"]:
            best = {"epoch": float(epoch), "hold_loss": hold_loss, "train_loss": sum(losses) / max(1, len(losses))}
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "mu": mu.cpu(), "sigma": sigma.cpu(), "best": best}, ckpt_path)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[train.py] epoch={epoch:03d}/{epochs} train_loss={sum(losses)/max(1, len(losses)):.6f} hold_loss={hold_loss:.6f}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model, ckpt["mu"].float(), ckpt["sigma"].float(), ckpt["best"]


def apply_alias_model(model: nn.Module, mu: torch.Tensor, sigma: torch.Tensor, rows: Sequence[dict[str, object]]) -> list[float]:
    boxes = []
    for row in rows:
        x1, y1, x2, y2 = [float(v) for v in row["box_xyxy"]]
        boxes.append(Box(int(row["class_id"]), x1, y1, x2, y2, float(row["final_score"])))
    features = [alias_box_features(box, int(row["image_width"]), int(row["image_height"]), ()) for box, row in zip(boxes, rows)]
    if not features:
        return []
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        x = ((x - mu.cpu()) / sigma.cpu()).to(DEVICE)
        return torch.sigmoid(model(x)).cpu().tolist()


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


def rich_label_and_weight(row: dict[str, object], gt_boxes: Sequence[Box]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in row["box_xyxy"]]
    box = Box(int(row["class_id"]), x1, y1, x2, y2, float(row["final_score"]))
    same_iou = best_iou(box, gt_boxes, same_class_only=True)
    conf_iou = confusion_iou(box, gt_boxes)
    label = 1.0 if same_iou >= 0.5 else 0.0
    weight = 1.0
    if box.class_id in SMALL_CLASSES:
        weight *= 1.8
    if box.class_id in WEAK_SMALL_CLASSES:
        weight *= 2.4
    if label > 0.5:
        weight *= 2.0
    elif float(row.get("final_score", 0.0)) >= 0.30 or conf_iou >= 0.30:
        weight *= 3.0
    if row.get("source_type") == "alias":
        weight *= 1.2
    return label, weight


def build_rich_dataset(split: str, stems: Sequence[str], alias_model: nn.Module, alias_mu: torch.Tensor, alias_sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    labels: list[float] = []
    weights: list[float] = []
    for stem in stems:
        rows = read_metadata(META_ROOT / split / f"{stem}.jsonl")
        gt_boxes = read_boxes(GT_ROOT / split / f"{stem}.txt", is_gt=True)
        alias_scores = apply_alias_model(alias_model, alias_mu, alias_sigma, rows)
        for row, alias_score in zip(rows, alias_scores):
            label, weight = rich_label_and_weight(row, gt_boxes)
            features.append(rich_features(row, alias_score))
            labels.append(label)
            weights.append(weight)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)


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
        for cls, box, score in sorted(pred_objs, key=lambda item: item[2], reverse=True):
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
            pred_boxes = []
            if pred_path.exists():
                for line in pred_path.read_text(encoding="utf-8").splitlines():
                    parsed = parse_line(line, True)
                    if parsed is not None:
                        pred_boxes.append(parsed)
            used = [False] * len(gt_boxes)
            for _, box, score in sorted([p for p in pred_boxes if p[0] == cls], key=lambda item: item[2], reverse=True):
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


def write_rich_predictions(model: nn.Module, mu: torch.Tensor, sigma: torch.Tensor, split: str, stems: Sequence[str], alias_model: nn.Module, alias_mu: torch.Tensor, alias_sigma: torch.Tensor, alpha: float, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        rows = read_metadata(META_ROOT / split / f"{stem}.jsonl")
        alias_scores = apply_alias_model(alias_model, alias_mu, alias_sigma, rows)
        features = [rich_features(row, alias_score) for row, alias_score in zip(rows, alias_scores)]
        if features:
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32)
                x = ((x - mu.cpu()) / sigma.cpu()).to(DEVICE)
                fusion_scores = torch.sigmoid(model(x)).cpu().tolist()
        else:
            fusion_scores = []
        with (out_dir / f"{stem}.txt").open("w", encoding="utf-8") as f:
            for row, alias_score, fusion_score in zip(rows, alias_scores, fusion_scores):
                base_blend = 0.5 * float(row["final_score"]) + 0.5 * alias_score
                score = alpha * base_blend + (1.0 - alpha) * float(fusion_score)
                x1, y1, x2, y2 = [float(v) for v in row["box_xyxy"]]
                f.write(f"{int(row['class_id'])} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.6f}\n")


def subset_gt(split: str, stems: Sequence[str], name: str) -> Path:
    out_dir = OUT_ROOT / "gt_subset" / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = GT_ROOT / split / f"{stem}.txt"
        if src.exists():
            (out_dir / f"{stem}.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return out_dir


def require_inputs() -> None:
    required = [
        SPLIT_MANIFEST,
        GT_ROOT / "val",
        GT_ROOT / "test",
        EXP35_PRED_ROOT / "val",
        EXP35_PRED_ROOT / "test",
        META_ROOT / "val",
        META_ROOT / "test",
        BASE_PRED_ROOT / "test",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing required fixed data/artifacts:\n  " + "\n  ".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-contained latest DroneP_VG training.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alias-epochs", type=int, default=ALIAS_EPOCHS)
    parser.add_argument("--fusion-epochs", type=int, default=FUSION_EPOCHS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    require_inputs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    split_map = read_split_map()
    val_stems = split_map["val"]
    test_stems = split_map["test"]
    random.Random(args.seed).shuffle(val_stems)
    holdout_count = max(20, int(len(val_stems) * 0.2))
    hold_stems = sorted(val_stems[:holdout_count])
    train_stems = sorted(val_stems[holdout_count:])

    print(f"[train.py] device={DEVICE} val_train={len(train_stems)} val_holdout={len(hold_stems)} test={len(test_stems)}")
    x_alias_train, y_alias_train, w_alias_train = build_alias_dataset("val", train_stems)
    x_alias_hold, y_alias_hold, _ = build_alias_dataset("val", hold_stems)
    alias_model = AliasReranker(input_dim=x_alias_train.shape[1])
    alias_model, alias_mu, alias_sigma, alias_best = train_binary_model(alias_model, ALIAS_CKPT, x_alias_train, y_alias_train, w_alias_train, x_alias_hold, y_alias_hold, args.alias_epochs, ALIAS_LR)

    x_rich_train, y_rich_train, w_rich_train = build_rich_dataset("val", train_stems, alias_model, alias_mu, alias_sigma)
    x_rich_hold, y_rich_hold, _ = build_rich_dataset("val", hold_stems, alias_model, alias_mu, alias_sigma)
    fusion_model = RichFusionModel(input_dim=x_rich_train.shape[1])
    fusion_model, fusion_mu, fusion_sigma, fusion_best = train_binary_model(fusion_model, FUSION_CKPT, x_rich_train, y_rich_train, w_rich_train, x_rich_hold, y_rich_hold, args.fusion_epochs, FUSION_LR)

    hold_gt = subset_gt("val", hold_stems, "val_holdout")
    alpha_rows = []
    best_alpha = 1.0
    best_metrics: dict[str, float] | None = None
    for alpha in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]:
        out_dir = PRED_DIR / "val_holdout_search" / f"alpha_{alpha:.2f}"
        write_rich_predictions(fusion_model, fusion_mu, fusion_sigma, "val", hold_stems, alias_model, alias_mu, alias_sigma, alpha, out_dir)
        metrics = evaluate_detections(hold_gt, out_dir)
        objective = 0.70 * metrics["map_05"] + 0.30 * metrics["acc_05"]
        row = {"alpha": alpha, "objective": objective, **metrics}
        alpha_rows.append(row)
        print(f"[train.py] alpha={alpha:.2f} acc05={metrics['acc_05']:.4f} map05={metrics['map_05']:.4f} objective={objective:.4f}")
        if best_metrics is None or objective > 0.70 * best_metrics["map_05"] + 0.30 * best_metrics["acc_05"]:
            best_alpha = alpha
            best_metrics = metrics

    test_out = PRED_DIR / "test"
    write_rich_predictions(fusion_model, fusion_mu, fusion_sigma, "test", test_stems, alias_model, alias_mu, alias_sigma, best_alpha, test_out)
    test_metrics = evaluate_detections(GT_ROOT / "test", test_out)
    base_metrics = evaluate_detections(GT_ROOT / "test", BASE_PRED_ROOT / "test")

    summary = {
        "seed": args.seed,
        "device": str(DEVICE),
        "train_split": {"val_train_count": len(train_stems), "val_holdout_count": len(hold_stems), "test_count": len(test_stems)},
        "checkpoints": {"alias": str(ALIAS_CKPT.relative_to(ROOT)), "fusion": str(FUSION_CKPT.relative_to(ROOT))},
        "config": {"alias_epochs": args.alias_epochs, "fusion_epochs": args.fusion_epochs, "alias_lr": ALIAS_LR, "fusion_lr": FUSION_LR, "best_alpha": best_alpha},
        "alias_best": alias_best,
        "fusion_best": fusion_best,
        "alpha_search": alpha_rows,
        "base_metadata_test_metrics": base_metrics,
        "test_metrics": test_metrics,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[train.py] Done. best_alpha={best_alpha:.2f} test_acc05={test_metrics['acc_05']:.4f} test_acc075={test_metrics['acc_075']:.4f} test_map05={test_metrics['map_05']:.4f}")
    print(f"[train.py] Summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
