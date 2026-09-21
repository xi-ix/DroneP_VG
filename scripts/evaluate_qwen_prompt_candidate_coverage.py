#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RL_ABLATION = ROOT / "RL_loop/outputs/query_prompt_ablation/predictions"
LLM_DIR = ROOT / "38+RL_loop/outputs/llm_dynamic"
PROMPT_CACHE = LLM_DIR / "llm_prompt_pools.json"
OUT_DIR = ROOT / "outputs/qwen_prompt_candidate_coverage"
SPLIT_MANIFEST = ROOT / "experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv"
GT_XYXY_ROOT = ROOT / "experiment/exp30_real_query_focal_rerank_20260711/log/gt_xyxy"
DATA_ROOT = ROOT / "dataset/VisDroneSplit1000Guarded"
GROUNDINGDINO_ROOT = Path("/home/wangzhe/GroundingDINO")
GDINO_CONFIG = GROUNDINGDINO_ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CHECKPOINT = GROUNDINGDINO_ROOT / "weights/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.03
TEXT_THRESHOLD = 0.15
DEDUP_IOU = 0.92
COVER_IOU = 0.5

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


def load_llm_dynamic_module():
    module_path = ROOT / "38+RL_loop/llm_dynamic.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("llm_dynamic_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_split_stems(split: str) -> list[str]:
    stems: list[str] = []
    with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["split"] == split:
                stems.append(row["stem"])
    return sorted(stems)


def read_gt(split: str, stem: str) -> list[Det]:
    path = GT_XYXY_ROOT / split / f"{stem}.txt"
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


def iou(a: Det, b: Det) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / max(area_a + area_b - inter, 1e-9)


def deduplicate(boxes: Sequence[Det], iou_thr: float = DEDUP_IOU) -> list[Det]:
    selected: list[Det] = []
    by_class: dict[int, list[Det]] = {}
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        same = by_class.setdefault(box.class_id, [])
        if all(iou(box, kept) < iou_thr for kept in same):
            selected.append(box)
            same.append(box)
    return selected


def evaluate_coverage(stems: Sequence[str], pred_root: Path) -> dict[str, float]:
    total_gt = 0
    hit_gt = 0
    total_candidates = 0
    file_count = 0
    for stem in stems:
        gt_boxes = read_gt("test", stem)
        pred_boxes = read_pred_file(pred_root / f"{stem}.txt")
        file_count += int((pred_root / f"{stem}.txt").exists())
        total_candidates += len(pred_boxes)
        for gt in gt_boxes:
            total_gt += 1
            if any(pred.class_id == gt.class_id and iou(pred, gt) >= COVER_IOU for pred in pred_boxes):
                hit_gt += 1
    return {
        "file_count": float(file_count),
        "candidate_count": float(total_candidates),
        "gt_count": float(total_gt),
        "covered_gt": float(hit_gt),
        "coverage": hit_gt / total_gt if total_gt else 0.0,
    }


def evaluate_union_coverage(stems: Sequence[str], pred_roots: Sequence[Path]) -> dict[str, float]:
    total_gt = 0
    hit_gt = 0
    total_candidates = 0
    file_count = 0
    for stem in stems:
        gt_boxes = read_gt("test", stem)
        pred_boxes: list[Det] = []
        existing_files = 0
        for pred_root in pred_roots:
            path = pred_root / f"{stem}.txt"
            existing_files += int(path.exists())
            pred_boxes.extend(read_pred_file(path))
        pred_boxes = deduplicate(pred_boxes)
        file_count += int(existing_files > 0)
        total_candidates += len(pred_boxes)
        for gt in gt_boxes:
            total_gt += 1
            if any(pred.class_id == gt.class_id and iou(pred, gt) >= COVER_IOU for pred in pred_boxes):
                hit_gt += 1
    return {
        "file_count": float(file_count),
        "candidate_count": float(total_candidates),
        "gt_count": float(total_gt),
        "covered_gt": float(hit_gt),
        "coverage": hit_gt / total_gt if total_gt else 0.0,
    }


def image_path(stem: str) -> Path:
    return DATA_ROOT / "VisDrone2019-DET-test/images" / f"{stem}.jpg"


def load_groundingdino_model(device: str):
    if str(GROUNDINGDINO_ROOT) not in sys.path:
        sys.path.insert(0, str(GROUNDINGDINO_ROOT))
    from groundingdino.util.inference import Model

    return Model(
        model_config_path=str(GDINO_CONFIG),
        model_checkpoint_path=str(GDINO_CHECKPOINT),
        device=device,
    )


def gdino_predict(model, image_bgr: np.ndarray, prompts: Sequence[str]):
    detections = model.predict_with_classes(
        image=image_bgr,
        classes=list(prompts),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    return np.asarray(detections.xyxy), np.asarray(detections.confidence), np.asarray(detections.class_id, dtype=object)


def write_predictions(path: Path, boxes: Iterable[Det]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for box in sorted(boxes, key=lambda item: item.score, reverse=True):
            f.write(f"{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {box.score:.6f}\n")


def qwen_actions_by_stem() -> dict[str, list[object]]:
    llm_dynamic = load_llm_dynamic_module()
    cache = json.loads(PROMPT_CACHE.read_text(encoding="utf-8"))
    out: dict[str, list[object]] = {}
    for stem in read_split_stems("test"):
        _, actions = llm_dynamic.candidate_actions(cache, stem)
        out[stem] = actions
    return out


def generate_qwen_predictions(stems: Sequence[str], mode: str, device: str, force: bool) -> Path:
    pred_root = OUT_DIR / f"qwen_prompts_{mode}"
    done = all((pred_root / f"{stem}.txt").exists() for stem in stems)
    if done and not force:
        return pred_root
    model = load_groundingdino_model(device)
    actions = qwen_actions_by_stem()
    for idx, stem in enumerate(stems, start=1):
        image_bgr = cv2.imread(str(image_path(stem)))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path(stem)}")
        boxes: list[Det] = []
        prompt_actions = actions.get(stem, [])
        if mode == "one_batch":
            prompts = [action.text for action in prompt_actions]
            if prompts:
                xyxy, scores, raw_ids = gdino_predict(model, image_bgr, prompts)
                for box, score, raw_id in zip(xyxy, scores, raw_ids):
                    if raw_id is None:
                        continue
                    prompt_idx = int(raw_id)
                    if 0 <= prompt_idx < len(prompt_actions):
                        action = prompt_actions[prompt_idx]
                        boxes.append(Det(action.class_id, float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score)))
        elif mode == "split":
            for action in prompt_actions:
                xyxy, scores, _ = gdino_predict(model, image_bgr, [action.text])
                for box, score in zip(xyxy, scores):
                    boxes.append(Det(action.class_id, float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score)))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        boxes = [box for box in boxes if box.x2 > box.x1 and box.y2 > box.y1]
        write_predictions(pred_root / f"{stem}.txt", deduplicate(boxes))
        if idx == 1 or idx % 10 == 0 or idx == len(stems):
            print(f"[{mode}] {idx}/{len(stems)} stem={stem} boxes={len(boxes)} dedup={len(deduplicate(boxes))}", flush=True)
    return pred_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate count and GT coverage for query-only and Qwen prompt variants.")
    parser.add_argument("--limit", type=int, default=0, help="Use first N test images; 0 means full test split.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generate-qwen", action="store_true", help="Generate one-batch and split Qwen prompt candidates with GroundingDINO.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stems = read_split_stems("test")
    if args.limit:
        stems = stems[: args.limit]
    targets: dict[str, Path] = {
        "query_only": RL_ABLATION / "test/query_only",
        "query_alias": RL_ABLATION / "test/query_alias",
        "qwen_dynamic_rl": LLM_DIR / "predictions",
        "qwen_query_base": LLM_DIR / "predictions_query_base",
    }
    if args.generate_qwen:
        targets["qwen_prompts_one_batch"] = generate_qwen_predictions(stems, "one_batch", args.device, args.force)
        targets["qwen_prompts_split"] = generate_qwen_predictions(stems, "split", args.device, args.force)
    metrics = {name: evaluate_coverage(stems, path) for name, path in targets.items()}
    query_only_root = targets["query_only"]
    if "qwen_prompts_one_batch" in targets:
        metrics["query_only_union_qwen_prompts_one_batch"] = evaluate_union_coverage(
            stems,
            [query_only_root, targets["qwen_prompts_one_batch"]],
        )
    if "qwen_prompts_split" in targets:
        metrics["query_only_union_qwen_prompts_split"] = evaluate_union_coverage(
            stems,
            [query_only_root, targets["qwen_prompts_split"]],
        )
    summary = {
        "split": "test",
        "image_count": len(stems),
        "cover_iou": COVER_IOU,
        "dedup_iou": DEDUP_IOU,
        "prompt_cache": str(PROMPT_CACHE),
        "metrics": metrics,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"summary_limit{args.limit}.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[Done] {out_json}", flush=True)


if __name__ == "__main__":
    main()
