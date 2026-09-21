#!/usr/bin/env python3
"""RL policy for inference-stage closed-loop candidate expansion.

This script is self-contained and does not import/call earlier experiment
scripts. It trains a lightweight policy network from validation rewards and uses
the learned policy at test time to choose a second-round feedback action.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
import torch.nn as nn
from torch.distributions import Categorical


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "RL_loop/outputs/closed_loop_rl"
PRED_ROOT = OUT_ROOT / "predictions"
CKPT_PATH = OUT_ROOT / "rl_policy_best.pt"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"

DATA_ROOT = ROOT / "dataset/VisDroneSplit1000Guarded"
SPLIT_MANIFEST = ROOT / "experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv"
CAND_ROOT = ROOT / "RL_loop/outputs/query_prompt_ablation/predictions"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
LR = 0.001
ENTROPY_COEF = 0.01
SMALL_CLASSES = {1, 2, 3, 7, 8, 10}
CLASS_IDS = list(range(1, 11))
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

ACTION_NAMES = [
    "STOP",
    "USE_QUERY_ONLY",
    "ADD_QUERY_ALIAS",
    "ADD_QUERY_ALIAS_FALLBACK",
]
ACTION_TO_VARIANT = {
    0: "fixed10",
    1: "query_only",
    2: "query_alias",
    3: "query_alias_fallback",
}


@dataclass
class Det:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x), self.value(x).squeeze(1)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device)


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
    path = DATA_ROOT / f"VisDrone2019-DET-{split}" / "annotations" / f"{stem}.txt"
    width, height = image_size(split, stem)
    boxes: list[Det] = []
    if not path.exists():
        return boxes
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


def read_pred(split: str, variant: str, stem: str) -> list[Det]:
    path = CAND_ROOT / split / variant / f"{stem}.txt"
    if not path.exists():
        # Backward-compatible layout from the first ablation run.
        path = CAND_ROOT / variant / f"{stem}.txt"
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


def deduplicate(boxes: Sequence[Det], iou_thr: float = 0.92) -> list[Det]:
    selected: list[Det] = []
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        if all(box.class_id != kept.class_id or iou(box, kept) < iou_thr for kept in selected):
            selected.append(box)
    return selected


def merge_action_pool(split: str, stem: str, action: int) -> list[Det]:
    base = read_pred(split, "fixed10", stem)
    if action == 0:
        return deduplicate(base)
    extra = read_pred(split, ACTION_TO_VARIANT[action], stem)
    return deduplicate(base + extra)


def state_from_boxes(boxes: Sequence[Det], width: int, height: int) -> list[float]:
    total = len(boxes)
    scores = [box.score for box in boxes]
    areas = [max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1) / max(1.0, width * height) for box in boxes]
    per_class_count = []
    per_class_top = []
    per_class_area = []
    for class_id in CLASS_IDS:
        cls_boxes = [box for box in boxes if box.class_id == class_id]
        per_class_count.append(min(len(cls_boxes) / 80.0, 1.0))
        per_class_top.append(max([box.score for box in cls_boxes] or [0.0]))
        per_class_area.append(sum([areas[idx] for idx, box in enumerate(boxes) if box.class_id == class_id]) / max(1, len(cls_boxes)))
    small_count = sum(1 for box in boxes if box.class_id in SMALL_CLASSES)
    mean_score = sum(scores) / max(1, total)
    score_std = math.sqrt(sum((s - mean_score) ** 2 for s in scores) / max(1, total))
    mean_area = sum(areas) / max(1, total)
    small_ratio = small_count / max(1, total)
    return [
        min(total / 300.0, 1.0),
        mean_score,
        max(scores or [0.0]),
        score_std,
        mean_area,
        small_ratio,
        *per_class_count,
        *per_class_top,
        *per_class_area,
        0.0,  # round id: Round 1 only in this contextual-bandit policy.
        1.0,  # remaining budget before feedback action.
    ]


def match_stats(pred_boxes: Sequence[Det], gt_boxes: Sequence[Det]) -> dict[str, float]:
    used = [False] * len(gt_boxes)
    tp = 0
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
            if gt_boxes[best_idx].class_id in SMALL_CLASSES:
                small_tp += 1
    gt_count = len(gt_boxes)
    small_gt = sum(1 for gt in gt_boxes if gt.class_id in SMALL_CLASSES)
    pred_count = len(pred_boxes)
    return {
        "recall": tp / gt_count if gt_count else 0.0,
        "precision": tp / pred_count if pred_count else 0.0,
        "small_recall": small_tp / small_gt if small_gt else 0.0,
        "pred_count": float(pred_count),
        "tp": float(tp),
    }


def reward_for_action(split: str, stem: str, action: int, base_stats: dict[str, float], gt_boxes: Sequence[Det]) -> float:
    pred = merge_action_pool(split, stem, action)
    stats = match_stats(pred, gt_boxes)
    delta_recall = stats["recall"] - base_stats["recall"]
    delta_small = stats["small_recall"] - base_stats["small_recall"]
    delta_precision = stats["precision"] - base_stats["precision"]
    added_boxes = max(0.0, stats["pred_count"] - base_stats["pred_count"])
    box_penalty = added_boxes / max(base_stats["pred_count"], 1.0)
    action_cost = [0.0, 0.08, 0.12, 0.18][action]
    return 0.50 * delta_recall + 0.25 * delta_small + 0.25 * delta_precision - 0.05 * box_penalty - 0.02 * action_cost


def build_rl_dataset(split: str, stems: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    states: list[list[float]] = []
    rewards: list[list[float]] = []
    action_counts = {name: 0 for name in ACTION_NAMES}
    for stem in stems:
        width, height = image_size(split, stem)
        base = merge_action_pool(split, stem, 0)
        gt_boxes = read_gt(split, stem)
        base_stats = match_stats(base, gt_boxes)
        state = state_from_boxes(base, width, height)
        reward_row = [reward_for_action(split, stem, action, base_stats, gt_boxes) for action in range(len(ACTION_NAMES))]
        states.append(state)
        rewards.append(reward_row)
        best_action = max(range(len(reward_row)), key=lambda idx: reward_row[idx])
        action_counts[ACTION_NAMES[best_action]] += 1
    meta = {"state_dim": len(states[0]) if states else 0, "action_counts_oracle": action_counts}
    return torch.tensor(states, dtype=torch.float32), torch.tensor(rewards, dtype=torch.float32), meta


def train_policy(device: torch.device, epochs: int, seed: int) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    stems = read_split_stems("val")
    states, reward_table, meta = build_rl_dataset("val", stems)
    states = states.to(device)
    reward_table = reward_table.to(device)
    policy = PolicyNet(states.shape[1], len(ACTION_NAMES)).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    best_objective = -1e9
    best_info: dict[str, float] = {}
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        logits, values = policy(states)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        rewards = reward_table[torch.arange(states.shape[0], device=device), actions]
        advantage = rewards - values.detach()
        policy_loss = -(dist.log_prob(actions) * advantage).mean()
        value_loss = nn.functional.mse_loss(values, rewards)
        entropy_loss = -dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss + ENTROPY_COEF * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        optimizer.step()
        with torch.no_grad():
            greedy = torch.argmax(policy(states)[0], dim=1)
            greedy_rewards = reward_table[torch.arange(states.shape[0], device=device), greedy]
            objective = float(greedy_rewards.mean().item())
        if objective > best_objective:
            best_objective = objective
            best_info = {"epoch": float(epoch), "val_mean_reward": objective, "loss": float(loss.item())}
            torch.save(
                {
                    "model": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
                    "state_dim": states.shape[1],
                    "action_names": ACTION_NAMES,
                    "best": best_info,
                },
                CKPT_PATH,
            )
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"[train] epoch={epoch:03d}/{epochs} loss={loss.item():.6f} greedy_reward={objective:.6f}")
    return {"split": "val", "image_count": len(stems), "best": best_info, **meta}


def load_policy(device: torch.device) -> PolicyNet:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    policy = PolicyNet(int(ckpt["state_dim"]), len(ckpt["action_names"])).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy


def write_predictions(path: Path, boxes: Sequence[Det]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for box in sorted(boxes, key=lambda item: item.score, reverse=True):
            f.write(f"{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {box.score:.6f}\n")


def run_closed_loop_test(device: torch.device) -> dict[str, object]:
    if not CKPT_PATH.exists():
        raise RuntimeError(f"Missing policy checkpoint: {CKPT_PATH}. Run --mode train first.")
    policy = load_policy(device)
    stems = read_split_stems("test")
    if PRED_ROOT.exists():
        for old in PRED_ROOT.glob("*.txt"):
            old.unlink()
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    action_counts = {name: 0 for name in ACTION_NAMES}
    trace = []
    for stem in stems:
        width, height = image_size("test", stem)
        round1 = merge_action_pool("test", stem, 0)
        state = torch.tensor([state_from_boxes(round1, width, height)], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = policy(state)
            action = int(torch.argmax(logits, dim=1).item())
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0).tolist()
        round2 = merge_action_pool("test", stem, action)
        write_predictions(PRED_ROOT / f"{stem}.txt", round2)
        action_counts[ACTION_NAMES[action]] += 1
        trace.append({"stem": stem, "action": ACTION_NAMES[action], "action_id": action, "probs": probs, "round1_count": len(round1), "round2_count": len(round2)})
    metrics = evaluate_split("test", PRED_ROOT)
    (OUT_ROOT / "closed_loop_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"split": "test", "image_count": len(stems), "action_counts": action_counts, "metrics": metrics}


def evaluate_split(split: str, pred_dir: Path) -> dict[str, float]:
    stems = read_split_stems(split)
    total_gt = total_pred = tp05 = tp075 = 0
    cls_gt_count = {class_id: 0 for class_id in CLASS_IDS}
    gt_cache = {stem: read_gt(split, stem) for stem in stems}
    pred_cache = {stem: read_pred_from_dir(pred_dir, stem) for stem in stems}
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
                    tp075 += 1
    aps = []
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
        "gt_box_count": float(total_gt),
        "prediction_box_count": float(total_pred),
        "tp_05": float(tp05),
        "tp_075": float(tp075),
        "acc_05": tp05 / total_gt if total_gt else 0.0,
        "acc_075": tp075 / total_gt if total_gt else 0.0,
        "map_05": sum(aps) / len(aps) if aps else 0.0,
    }


def read_pred_from_dir(pred_dir: Path, stem: str) -> list[Det]:
    path = pred_dir / f"{stem}.txt"
    boxes = []
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


def require_candidates(split: str) -> None:
    missing = []
    for variant in ACTION_TO_VARIANT.values():
        path = CAND_ROOT / split / variant
        fallback_path = CAND_ROOT / variant
        if not path.exists() and not fallback_path.exists():
            missing.append(str(path))
    if missing:
        raise RuntimeError("Missing candidate prediction directories. Run query_prompt_ablation.py for this split first:\n  " + "\n  ".join(missing))


def write_summary(train_info: dict[str, object] | None, test_info: dict[str, object] | None) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"train": train_info, "test": test_info, "checkpoint": str(CKPT_PATH.relative_to(ROOT))}
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Closed-loop RL Summary", "", f"- checkpoint: `{CKPT_PATH.relative_to(ROOT)}`"]
    if train_info:
        lines.extend([
            "",
            "## Train",
            f"- split: `{train_info['split']}`",
            f"- image_count: `{train_info['image_count']}`",
            f"- best: `{train_info['best']}`",
            f"- oracle_action_counts: `{train_info['action_counts_oracle']}`",
        ])
    if test_info:
        m = test_info["metrics"]
        lines.extend([
            "",
            "## Test",
            f"- action_counts: `{test_info['action_counts']}`",
            "",
            "| Method | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| closed_loop_rl | {int(m['prediction_box_count'])} | {m['acc_05']:.4f} | {m['acc_075']:.4f} | {m['map_05']:.4f} |",
        ])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test RL-driven inference-stage closed loop.")
    parser.add_argument("--mode", choices=["train", "test", "all"], default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[closed_loop_rl] device={device}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_info = None
    test_info = None
    if args.mode in {"train", "all"}:
        require_candidates("val")
        train_info = train_policy(device, args.epochs, args.seed)
    if args.mode in {"test", "all"}:
        require_candidates("test")
        test_info = run_closed_loop_test(device)
        m = test_info["metrics"]
        print(f"[test] acc05={m['acc_05']:.4f} acc075={m['acc_075']:.4f} map05={m['map_05']:.4f} pred={int(m['prediction_box_count'])}")
    write_summary(train_info, test_info)
    print(f"[closed_loop_rl] summary={SUMMARY_MD}")


if __name__ == "__main__":
    main()
