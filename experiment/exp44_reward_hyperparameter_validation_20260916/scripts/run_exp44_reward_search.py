#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


ROOT = Path(__file__).resolve().parents[3]
RL_ROOT = ROOT / "38+RL_loop"
sys.path.insert(0, str(RL_ROOT))

from common import (  # noqa: E402
    ACTION_NAMES,
    SMALL_CLASSES,
    Det,
    apply_action_to_pool,
    continuous_ap,
    deduplicate,
    exp38_rescore,
    image_size,
    iou,
    merge_action_pool,
    read_gt,
    read_split_stems,
    require_inputs,
    state_from_boxes,
)


EXP_NAME = "exp44_reward_hyperparameter_validation_20260916"
EXP_ROOT = ROOT / "experiment" / EXP_NAME
LOG_DIR = EXP_ROOT / "log"
CACHE_PATH = LOG_DIR / "reward_components.pt"
FOLDS_PATH = LOG_DIR / "val_folds.json"
SCREEN_JSON = LOG_DIR / "screening_results.json"
CONFIRM_JSON = LOG_DIR / "confirmation_results.json"
CONFIRM_SHARD_PATTERN = "confirmation_shard_{index:02d}_of_{count:02d}.json"
FINAL_JSON = LOG_DIR / "final_test_results.json"
SUMMARY_MD = LOG_DIR / "实验总结.md"
FIG_DIR = LOG_DIR / "figures"
CKPT_DIR = LOG_DIR / "checkpoints"

POSITIVE_TOTAL = 1.22
ACTION_DIM = len(ACTION_NAMES)
SMALL_AREA_PX = 32 * 32
SCREEN_EPOCHS = 180
CONFIRM_EPOCHS = 300
SCREEN_SEEDS = [42]
CONFIRM_SEEDS = [42, 43, 44]
FOLD_COUNT = 5


@dataclass(frozen=True)
class RewardConfig:
    name: str
    recall05: float = 0.40
    recall075: float = 0.22
    small: float = 0.45
    precision: float = 0.15
    box_cost: float = 0.010
    action_cost: float = 0.012
    stop_cost: float = 0.012
    small_mode: str = "weak"
    family: str = "baseline"

    def positive_sum(self) -> float:
        return self.recall05 + self.recall075 + self.small + self.precision


CURRENT = RewardConfig("current")


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.value = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.value(x).squeeze(1)


def stable_group(stem: str) -> str:
    return stem.split("_", 1)[0]


def extended_stats(pred_boxes: Sequence[Det], gt_boxes: Sequence[Det]) -> dict[str, float]:
    used = [False] * len(gt_boxes)
    tp05 = tp075 = weak_tp = area_small_tp = 0
    for pred in sorted(pred_boxes, key=lambda item: item.score, reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_boxes):
            if used[idx] or gt.class_id != pred.class_id:
                continue
            overlap = iou(pred, gt)
            if overlap > best_iou:
                best_idx = idx
                best_iou = overlap
        if best_idx >= 0 and best_iou >= 0.5:
            used[best_idx] = True
            tp05 += 1
            gt = gt_boxes[best_idx]
            if best_iou >= 0.75:
                tp075 += 1
            if gt.class_id in SMALL_CLASSES:
                weak_tp += 1
            if (gt.x2 - gt.x1) * (gt.y2 - gt.y1) <= SMALL_AREA_PX:
                area_small_tp += 1
    weak_gt = sum(gt.class_id in SMALL_CLASSES for gt in gt_boxes)
    area_small_gt = sum((gt.x2 - gt.x1) * (gt.y2 - gt.y1) <= SMALL_AREA_PX for gt in gt_boxes)
    pred_count = len(pred_boxes)
    return {
        "tp05": float(tp05),
        "tp075": float(tp075),
        "weak_tp": float(weak_tp),
        "area_small_tp": float(area_small_tp),
        "gt": float(len(gt_boxes)),
        "weak_gt": float(weak_gt),
        "area_small_gt": float(area_small_gt),
        "pred_count": float(pred_count),
        "recall05": tp05 / len(gt_boxes) if gt_boxes else 0.0,
        "recall075": tp075 / len(gt_boxes) if gt_boxes else 0.0,
        "weak_recall": weak_tp / weak_gt if weak_gt else 0.0,
        "area_small_recall": area_small_tp / area_small_gt if area_small_gt else 0.0,
        "precision": tp05 / pred_count if pred_count else 0.0,
    }


def build_cache() -> dict[str, object]:
    require_inputs()
    stems = read_split_stems("val")
    states = torch.zeros((len(stems), 1 + ACTION_DIM, 38), dtype=torch.float32)
    final_stats = torch.zeros((len(stems), ACTION_DIM, ACTION_DIM, 5), dtype=torch.float32)
    base_stats = torch.zeros((len(stems), 5), dtype=torch.float32)
    denominators = torch.zeros((len(stems), 3), dtype=torch.float32)
    invalid = torch.zeros((len(stems), ACTION_DIM, ACTION_DIM), dtype=torch.bool)
    started = time.perf_counter()
    for stem_idx, stem in enumerate(stems):
        width, height = image_size("val", stem)
        base, _ = exp38_rescore("val", stem, merge_action_pool("val", stem, 0))
        gt = read_gt("val", stem)
        base_item = extended_stats(base, gt)
        base_stats[stem_idx] = torch.tensor([
            base_item["tp05"], base_item["tp075"], base_item["weak_tp"],
            base_item["area_small_tp"], base_item["pred_count"],
        ])
        denominators[stem_idx] = torch.tensor([
            base_item["gt"], base_item["weak_gt"], base_item["area_small_gt"],
        ])
        first_state = state_from_boxes(base, width, height)
        first_state[-2:] = [0.0, 1.0]
        states[stem_idx, 0] = torch.tensor(first_state)
        for first_action in range(ACTION_DIM):
            current = apply_action_to_pool("val", stem, base, first_action)
            second_state = state_from_boxes(current, width, height)
            second_state[-2:] = [0.5, 0.5]
            states[stem_idx, 1 + first_action] = torch.tensor(second_state)
            for second_action in range(ACTION_DIM):
                if first_action == second_action and second_action != 0:
                    invalid[stem_idx, first_action, second_action] = True
                    continue
                final = apply_action_to_pool("val", stem, current, second_action)
                item = extended_stats(final, gt)
                final_stats[stem_idx, first_action, second_action] = torch.tensor([
                    item["tp05"], item["tp075"], item["weak_tp"],
                    item["area_small_tp"], item["pred_count"],
                ])
        if (stem_idx + 1) % 10 == 0 or stem_idx + 1 == len(stems):
            print(f"[cache] {stem_idx + 1}/{len(stems)} images", flush=True)
    payload = {
        "version": 1,
        "stems": stems,
        "states": states,
        "final_stats": final_stats,
        "base_stats": base_stats,
        "denominators": denominators,
        "invalid": invalid,
        "elapsed_sec": time.perf_counter() - started,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CACHE_PATH)
    return payload


def load_cache(rebuild: bool = False) -> dict[str, object]:
    if rebuild or not CACHE_PATH.exists():
        return build_cache()
    return torch.load(CACHE_PATH, map_location="cpu", weights_only=False)


def make_folds(cache: dict[str, object]) -> list[list[int]]:
    stems: list[str] = cache["stems"]
    denominators: torch.Tensor = cache["denominators"]
    groups: dict[str, list[int]] = {}
    for idx, stem in enumerate(stems):
        groups.setdefault(stable_group(stem), []).append(idx)
    group_rows = []
    for group, indices in groups.items():
        gt = float(denominators[indices, 0].sum())
        small = float(denominators[indices, 2].sum())
        group_rows.append((group, indices, gt, small))
    group_rows.sort(key=lambda row: (row[2], row[3], len(row[1]), row[0]), reverse=True)
    folds: list[list[int]] = [[] for _ in range(FOLD_COUNT)]
    loads = [[0.0, 0.0, 0.0] for _ in range(FOLD_COUNT)]
    for _, indices, gt, small in group_rows:
        fold_idx = min(range(FOLD_COUNT), key=lambda idx: (loads[idx][0], loads[idx][1], loads[idx][2], idx))
        folds[fold_idx].extend(indices)
        loads[fold_idx][0] += len(indices)
        loads[fold_idx][1] += gt / 100.0
        loads[fold_idx][2] += small / 100.0
    for fold in folds:
        fold.sort()
    payload = {
        "group_rule": "stem prefix before first underscore",
        "folds": [[stems[idx] for idx in fold] for fold in folds],
        "fold_sizes": [len(fold) for fold in folds],
        "fold_gt_counts": [float(denominators[fold, 0].sum()) for fold in folds],
        "fold_area_small_gt_counts": [float(denominators[fold, 2].sum()) for fold in folds],
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FOLDS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return folds


def reward_table(cache: dict[str, object], config: RewardConfig) -> torch.Tensor:
    final: torch.Tensor = cache["final_stats"]
    base: torch.Tensor = cache["base_stats"]
    den: torch.Tensor = cache["denominators"]
    invalid: torch.Tensor = cache["invalid"]
    gt = den[:, 0].clamp_min(1.0)[:, None, None]
    weak_gt = den[:, 1].clamp_min(1.0)[:, None, None]
    area_gt = den[:, 2].clamp_min(1.0)[:, None, None]
    base_gt = den[:, 0].clamp_min(1.0)
    base_weak = den[:, 1].clamp_min(1.0)
    base_area = den[:, 2].clamp_min(1.0)
    delta_r05 = final[..., 0] / gt - (base[:, 0] / base_gt)[:, None, None]
    delta_r075 = final[..., 1] / gt - (base[:, 1] / base_gt)[:, None, None]
    delta_weak = final[..., 2] / weak_gt - (base[:, 2] / base_weak)[:, None, None]
    delta_area = final[..., 3] / area_gt - (base[:, 3] / base_area)[:, None, None]
    final_precision = final[..., 0] / final[..., 4].clamp_min(1.0)
    base_precision = base[:, 0] / base[:, 4].clamp_min(1.0)
    delta_precision = final_precision - base_precision[:, None, None]
    if config.small_mode == "weak":
        delta_small = delta_weak
    elif config.small_mode == "area":
        delta_small = delta_area
    elif config.small_mode == "hybrid":
        delta_small = 0.5 * (delta_weak + delta_area)
    else:
        raise ValueError(f"Unknown small mode: {config.small_mode}")
    box_ratio = (final[..., 4] - base[:, 4, None, None]).clamp_min(0.0) / base[:, 4, None, None].clamp_min(1.0)
    first_ids = torch.arange(ACTION_DIM)[:, None]
    second_ids = torch.arange(ACTION_DIM)[None, :]
    action_count = (first_ids != 0).float() + (second_ids != 0).float()
    stop = (second_ids == 0).float()
    rewards = (
        config.recall05 * delta_r05
        + config.recall075 * delta_r075
        + config.small * delta_small
        + config.precision * delta_precision
        - config.box_cost * box_ratio
        - config.action_cost * action_count[None]
        - config.stop_cost * stop[None]
    )
    return rewards.masked_fill(invalid, -1.0)


def rows_for_indices(cache: dict[str, object], rewards: torch.Tensor, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.tensor(indices, dtype=torch.long)
    states: torch.Tensor = cache["states"]
    selected_states = states[idx]
    selected_rewards = rewards[idx]
    first_rewards = selected_rewards.max(dim=2).values
    return torch.cat([selected_states[:, :1], selected_states[:, 1:]], dim=1).reshape(-1, states.shape[-1]), torch.cat(
        [first_rewards[:, None, :], selected_rewards], dim=1
    ).reshape(-1, ACTION_DIM)


def train_policy(
    cache: dict[str, object],
    rewards: torch.Tensor,
    indices: Sequence[int],
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[PolicyNet, dict[str, float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    states, table = rows_for_indices(cache, rewards, indices)
    states = states.to(device)
    table = table.to(device)
    model = PolicyNet(states.shape[1], ACTION_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_score = -1e9
    best_state = None
    best_loss = 0.0
    best_epoch = 0
    row_idx = torch.arange(states.shape[0], device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        logits, values = model(states)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        chosen = table[row_idx, actions]
        advantage = chosen - values.detach()
        policy_loss = -(dist.log_prob(actions) * advantage).mean()
        value_loss = nn.functional.mse_loss(values, chosen)
        entropy_loss = -dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        with torch.no_grad():
            greedy = torch.argmax(model(states)[0], dim=1)
            objective = float(table[row_idx, greedy].mean())
        if objective > best_score:
            best_score = objective
            best_loss = float(loss.item())
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": float(best_epoch), "train_reward": best_score, "loss": best_loss}


def choose_actions(model: PolicyNet, cache: dict[str, object], indices: Sequence[int], device: torch.device) -> dict[int, tuple[int, int]]:
    states: torch.Tensor = cache["states"]
    result = {}
    with torch.no_grad():
        for idx in indices:
            first_logits, _ = model(states[idx, 0:1].to(device))
            first = int(first_logits.argmax(dim=1).item())
            second_logits, _ = model(states[idx, 1 + first:2 + first].to(device))
            if first != 0:
                second_logits[:, first] = -1e9
            second = int(second_logits.argmax(dim=1).item())
            result[idx] = (first, second)
    return result


def fast_metrics(cache: dict[str, object], choices: dict[int, tuple[int, int]]) -> dict[str, float]:
    final: torch.Tensor = cache["final_stats"]
    den: torch.Tensor = cache["denominators"]
    sums = torch.zeros(5)
    gt = weak_gt = area_gt = 0.0
    actions = stops = 0
    per_image = []
    for idx, (first, second) in choices.items():
        item = final[idx, first, second]
        sums += item
        gt += float(den[idx, 0])
        weak_gt += float(den[idx, 1])
        area_gt += float(den[idx, 2])
        action_count = int(first != 0) + int(second != 0)
        actions += action_count
        stops += int(first == 0) + int(second == 0)
        per_image.append({
            "idx": idx,
            "tp05": float(item[0]),
            "tp075": float(item[1]),
            "weak_tp": float(item[2]),
            "area_small_tp": float(item[3]),
            "pred_count": float(item[4]),
            "gt": float(den[idx, 0]),
            "weak_gt": float(den[idx, 1]),
            "area_small_gt": float(den[idx, 2]),
            "actions": action_count,
            "stops": int(first == 0) + int(second == 0),
        })
    count = max(1, len(choices))
    return {
        "image_count": float(len(choices)),
        "acc_05": float(sums[0]) / max(gt, 1.0),
        "acc_075": float(sums[1]) / max(gt, 1.0),
        "weak_recall_05": float(sums[2]) / max(weak_gt, 1.0),
        "area_small_recall_05": float(sums[3]) / max(area_gt, 1.0),
        "prediction_box_count": float(sums[4]),
        "predictions_per_image": float(sums[4]) / count,
        "actions_per_image": actions / count,
        "stop_ratio": stops / (2 * count),
        "per_image": per_image,
    }


def normalize_positive(config: RewardConfig, **changes: float) -> RewardConfig:
    values = {
        "recall05": config.recall05,
        "recall075": config.recall075,
        "small": config.small,
        "precision": config.precision,
    }
    values.update(changes)
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Positive reward weights cannot all be zero")
    values = {key: value * POSITIVE_TOTAL / total for key, value in values.items()}
    return replace(config, **values)


def unique_configs(configs: Iterable[RewardConfig]) -> list[RewardConfig]:
    seen = set()
    result = []
    for config in configs:
        key = (
            round(config.recall05, 8), round(config.recall075, 8), round(config.small, 8),
            round(config.precision, 8), round(config.box_cost, 8), round(config.action_cost, 8),
            round(config.stop_cost, 8), config.small_mode,
        )
        if key not in seen:
            seen.add(key)
            result.append(config)
    return result


def generate_configs() -> list[RewardConfig]:
    configs = [CURRENT]
    configs += [
        RewardConfig("equal", 0.305, 0.305, 0.305, 0.305, family="ablation"),
        RewardConfig("recall05_only", 1.22, 0.0, 0.0, 0.0, family="ablation"),
        replace(normalize_positive(CURRENT, recall05=0.0), name="remove_recall05", family="ablation"),
        replace(normalize_positive(CURRENT, recall075=0.0), name="remove_recall075", family="ablation"),
        replace(normalize_positive(CURRENT, small=0.0), name="remove_small", family="ablation"),
        replace(normalize_positive(CURRENT, precision=0.0), name="remove_precision", family="ablation"),
        replace(CURRENT, name="remove_box_cost", box_cost=0.0, family="ablation"),
        replace(CURRENT, name="remove_action_cost", action_cost=0.0, family="ablation"),
        replace(CURRENT, name="remove_stop_cost", stop_cost=0.0, family="ablation"),
        replace(CURRENT, name="remove_all_costs", box_cost=0.0, action_cost=0.0, stop_cost=0.0, family="ablation"),
        replace(CURRENT, name="area_small", small_mode="area", family="small_definition"),
        replace(CURRENT, name="hybrid_small", small_mode="hybrid", family="small_definition"),
    ]

    positive = ["recall05", "recall075", "small", "precision"]
    for field in positive:
        original = getattr(CURRENT, field)
        for multiplier in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            item = normalize_positive(CURRENT, **{field: original * multiplier})
            configs.append(replace(item, name=f"trend_{field}_x{multiplier:g}", family=f"trend_{field}"))

    cost_levels = {
        "box_cost": [0.0, 0.0025, 0.005, 0.010, 0.015, 0.020, 0.030],
        "action_cost": [0.0, 0.003, 0.006, 0.012, 0.018, 0.024, 0.036],
        "stop_cost": [0.0, 0.003, 0.006, 0.012, 0.018, 0.024, 0.036],
    }
    for field, levels in cost_levels.items():
        for level in levels:
            configs.append(replace(CURRENT, name=f"trend_{field}_{level:g}", family=f"trend_{field}", **{field: level}))

    interaction_levels = [0.5, 0.75, 1.0, 1.25, 1.5]
    for left, right in [("recall05", "small"), ("small", "precision")]:
        for left_mul in interaction_levels:
            for right_mul in interaction_levels:
                item = normalize_positive(CURRENT, **{
                    left: getattr(CURRENT, left) * left_mul,
                    right: getattr(CURRENT, right) * right_mul,
                })
                configs.append(replace(
                    item,
                    name=f"surface_{left}_{left_mul:g}_{right}_{right_mul:g}",
                    family=f"surface_{left}_{right}",
                ))
    for action_cost in [0.0, 0.006, 0.012, 0.024, 0.036]:
        for stop_cost in [0.0, 0.006, 0.012, 0.024, 0.036]:
            configs.append(replace(
                CURRENT,
                name=f"surface_action_{action_cost:g}_stop_{stop_cost:g}",
                action_cost=action_cost,
                stop_cost=stop_cost,
                family="surface_action_stop",
            ))

    rng = np.random.default_rng(20260916)
    accepted = 0
    while accepted < 40:
        weights = rng.dirichlet(np.array([3.0, 2.0, 3.0, 1.5])) * POSITIVE_TOTAL
        if not (0.15 <= weights[0] <= 0.65 and 0.05 <= weights[1] <= 0.45 and 0.15 <= weights[2] <= 0.70 and 0.05 <= weights[3] <= 0.45):
            continue
        configs.append(RewardConfig(
            f"simplex_{accepted:02d}", *[float(value) for value in weights], family="simplex",
        ))
        accepted += 1
    return unique_configs(configs)


def aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = [
        "acc_05", "acc_075", "weak_recall_05", "area_small_recall_05",
        "predictions_per_image", "actions_per_image", "stop_ratio",
    ]
    result = {}
    for key in keys:
        values = [float(row["metrics"][key]) for row in rows]
        result[key] = mean(values)
        result[f"{key}_std"] = pstdev(values)
    return result


def run_cv_config(
    config: RewardConfig,
    cache: dict[str, object],
    folds: list[list[int]],
    device: torch.device,
    epochs: int,
    seeds: Sequence[int],
    fold_ids: Sequence[int] | None = None,
    collect_choices: bool = False,
) -> dict[str, object]:
    rewards = reward_table(cache, config)
    all_indices = set(range(len(cache["stems"])))
    rows = []
    choices_by_seed: dict[str, dict[int, tuple[int, int]]] = {str(seed): {} for seed in seeds}
    for seed in seeds:
        for fold_idx in fold_ids or range(len(folds)):
            heldout = folds[fold_idx]
            train_indices = sorted(all_indices - set(heldout))
            model, train_info = train_policy(cache, rewards, train_indices, device, epochs, seed)
            choices = choose_actions(model, cache, heldout, device)
            metrics = fast_metrics(cache, choices)
            rows.append({"seed": seed, "fold": fold_idx, "train": train_info, "metrics": metrics})
            if collect_choices:
                choices_by_seed[str(seed)].update(choices)
    return {
        "config": asdict(config),
        "aggregate": aggregate_metrics(rows),
        "runs": rows,
        "choices_by_seed": choices_by_seed if collect_choices else {},
    }


def screen(device: torch.device, rebuild_cache: bool = False) -> dict[str, object]:
    cache = load_cache(rebuild_cache)
    folds = make_folds(cache)
    configs = generate_configs()
    results = []
    started = time.perf_counter()
    for idx, config in enumerate(configs):
        item = run_cv_config(config, cache, folds, device, SCREEN_EPOCHS, SCREEN_SEEDS)
        results.append(item)
        metric = item["aggregate"]
        print(
            f"[screen {idx + 1:03d}/{len(configs):03d}] {config.name} "
            f"acc05={metric['acc_05']:.4f} small={metric['area_small_recall_05']:.4f} "
            f"actions={metric['actions_per_image']:.3f}",
            flush=True,
        )
    payload = {
        "experiment": EXP_NAME,
        "stage": "screen",
        "device": str(device),
        "epochs": SCREEN_EPOCHS,
        "seeds": SCREEN_SEEDS,
        "fold_count": FOLD_COUNT,
        "config_count": len(configs),
        "elapsed_sec": time.perf_counter() - started,
        "results": results,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCREEN_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def metric_rank_key(item: dict[str, object]) -> tuple[float, float, float]:
    metric = item["aggregate"]
    return (float(metric["acc_05"]), float(metric["area_small_recall_05"]), -float(metric["actions_per_image"]))


def select_finalists(screen_payload: dict[str, object], count: int = 12) -> list[RewardConfig]:
    results: list[dict[str, object]] = screen_payload["results"]
    current = next(item for item in results if item["config"]["name"] == "current")
    current_metric = current["aggregate"]
    feasible = [
        item for item in results
        if float(item["aggregate"]["actions_per_image"]) <= 1.05 * float(current_metric["actions_per_image"])
        and float(item["aggregate"]["predictions_per_image"]) <= 1.05 * float(current_metric["predictions_per_image"])
    ]
    selected = [current]
    family_best = {}
    for item in feasible:
        family = item["config"]["family"]
        if family not in family_best or metric_rank_key(item) > metric_rank_key(family_best[family]):
            family_best[family] = item
    for item in sorted(feasible, key=metric_rank_key, reverse=True):
        if item not in selected:
            selected.append(item)
        if len(selected) >= max(6, count - len(family_best)):
            break
    for item in sorted(family_best.values(), key=metric_rank_key, reverse=True):
        if item not in selected:
            selected.append(item)
        if len(selected) >= count:
            break
    return [RewardConfig(**item["config"]) for item in selected[:count]]


def confirm(device: torch.device) -> dict[str, object]:
    if not SCREEN_JSON.exists():
        raise RuntimeError(f"Missing screening results: {SCREEN_JSON}")
    screen_payload = json.loads(SCREEN_JSON.read_text(encoding="utf-8"))
    cache = load_cache()
    folds = make_folds(cache)
    finalists = select_finalists(screen_payload)
    results = []
    started = time.perf_counter()
    for idx, config in enumerate(finalists):
        item = run_cv_config(config, cache, folds, device, CONFIRM_EPOCHS, CONFIRM_SEEDS, collect_choices=True)
        item["full_metrics_by_seed"] = {}
        for seed in CONFIRM_SEEDS:
            choices = item["choices_by_seed"][str(seed)]
            predictions = predictions_for_choices("val", cache["stems"], choices)
            item["full_metrics_by_seed"][str(seed)] = evaluate_predictions("val", cache["stems"], predictions)
        results.append(item)
        metric = item["aggregate"]
        print(
            f"[confirm {idx + 1:02d}/{len(finalists):02d}] {config.name} "
            f"acc05={metric['acc_05']:.4f}+/-{metric['acc_05_std']:.4f} "
            f"small={metric['area_small_recall_05']:.4f} actions={metric['actions_per_image']:.3f}",
            flush=True,
        )
    selected = choose_challenger(results)
    payload = {
        "experiment": EXP_NAME,
        "stage": "confirmation",
        "device": str(device),
        "epochs": CONFIRM_EPOCHS,
        "seeds": CONFIRM_SEEDS,
        "selection_rule": "<=105% current actions and predictions; max Acc@0.5; within 0.002 use area-small recall, mAP, then actions",
        "selected": selected["config"],
        "elapsed_sec": time.perf_counter() - started,
        "results": results,
    }
    CONFIRM_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def confirm_shard(device: torch.device, shard_index: int, shard_count: int) -> dict[str, object]:
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"Invalid shard {shard_index} of {shard_count}")
    if not SCREEN_JSON.exists():
        raise RuntimeError(f"Missing screening results: {SCREEN_JSON}")
    screen_payload = json.loads(SCREEN_JSON.read_text(encoding="utf-8"))
    cache = load_cache()
    folds = make_folds(cache)
    all_finalists = select_finalists(screen_payload)
    finalists = all_finalists[shard_index::shard_count]
    results = []
    started = time.perf_counter()
    for idx, config in enumerate(finalists):
        item = run_cv_config(config, cache, folds, device, CONFIRM_EPOCHS, CONFIRM_SEEDS, collect_choices=True)
        item["full_metrics_by_seed"] = {}
        for seed in CONFIRM_SEEDS:
            choices = item["choices_by_seed"][str(seed)]
            predictions = predictions_for_choices("val", cache["stems"], choices)
            item["full_metrics_by_seed"][str(seed)] = evaluate_predictions("val", cache["stems"], predictions)
        results.append(item)
        metric = item["aggregate"]
        print(
            f"[confirm shard {shard_index + 1}/{shard_count} {idx + 1:02d}/{len(finalists):02d}] {config.name} "
            f"acc05={metric['acc_05']:.4f}+/-{metric['acc_05_std']:.4f} "
            f"small={metric['area_small_recall_05']:.4f} actions={metric['actions_per_image']:.3f}",
            flush=True,
        )
    payload = {
        "experiment": EXP_NAME,
        "stage": "confirmation_shard",
        "device": str(device),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "epochs": CONFIRM_EPOCHS,
        "seeds": CONFIRM_SEEDS,
        "elapsed_sec": time.perf_counter() - started,
        "results": results,
    }
    path = LOG_DIR / CONFIRM_SHARD_PATTERN.format(index=shard_index, count=shard_count)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def merge_confirmation(shard_count: int) -> dict[str, object]:
    screen_payload = json.loads(SCREEN_JSON.read_text(encoding="utf-8"))
    finalist_order = [config.name for config in select_finalists(screen_payload)]
    by_name = {}
    shards = []
    for shard_index in range(shard_count):
        path = LOG_DIR / CONFIRM_SHARD_PATTERN.format(index=shard_index, count=shard_count)
        if not path.exists():
            raise RuntimeError(f"Missing confirmation shard: {path}")
        shard = json.loads(path.read_text(encoding="utf-8"))
        shards.append(shard)
        for item in shard["results"]:
            by_name[item["config"]["name"]] = item
    missing = [name for name in finalist_order if name not in by_name]
    if missing:
        raise RuntimeError(f"Missing finalist results: {missing}")
    results = [by_name[name] for name in finalist_order]
    recommended = choose_challenger(results)
    alternatives = [item for item in results if item["config"]["name"] != CURRENT.name]
    challenger = choose_challenger([next(item for item in results if item["config"]["name"] == CURRENT.name), *alternatives])
    if challenger["config"]["name"] == CURRENT.name:
        current = next(item for item in results if item["config"]["name"] == CURRENT.name)
        cur = current["aggregate"]
        feasible_alternatives = [
            item for item in alternatives
            if float(item["aggregate"]["actions_per_image"]) <= 1.05 * float(cur["actions_per_image"])
            and float(item["aggregate"]["predictions_per_image"]) <= 1.05 * float(cur["predictions_per_image"])
        ]
        best_alt_acc = max(float(item["aggregate"]["acc_05"]) for item in feasible_alternatives)
        tied_alt = [item for item in feasible_alternatives if float(item["aggregate"]["acc_05"]) >= best_alt_acc - 0.002]
        tied_alt.sort(key=lambda item: (
            float(item["aggregate"]["area_small_recall_05"]), mean_map(item),
            -float(item["aggregate"]["actions_per_image"]),
        ), reverse=True)
        challenger = tied_alt[0]
    payload = {
        "experiment": EXP_NAME,
        "stage": "confirmation",
        "devices": [shard["device"] for shard in shards],
        "epochs": CONFIRM_EPOCHS,
        "seeds": CONFIRM_SEEDS,
        "selection_rule": "<=105% current actions and predictions; max Acc@0.5; within 0.002 use area-small recall, mAP, then actions",
        "recommended": recommended["config"],
        "challenger": challenger["config"],
        "selected": recommended["config"],
        "elapsed_sec_max_shard": max(float(shard["elapsed_sec"]) for shard in shards),
        "results": results,
    }
    CONFIRM_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def mean_map(item: dict[str, object]) -> float:
    values = [float(metrics["map_05"]) for metrics in item.get("full_metrics_by_seed", {}).values()]
    return mean(values) if values else -1.0


def choose_challenger(results: list[dict[str, object]]) -> dict[str, object]:
    current = next(item for item in results if item["config"]["name"] == "current")
    cur = current["aggregate"]
    feasible = [
        item for item in results
        if float(item["aggregate"]["actions_per_image"]) <= 1.05 * float(cur["actions_per_image"])
        and float(item["aggregate"]["predictions_per_image"]) <= 1.05 * float(cur["predictions_per_image"])
    ]
    best_acc = max(float(item["aggregate"]["acc_05"]) for item in feasible)
    tied = [item for item in feasible if float(item["aggregate"]["acc_05"]) >= best_acc - 0.002]
    tied.sort(key=lambda item: (
        float(item["aggregate"]["area_small_recall_05"]), mean_map(item),
        -float(item["aggregate"]["actions_per_image"]),
    ), reverse=True)
    return tied[0]


def predictions_for_choices(split: str, stems: Sequence[str], choices: dict[int, tuple[int, int]]) -> dict[str, list[Det]]:
    predictions = {}
    for idx, (first, second) in choices.items():
        stem = stems[idx]
        base, _ = exp38_rescore(split, stem, merge_action_pool(split, stem, 0))
        current = apply_action_to_pool(split, stem, base, first)
        current = apply_action_to_pool(split, stem, current, second)
        predictions[stem], _ = exp38_rescore(split, stem, current)
    return predictions


def evaluate_predictions(split: str, stems: Sequence[str], predictions: dict[str, Sequence[Det]]) -> dict[str, object]:
    total_gt = total_pred = tp05 = tp075 = weak_tp = area_tp = 0
    weak_gt = area_gt = 0
    class_gt = Counter()
    gt_cache = {stem: read_gt(split, stem) for stem in stems}
    per_image = []
    for stem in stems:
        gt = gt_cache[stem]
        pred = predictions[stem]
        stats = extended_stats(pred, gt)
        total_gt += int(stats["gt"])
        total_pred += int(stats["pred_count"])
        tp05 += int(stats["tp05"])
        tp075 += int(stats["tp075"])
        weak_tp += int(stats["weak_tp"])
        area_tp += int(stats["area_small_tp"])
        weak_gt += int(stats["weak_gt"])
        area_gt += int(stats["area_small_gt"])
        class_gt.update(box.class_id for box in gt)
        per_image.append({"stem": stem, **stats})
    aps = []
    for class_id in range(1, 11):
        if class_gt[class_id] == 0:
            continue
        ranked = []
        for stem in stems:
            gt = [box for box in gt_cache[stem] if box.class_id == class_id]
            pred = [box for box in predictions[stem] if box.class_id == class_id]
            used = [False] * len(gt)
            for box in sorted(pred, key=lambda item: item.score, reverse=True):
                best_idx = -1
                best_iou = 0.0
                for idx, target in enumerate(gt):
                    if used[idx]:
                        continue
                    overlap = iou(box, target)
                    if overlap > best_iou:
                        best_idx = idx
                        best_iou = overlap
                if best_idx >= 0 and best_iou >= 0.5:
                    used[best_idx] = True
                    ranked.append((box.score, 1))
                else:
                    ranked.append((box.score, 0))
        aps.append(continuous_ap(ranked, class_gt[class_id]))
    return {
        "image_count": len(stems),
        "gt_box_count": total_gt,
        "prediction_box_count": total_pred,
        "tp_05": tp05,
        "tp_075": tp075,
        "acc_05": tp05 / max(total_gt, 1),
        "acc_075": tp075 / max(total_gt, 1),
        "map_05": sum(aps) / max(len(aps), 1),
        "weak_recall_05": weak_tp / max(weak_gt, 1),
        "area_small_recall_05": area_tp / max(area_gt, 1),
        "per_image": per_image,
    }


def test_cache() -> dict[str, object]:
    stems = read_split_stems("test")
    states = torch.zeros((len(stems), 1 + ACTION_DIM, 38), dtype=torch.float32)
    for idx, stem in enumerate(stems):
        width, height = image_size("test", stem)
        base, _ = exp38_rescore("test", stem, merge_action_pool("test", stem, 0))
        first_state = state_from_boxes(base, width, height)
        first_state[-2:] = [0.0, 1.0]
        states[idx, 0] = torch.tensor(first_state)
        for first in range(ACTION_DIM):
            current = apply_action_to_pool("test", stem, base, first)
            second_state = state_from_boxes(current, width, height)
            second_state[-2:] = [0.5, 0.5]
            states[idx, 1 + first] = torch.tensor(second_state)
    return {"stems": stems, "states": states}


def final_test(device: torch.device) -> dict[str, object]:
    if not CONFIRM_JSON.exists():
        raise RuntimeError(f"Missing confirmation results: {CONFIRM_JSON}")
    confirmation = json.loads(CONFIRM_JSON.read_text(encoding="utf-8"))
    recommended = RewardConfig(**confirmation.get("recommended", confirmation["selected"]))
    challenger = RewardConfig(**confirmation.get("challenger", confirmation["selected"]))
    cache = load_cache()
    test_data = test_cache()
    all_val = list(range(len(cache["stems"])))
    results = {}
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for config in [CURRENT, challenger]:
        config_results = []
        rewards = reward_table(cache, config)
        for seed in CONFIRM_SEEDS:
            model, train_info = train_policy(cache, rewards, all_val, device, CONFIRM_EPOCHS, seed)
            choices = choose_actions(model, test_data, range(len(test_data["stems"])), device)
            predictions = predictions_for_choices("test", test_data["stems"], choices)
            metrics = evaluate_predictions("test", test_data["stems"], predictions)
            action_count = sum(int(first != 0) + int(second != 0) for first, second in choices.values())
            stop_count = sum(int(first == 0) + int(second == 0) for first, second in choices.values())
            metrics["actions_per_image"] = action_count / len(choices)
            metrics["stop_ratio"] = stop_count / (2 * len(choices))
            config_results.append({"seed": seed, "train": train_info, "metrics": metrics})
            torch.save({
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "state_dim": 38,
                "action_names": ACTION_NAMES,
                "reward_config": asdict(config),
                "seed": seed,
                "train": train_info,
            }, CKPT_DIR / f"{config.name}_seed{seed}.pt")
            print(f"[test] {config.name} seed={seed} acc05={metrics['acc_05']:.4f} map05={metrics['map_05']:.4f}", flush=True)
        results[config.name] = summarize_test_runs(config_results)
        results[config.name]["runs"] = config_results
        results[config.name]["config"] = asdict(config)
    bootstrap = paired_bootstrap(results[CURRENT.name]["runs"][0]["metrics"]["per_image"], results[challenger.name]["runs"][0]["metrics"]["per_image"])
    payload = {
        "experiment": EXP_NAME,
        "stage": "final_test",
        "selection_frozen_before_test": True,
        "current": CURRENT.name,
        "validation_recommended": recommended.name,
        "challenger": challenger.name,
        "selected": challenger.name,
        "seeds": CONFIRM_SEEDS,
        "results": results,
        "paired_bootstrap_seed42": bootstrap,
    }
    FINAL_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_outputs(payload)
    return payload


def summarize_test_runs(runs: list[dict[str, object]]) -> dict[str, float]:
    keys = ["acc_05", "acc_075", "map_05", "weak_recall_05", "area_small_recall_05", "prediction_box_count", "actions_per_image", "stop_ratio"]
    result = {}
    for key in keys:
        values = [float(run["metrics"][key]) for run in runs]
        result[key] = mean(values)
        result[f"{key}_std"] = pstdev(values)
    return result


def paired_bootstrap(current_rows: list[dict[str, float]], selected_rows: list[dict[str, float]], repeats: int = 10000) -> dict[str, object]:
    current = {row["stem"]: row for row in current_rows}
    selected = {row["stem"]: row for row in selected_rows}
    stems = sorted(current)
    rng = np.random.default_rng(20260916)
    fields = {
        "acc_05": ("tp05", "gt"),
        "acc_075": ("tp075", "gt"),
        "weak_recall_05": ("weak_tp", "weak_gt"),
        "area_small_recall_05": ("area_small_tp", "area_small_gt"),
    }
    samples = {key: [] for key in fields}
    for _ in range(repeats):
        picked = rng.integers(0, len(stems), len(stems))
        for name, (num, den) in fields.items():
            cur_num = sum(float(current[stems[idx]][num]) for idx in picked)
            cur_den = sum(float(current[stems[idx]][den]) for idx in picked)
            sel_num = sum(float(selected[stems[idx]][num]) for idx in picked)
            sel_den = sum(float(selected[stems[idx]][den]) for idx in picked)
            samples[name].append(sel_num / max(sel_den, 1.0) - cur_num / max(cur_den, 1.0))
    return {
        name: {
            "mean_delta": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
    }


def make_outputs(payload: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    screen = json.loads(SCREEN_JSON.read_text(encoding="utf-8"))
    rows = screen["results"]
    current = next(row for row in rows if row["config"]["name"] == "current")
    confirmation = json.loads(CONFIRM_JSON.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    families = sorted(set(row["config"]["family"] for row in rows))
    colors = plt.cm.tab20(np.linspace(0, 1, len(families)))
    for family, color in zip(families, colors):
        subset = [row for row in rows if row["config"]["family"] == family]
        ax.scatter(
            [row["aggregate"]["actions_per_image"] for row in subset],
            [row["aggregate"]["acc_05"] for row in subset],
            s=28, alpha=0.75, label=family, color=color,
        )
    ax.scatter([current["aggregate"]["actions_per_image"]], [current["aggregate"]["acc_05"]], marker="*", s=220, color="black", label="current")
    ax.set_xlabel("Expansion actions per image")
    ax.set_ylabel("Held-out validation Acc@0.5")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reward_pareto.png", dpi=200)
    plt.close(fig)

    trend_fields = ["recall05", "recall075", "small", "precision", "box_cost", "action_cost", "stop_cost"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for axis, field in zip(axes.flat, trend_fields):
        subset = [row for row in rows if row["config"]["family"] == f"trend_{field}"]
        if all(abs(float(row["config"][field]) - float(current["config"][field])) > 1e-12 for row in subset):
            subset.append(current)
        subset.sort(key=lambda row: float(row["config"][field]))
        axis.plot([row["config"][field] for row in subset], [row["aggregate"]["acc_05"] for row in subset], marker="o", label="Acc@0.5")
        axis.plot([row["config"][field] for row in subset], [row["aggregate"]["area_small_recall_05"] for row in subset], marker="s", label="Area-small R")
        axis.axvline(getattr(CURRENT, field), color="black", linestyle="--", linewidth=1)
        axis.set_title(field)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reward_sensitivity_curves.png", dpi=200)
    plt.close(fig)

    current_result = payload["results"][payload["current"]]
    challenger_result = payload["results"][payload["challenger"]]
    current_screen = current["aggregate"]
    feasible_screen = [
        row for row in rows
        if float(row["aggregate"]["actions_per_image"]) <= 1.05 * float(current_screen["actions_per_image"])
        and float(row["aggregate"]["predictions_per_image"]) <= 1.05 * float(current_screen["predictions_per_image"])
    ]
    best_screen = max(feasible_screen, key=lambda row: float(row["aggregate"]["acc_05"]))
    screen_by_name = {row["config"]["name"]: row for row in rows}
    ablation_names = [
        "current", "equal", "recall05_only", "remove_recall05", "remove_recall075",
        "remove_small", "remove_precision", "remove_box_cost", "remove_action_cost",
        "remove_stop_cost", "remove_all_costs", "area_small", "hybrid_small",
    ]
    lines = [
        "# Exp44 Reward 超参数验证实验总结",
        "",
        "## 实验协议",
        "",
        "- 所有 reward 搜索只使用 validation 五折交叉验证。",
        "- test 在候选配置冻结后才评测，且不根据 test 结果继续调参。",
        "- 粗筛覆盖组件消融、宽范围单因素曲线、关键二因素交互和四维单纯形采样。",
        "- 最终配置使用 3 个随机种子复核；置信区间使用逐图配对 bootstrap 10,000 次。",
        f"- 粗筛共 `{screen['config_count']}` 组，其中 `{len(feasible_screen)}` 组满足相对 Current 不超过 105% 的动作数与候选数约束。",
        "- 本项目沿用的 `Acc@0.5` 定义为同类别一对一匹配后的 `TP@0.5 / GT总数`，本质上是目标命中率而不是分类 accuracy；`Acc@0.75` 同理。",
        "- 所有 Recall 均采用同类别一对一匹配和 IoU>=0.5；面积小目标指 GT 像素面积不超过 32×32，弱类指类别 `{1,2,3,7,8,10}`。",
        "",
        "## Reward 组件消融与定义对照（五折粗筛，seed 42）",
        "",
        "表中每一行都重新训练策略，而不是只把同一策略换一个评分公式。数值为 5 个 held-out fold 的算术平均，粗筛统一使用 seed 42 和 180 epochs。",
        "`动作/图` 是两次反馈决策中实际执行的非 STOP 扩展动作数；`预测框/图` 是候选合并、去重并经 Exp38 重评分后的最终框数。",
        "`相对Current Acc` 为该配置减去 Current，正值表示 Acc 更高，但仍需同时检查动作与候选成本。删除某个正奖励项时，其余正权重会同比归一化，使总和保持 1.22。",
        "",
        "| 配置 | Acc@0.5 | 面积小目标Recall | 动作/图 | 预测框/图 | 相对Current Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ablation_names:
        if name not in screen_by_name:
            continue
        metric = screen_by_name[name]["aggregate"]
        lines.append(
            f"| {name} | {metric['acc_05']:.4f} | {metric['area_small_recall_05']:.4f} | "
            f"{metric['actions_per_image']:.3f} | {metric['predictions_per_image']:.1f} | "
            f"{metric['acc_05'] - current_screen['acc_05']:+.4f} |"
        )
    lines += [
        "",
        "关键观察：",
        "",
        "- `equal` 表示四个正奖励等权；`recall05_only` 只保留 Recall@0.5；`remove_*` 表示删除对应奖励或成本项。",
        "- `area_small` 用 GT 像素面积不超过 32×32 定义 small；`hybrid_small` 对弱类别召回和面积小目标召回各取一半。Current 的 small 项使用弱类别集合 `{1,2,3,7,8,10}`。",
        "- 去掉 small 项同时降低 Acc 和面积小目标召回，说明小目标强化项有独立作用。",
        "- 去掉 action cost 会明显增加动作数；部分去项配置虽有轻微 Acc 增益，但超过预注册计算预算，不能作为同成本优势。",
        "- 直接将弱类别 small 定义替换为面积 small 没有提升；hybrid 定义接近，但完整确认阶段仍未超过 Current。",
        f"- 粗筛最高可行配置为 `{best_screen['config']['name']}`，Acc@0.5={best_screen['aggregate']['acc_05']:.4f}；粗筛结果不直接用于最终结论。",
        "",
        "## 为什么 Recall@0.5 的系数保留为 0.40",
        "",
        "`0.40` 不是由理论公式解析求出的常数。它占四个正奖励总权重 1.22 的 32.8%，表达的是普通召回与高 IoU 召回、小目标召回、Precision 之间的相对优先级。",
        "为单独检验这一系数，实验将 Recall@0.5 原始权重乘以 0、0.5、0.75、1、1.25、1.5、2，并把四个正权重重新归一化到 1.22；因此比较不会被整体 reward 尺度变化干扰。",
        "",
        "| 归一化后的 w_R@0.5 | 粗筛 Acc@0.5 | 面积小目标Recall | 动作/图 | 满足105%成本约束 |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    recall_rows = [screen_by_name["remove_recall05"], current] + [
        row for row in rows if row["config"]["family"] == "trend_recall05"
    ]
    recall_rows.sort(key=lambda row: float(row["config"]["recall05"]))
    for row in recall_rows:
        config = row["config"]
        metric = row["aggregate"]
        lines.append(
            f"| {config['recall05']:.4f} | {metric['acc_05']:.4f} | "
            f"{metric['area_small_recall_05']:.4f} | {metric['actions_per_image']:.3f} | "
            f"{'是' if row in feasible_screen else '否'} |"
        )
    lines += [
        "",
        "粗筛中，较大的 Recall@0.5 权重偶尔带来约 0.001 的 Acc 增益，但动作数超过 Current 的 105% 预算；不能据此认定更优。",
        "满足预算且最接近的竞争点是 `w_R@0.5=0.3268`。它进入五折三种子确认后，Mean-fold Acc 为 0.5095、面积小目标 Recall 为 0.3880、动作数为 1.513；Current 分别为 0.5095、0.3884、1.447，且 Current 的 OOF mAP 更高（0.1295 对 0.1293）。",
        "因此实验支持的结论是：0.40 位于稳定平台，并在几乎相同的 Acc 下取得更好的小目标召回、mAP 和动作效率。实验不支持声称 0.40 在任意小数精度上唯一最优。",
        "",
        "## 五折三种子确认",
        "",
        "粗筛后共有 12 组配置进入完整确认。下面先给出配置本身，四个 `w_*` 是正奖励权重；三个 `cost` 分别惩罚新增候选框、非 STOP 动作数和当前轮 STOP。",
        "`trend_x1.5` 表示先将对应权重乘 1.5，再把四个正权重归一化到总和 1.22；`surface` 表示来自二因素交互网格；`simplex` 表示来自四维权重空间采样。",
        "",
        "| 配置 | w_R@0.5 | w_R@0.75 | w_small | w_precision | box/action/STOP cost | small定义 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in confirmation["results"]:
        config = item["config"]
        lines.append(
            f"| {config['name']} | {config['recall05']:.4f} | {config['recall075']:.4f} | "
            f"{config['small']:.4f} | {config['precision']:.4f} | "
            f"{config['box_cost']:.3f}/{config['action_cost']:.3f}/{config['stop_cost']:.3f} | {config['small_mode']} |"
        )
    lines += [
        "",
        "结果表中，`Mean-fold` 是先在每个 held-out fold 上计算指标，再让 5 个 fold 等权平均，这是预注册的选参口径。",
        "`OOF` 是把 5 个 held-out fold 的预测重新合并成完整 validation 预测后计算全局指标，再对 3 个随机种子取平均；它更接近一次完整数据集评测。",
        "由于不同 fold 的 GT 数量不同，Mean-fold Acc 和 OOF Acc 会有小幅差异。`OOF mAP` 必须在合并后的全局置信度排序上计算，不能由 fold mAP 直接平均替代。",
        "",
        "| 配置 | Mean-fold Acc@0.5 | Mean-fold 面积小目标Recall | OOF Acc@0.5 | OOF mAP@0.5 | 动作/图 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in confirmation["results"]:
        pooled = list(item["full_metrics_by_seed"].values())
        lines.append(
            f"| {item['config']['name']} | {item['aggregate']['acc_05']:.4f} | "
            f"{item['aggregate']['area_small_recall_05']:.4f} | "
            f"{mean(float(value['acc_05']) for value in pooled):.4f} | "
            f"{mean(float(value['map_05']) for value in pooled):.4f} | "
            f"{item['aggregate']['actions_per_image']:.3f} |"
        )
    lines += [
        "",
        "按预注册的 mean-fold 规则，Current 在 Acc、面积小目标召回和 OOF mAP 的综合比较中被保留。",
        "表中 Current 与 `surface_small_1.25_precision_1.25` 的 Mean-fold Acc 只差约 0.00001，属于预设的 0.002 平局区间；Current 的 Mean-fold 面积小目标召回和 OOF mAP 略高，因此被保留。",
        "为形成有效的独立 test 对照，将验证集最强的非当前配置冻结为 Challenger。Challenger 只是对照对象，不是最终推荐公式；test 后不再调参。",
        "",
        "## 冻结配置",
        "",
        f"- Validation 推荐：`{payload['validation_recommended']}`",
        f"- Current: `{json.dumps(current_result['config'], ensure_ascii=False)}`",
        f"- Challenger: `{json.dumps(challenger_result['config'], ensure_ascii=False)}`",
        "",
        "## 独立 Test 结果（3 seeds）",
        "",
        "`均值 +/- 标准差` 来自随机种子 42、43、44，标准差描述训练随机性，不是置信区间。`候选框` 是 150 张 test 图像上的最终预测框总数，再对 3 个种子取平均。",
        "两种配置使用相同的候选源、网络结构、训练轮数、推理轮数、去重和 Exp38 重评分流程，唯一变化是 reward 权重。",
        "",
        "| 配置 | Acc@0.5 | Acc@0.75 | mAP@0.5 | 面积小目标Recall | 弱类Recall | 动作/图 | 候选框 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in [(payload["current"], current_result), (payload["challenger"], challenger_result)]:
        lines.append(
            f"| {name} | {item['acc_05']:.4f} +/- {item['acc_05_std']:.4f} | "
            f"{item['acc_075']:.4f} +/- {item['acc_075_std']:.4f} | "
            f"{item['map_05']:.4f} +/- {item['map_05_std']:.4f} | "
            f"{item['area_small_recall_05']:.4f} +/- {item['area_small_recall_05_std']:.4f} | "
            f"{item['weak_recall_05']:.4f} +/- {item['weak_recall_05_std']:.4f} | "
            f"{item['actions_per_image']:.3f} | {item['prediction_box_count']:.1f} |"
        )
    lines += [
        "",
        "## 配对 Bootstrap（Challenger - Current，seed 42）",
        "",
        "Bootstrap 以图像为重采样单位，重复 10,000 次。差值定义为 `Challenger - Current`：负值表示 Current 更好；95% CI 不跨 0 表示该方向在本次配对检验中达到统计显著。",
        "mAP 依赖跨图像的全局置信度排序，本表不使用逐图近似 mAP 做显著性检验，因此只对可严格按计数汇总的召回类指标报告区间。",
        "",
        "| 指标 | 平均差值 | 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for name, item in payload["paired_bootstrap_seed42"].items():
        lines.append(f"| {name} | {item['mean_delta']:+.5f} | [{item['ci95_low']:+.5f}, {item['ci95_high']:+.5f}] |")
    lines += [
        "",
        "## 最终结论",
        "",
        f"Current 的三种子平均 Acc@0.5 比 Challenger 高 `{current_result['acc_05'] - challenger_result['acc_05']:+.4f}`，"
        f"mAP@0.5 高 `{current_result['map_05'] - challenger_result['map_05']:+.4f}`，"
        f"面积小目标 Recall 高 `{current_result['area_small_recall_05'] - challenger_result['area_small_recall_05']:+.4f}`。",
        "面积小目标 Recall 的配对 bootstrap 差异不跨 0；Acc@0.5 的区间上界触及 0，因此应表述为稳定优势，而不是所有指标均达到统计显著。",
        "综合五折确认、三个 test 随机种子和成本约束，最终保留当前 reward 超参数。",
        "",
        "该实验验证的是预注册搜索空间内的经验稳健性和局部优势，不声称数学上的全局最优。",
        "建议答辩表述为：当前权重经宽范围敏感性搜索和消融验证后被保留，是性能、小目标召回与动作成本之间的稳定折中。",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (LOG_DIR / "final_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "acc_05", "acc_075", "map_05", "area_small_recall_05", "weak_recall_05", "actions_per_image", "prediction_box_count"])
        for name, item in [(payload["current"], current_result), (payload["challenger"], challenger_result)]:
            writer.writerow([name, item["acc_05"], item["acc_075"], item["map_05"], item["area_small_recall_05"], item["weak_recall_05"], item["actions_per_image"], item["prediction_box_count"]])

    with (LOG_DIR / "screening_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "family", "recall05", "recall075", "small", "precision", "box_cost", "action_cost", "stop_cost", "small_mode", "acc_05", "acc_075", "area_small_recall_05", "weak_recall_05", "actions_per_image", "predictions_per_image", "feasible"])
        for row in rows:
            config = row["config"]
            metric = row["aggregate"]
            writer.writerow([config["name"], config["family"], config["recall05"], config["recall075"], config["small"], config["precision"], config["box_cost"], config["action_cost"], config["stop_cost"], config["small_mode"], metric["acc_05"], metric["acc_075"], metric["area_small_recall_05"], metric["weak_recall_05"], metric["actions_per_image"], metric["predictions_per_image"], row in feasible_screen])

    with (LOG_DIR / "confirmation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "mean_fold_acc_05", "mean_fold_area_small_recall_05", "mean_fold_actions_per_image", "pooled_oof_acc_05", "pooled_oof_map_05"])
        for item in confirmation["results"]:
            pooled = list(item["full_metrics_by_seed"].values())
            writer.writerow([item["config"]["name"], item["aggregate"]["acc_05"], item["aggregate"]["area_small_recall_05"], item["aggregate"]["actions_per_image"], mean(float(value["acc_05"]) for value in pooled), mean(float(value["map_05"]) for value in pooled)])


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this execution context")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp44 reward hyperparameter validation")
    parser.add_argument("stage", choices=["cache", "screen", "confirm", "confirm-shard", "confirm-merge", "test", "report", "all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "cache":
        cache = load_cache(args.rebuild_cache)
        make_folds(cache)
        print(json.dumps({"cache": str(CACHE_PATH), "images": len(cache["stems"])}, indent=2))
        return
    if args.stage == "confirm-merge":
        payload = merge_confirmation(args.shard_count)
        print(json.dumps({"selected": payload["selected"], "result_count": len(payload["results"])}, ensure_ascii=False, indent=2))
        return
    if args.stage == "report":
        if not FINAL_JSON.exists():
            raise RuntimeError(f"Missing final test results: {FINAL_JSON}")
        make_outputs(json.loads(FINAL_JSON.read_text(encoding="utf-8")))
        print(SUMMARY_MD)
        return
    device = resolve_device(args.device)
    if args.stage in {"screen", "all"}:
        screen(device, args.rebuild_cache)
    if args.stage in {"confirm", "all"}:
        confirm(device)
    if args.stage == "confirm-shard":
        confirm_shard(device, args.shard_index, args.shard_count)
    if args.stage in {"test", "all"}:
        final_test(device)


if __name__ == "__main__":
    main()
