#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

from common import (
    ACTION_NAMES,
    CKPT_PATH,
    OUT_ROOT,
    SEED,
    apply_action_to_pool,
    exp38_rescore,
    image_size,
    match_stats,
    merge_action_pool,
    read_gt,
    read_split_stems,
    require_inputs,
    reward_from_stats,
    state_from_boxes,
)


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


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA for training, but torch.cuda.is_available() is False.")
    return torch.device(name)


def build_dataset(split: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    states: list[list[float]] = []
    rewards: list[list[float]] = []
    oracle_counts_round2 = {name: 0 for name in ACTION_NAMES}
    oracle_counts_round3 = {name: 0 for name in ACTION_NAMES}
    match_total = 0
    box_total = 0
    for stem in read_split_stems(split):
        width, height = image_size(split, stem)
        base_scored, matched = exp38_rescore(split, stem, merge_action_pool(split, stem, 0))
        gt_boxes = read_gt(split, stem)
        base_stats = match_stats(base_scored, gt_boxes)
        first_reward_row: list[float] = []
        second_rows: list[tuple[int, list[float], list[float]]] = []
        for first_action in range(len(ACTION_NAMES)):
            current = apply_action_to_pool(split, stem, base_scored, first_action)
            state = state_from_boxes(current, width, height)
            state[-2] = 0.5
            state[-1] = 0.5
            second_reward_row: list[float] = []
            for second_action in range(len(ACTION_NAMES)):
                if second_action == first_action and second_action != 0:
                    second_reward_row.append(-1.0)
                    continue
                final_boxes = apply_action_to_pool(split, stem, current, second_action)
                final_stats = match_stats(final_boxes, gt_boxes)
                action_count = int(first_action != 0) + int(second_action != 0)
                second_reward_row.append(reward_from_stats(final_stats, base_stats, action_count, used_stop=(second_action == 0)))
            second_rows.append((first_action, state, second_reward_row))
            first_reward_row.append(max(second_reward_row))
        first_state = state_from_boxes(base_scored, width, height)
        first_state[-2] = 0.0
        first_state[-1] = 1.0
        states.append(first_state)
        rewards.append(first_reward_row)
        oracle_counts_round2[ACTION_NAMES[max(range(len(first_reward_row)), key=lambda idx: first_reward_row[idx])]] += 1
        for _, second_state, second_reward_row in second_rows:
            states.append(second_state)
            rewards.append(second_reward_row)
            oracle_counts_round3[ACTION_NAMES[max(range(len(second_reward_row)), key=lambda idx: second_reward_row[idx])]] += 1
        match_total += matched
        box_total += len(base_scored)
    return torch.tensor(states, dtype=torch.float32), torch.tensor(rewards, dtype=torch.float32), {
        "state_dim": len(states[0]) if states else 0,
        "transition_count": len(states),
        "oracle_action_counts_round2": oracle_counts_round2,
        "oracle_action_counts_round3": oracle_counts_round3,
        "base_exp38_match_ratio": match_total / max(1, box_total),
    }


def train(device: torch.device, epochs: int, seed: int) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    states, reward_table, meta = build_dataset("val")
    states = states.to(device)
    reward_table = reward_table.to(device)
    policy = PolicyNet(states.shape[1], len(ACTION_NAMES)).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001, weight_decay=1e-4)
    best = {"epoch": 0.0, "val_mean_reward": -1e9, "loss": 0.0}
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        logits, values = policy(states)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        chosen_rewards = reward_table[torch.arange(states.shape[0], device=device), actions]
        advantage = chosen_rewards - values.detach()
        policy_loss = -(dist.log_prob(actions) * advantage).mean()
        value_loss = nn.functional.mse_loss(values, chosen_rewards)
        entropy_loss = -dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        optimizer.step()
        with torch.no_grad():
            greedy = torch.argmax(policy(states)[0], dim=1)
            objective = float(reward_table[torch.arange(states.shape[0], device=device), greedy].mean().item())
        if objective > best["val_mean_reward"]:
            best = {"epoch": float(epoch), "val_mean_reward": objective, "loss": float(loss.item())}
            torch.save(
                {
                    "model": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
                    "state_dim": states.shape[1],
                    "action_names": ACTION_NAMES,
                    "best": best,
                    "device_used": str(device),
                },
                CKPT_PATH,
            )
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"[38+RL train] epoch={epoch:03d}/{epochs} loss={loss.item():.6f} greedy_reward={objective:.6f}", flush=True)
    payload = {"split": "val", "transition_count": int(states.shape[0]), "best": best, "device": str(device), **meta}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp38-scored RL closed-loop policy on GPU.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_inputs()
    device = resolve_device(args.device)
    print(f"[38+RL train] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)
    info = train(device, args.epochs, args.seed)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
