#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "38+RL_loop"))

from common import (  # noqa: E402
    ACTION_NAMES,
    CKPT_PATH,
    OUT_ROOT,
    Det,
    deduplicate,
    evaluate_split,
    exp38_rescore,
    image_size,
    merge_action_pool,
    read_split_stems,
    require_inputs,
    state_from_boxes,
    write_predictions,
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


def load_policy(device: torch.device) -> PolicyNet:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    policy = PolicyNet(int(ckpt["state_dim"]), len(ckpt["action_names"])).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy


def merge_boxes(a: list[Det], b: list[Det]) -> list[Det]:
    return deduplicate(a + b)


def main() -> None:
    require_inputs()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = load_policy(device)
    predictions = {}
    trace = []
    action_counts = {name: 0 for name in ACTION_NAMES}
    action_pair_counts = {}
    for stem in read_split_stems("test"):
        width, height = image_size("test", stem)
        current, _ = exp38_rescore("test", stem, merge_action_pool("test", stem, 0))
        chosen: list[int] = []
        for round_idx in range(2):
            state_values = state_from_boxes(current, width, height)
            state_values[-2] = float(round_idx + 1) / 2.0
            state_values[-1] = float(1 - round_idx) / 2.0
            state = torch.tensor([state_values], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = policy(state)
                logits = logits.clone()
                logits[:, 0] -= 1.0
                for old_action in chosen:
                    logits[:, old_action] = -1e9
                action = int(torch.argmax(logits, dim=1).item())
                probs = torch.softmax(logits, dim=1).cpu().squeeze(0).tolist()
            extra, _ = exp38_rescore("test", stem, merge_action_pool("test", stem, action))
            current = merge_boxes(current, extra)
            chosen.append(action)
            action_counts[ACTION_NAMES[action]] += 1
            trace.append({"stem": stem, "round": round_idx + 2, "action": ACTION_NAMES[action], "probs": probs, "count_after": len(current)})
        key = " -> ".join(ACTION_NAMES[a] for a in chosen)
        action_pair_counts[key] = action_pair_counts.get(key, 0) + 1
        predictions[stem] = current
    out_dir = OUT_ROOT / "predictions_three_round_probe"
    write_predictions(out_dir, predictions)
    metrics = evaluate_split("test", out_dir)
    payload = {
        "method": "three-round probe using current 38+RL policy, repeated two feedback actions",
        "device": str(device),
        "action_counts": action_counts,
        "top_action_pairs": sorted(action_pair_counts.items(), key=lambda item: item[1], reverse=True)[:20],
        "metrics": metrics,
        "prediction_dir": str(out_dir),
    }
    out_path = ROOT / "test/outputs/three_round_rl_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
