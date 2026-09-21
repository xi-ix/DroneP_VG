#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from common import (
    ACTION_NAMES,
    CKPT_PATH,
    EXP38_PRED_ROOT,
    OUT_ROOT,
    RL_PLAIN_ROOT,
    SUMMARY_JSON,
    SUMMARY_MD,
    apply_action_to_pool,
    evaluate_split,
    exp38_rescore,
    image_size,
    merge_action_pool,
    read_pred_file,
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


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def load_policy(device: torch.device) -> PolicyNet:
    if not CKPT_PATH.exists():
        raise RuntimeError(f"Missing checkpoint: {CKPT_PATH}. Run 38+RL_loop/train.py first.")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    policy = PolicyNet(int(ckpt["state_dim"]), len(ckpt["action_names"])).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy


def run_test(device: torch.device, rounds: int = 3, output_dir: Path = OUT_ROOT) -> dict[str, object]:
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    policy = load_policy(device)
    inference_started = time.perf_counter()
    predictions = {}
    trace = []
    action_counts = {name: 0 for name in ACTION_NAMES}
    matched_total = 0
    box_total = 0
    for stem in read_split_stems("test"):
        width, height = image_size("test", stem)
        round1, _ = exp38_rescore("test", stem, merge_action_pool("test", stem, 0))
        current = round1
        chosen_actions: list[int] = []
        round_trace = []
        for feedback_idx in range(rounds - 1):
            state_values = state_from_boxes(current, width, height)
            # The checkpoint was trained on two feedback steps. Preserve that
            # state scale and use its terminal state for any additional steps.
            state_values[-2] = min(float(feedback_idx) / 2.0, 1.0)
            state_values[-1] = max(float(2 - feedback_idx) / 2.0, 0.0)
            state = torch.tensor([state_values], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = policy(state)
                logits = logits.clone()
                for old_action in chosen_actions:
                    if old_action != 0:
                        logits[:, old_action] = -1e9
                probs = torch.softmax(logits, dim=1).cpu().squeeze(0).tolist()
                action = int(torch.argmax(logits, dim=1).item())
            current = apply_action_to_pool("test", stem, current, action)
            chosen_actions.append(action)
            action_counts[ACTION_NAMES[action]] += 1
            round_trace.append({"round": feedback_idx + 2, "action": ACTION_NAMES[action], "action_id": action, "probs": probs, "count_after_action": len(current)})
        boxes, matched = exp38_rescore("test", stem, current)
        predictions[stem] = boxes
        matched_total += matched
        box_total += len(boxes)
        trace.append({
            "stem": stem,
            "actions": [ACTION_NAMES[action] for action in chosen_actions],
            "action_ids": chosen_actions,
            "rounds": round_trace,
            "round1_count": len(round1),
            "final_count": len(boxes),
            "exp38_matched": matched,
        })
    inference_sec = time.perf_counter() - inference_started
    pred_dir = output_dir / "predictions"
    write_predictions(pred_dir, predictions)
    evaluation_started = time.perf_counter()
    metrics = evaluate_split("test", pred_dir)
    baseline_query_predictions = {}
    for stem in read_split_stems("test"):
        baseline_query_predictions[stem], _ = exp38_rescore("test", stem, merge_action_pool("test", stem, 0))
    query_base_dir = output_dir / "predictions_query_base_exp38_scored"
    write_predictions(query_base_dir, baseline_query_predictions)
    query_base_metrics = evaluate_split("test", query_base_dir)
    rl_plain_metrics = evaluate_split("test", RL_PLAIN_ROOT)
    exp38_full_metrics = evaluate_split("test", EXP38_PRED_ROOT)
    evaluation_sec = time.perf_counter() - evaluation_started
    payload = {
        "method": f"{rounds}-round Exp38 rescoring over RL inference-loop candidates",
        "loop_rounds": rounds,
        "checkpoint": str(CKPT_PATH),
        "device": str(device),
        "timing": {
            "closed_loop_inference_sec": inference_sec,
            "evaluation_and_reference_sec": evaluation_sec,
            "total_sec": inference_sec + evaluation_sec,
            "images_per_sec": len(trace) / inference_sec if inference_sec else 0.0,
            "inference_ms_per_image": inference_sec * 1000.0 / len(trace) if trace else 0.0,
            "scope": "cached candidate loading, policy decisions, merging, deduplication, and Exp38 rescoring; excludes GroundingDINO candidate generation",
        },
        "action_counts": action_counts,
        "exp38_match_ratio": matched_total / max(1, box_total),
        "query_base_exp38_scored_metrics": query_base_metrics,
        "rl_plain_metrics": rl_plain_metrics,
        "exp38_full_metrics": exp38_full_metrics,
        "test_metrics": metrics,
        "trace_path": str(output_dir / "closed_loop_trace.json"),
        "prediction_dir": str(pred_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "closed_loop_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_json = SUMMARY_JSON if output_dir == OUT_ROOT else output_dir / "summary.json"
    summary_md = SUMMARY_MD if output_dir == OUT_ROOT else output_dir / "summary.md"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# 38 + RL_loop {rounds}-round Summary",
        "",
        f"- method: `{payload['method']}`",
        f"- checkpoint: `{CKPT_PATH}`",
        f"- device: `{device}`",
        f"- action_counts: `{action_counts}`",
        f"- loop_rounds: `{rounds}`",
        f"- closed_loop_inference_sec: `{inference_sec:.4f}`",
        "- timing_scope: cached candidates + policy + merge/dedup + Exp38 rescoring; excludes GroundingDINO",
        f"- exp38_match_ratio: `{payload['exp38_match_ratio']:.4f}`",
        "",
        "| Method | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in [
        ("query_base_exp38_scored", query_base_metrics),
        ("rl_plain", rl_plain_metrics),
        ("38+RL_loop", metrics),
        ("exp38_full_reference", exp38_full_metrics),
    ]:
        lines.append(f"| {name} | {int(item['prediction_box_count'])} | {item['acc_05']:.4f} | {item['acc_075']:.4f} | {item['map_05']:.4f} |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Exp38 + RL closed-loop inference.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_inputs()
    device = resolve_device(args.device)
    print(f"[38+RL test] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)
    output_dir = args.output_dir or OUT_ROOT
    payload = run_test(device, rounds=args.rounds, output_dir=output_dir)
    print(json.dumps(payload["test_metrics"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
