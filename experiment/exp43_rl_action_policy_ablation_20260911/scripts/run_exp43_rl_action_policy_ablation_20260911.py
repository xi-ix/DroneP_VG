#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence


ROOT = Path(__file__).resolve().parents[3]
RL38_ROOT = ROOT / "38+RL_loop"
sys.path.insert(0, str(RL38_ROOT))

from common import (  # noqa: E402
    ACTION_NAMES,
    CLASS_NAMES,
    OUT_ROOT as RL38_OUT,
    SMALL_CLASSES,
    Det,
    apply_action_to_pool,
    evaluate_split,
    exp38_rescore,
    image_size,
    iou,
    merge_action_pool,
    read_gt,
    read_pred_file,
    read_split_stems,
    require_inputs,
    write_predictions,
)


EXP_NAME = "exp43_rl_action_policy_ablation_20260911"
EXP_ROOT = ROOT / "experiment" / EXP_NAME
LOG_DIR = EXP_ROOT / "log"
PRED_ROOT = LOG_DIR / "predictions"
SUMMARY_JSON = LOG_DIR / f"{EXP_NAME}_summary.json"
SUMMARY_MD = LOG_DIR / "evaluation_summary_test_class_aware.md"
TRACE_JSON = LOG_DIR / "ablation_trace.json"
METRICS_CSV = LOG_DIR / "metrics.csv"
FIG_DIR = LOG_DIR / "figures"

FIXED_TRACE = RL38_OUT / "closed_loop_trace.json"
DYNAMIC_TRACE = RL38_OUT / "llm_dynamic" / "closed_loop_trace.json"
FIXED_CKPT = RL38_OUT / "rl_policy_exp38_reward.pt"
DYNAMIC_CKPT = RL38_OUT / "llm_dynamic" / "dynamic_prompt_policy.pt"
PROMPT_CACHE = RL38_OUT / "llm_dynamic" / "llm_prompt_pools.json"

RANDOM_SEEDS = [42, 43, 44, 45, 46]
SMALL_AREA_PX = 32 * 32
FIXED_ACTION_IDS = list(range(1, len(ACTION_NAMES)))
SOURCE_TO_ACTION = {
    "gdino_base": 1,
    "person": 2,
    "people": 3,
    "group of people": 4,
    "tricycle": 5,
    "covered tricycle": 6,
    "awning tricycle": 7,
    "motorcycle": 8,
    "motorbike": 9,
    "scooter": 10,
    "bicycle": 11,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_exp43_inputs() -> None:
    require_inputs()
    missing = [
        path
        for path in [FIXED_TRACE, DYNAMIC_TRACE, FIXED_CKPT, DYNAMIC_CKPT, PROMPT_CACHE]
        if not path.exists()
    ]
    if missing:
        raise RuntimeError("Missing Exp43 dependency:\n  " + "\n  ".join(str(path) for path in missing))


def fixed_action_ids(trace_row: dict[str, object]) -> list[int]:
    if "action_ids" in trace_row:
        return [int(value) for value in trace_row["action_ids"]]
    return [ACTION_NAMES.index(str(name)) for name in trace_row.get("actions", [])]


def ranked_sources(trace_row: dict[str, object]) -> list[str]:
    ranked = [str(value) for value in trace_row.get("mapped_sources", [])]
    result: list[str] = []
    for source in ranked:
        if source in SOURCE_TO_ACTION and source not in result:
            result.append(source)
    return result


def apply_actions(stem: str, action_ids: Sequence[int]) -> tuple[list[Det], list[list[Det]]]:
    current = merge_action_pool("test", stem, 0)
    rounds = [list(current)]
    for action_id in action_ids:
        current = apply_action_to_pool("test", stem, current, action_id)
        rounds.append(list(current))
    return list(current), rounds


def is_small_gt(box: Det) -> bool:
    return (box.x2 - box.x1) * (box.y2 - box.y1) <= SMALL_AREA_PX


def subset_recall(predictions: dict[str, Sequence[Det]], predicate, oracle: bool = False) -> dict[str, float]:
    matched = total = 0
    for stem in read_split_stems("test"):
        gt_boxes = [box for box in read_gt("test", stem) if predicate(box)]
        total += len(gt_boxes)
        if oracle:
            matched += sum(
                any(pred.class_id == gt.class_id and iou(pred, gt) >= 0.5 for pred in predictions[stem])
                for gt in gt_boxes
            )
            continue
        used = [False] * len(gt_boxes)
        for pred in sorted(predictions[stem], key=lambda item: item.score, reverse=True):
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(gt_boxes):
                if used[idx] or pred.class_id != gt.class_id:
                    continue
                overlap = iou(pred, gt)
                if overlap > best_iou:
                    best_idx = idx
                    best_iou = overlap
            if best_idx >= 0 and best_iou >= 0.5:
                used[best_idx] = True
                matched += 1
    return {"matched": float(matched), "gt_count": float(total), "recall_05": matched / total if total else 0.0}


def normalize_three_rounds(rounds: list[list[Det]]) -> list[list[Det]]:
    result = list(rounds[:3])
    while len(result) < 3:
        result.append(list(result[-1]))
    return result


def evaluate_method(
    name: str,
    action_plan: dict[str, list[int]],
    stop_decisions: int,
    total_decisions: int,
    save_predictions: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidate_predictions: dict[str, list[Det]] = {}
    final_predictions: dict[str, list[Det]] = {}
    trace: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()
    round_candidate_totals = [0, 0, 0]
    round_action_totals = [0, 0]
    for stem in read_split_stems("test"):
        action_ids = action_plan[stem]
        candidate_boxes, rounds = apply_actions(stem, action_ids)
        if name == "all_expansion":
            rounds = [rounds[0], rounds[-1], rounds[-1]]
        else:
            rounds = normalize_three_rounds(rounds)
        final_boxes, _ = exp38_rescore("test", stem, candidate_boxes)
        candidate_predictions[stem] = candidate_boxes
        final_predictions[stem] = final_boxes
        for idx, boxes in enumerate(rounds):
            round_candidate_totals[idx] += len(boxes)
        if name == "all_expansion":
            round_action_totals[0] += len([action for action in action_ids if action != 0])
        else:
            for idx, action_id in enumerate(action_ids[:2]):
                round_action_totals[idx] += int(action_id != 0)
        action_counts.update(ACTION_NAMES[action_id] for action_id in action_ids)
        trace.append(
            {
                "stem": stem,
                "actions": [ACTION_NAMES[action_id] for action_id in action_ids],
                "candidate_count_by_round": [len(boxes) for boxes in rounds],
                "final_count": len(final_boxes),
            }
        )

    pred_dir = PRED_ROOT / name
    if save_predictions:
        write_predictions(pred_dir, final_predictions)
        metrics = evaluate_split("test", pred_dir)
    else:
        metrics = evaluate_predictions(final_predictions)
    small_candidate = subset_recall(candidate_predictions, is_small_gt, oracle=True)
    small_final = subset_recall(final_predictions, is_small_gt)
    weak_candidate = subset_recall(candidate_predictions, lambda box: box.class_id in SMALL_CLASSES, oracle=True)
    weak_final = subset_recall(final_predictions, lambda box: box.class_id in SMALL_CLASSES)
    image_count = len(final_predictions)
    result: dict[str, object] = {
        "method": name,
        "metrics": metrics,
        "small_candidate_recall_05": small_candidate,
        "small_object_final_recall_05": small_final,
        "weak_class_candidate_recall_05": weak_candidate,
        "weak_class_final_recall_05": weak_final,
        "action_counts": dict(sorted(action_counts.items())),
        "expansion_action_count": sum(action != 0 for actions in action_plan.values() for action in actions),
        "average_actions_per_image": sum(action != 0 for actions in action_plan.values() for action in actions) / max(1, image_count),
        "stop_decisions": stop_decisions,
        "total_decisions": total_decisions,
        "stop_ratio": stop_decisions / total_decisions if total_decisions else None,
        "round_action_counts": [0, *round_action_totals],
        "round_candidate_counts": round_candidate_totals,
    }
    return result, trace


def evaluate_predictions(predictions: dict[str, Sequence[Det]]) -> dict[str, float]:
    temp_dir = LOG_DIR / ".tmp_evaluation"
    write_predictions(temp_dir, predictions)
    read_pred_file.cache_clear()
    metrics = evaluate_split("test", temp_dir)
    read_pred_file.cache_clear()
    for path in temp_dir.glob("*.txt"):
        path.unlink()
    temp_dir.rmdir()
    return metrics


def aggregate_random_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    fields = [
        ("metrics", "prediction_box_count"),
        ("metrics", "acc_05"),
        ("metrics", "acc_075"),
        ("metrics", "map_05"),
        ("small_candidate_recall_05", "recall_05"),
        ("small_object_final_recall_05", "recall_05"),
        ("weak_class_candidate_recall_05", "recall_05"),
        ("weak_class_final_recall_05", "recall_05"),
    ]
    aggregate: dict[str, object] = {"method": "random_action", "seeds": RANDOM_SEEDS, "runs": runs}
    for section, field in fields:
        values = [float(run[section][field]) for run in runs]
        aggregate.setdefault(section, {})[field] = mean(values)
        aggregate[section][f"{field}_std"] = pstdev(values)
    for field in [
        "expansion_action_count",
        "average_actions_per_image",
        "stop_decisions",
        "total_decisions",
        "stop_ratio",
        "round_action_counts",
        "round_candidate_counts",
    ]:
        aggregate[field] = runs[0][field]
    combined = Counter()
    for run in runs:
        combined.update(run["action_counts"])
    aggregate["action_counts"] = {key: value / len(runs) for key, value in sorted(combined.items())}
    return aggregate


def source_group(action_name: str) -> str:
    name = action_name.lower().replace("add_", "").replace("_", " ")
    if name in {"person"}:
        return "person"
    if name in {"people", "group of people"}:
        return "people"
    if name in {"bicycle"}:
        return "bicycle"
    if name in {"motorcycle", "motorbike", "scooter"}:
        return "motor"
    if name in {"tricycle", "covered tricycle", "awning tricycle"}:
        return "tricycle"
    return name


def make_figures(results: dict[str, dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["No exp.", "All exp.", "Random", "Prompt-rank", "Fixed RL", "Qwen dyn. RL"]
    keys = ["no_expansion", "all_expansion", "random_action", "prompt_rank", "fixed_action_rl", "qwen_dynamic_rl"]
    colors = ["#777777", "#D55E00", "#E69F00", "#009E73", "#0072B2", "#CC79A7"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = list(range(len(keys)))
    width = 0.36
    axes[0].bar([v - width / 2 for v in x], [results[k]["metrics"]["acc_05"] for k in keys], width, label="Acc@0.5", color="#0072B2")
    axes[0].bar([v + width / 2 for v in x], [results[k]["metrics"]["map_05"] for k in keys], width, label="mAP@0.5", color="#E69F00")
    axes[0].set_ylim(0, 0.6)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False)
    axes[1].bar(x, [results[k]["metrics"]["prediction_box_count"] for k in keys], color=colors)
    axes[1].set_ylabel("Prediction boxes")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "main_metrics_and_candidates.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    rounds = [1, 2, 3]
    for key, label, color in zip(keys, labels, colors):
        axes[0].plot(rounds, results[key]["round_candidate_counts"], marker="o", label=label, color=color)
        axes[1].plot(rounds, results[key]["round_action_counts"], marker="o", label=label, color=color)
    axes[0].set_ylabel("Candidate boxes")
    axes[1].set_ylabel("Actions executed")
    for axis in axes:
        axis.set_xlabel("Round")
        axis.set_xticks(rounds)
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "round_actions_and_candidates.png", dpi=200)
    plt.close(fig)

    groups = ["person", "people", "bicycle", "motor", "tricycle", "gdino base"]
    policies = ["fixed_action_rl", "qwen_dynamic_rl"]
    grouped_counts: dict[str, Counter[str]] = {}
    for policy in policies:
        grouped_counts[policy] = Counter()
        for action, count in results[policy]["action_counts"].items():
            if action != "STOP":
                grouped_counts[policy][source_group(action)] += count
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, policy in enumerate(policies):
        offset = -0.18 if idx == 0 else 0.18
        ax.bar([value + offset for value in range(len(groups))], [grouped_counts[policy][group] for group in groups], 0.36, label=policy.replace("_", " "))
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel("Selected expansion actions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rl_action_class_distribution.png", dpi=200)
    plt.close(fig)


def fmt(value: float, std: float | None = None) -> str:
    if std is None:
        return f"{value:.4f}"
    return f"{value:.4f} +/- {std:.4f}"


def write_reports(results: dict[str, dict[str, object]], traces: dict[str, object]) -> None:
    order = ["no_expansion", "all_expansion", "random_action", "prompt_rank", "fixed_action_rl", "qwen_dynamic_rl"]
    labels = {
        "no_expansion": "No expansion",
        "all_expansion": "All expansion",
        "random_action": "Random action (5 seeds)",
        "prompt_rank": "Prompt-rank",
        "fixed_action_rl": "Fixed-action RL",
        "qwen_dynamic_rl": "Qwen dynamic RL",
    }
    payload = {
        "experiment": EXP_NAME,
        "split": "VisDroneSplit1000Guarded test (150 images)",
        "protocol": {
            "class_aware": True,
            "small_object_definition": f"GT pixel area <= {SMALL_AREA_PX} (COCO small, 32x32)",
            "small_candidate_stage": "oracle candidate coverage in the deduplicated pool before Exp38 rescoring",
            "small_final_stage": "prediction pool after Exp38 rescoring",
            "weak_classes": {str(class_id): CLASS_NAMES[class_id] for class_id in sorted(SMALL_CLASSES)},
            "random_seeds": RANDOM_SEEDS,
            "random_budget": "per-image count of non-STOP actions selected by Fixed-action RL",
            "prompt_rank_budget": "same per-image action budget as Fixed-action RL; select unique mapped sources in Qwen output order",
            "stop_ratio": "STOP decisions / policy decisions; non-policy baselines report 0 except No expansion, represented as immediate STOP",
        },
        "dependencies": {
            "fixed_checkpoint": str(FIXED_CKPT),
            "dynamic_checkpoint": str(DYNAMIC_CKPT),
            "fixed_trace": str(FIXED_TRACE),
            "dynamic_trace": str(DYNAMIC_TRACE),
            "prompt_cache": str(PROMPT_CACHE),
        },
        "results": results,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_JSON.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with METRICS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "pred_boxes", "acc_05", "acc_075", "map_05", "avg_actions", "stop_ratio", "small_candidate_recall_05", "small_final_recall_05"])
        for key in order:
            row = results[key]
            writer.writerow([key, row["metrics"]["prediction_box_count"], row["metrics"]["acc_05"], row["metrics"]["acc_075"], row["metrics"]["map_05"], row["average_actions_per_image"], row["stop_ratio"], row["small_candidate_recall_05"]["recall_05"], row["small_object_final_recall_05"]["recall_05"]])

    lines = [
        "# Exp43 RL Action-policy Ablation",
        "",
        "- Split: `VisDroneSplit1000Guarded/test`, 150 images, class-aware matching.",
        "- Every method starts from the same first-round query-aware candidates and uses the same Exp38 rescoring.",
        "- Small object: GT pixel area `<= 32 x 32`; candidate recall is measured before rescoring, final recall after rescoring.",
        "- Random action reports population mean and standard deviation over seeds `42..46`.",
        "",
        "| Method | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 | Avg actions/image | STOP ratio | Small cand. R@0.5 | Small final R@0.5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in order:
        row = results[key]
        metric_std = row["metrics"].get("acc_05_std")
        pred_std = row["metrics"].get("prediction_box_count_std")
        stop = "N/A" if row["stop_ratio"] is None else f"{100 * row['stop_ratio']:.1f}%"
        lines.append(
            f"| {labels[key]} | {fmt(row['metrics']['prediction_box_count'], pred_std)} | "
            f"{fmt(row['metrics']['acc_05'], metric_std)} | {fmt(row['metrics']['acc_075'], row['metrics'].get('acc_075_std'))} | "
            f"{fmt(row['metrics']['map_05'], row['metrics'].get('map_05_std'))} | {row['average_actions_per_image']:.3f} | {stop} | "
            f"{fmt(row['small_candidate_recall_05']['recall_05'], row['small_candidate_recall_05'].get('recall_05_std'))} | "
            f"{fmt(row['small_object_final_recall_05']['recall_05'], row['small_object_final_recall_05'].get('recall_05_std'))} |"
        )
    fixed = results["fixed_action_rl"]
    random_result = results["random_action"]
    lines += [
        "",
        "## Controlled comparison",
        "",
        f"Fixed-action RL and Random action use the same mean budget ({fixed['average_actions_per_image']:.3f} expansion actions/image). "
        f"Their Acc@0.5 values are {fixed['metrics']['acc_05']:.4f} and {random_result['metrics']['acc_05']:.4f}, respectively; "
        f"their mAP@0.5 values are {fixed['metrics']['map_05']:.4f} and {random_result['metrics']['map_05']:.4f}.",
        f"Fixed-action RL improves Acc@0.5 by {(fixed['metrics']['acc_05'] - random_result['metrics']['acc_05']) * 100:.2f} percentage points and "
        f"small final recall by {(fixed['small_object_final_recall_05']['recall_05'] - random_result['small_object_final_recall_05']['recall_05']) * 100:.2f} points over Random action at the same action budget.",
        f"All expansion reaches the highest absolute scores, but uses {results['all_expansion']['average_actions_per_image'] / fixed['average_actions_per_image']:.2f}x as many actions per image and "
        f"produces {results['all_expansion']['metrics']['prediction_box_count'] / fixed['metrics']['prediction_box_count']:.2f}x as many final boxes as Fixed-action RL. "
        "The supported conclusion is therefore action efficiency, not that exhaustive expansion cannot improve accuracy.",
        f"Prompt-rank obtains mAP@0.5={results['prompt_rank']['metrics']['map_05']:.4f}, while Qwen dynamic RL obtains "
        f"mAP@0.5={results['qwen_dynamic_rl']['metrics']['map_05']:.4f} with {int(results['qwen_dynamic_rl']['metrics']['prediction_box_count'])} boxes; "
        "neither dynamic method dominates Fixed-action RL on every metric.",
        "",
        "## Figures",
        "",
        "- `figures/main_metrics_and_candidates.png`",
        "- `figures/round_actions_and_candidates.png`",
        "- `figures/rl_action_class_distribution.png`",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, dict[str, object]]:
    require_exp43_inputs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fixed_rows = {row["stem"]: row for row in load_json(FIXED_TRACE)}
    dynamic_rows = {row["stem"]: row for row in load_json(DYNAMIC_TRACE)}
    stems = read_split_stems("test")
    if set(stems) != set(fixed_rows) or set(stems) != set(dynamic_rows):
        raise RuntimeError("Saved RL traces do not exactly match the test split.")

    fixed_plan = {stem: fixed_action_ids(fixed_rows[stem]) for stem in stems}
    fixed_budgets = {stem: sum(action != 0 for action in fixed_plan[stem]) for stem in stems}
    no_plan = {stem: [0] for stem in stems}
    all_plan = {stem: list(FIXED_ACTION_IDS) for stem in stems}
    prompt_plan = {
        stem: [SOURCE_TO_ACTION[source] for source in ranked_sources(dynamic_rows[stem])[: fixed_budgets[stem]]]
        for stem in stems
    }
    dynamic_plan = {
        stem: [0 if str(source) == "STOP" else SOURCE_TO_ACTION[str(source)] for source in dynamic_rows[stem].get("actions", [])]
        for stem in stems
    }

    results: dict[str, dict[str, object]] = {}
    traces: dict[str, object] = {}
    results["no_expansion"], traces["no_expansion"] = evaluate_method("no_expansion", no_plan, len(stems), len(stems))
    results["all_expansion"], traces["all_expansion"] = evaluate_method("all_expansion", all_plan, 0, len(stems) * len(FIXED_ACTION_IDS))

    random_runs = []
    for seed in RANDOM_SEEDS:
        rng = random.Random(seed)
        plan = {stem: rng.sample(FIXED_ACTION_IDS, fixed_budgets[stem]) for stem in stems}
        run_result, run_trace = evaluate_method(f"random_action_seed{seed}", plan, 0, sum(len(v) for v in plan.values()), save_predictions=seed == RANDOM_SEEDS[0])
        random_runs.append(run_result)
        traces[f"random_action_seed{seed}"] = run_trace
    results["random_action"] = aggregate_random_runs(random_runs)

    results["prompt_rank"], traces["prompt_rank"] = evaluate_method("prompt_rank", prompt_plan, 0, sum(len(v) for v in prompt_plan.values()))
    fixed_stops = sum(action == 0 for stem in stems for action in fixed_action_ids(fixed_rows[stem]))
    fixed_decisions = sum(len(fixed_action_ids(fixed_rows[stem])) for stem in stems)
    results["fixed_action_rl"], traces["fixed_action_rl"] = evaluate_method("fixed_action_rl", fixed_plan, fixed_stops, fixed_decisions)
    dynamic_stops = sum(str(action) == "STOP" for stem in stems for action in dynamic_rows[stem].get("actions", []))
    dynamic_decisions = sum(len(dynamic_rows[stem].get("actions", [])) for stem in stems)
    results["qwen_dynamic_rl"], traces["qwen_dynamic_rl"] = evaluate_method("qwen_dynamic_rl", dynamic_plan, dynamic_stops, dynamic_decisions)

    make_figures(results)
    write_reports(results, traces)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Exp43 RL action-policy ablation on cached candidates")
    parser.add_argument("--print-json", action="store_true", help="Print the complete result payload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run()
    if args.print_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for name, result in results.items():
            metrics = result["metrics"]
            print(f"{name:20s} pred={metrics['prediction_box_count']:.1f} Acc@0.5={metrics['acc_05']:.4f} Acc@0.75={metrics['acc_075']:.4f} mAP@0.5={metrics['map_05']:.4f}")
        print(f"summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
