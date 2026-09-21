#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "experiment/exp43_rl_action_policy_ablation_20260911/scripts/run_exp43_rl_action_policy_ablation_20260911.py"
OUT_DIR = Path(__file__).resolve().parent / "log_performance_cost"


def load_base():
    spec = importlib.util.spec_from_file_location("exp43_performance_cost_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact(result: dict[str, object]) -> dict[str, float]:
    metrics = result["metrics"]
    small = result["small_object_final_recall_05"]
    return {
        "prediction_boxes": float(metrics["prediction_box_count"]),
        "acc_05": float(metrics["acc_05"]),
        "acc_075": float(metrics["acc_075"]),
        "map_05": float(metrics["map_05"]),
        "small_final_recall_05": float(small["recall_05"]),
        "actions_per_image": float(result["average_actions_per_image"]),
    }


def main() -> None:
    m = load_base()
    m.LOG_DIR = OUT_DIR
    m.PRED_ROOT = OUT_DIR / "predictions"
    m.require_exp43_inputs()

    rows = {row["stem"]: row for row in m.load_json(m.DYNAMIC_TRACE)}
    stems = m.read_split_stems("test")
    query_plan = {stem: [0] for stem in stems}
    all_qwen_plan = {
        stem: [m.SOURCE_TO_ACTION[source] for source in m.ranked_sources(rows[stem])]
        for stem in stems
    }
    rl_plan = {
        stem: [
            0 if str(action) == "STOP" else m.SOURCE_TO_ACTION[str(action)]
            for action in rows[stem].get("actions", [])
        ]
        for stem in stems
    }

    plans = {
        "query_only": query_plan,
        "all_qwen_mapped_actions": all_qwen_plan,
        "qwen_dynamic_rl": rl_plan,
    }
    results: dict[str, dict[str, float]] = {}
    traces: dict[str, object] = {}
    for name, plan in plans.items():
        total_actions = sum(action != 0 for actions in plan.values() for action in actions)
        raw, trace = m.evaluate_method(
            name,
            plan,
            len(stems) if name == "query_only" else 0,
            max(len(stems), total_actions),
            save_predictions=True,
        )
        results[name] = compact(raw)
        traces[name] = trace

    query = results["query_only"]
    full = results["all_qwen_mapped_actions"]
    rl = results["qwen_dynamic_rl"]
    full_gain = full["acc_05"] - query["acc_05"]
    derived = {
        "rl_acc_gain_over_query_pp": (rl["acc_05"] - query["acc_05"]) * 100,
        "rl_map_gain_over_query": rl["map_05"] - query["map_05"],
        "rl_small_recall_gain_over_query_pp": (rl["small_final_recall_05"] - query["small_final_recall_05"]) * 100,
        "rl_retained_acc_gain_pct": (rl["acc_05"] - query["acc_05"]) / full_gain * 100 if full_gain else 0.0,
        "rl_action_reduction_vs_full_pct": (1 - rl["actions_per_image"] / full["actions_per_image"]) * 100,
        "rl_box_reduction_vs_full_pct": (1 - rl["prediction_boxes"] / full["prediction_boxes"]) * 100,
    }

    payload = {
        "experiment": "qwen_dynamic_rl_performance_cost",
        "protocol": {
            "split": "VisDroneSplit1000Guarded/test",
            "images": len(stems),
            "full_baseline": "execute every unique mapped source in each image's cached Qwen output",
            "shared_pipeline": "same initial candidates, merge/dedup, rich-metadata rescoring, class-aware evaluator",
        },
        "results": results,
        "derived": derived,
        "traces": traces,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    labels = {
        "query_only": "Original query",
        "all_qwen_mapped_actions": "All Qwen mapped actions",
        "qwen_dynamic_rl": "Qwen dynamic RL",
    }
    lines = [
        "# Qwen dynamic RL performance-cost comparison",
        "",
        "| Method | Boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 | Small Recall | Actions/image |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in plans:
        row = results[key]
        lines.append(
            f"| {labels[key]} | {row['prediction_boxes']:.0f} | {row['acc_05']:.4f} | "
            f"{row['acc_075']:.4f} | {row['map_05']:.4f} | {row['small_final_recall_05']:.4f} | "
            f"{row['actions_per_image']:.3f} |"
        )
    lines += [
        "",
        f"- RL retains {derived['rl_retained_acc_gain_pct']:.1f}% of the full Qwen Acc@0.5 gain over query-only.",
        f"- RL reduces actions by {derived['rl_action_reduction_vs_full_pct']:.1f}% and final boxes by {derived['rl_box_reduction_vs_full_pct']:.1f}% versus full Qwen expansion.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
