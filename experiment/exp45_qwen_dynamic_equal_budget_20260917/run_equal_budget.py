#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "experiment/exp43_rl_action_policy_ablation_20260911/scripts/run_exp43_rl_action_policy_ablation_20260911.py"
OUT_DIR = Path(__file__).resolve().parent / "log"
SEEDS = list(range(42, 62))


def load_base():
    spec = importlib.util.spec_from_file_location("exp43_equal_budget_base", BASE_SCRIPT)
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


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        out[key] = mean(values)
        out[f"{key}_std"] = pstdev(values)
    return out


def main():
    m = load_base()
    m.LOG_DIR = OUT_DIR
    m.PRED_ROOT = OUT_DIR / "predictions"
    m.require_exp43_inputs()
    dynamic_rows = {row["stem"]: row for row in m.load_json(m.DYNAMIC_TRACE)}
    stems = m.read_split_stems("test")
    budgets = {
        stem: sum(str(action) != "STOP" for action in dynamic_rows[stem].get("actions", []))
        for stem in stems
    }
    pools = {stem: m.ranked_sources(dynamic_rows[stem]) for stem in stems}
    insufficient = [stem for stem in stems if len(pools[stem]) < budgets[stem]]
    if insufficient:
        raise RuntimeError(f"Mapped Qwen pool is smaller than the dynamic budget: {insufficient[:5]}")

    dynamic_plan = {
        stem: [
            0 if str(action) == "STOP" else m.SOURCE_TO_ACTION[str(action)]
            for action in dynamic_rows[stem].get("actions", [])
        ]
        for stem in stems
    }
    prompt_rank_plan = {
        stem: [m.SOURCE_TO_ACTION[source] for source in pools[stem][: budgets[stem]]]
        for stem in stems
    }

    total_budget = sum(budgets.values())
    dynamic_result, dynamic_trace = m.evaluate_method(
        "qwen_dynamic_rl_equal_budget", dynamic_plan, 0, total_budget, save_predictions=True
    )
    prompt_result, prompt_trace = m.evaluate_method(
        "prompt_rank_equal_budget", prompt_rank_plan, 0, total_budget, save_predictions=True
    )

    random_results = []
    for seed in SEEDS:
        rng = random.Random(seed)
        plan = {
            stem: [m.SOURCE_TO_ACTION[source] for source in rng.sample(pools[stem], budgets[stem])]
            for stem in stems
        }
        result, trace = m.evaluate_method(
            f"random_dynamic_pool_seed{seed}", plan, 0, total_budget, save_predictions=False
        )
        random_results.append(compact(result))

    payload = {
        "experiment": "qwen_dynamic_equal_budget",
        "protocol": {
            "split": "VisDroneSplit1000Guarded/test",
            "images": len(stems),
            "action_pool": "unique mapped sources in each image's Qwen output",
            "budget": "per-image non-STOP action count executed by Qwen dynamic RL",
            "shared_pipeline": "same initial candidates, merge/dedup, rich-metadata rescoring, class-aware evaluator",
            "random_seeds": SEEDS,
            "total_non_stop_actions": total_budget,
            "average_actions_per_image": total_budget / len(stems),
            "budget_distribution": {str(value): list(budgets.values()).count(value) for value in sorted(set(budgets.values()))},
        },
        "results": {
            "qwen_dynamic_rl": compact(dynamic_result),
            "prompt_rank_equal_budget": compact(prompt_result),
            "random_equal_budget": aggregate(random_results),
            "random_runs": random_results,
        },
        "traces": {
            "qwen_dynamic_rl": dynamic_trace,
            "prompt_rank_equal_budget": prompt_trace,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    r = payload["results"]
    lines = [
        "# Qwen dynamic RL equal-budget comparison",
        "",
        "All methods use the same per-image Qwen action pool and the exact non-STOP action budget executed by Qwen dynamic RL.",
        "",
        "| Method | Boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 | Small final recall | Actions/image |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Qwen dynamic RL | {r['qwen_dynamic_rl']['prediction_boxes']:.0f} | {r['qwen_dynamic_rl']['acc_05']:.4f} | {r['qwen_dynamic_rl']['acc_075']:.4f} | {r['qwen_dynamic_rl']['map_05']:.4f} | {r['qwen_dynamic_rl']['small_final_recall_05']:.4f} | {r['qwen_dynamic_rl']['actions_per_image']:.3f} |",
        f"| Prompt-rank (equal budget) | {r['prompt_rank_equal_budget']['prediction_boxes']:.0f} | {r['prompt_rank_equal_budget']['acc_05']:.4f} | {r['prompt_rank_equal_budget']['acc_075']:.4f} | {r['prompt_rank_equal_budget']['map_05']:.4f} | {r['prompt_rank_equal_budget']['small_final_recall_05']:.4f} | {r['prompt_rank_equal_budget']['actions_per_image']:.3f} |",
        f"| Random (20 seeds, equal budget) | {r['random_equal_budget']['prediction_boxes']:.1f} +/- {r['random_equal_budget']['prediction_boxes_std']:.1f} | {r['random_equal_budget']['acc_05']:.4f} +/- {r['random_equal_budget']['acc_05_std']:.4f} | {r['random_equal_budget']['acc_075']:.4f} +/- {r['random_equal_budget']['acc_075_std']:.4f} | {r['random_equal_budget']['map_05']:.4f} +/- {r['random_equal_budget']['map_05_std']:.4f} | {r['random_equal_budget']['small_final_recall_05']:.4f} +/- {r['random_equal_budget']['small_final_recall_05_std']:.4f} | {r['random_equal_budget']['actions_per_image']:.3f} |",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
