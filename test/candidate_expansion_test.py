#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "test/outputs"
SUMMARY_JSON = OUT_ROOT / "candidate_expansion_summary.json"
SUMMARY_MD = OUT_ROOT / "candidate_expansion_summary.md"
PER_CLASS_CSV = OUT_ROOT / "candidate_expansion_per_class.csv"

SPLIT = "test"
SPLIT_MANIFEST = ROOT / "experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv"
GT_ROOT = ROOT / "experiment/exp30_real_query_focal_rerank_20260711/log/gt_xyxy" / SPLIT
CAND_ROOT = ROOT / "RL_loop/outputs/query_prompt_ablation/predictions" / SPLIT
VARIANTS = ["fixed10", "query_only", "query_alias", "query_alias_fallback"]
BASELINE = "fixed10"
EXPANDED = "query_alias_fallback"
CLASS_IDS = list(range(1, 11))
SMALL_CLASSES = {1, 2, 3, 7, 8, 10}
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


def read_stems() -> list[str]:
    stems: list[str] = []
    with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["split"] == SPLIT:
                stems.append(row["stem"])
    return sorted(stems)


def read_dets(path: Path, is_gt: bool = False) -> list[Det]:
    boxes: list[Det] = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if is_gt and len(parts) != 5:
            continue
        if not is_gt and len(parts) != 6:
            continue
        class_id = int(float(parts[0]))
        x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
        score = 1.0 if is_gt else float(parts[5])
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


def covered_gt_indices(gt_boxes: Sequence[Det], pred_boxes: Sequence[Det], thr: float = 0.5) -> set[int]:
    covered: set[int] = set()
    for gt_idx, gt in enumerate(gt_boxes):
        for pred in pred_boxes:
            if pred.class_id == gt.class_id and iou(pred, gt) >= thr:
                covered.add(gt_idx)
                break
    return covered


def hit_pred_indices(pred_boxes: Sequence[Det], gt_boxes: Sequence[Det], thr: float = 0.5) -> set[int]:
    hits: set[int] = set()
    for pred_idx, pred in enumerate(pred_boxes):
        for gt in gt_boxes:
            if pred.class_id == gt.class_id and iou(pred, gt) >= thr:
                hits.add(pred_idx)
                break
    return hits


def unique_added_boxes(expanded: Sequence[Det], baseline: Sequence[Det], dup_thr: float = 0.92) -> list[Det]:
    added: list[Det] = []
    for box in expanded:
        duplicated = False
        for base in baseline:
            if box.class_id == base.class_id and iou(box, base) >= dup_thr:
                duplicated = True
                break
        if not duplicated:
            added.append(box)
    return added


def ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def main() -> None:
    stems = read_stems()
    variant_totals = {
        variant: {"candidates": 0, "covered_gt": 0, "hit_candidates": 0, "small_covered_gt": 0}
        for variant in VARIANTS
    }
    totals = {
        "image_count": len(stems),
        "gt": 0,
        "small_gt": 0,
        "baseline_candidates": 0,
        "expanded_candidates": 0,
        "added_unique_candidates": 0,
        "baseline_covered_gt": 0,
        "expanded_covered_gt": 0,
        "newly_covered_gt": 0,
        "baseline_hit_candidates": 0,
        "expanded_hit_candidates": 0,
        "added_hit_candidates": 0,
        "baseline_small_covered_gt": 0,
        "expanded_small_covered_gt": 0,
        "newly_covered_small_gt": 0,
    }
    per_class = {
        class_id: {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "gt": 0,
            "baseline_covered": 0,
            "expanded_covered": 0,
            "newly_covered": 0,
            "baseline_candidates": 0,
            "expanded_candidates": 0,
            "added_unique_candidates": 0,
            "added_hit_candidates": 0,
        }
        for class_id in CLASS_IDS
    }
    strongest_examples: list[dict[str, object]] = []
    for stem in stems:
        gt_boxes = read_dets(GT_ROOT / f"{stem}.txt", is_gt=True)
        variant_boxes = {variant: read_dets(CAND_ROOT / variant / f"{stem}.txt") for variant in VARIANTS}
        base_boxes = variant_boxes[BASELINE]
        exp_boxes = variant_boxes[EXPANDED]
        added_boxes = unique_added_boxes(exp_boxes, base_boxes)
        base_cov = covered_gt_indices(gt_boxes, base_boxes)
        exp_cov = covered_gt_indices(gt_boxes, exp_boxes)
        new_cov = exp_cov - base_cov
        base_hits = hit_pred_indices(base_boxes, gt_boxes)
        exp_hits = hit_pred_indices(exp_boxes, gt_boxes)
        added_hits = hit_pred_indices(added_boxes, gt_boxes)
        small_gt_indices = {idx for idx, gt in enumerate(gt_boxes) if gt.class_id in SMALL_CLASSES}
        for variant, boxes in variant_boxes.items():
            cov = covered_gt_indices(gt_boxes, boxes)
            hits = hit_pred_indices(boxes, gt_boxes)
            variant_totals[variant]["candidates"] += len(boxes)
            variant_totals[variant]["covered_gt"] += len(cov)
            variant_totals[variant]["hit_candidates"] += len(hits)
            variant_totals[variant]["small_covered_gt"] += len(cov & small_gt_indices)
        totals["gt"] += len(gt_boxes)
        totals["small_gt"] += len(small_gt_indices)
        totals["baseline_candidates"] += len(base_boxes)
        totals["expanded_candidates"] += len(exp_boxes)
        totals["added_unique_candidates"] += len(added_boxes)
        totals["baseline_covered_gt"] += len(base_cov)
        totals["expanded_covered_gt"] += len(exp_cov)
        totals["newly_covered_gt"] += len(new_cov)
        totals["baseline_hit_candidates"] += len(base_hits)
        totals["expanded_hit_candidates"] += len(exp_hits)
        totals["added_hit_candidates"] += len(added_hits)
        totals["baseline_small_covered_gt"] += len(base_cov & small_gt_indices)
        totals["expanded_small_covered_gt"] += len(exp_cov & small_gt_indices)
        totals["newly_covered_small_gt"] += len(new_cov & small_gt_indices)
        if new_cov:
            strongest_examples.append(
                {
                    "stem": stem,
                    "gt_count": len(gt_boxes),
                    "baseline_candidates": len(base_boxes),
                    "expanded_candidates": len(exp_boxes),
                    "added_unique_candidates": len(added_boxes),
                    "newly_covered_gt": len(new_cov),
                    "newly_covered_small_gt": len(new_cov & small_gt_indices),
                }
            )
        for class_id in CLASS_IDS:
            cls_gt = {idx for idx, gt in enumerate(gt_boxes) if gt.class_id == class_id}
            row = per_class[class_id]
            row["gt"] += len(cls_gt)
            row["baseline_covered"] += len(base_cov & cls_gt)
            row["expanded_covered"] += len(exp_cov & cls_gt)
            row["newly_covered"] += len(new_cov & cls_gt)
            row["baseline_candidates"] += sum(1 for box in base_boxes if box.class_id == class_id)
            row["expanded_candidates"] += sum(1 for box in exp_boxes if box.class_id == class_id)
            cls_added = [box for box in added_boxes if box.class_id == class_id]
            row["added_unique_candidates"] += len(cls_added)
            row["added_hit_candidates"] += len(hit_pred_indices(cls_added, gt_boxes))
    metrics = {
        **totals,
        "candidate_expansion_factor": ratio(totals["expanded_candidates"], totals["baseline_candidates"]),
        "candidate_increase": totals["expanded_candidates"] - totals["baseline_candidates"],
        "candidate_increase_rate": ratio(totals["expanded_candidates"] - totals["baseline_candidates"], totals["baseline_candidates"]),
        "unique_added_rate_vs_expanded": ratio(totals["added_unique_candidates"], totals["expanded_candidates"]),
        "baseline_gt_coverage": ratio(totals["baseline_covered_gt"], totals["gt"]),
        "expanded_gt_coverage": ratio(totals["expanded_covered_gt"], totals["gt"]),
        "coverage_gain": ratio(totals["expanded_covered_gt"], totals["gt"]) - ratio(totals["baseline_covered_gt"], totals["gt"]),
        "relative_coverage_gain": ratio(totals["expanded_covered_gt"] - totals["baseline_covered_gt"], totals["baseline_covered_gt"]),
        "baseline_candidate_hit_ratio": ratio(totals["baseline_hit_candidates"], totals["baseline_candidates"]),
        "expanded_candidate_hit_ratio": ratio(totals["expanded_hit_candidates"], totals["expanded_candidates"]),
        "added_candidate_hit_ratio": ratio(totals["added_hit_candidates"], totals["added_unique_candidates"]),
        "baseline_small_gt_coverage": ratio(totals["baseline_small_covered_gt"], totals["small_gt"]),
        "expanded_small_gt_coverage": ratio(totals["expanded_small_covered_gt"], totals["small_gt"]),
        "small_coverage_gain": ratio(totals["expanded_small_covered_gt"], totals["small_gt"]) - ratio(totals["baseline_small_covered_gt"], totals["small_gt"]),
    }
    variant_metrics = {}
    fixed = variant_totals[BASELINE]
    for variant, item in variant_totals.items():
        variant_metrics[variant] = {
            **item,
            "candidate_factor_vs_fixed10": ratio(item["candidates"], fixed["candidates"]),
            "candidate_increase_vs_fixed10": item["candidates"] - fixed["candidates"],
            "gt_coverage": ratio(item["covered_gt"], totals["gt"]),
            "coverage_gain_vs_fixed10": ratio(item["covered_gt"], totals["gt"]) - ratio(fixed["covered_gt"], totals["gt"]),
            "candidate_hit_ratio": ratio(item["hit_candidates"], item["candidates"]),
            "small_gt_coverage": ratio(item["small_covered_gt"], totals["small_gt"]),
            "small_coverage_gain_vs_fixed10": ratio(item["small_covered_gt"], totals["small_gt"]) - ratio(fixed["small_covered_gt"], totals["small_gt"]),
        }
    for row in per_class.values():
        row["candidate_factor"] = ratio(row["expanded_candidates"], row["baseline_candidates"])
        row["baseline_coverage"] = ratio(row["baseline_covered"], row["gt"])
        row["expanded_coverage"] = ratio(row["expanded_covered"], row["gt"])
        row["coverage_gain"] = row["expanded_coverage"] - row["baseline_coverage"]
        row["added_hit_ratio"] = ratio(row["added_hit_candidates"], row["added_unique_candidates"])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "baseline": BASELINE,
                "expanded": EXPANDED,
                "split": SPLIT,
                "metrics": metrics,
                "variant_metrics": variant_metrics,
                "top_new_coverage_examples": sorted(strongest_examples, key=lambda item: item["newly_covered_gt"], reverse=True)[:10],
                "per_class": list(per_class.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with PER_CLASS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(per_class.values())).keys()))
        writer.writeheader()
        writer.writerows(per_class.values())
    lines = [
        "# 第一轮 Query-aware 候选扩充测试",
        "",
        f"- split: `{SPLIT}`",
        f"- baseline: `{BASELINE}`",
        f"- expanded: `{EXPANDED}`",
        f"- images: `{metrics['image_count']}`",
        "",
        "## 总体结果",
        "",
        "| 指标 | fixed10 | query_alias_fallback | 变化 |",
        "| --- | ---: | ---: | ---: |",
        f"| 候选框数量 | {metrics['baseline_candidates']} | {metrics['expanded_candidates']} | +{metrics['candidate_increase']} / {metrics['candidate_increase_rate']:.2%} |",
        f"| GT覆盖数@0.5 | {metrics['baseline_covered_gt']} | {metrics['expanded_covered_gt']} | +{metrics['newly_covered_gt']} |",
        f"| GT覆盖率@0.5 | {metrics['baseline_gt_coverage']:.4f} | {metrics['expanded_gt_coverage']:.4f} | +{metrics['coverage_gain']:.4f} |",
        f"| 候选命中数@0.5 | {metrics['baseline_hit_candidates']} | {metrics['expanded_hit_candidates']} | +{metrics['expanded_hit_candidates'] - metrics['baseline_hit_candidates']} |",
        f"| 候选命中率@0.5 | {metrics['baseline_candidate_hit_ratio']:.4f} | {metrics['expanded_candidate_hit_ratio']:.4f} | {metrics['expanded_candidate_hit_ratio'] - metrics['baseline_candidate_hit_ratio']:+.4f} |",
        f"| 小目标GT覆盖率@0.5 | {metrics['baseline_small_gt_coverage']:.4f} | {metrics['expanded_small_gt_coverage']:.4f} | +{metrics['small_coverage_gain']:.4f} |",
        "",
        "## Prompt变体对比",
        "",
        "| 变体 | 候选框 | 相对fixed10倍数 | GT覆盖数@0.5 | GT覆盖率@0.5 | 候选命中数@0.5 | 候选命中率@0.5 | 小目标GT覆盖率@0.5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        item = variant_metrics[variant]
        lines.append(
            f"| {variant} | {item['candidates']} | {item['candidate_factor_vs_fixed10']:.2f} | {item['covered_gt']} | {item['gt_coverage']:.4f} | {item['hit_candidates']} | {item['candidate_hit_ratio']:.4f} | {item['small_gt_coverage']:.4f} |"
        )
    lines.extend([
        "",
        "## 新增候选贡献",
        "",
        f"- 去掉与 fixed10 同类 IoU>=0.92 的重复框后，新增独有候选：`{metrics['added_unique_candidates']}`",
        f"- 新增独有候选命中 GT 的数量：`{metrics['added_hit_candidates']}`",
        f"- 新增独有候选命中率：`{metrics['added_candidate_hit_ratio']:.4f}`",
        f"- 被 expanded 新覆盖、但 fixed10 没覆盖到的 GT：`{metrics['newly_covered_gt']}`",
        f"- 其中小目标新增覆盖：`{metrics['newly_covered_small_gt']}`",
        "",
        "## 按类别覆盖率",
        "",
        "| 类别 | GT | fixed10覆盖率 | expanded覆盖率 | 提升 | 候选扩充倍数 | 新增候选命中率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in per_class.values():
        lines.append(
            f"| {row['class_id']} {row['class_name']} | {row['gt']} | {row['baseline_coverage']:.4f} | {row['expanded_coverage']:.4f} | {row['coverage_gain']:+.4f} | {row['candidate_factor']:.2f} | {row['added_hit_ratio']:.4f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_MD)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
