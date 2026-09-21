import csv
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp33_small_candidate_recall_union_20260711'
LOG_DIR = EXP_ROOT / 'log'
RUN_LOG = LOG_DIR / 'run_log.txt'
PRED_ROOT = LOG_DIR / 'predictions'
GT_ROOT = LOG_DIR / 'gt_xyxy'
SUMMARY_JSON = LOG_DIR / 'exp33_small_candidate_recall_union_20260711_summary.json'
SEARCH_CSV = LOG_DIR / 'small_candidate_union_search.csv'
GROUNDING_FULL_JSON = LOG_DIR / 'grounding_eval_full_topk.json'
GROUNDING_FULL_MD = LOG_DIR / 'grounding_eval_full_topk.md'

DATA_ROOT = ROOT / 'dataset/VisDroneSplit1000Guarded'
EXP17_SPLIT_MANIFEST = ROOT / 'experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv'
EXP23_SCRIPT = ROOT / 'experiment/exp23_metric_driven_fusion_20260512/scripts/run_exp23_metric_driven_fusion_20260512.py'

EXP30_ROOT = ROOT / 'experiment/exp30_real_query_focal_rerank_20260711/log'
EXP30_PRED_ROOT = EXP30_ROOT / 'predictions'
EXP30_GT_ROOT = EXP30_ROOT / 'gt_xyxy'

EXP18_PRED_ROOT = ROOT / 'experiment/exp18_exp14_method_large_data_20260423/log/predictions'
GDINO_PRED_ROOT = ROOT / 'experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions'

QUERY_CLASSES = list(range(1, 11))
SMALL_CLASSES = {1, 2, 3, 7, 8, 10}
WEAK_CLASSES = {2, 7, 8, 10}
RECOVERY_CLASSES = {2, 7, 8, 10}

AREA_THRESHOLDS = {
    2: 0.0040,
    7: 0.0045,
    8: 0.0060,
    10: 0.0035,
}

PROFILES = [
    {'name': 'none', 'score_thr': 1.01, 'topk': 0, 'base_score': 0.0, 'area_bonus': 0.0, 'nms': 0.70},
    {'name': 'very_safe', 'score_thr': 0.18, 'topk': 10, 'base_score': 0.018, 'area_bonus': 0.025, 'nms': 0.65},
    {'name': 'safe', 'score_thr': 0.12, 'topk': 20, 'base_score': 0.016, 'area_bonus': 0.035, 'nms': 0.65},
    {'name': 'balanced', 'score_thr': 0.08, 'topk': 35, 'base_score': 0.014, 'area_bonus': 0.050, 'nms': 0.65},
    {'name': 'recall', 'score_thr': 0.04, 'topk': 60, 'base_score': 0.012, 'area_bonus': 0.065, 'nms': 0.70},
    {'name': 'motor_recall', 'score_thr': 0.03, 'topk': 80, 'base_score': 0.010, 'area_bonus': 0.080, 'nms': 0.70, 'class_bonus': {10: 0.035}},
    {'name': 'awning_recall', 'score_thr': 0.04, 'topk': 80, 'base_score': 0.010, 'area_bonus': 0.080, 'nms': 0.70, 'class_bonus': {8: 0.040}},
    {'name': 'rank_safe', 'score_thr': 0.08, 'topk': 25, 'base_score': 0.18, 'area_bonus': 0.12, 'nms': 0.65},
    {'name': 'rank_balanced', 'score_thr': 0.05, 'topk': 45, 'base_score': 0.24, 'area_bonus': 0.18, 'nms': 0.65},
    {'name': 'rank_motor_alias', 'score_thr': 0.03, 'topk': 60, 'base_score': 0.22, 'area_bonus': 0.22, 'nms': 0.70, 'class_bonus': {10: 0.08}},
    {'name': 'rank_awning_alias', 'score_thr': 0.03, 'topk': 60, 'base_score': 0.22, 'area_bonus': 0.22, 'nms': 0.70, 'class_bonus': {8: 0.08}},
]

MAP_DROP_LIMIT = 0.006
R1_DROP_LIMIT = 0.006

RECOVERY_ALIASES = {
    2: {1, 2},
    7: {7, 8},
    8: {7, 8},
    10: {3, 7, 10},
}


@dataclass
class DetBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str = 'base'


def log(message: str) -> None:
    print(message, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(message + '\n')


def load_exp23_helpers():
    spec = importlib.util.spec_from_file_location('exp23_helpers_for_exp33', str(EXP23_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import Exp23 helpers from {EXP23_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp23_helpers_for_exp33'] = module
    spec.loader.exec_module(module)
    return module


EXP23 = load_exp23_helpers()


def read_split_map() -> Dict[str, List[str]]:
    split_map: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}
    with EXP17_SPLIT_MANIFEST.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            split = row['split']
            if split in split_map:
                split_map[split].append(row['stem'])
    return {split: sorted(stems) for split, stems in split_map.items()}


def build_stem_to_split() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for split, stems in read_split_map().items():
        for stem in stems:
            mapping[stem] = split
    return mapping


def image_size(stem: str, stem_to_split: Dict[str, str]) -> Tuple[int, int]:
    split = stem_to_split[stem]
    path = DATA_ROOT / f'VisDrone2019-DET-{split}' / 'images' / f'{stem}.jpg'
    with Image.open(path) as image:
        return image.size


def read_pred_boxes(path: Path, source: str = 'base') -> List[DetBox]:
    boxes: List[DetBox] = []
    if not path.exists():
        return boxes
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) != 6:
                continue
            class_id = int(float(parts[0]))
            x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
            score = float(parts[5])
            if x2 > x1 and y2 > y1:
                boxes.append(DetBox(class_id, x1, y1, x2, y2, score, source))
    return boxes


def read_gt_boxes(path: Path) -> List[DetBox]:
    boxes: List[DetBox] = []
    if not path.exists():
        return boxes
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
            if x2 > x1 and y2 > y1:
                boxes.append(DetBox(class_id, x1, y1, x2, y2, 1.0, 'gt'))
    return boxes


def box_area_ratio(box: DetBox, width: int, height: int) -> float:
    return ((box.x2 - box.x1) * (box.y2 - box.y1)) / max(1.0, width * height)


def small_area_prior(box: DetBox, width: int, height: int) -> float:
    threshold = AREA_THRESHOLDS.get(box.class_id, 0.0)
    if threshold <= 0:
        return 0.0
    area_ratio = box_area_ratio(box, width, height)
    return max(0.0, min(1.0, (threshold - area_ratio) / threshold))


def iou(a: DetBox, b: DetBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def class_aware_nms(boxes: List[DetBox], nms_thr: float) -> List[DetBox]:
    selected: List[DetBox] = []
    by_class: Dict[int, List[DetBox]] = {}
    for box in boxes:
        by_class.setdefault(box.class_id, []).append(box)
    for class_id, items in by_class.items():
        items = sorted(items, key=lambda box: box.score, reverse=True)
        kept: List[DetBox] = []
        for box in items:
            if all(iou(box, prev) < nms_thr for prev in kept):
                kept.append(box)
        selected.extend(kept)
    return sorted(selected, key=lambda box: box.score, reverse=True)


def source_candidate_paths(scope: str, stem: str) -> List[Tuple[Path, str]]:
    exp18_dir = EXP18_PRED_ROOT / scope
    paths = []
    if exp18_dir.exists():
        paths.append((exp18_dir / f'{stem}.txt', 'exp18'))
    paths.append((GDINO_PRED_ROOT / f'{stem}.txt', 'gdino'))
    return paths


def recover_candidates(stem: str, scope: str, profile: Dict[str, object], stem_to_split: Dict[str, str]) -> List[DetBox]:
    width, height = image_size(stem, stem_to_split)
    candidates: List[DetBox] = []
    for path, source in source_candidate_paths(scope, stem):
        for source_box in read_pred_boxes(path, source):
            target_classes = [
                target_class for target_class, source_classes in RECOVERY_ALIASES.items()
                if source_box.class_id in source_classes
            ]
            if not target_classes:
                continue
            if source_box.score < float(profile['score_thr']):
                continue
            for target_class in target_classes:
                box = DetBox(
                    target_class,
                    source_box.x1,
                    source_box.y1,
                    source_box.x2,
                    source_box.y2,
                    source_box.score,
                    source,
                )
                if small_area_prior(box, width, height) <= 0.0:
                    continue
                candidates.append(box)

    class_bonus = profile.get('class_bonus', {})
    rescored: List[DetBox] = []
    for box in candidates:
        prior = small_area_prior(box, width, height)
        bonus = float(class_bonus.get(box.class_id, 0.0))
        new_score = float(profile['base_score']) + 0.05 * box.score + float(profile['area_bonus']) * prior + bonus
        rescored.append(DetBox(box.class_id, box.x1, box.y1, box.x2, box.y2, new_score, box.source))

    selected: List[DetBox] = []
    topk = int(profile['topk'])
    for class_id in RECOVERY_CLASSES:
        class_items = sorted([box for box in rescored if box.class_id == class_id], key=lambda box: box.score, reverse=True)
        selected.extend(class_items[:topk])
    return selected


def union_prediction_dir(scope: str, dst_dir: Path, profile: Dict[str, object], stem_to_split: Dict[str, str]) -> Dict[str, int]:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_dir = EXP30_PRED_ROOT / scope
    added_counts = {str(class_id): 0 for class_id in RECOVERY_CLASSES}
    for src_file in sorted(src_dir.glob('*.txt')):
        stem = src_file.stem
        base_boxes = read_pred_boxes(src_file, 'exp30')
        recovered = recover_candidates(stem, scope, profile, stem_to_split)
        for box in recovered:
            added_counts[str(box.class_id)] += 1
        merged = class_aware_nms(base_boxes + recovered, float(profile['nms']))
        with (dst_dir / src_file.name).open('w', encoding='utf-8') as f:
            for box in merged:
                f.write(f'{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {box.score:.6f}\n')
    return added_counts


def query_hit_at_k(preds: Sequence[DetBox], gt_boxes: Sequence[DetBox], k: int) -> bool:
    for pred in preds[:k]:
        for gt in gt_boxes:
            if pred.class_id == gt.class_id and iou(pred, gt) >= 0.5:
                return True
    return False


def evaluate_prediction_grounding(gt_dir: Path, pred_dir: Path) -> Dict[str, object]:
    total_queries = 0
    hit_counts = {1: 0, 5: 0, 10: 0}
    by_class = {
        str(class_id): {'queries': 0.0, 'recall_at_1': 0.0, 'recall_at_5': 0.0, 'recall_at_10': 0.0}
        for class_id in QUERY_CLASSES
    }
    rank_sum = 0.0
    found_count = 0
    for gt_file in sorted(gt_dir.glob('*.txt')):
        gt_boxes = read_gt_boxes(gt_file)
        pred_boxes = read_pred_boxes(pred_dir / gt_file.name)
        for class_id in QUERY_CLASSES:
            class_gt = [box for box in gt_boxes if box.class_id == class_id]
            if not class_gt:
                continue
            class_preds = sorted([box for box in pred_boxes if box.class_id == class_id], key=lambda box: box.score, reverse=True)
            total_queries += 1
            class_key = str(class_id)
            by_class[class_key]['queries'] += 1.0
            for k in hit_counts:
                if query_hit_at_k(class_preds, class_gt, k):
                    hit_counts[k] += 1
                    by_class[class_key][f'recall_at_{k}'] += 1.0
            for idx, pred in enumerate(class_preds, start=1):
                if query_hit_at_k([pred], class_gt, 1):
                    rank_sum += idx
                    found_count += 1
                    break
    for item in by_class.values():
        queries = item['queries']
        if queries > 0:
            item['recall_at_1'] /= queries
            item['recall_at_5'] /= queries
            item['recall_at_10'] /= queries
    return {
        'query_count': total_queries,
        'recall_at_1': hit_counts[1] / total_queries if total_queries else 0.0,
        'recall_at_5': hit_counts[5] / total_queries if total_queries else 0.0,
        'recall_at_10': hit_counts[10] / total_queries if total_queries else 0.0,
        'mean_first_hit_rank': rank_sum / found_count if found_count else 0.0,
        'by_class': by_class,
    }


def small_metric(metrics: Dict[str, object], key: str, classes=SMALL_CLASSES) -> float:
    total = 0.0
    weighted = 0.0
    for class_id in classes:
        item = metrics['by_class'].get(str(class_id), {})
        queries = float(item.get('queries', 0.0))
        total += queries
        weighted += queries * float(item.get(key, 0.0))
    return weighted / total if total > 0 else 0.0


def write_grounding_summary(path: Path, metrics: Dict[str, object]) -> None:
    lines = [
        '# Exp33 Small Candidate Recall Union Grounding TopK',
        '',
        f"- query_count: {metrics['query_count']}",
        f"- Recall@1: {metrics['recall_at_1']:.4f}",
        f"- Recall@5: {metrics['recall_at_5']:.4f}",
        f"- Recall@10: {metrics['recall_at_10']:.4f}",
        f"- mean_first_hit_rank: {metrics['mean_first_hit_rank']:.2f}",
        '',
        '| class_id | queries | R@1 | R@5 | R@10 |',
        '| ---: | ---: | ---: | ---: | ---: |',
    ]
    for class_id, item in metrics['by_class'].items():
        if item['queries'] <= 0:
            continue
        lines.append(
            f"| {class_id} | {int(item['queries'])} | "
            f"{item['recall_at_1']:.4f} | {item['recall_at_5']:.4f} | {item['recall_at_10']:.4f} |"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def copy_gt_dirs() -> None:
    if GT_ROOT.exists():
        shutil.rmtree(GT_ROOT)
    shutil.copytree(EXP30_GT_ROOT, GT_ROOT)


def search_val(stem_to_split: Dict[str, str]) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, object]]:
    baseline_pred_dir = EXP30_PRED_ROOT / 'val'
    baseline_det = EXP23.evaluate_detections(GT_ROOT / 'val', baseline_pred_dir, iou_thr=0.5)
    baseline_grounding = evaluate_prediction_grounding(GT_ROOT / 'val', baseline_pred_dir)
    baseline_map = float(baseline_det['map_05'])
    baseline_r1 = float(baseline_grounding['recall_at_1'])
    rows: List[Dict[str, object]] = []
    best = None
    for profile in PROFILES:
        out_dir = PRED_ROOT / 'val_search' / profile['name']
        added = union_prediction_dir('val', out_dir, profile, stem_to_split)
        det = EXP23.evaluate_detections(GT_ROOT / 'val', out_dir, iou_thr=0.5)
        grounding = evaluate_prediction_grounding(GT_ROOT / 'val', out_dir)
        small_r1 = small_metric(grounding, 'recall_at_1')
        small_r5 = small_metric(grounding, 'recall_at_5')
        weak_r1 = small_metric(grounding, 'recall_at_1', WEAK_CLASSES)
        weak_r5 = small_metric(grounding, 'recall_at_5', WEAK_CLASSES)
        keep = det['map_05'] >= baseline_map - MAP_DROP_LIMIT and grounding['recall_at_1'] >= baseline_r1 - R1_DROP_LIMIT
        objective = 0.45 * small_r1 + 0.35 * small_r5 + 0.20 * weak_r5 if keep else -1.0
        row = {
            'profile': profile['name'],
            'keep': keep,
            'objective': objective,
            'acc_05': det['acc_05'],
            'acc_075': det['acc_075'],
            'map_05': det['map_05'],
            'prediction_box_count': det['prediction_box_count'],
            'overall_r1': grounding['recall_at_1'],
            'overall_r5': grounding['recall_at_5'],
            'overall_r10': grounding['recall_at_10'],
            'small_r1': small_r1,
            'small_r5': small_r5,
            'weak_r1': weak_r1,
            'weak_r5': weak_r5,
            'added_counts': added,
            'profile_config': profile,
        }
        rows.append(row)
        log(
            f"[Search] {profile['name']} keep={keep} map={det['map_05']:.4f} "
            f"overall_r1={grounding['recall_at_1']:.4f} small_r1={small_r1:.4f} "
            f"small_r5={small_r5:.4f} weak_r5={weak_r5:.4f} added={added}"
        )
        if best is None or row['objective'] > best['objective']:
            best = row
    if best is None or best['objective'] < 0:
        best = max(rows, key=lambda row: row['small_r5'])

    with SEARCH_CSV.open('w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'profile', 'keep', 'objective', 'acc_05', 'acc_075', 'map_05', 'prediction_box_count',
            'overall_r1', 'overall_r5', 'overall_r10', 'small_r1', 'small_r5', 'weak_r1', 'weak_r5',
            'added_counts', 'profile_config',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    baseline = {
        'detection': baseline_det,
        'grounding': baseline_grounding,
        'small_r1': small_metric(baseline_grounding, 'recall_at_1'),
        'small_r5': small_metric(baseline_grounding, 'recall_at_5'),
        'weak_r1': small_metric(baseline_grounding, 'recall_at_1', WEAK_CLASSES),
        'weak_r5': small_metric(baseline_grounding, 'recall_at_5', WEAK_CLASSES),
    }
    return best, rows, baseline


def evaluate_scope(scope: str, profile: Dict[str, object], stem_to_split: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, object], Dict[str, int]]:
    dst_dir = PRED_ROOT / scope
    added = union_prediction_dir(scope, dst_dir, profile['profile_config'], stem_to_split)
    det = EXP23.evaluate_detections(GT_ROOT / scope, dst_dir, iou_thr=0.5)
    grounding = evaluate_prediction_grounding(GT_ROOT / scope, dst_dir)
    return det, grounding, added


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')
    log('[Start] Exp33 small candidate recall union')
    copy_gt_dirs()
    stem_to_split = build_stem_to_split()
    best, rows, baseline = search_val(stem_to_split)
    log(f"[Best] profile={best['profile']} objective={best['objective']:.4f}")

    val_det, val_grounding, val_added = evaluate_scope('val', best, stem_to_split)
    test_det, test_grounding, test_added = evaluate_scope('test', best, stem_to_split)
    trainval_det, trainval_grounding, trainval_added = evaluate_scope('trainval', best, stem_to_split)
    full_det, full_grounding, full_added = evaluate_scope('full', best, stem_to_split)

    GROUNDING_FULL_JSON.write_text(json.dumps(full_grounding, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_grounding_summary(GROUNDING_FULL_MD, full_grounding)
    summary = {
        'method': 'Exp30 final prediction + weak small class candidate recovery from Exp18/GDINO',
        'baseline_exp30_val_prediction_grounding': baseline,
        'constraints': {'map_drop_limit': MAP_DROP_LIMIT, 'r1_drop_limit': R1_DROP_LIMIT},
        'area_thresholds': AREA_THRESHOLDS,
        'best_profile': best,
        'search_rows': rows,
        'val_metrics': val_det,
        'test_metrics': test_det,
        'trainval_metrics': trainval_det,
        'full_metrics': full_det,
        'val_prediction_grounding': val_grounding,
        'test_prediction_grounding': test_grounding,
        'trainval_prediction_grounding': trainval_grounding,
        'full_prediction_grounding': full_grounding,
        'added_counts': {'val': val_added, 'test': test_added, 'trainval': trainval_added, 'full': full_added},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log(
        f"[Done] full map={full_det['map_05']:.4f} acc05={full_det['acc_05']:.4f} "
        f"pred_r1={full_grounding['recall_at_1']:.4f} small_r1={small_metric(full_grounding, 'recall_at_1'):.4f} "
        f"small_r5={small_metric(full_grounding, 'recall_at_5'):.4f} added={full_added}"
    )


if __name__ == '__main__':
    main()
