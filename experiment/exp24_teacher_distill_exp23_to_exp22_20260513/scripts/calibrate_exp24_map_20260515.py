import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP24_ROOT = ROOT / 'experiment/exp24_teacher_distill_exp23_to_exp22_20260513'
EXP23_ROOT = ROOT / 'experiment/exp23_metric_driven_fusion_20260512'

CASES = {
    'exp23_teacher': {
        'gt_dir': EXP23_ROOT / 'log/gt_xyxy/full',
        'pred_dir': EXP23_ROOT / 'log/predictions/full',
        'historical_map_05': 0.1295,
    },
    'exp24_student': {
        'gt_dir': EXP24_ROOT / 'log/gt_xyxy/full',
        'pred_dir': EXP24_ROOT / 'log/predictions/full',
        'historical_map_05': 0.1234,
    },
}

OUT_JSON = EXP24_ROOT / 'log/map_calibration_exp24_20260515.json'
OUT_MD = EXP24_ROOT / 'log/map_calibration_exp24_20260515.md'

Box = Tuple[float, float, float, float]


def parse_det_line(line: str, is_prediction: bool) -> Optional[Tuple[int, Box, float]]:
    parts = line.strip().split()
    if len(parts) not in (5, 6):
        return None
    class_id = int(float(parts[0]))
    box = tuple(float(v) for v in parts[1:5])
    score = float(parts[5]) if is_prediction and len(parts) == 6 else 1.0
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return class_id, box, score


def read_dets(path: Path, is_prediction: bool) -> List[Tuple[int, Box, float]]:
    if not path.exists():
        return []
    items = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_det_line(line, is_prediction=is_prediction)
            if parsed is not None:
                items.append(parsed)
    return items


def iou(a: Box, b: Box) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def ap_from_labels(labels: Sequence[int], positive_count: int, mode: str) -> float:
    if positive_count <= 0:
        return 0.0
    if not labels:
        return 0.0

    tp = 0.0
    fp = 0.0
    recalls: List[float] = []
    precisions: List[float] = []
    for label in labels:
        if label:
            tp += 1.0
        else:
            fp += 1.0
        recalls.append(tp / positive_count)
        precisions.append(tp / max(tp + fp, 1e-12))

    if mode == 'voc2007_11point':
        thresholds = [idx / 10.0 for idx in range(11)]
        return sum(max([p for r, p in zip(recalls, precisions) if r >= thr] or [0.0]) for thr in thresholds) / 11.0

    if mode == 'continuous':
        mrec = [0.0] + recalls + [1.0]
        mpre = [0.0] + precisions + [0.0]
        for idx in range(len(mpre) - 2, -1, -1):
            mpre[idx] = max(mpre[idx], mpre[idx + 1])
        return sum(
            (mrec[idx] - mrec[idx - 1]) * mpre[idx]
            for idx in range(1, len(mrec))
            if mrec[idx] != mrec[idx - 1]
        )

    raise ValueError(f'Unknown AP mode: {mode}')


def evaluate_map(gt_dir: Path, pred_dir: Path, mode: str) -> Dict[str, object]:
    stems = sorted(path.stem for path in gt_dir.glob('*.txt'))
    gt_by_stem = {stem: read_dets(gt_dir / f'{stem}.txt', is_prediction=False) for stem in stems}
    pred_by_stem = {stem: read_dets(pred_dir / f'{stem}.txt', is_prediction=True) for stem in stems}
    classes = sorted({class_id for items in gt_by_stem.values() for class_id, _, _ in items})

    ap_by_class: Dict[str, float] = {}
    for class_id in classes:
        positive_count = sum(1 for items in gt_by_stem.values() for c, _, _ in items if c == class_id)
        pred_items: List[Tuple[float, str, Box]] = []
        gt_boxes_by_stem: Dict[str, List[Box]] = {}
        used_by_stem: Dict[str, List[bool]] = {}

        for stem in stems:
            gt_boxes = [box for c, box, _ in gt_by_stem[stem] if c == class_id]
            gt_boxes_by_stem[stem] = gt_boxes
            used_by_stem[stem] = [False] * len(gt_boxes)
            for c, box, score in pred_by_stem[stem]:
                if c == class_id:
                    pred_items.append((score, stem, box))

        labels: List[int] = []
        for _, stem, pred_box in sorted(pred_items, key=lambda item: item[0], reverse=True):
            best_iou = 0.0
            best_idx = -1
            for idx, gt_box in enumerate(gt_boxes_by_stem[stem]):
                if used_by_stem[stem][idx]:
                    continue
                cur_iou = iou(pred_box, gt_box)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= 0.5:
                used_by_stem[stem][best_idx] = True
                labels.append(1)
            else:
                labels.append(0)

        ap_by_class[str(class_id)] = ap_from_labels(labels, positive_count, mode)

    return {
        'mode': mode,
        'map_05': sum(ap_by_class.values()) / len(ap_by_class) if ap_by_class else 0.0,
        'ap_by_class': ap_by_class,
    }


def main() -> None:
    results: Dict[str, object] = {}
    for name, config in CASES.items():
        gt_dir = config['gt_dir']
        pred_dir = config['pred_dir']
        if not gt_dir.exists() or not pred_dir.exists():
            raise RuntimeError(f'Missing dirs for {name}: gt={gt_dir} pred={pred_dir}')
        continuous = evaluate_map(gt_dir, pred_dir, mode='continuous')
        voc2007 = evaluate_map(gt_dir, pred_dir, mode='voc2007_11point')
        results[name] = {
            'gt_dir': str(gt_dir),
            'pred_dir': str(pred_dir),
            'historical_map_05': config['historical_map_05'],
            'continuous_ap': continuous,
            'voc2007_11point_ap': voc2007,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# Exp24 mAP Calibration',
        '',
        'Conclusion: historical Exp23/Exp24 records use VOC 2007 11-point AP for mAP@0.5, not continuous AP.',
        '',
        '| case | historical mAP@0.5 | continuous AP mAP@0.5 | VOC2007 11-point mAP@0.5 |',
        '| --- | ---: | ---: | ---: |',
    ]
    for name, item in results.items():
        lines.append(
            f"| {name} | {item['historical_map_05']:.4f} | "
            f"{item['continuous_ap']['map_05']:.4f} | "
            f"{item['voc2007_11point_ap']['map_05']:.4f} |"
        )
    lines.extend([
        '',
        'Use `voc2007_11point_ap.map_05` when comparing against the historical Exp23/Exp24 table.',
    ])
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(OUT_MD.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
