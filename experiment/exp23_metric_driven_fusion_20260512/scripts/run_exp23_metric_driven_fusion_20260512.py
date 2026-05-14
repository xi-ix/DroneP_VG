import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torchvision.ops import nms


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp23_metric_driven_fusion_20260512'
LOG_DIR = EXP_ROOT / 'log'
DATA_ROOT = ROOT / 'dataset/VisDroneSplit1000Guarded'
EXP17_SPLIT_MANIFEST = ROOT / 'experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv'
EXP18_FULL = ROOT / 'experiment/exp18_exp14_method_large_data_20260423/log/predictions/full'
GDINO_FULL = ROOT / 'experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions'

RUN_LOG = LOG_DIR / 'run_log.txt'
PRED_ROOT = LOG_DIR / 'predictions'
GT_ROOT = LOG_DIR / 'gt_xyxy'
SUMMARY_JSON = LOG_DIR / 'exp23_metric_driven_fusion_20260512_summary.json'
SEARCH_CSV = LOG_DIR / 'exp23_metric_driven_fusion_20260512_search_results.csv'
SUMMARY_TRAINVAL_MD = LOG_DIR / 'evaluation_summary_exp23_metric_driven_fusion_trainval_class_aware.md'
SUMMARY_TEST_MD = LOG_DIR / 'evaluation_summary_exp23_metric_driven_fusion_test_class_aware.md'
SUMMARY_FULL_MD = LOG_DIR / 'evaluation_summary_exp23_metric_driven_fusion_full_class_aware.md'

THRESHOLD = 0.01
NMS_THRESHOLD = 0.45
EXP18_SCALE = 1.0
GDINO_SCALE = 1.0


@dataclass
class DetBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str


Box = Tuple[float, float, float, float]


def log(message: str) -> None:
    print(message, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(message + '\n')


def read_split_map() -> Dict[str, List[str]]:
    split_map: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}
    with EXP17_SPLIT_MANIFEST.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            split = row['split']
            if split in split_map:
                split_map[split].append(row['stem'])
    return {split: sorted(stems) for split, stems in split_map.items()}


def read_boxes(path: Path, scale: float, source: str) -> List[DetBox]:
    boxes: List[DetBox] = []
    if not path.exists():
        return boxes
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                continue
            class_id = int(float(parts[0]))
            x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
            score = float(parts[5]) * scale
            if x2 <= x1 or y2 <= y1 or score < THRESHOLD:
                continue
            boxes.append(DetBox(class_id, x1, y1, x2, y2, score, source))
    return boxes


def convert_visdrone_gt_to_xyxy(gt_dir: Path, image_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for label_path in sorted(gt_dir.glob('*.txt')):
        image_path = image_dir / f'{label_path.stem}.jpg'
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        rows: List[str] = []
        with label_path.open('r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 6:
                    continue
                x, y, bw, bh = map(float, parts[0:4])
                class_id = int(float(parts[5]))
                if not (1 <= class_id <= 10):
                    continue
                x1 = max(0.0, min(float(width) - 1.0, x))
                y1 = max(0.0, min(float(height) - 1.0, y))
                x2 = max(x1 + 1e-6, min(float(width), x + bw))
                y2 = max(y1 + 1e-6, min(float(height), y + bh))
                rows.append(f'{class_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n')
        (out_dir / f'{label_path.stem}.txt').write_text(''.join(rows), encoding='utf-8')


def parse_det_line(line: str, is_prediction: bool) -> Optional[Tuple[int, float, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) not in (5, 6):
        return None
    class_id = int(float(parts[0]))
    x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
    score = float(parts[5]) if is_prediction and len(parts) == 6 else 1.0
    if x2 <= x1 or y2 <= y1:
        return None
    return class_id, x1, y1, x2, y2, score


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_detections(gt_dir: Path, pred_dir: Path, iou_thr: float = 0.5) -> Dict[str, float]:
    gt_files = sorted(gt_dir.glob('*.txt'))
    total_gt = 0
    total_pred = 0
    tp_05 = 0
    tp_075 = 0
    cls_gt_count = {c: 0 for c in range(1, 11)}

    for gt_path in gt_files:
        pred_path = pred_dir / f'{gt_path.stem}.txt'
        gt_objs = []
        for line in gt_path.read_text(encoding='utf-8').splitlines():
            parsed = parse_det_line(line, is_prediction=False)
            if parsed is None:
                continue
            cls, x1, y1, x2, y2, _ = parsed
            gt_objs.append((cls, (x1, y1, x2, y2)))
        pred_objs = []
        if pred_path.exists():
            for line in pred_path.read_text(encoding='utf-8').splitlines():
                parsed = parse_det_line(line, is_prediction=True)
                if parsed is None:
                    continue
                cls, x1, y1, x2, y2, score = parsed
                pred_objs.append((cls, (x1, y1, x2, y2), score))
        total_gt += len(gt_objs)
        total_pred += len(pred_objs)
        for cls, _ in gt_objs:
            if cls in cls_gt_count:
                cls_gt_count[cls] += 1

        used = [False] * len(gt_objs)
        for cls, box, _ in sorted(pred_objs, key=lambda item: item[2], reverse=True):
            best_iou = 0.0
            best_idx = -1
            for idx, (gt_cls, gt_box) in enumerate(gt_objs):
                if used[idx] or gt_cls != cls:
                    continue
                cur_iou = iou(box, gt_box)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= 0.5:
                used[best_idx] = True
                tp_05 += 1
                if best_iou >= 0.75:
                    tp_075 += 1

    aps = []
    for cls in range(1, 11):
        gt_count = cls_gt_count[cls]
        if gt_count == 0:
            continue
        per_cls_preds: List[Tuple[float, int]] = []
        for gt_path in gt_files:
            pred_path = pred_dir / f'{gt_path.stem}.txt'
            gt_boxes = []
            for line in gt_path.read_text(encoding='utf-8').splitlines():
                parsed = parse_det_line(line, is_prediction=False)
                if parsed is None:
                    continue
                c, x1, y1, x2, y2, _ = parsed
                if c == cls:
                    gt_boxes.append((x1, y1, x2, y2))
            pred_boxes = []
            if pred_path.exists():
                for line in pred_path.read_text(encoding='utf-8').splitlines():
                    parsed = parse_det_line(line, is_prediction=True)
                    if parsed is None:
                        continue
                    c, x1, y1, x2, y2, score = parsed
                    if c == cls:
                        pred_boxes.append((score, (x1, y1, x2, y2)))
            used = [False] * len(gt_boxes)
            for score, box in sorted(pred_boxes, key=lambda item: item[0], reverse=True):
                best_iou = 0.0
                best_idx = -1
                for idx, gt_box in enumerate(gt_boxes):
                    if used[idx]:
                        continue
                    cur_iou = iou(box, gt_box)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_idx = idx
                if best_idx >= 0 and best_iou >= iou_thr:
                    used[best_idx] = True
                    per_cls_preds.append((score, 1))
                else:
                    per_cls_preds.append((score, 0))
        if not per_cls_preds:
            aps.append(0.0)
            continue
        tp_cum = 0.0
        fp_cum = 0.0
        recalls = [0.0]
        precisions = [0.0]
        for _, label in sorted(per_cls_preds, key=lambda item: item[0], reverse=True):
            if label:
                tp_cum += 1.0
            else:
                fp_cum += 1.0
            recalls.append(tp_cum / max(1, gt_count))
            precisions.append(tp_cum / max(tp_cum + fp_cum, 1e-9))
        recalls.append(1.0)
        precisions.append(0.0)
        for idx in range(len(precisions) - 2, -1, -1):
            precisions[idx] = max(precisions[idx], precisions[idx + 1])
        ap = 0.0
        for idx in range(1, len(recalls)):
            if recalls[idx] != recalls[idx - 1]:
                ap += (recalls[idx] - recalls[idx - 1]) * precisions[idx]
        aps.append(ap)

    return {
        'file_count': float(len(gt_files)),
        'gt_box_count': float(total_gt),
        'prediction_box_count': float(total_pred),
        'acc_05': float(tp_05 / total_gt) if total_gt > 0 else 0.0,
        'acc_075': float(tp_075 / total_gt) if total_gt > 0 else 0.0,
        'map_05': float(sum(aps) / len(aps)) if aps else 0.0,
        'tp_05': float(tp_05),
        'tp_075': float(tp_075),
    }


def class_aware_nms(boxes: List[DetBox], thr: float) -> List[DetBox]:
    kept: List[DetBox] = []
    by_class: Dict[int, List[DetBox]] = {}
    for box in boxes:
        by_class.setdefault(box.class_id, []).append(box)
    for items in by_class.values():
        if not items:
            continue
        t_boxes = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in items], dtype=torch.float32)
        t_scores = torch.tensor([b.score for b in items], dtype=torch.float32)
        keep = nms(t_boxes, t_scores, thr).cpu().tolist()
        kept.extend(items[idx] for idx in keep)
    return sorted(kept, key=lambda b: b.score, reverse=True)


def fuse_stem(stem: str) -> List[DetBox]:
    boxes = []
    boxes.extend(read_boxes(EXP18_FULL / f'{stem}.txt', EXP18_SCALE, 'exp18'))
    boxes.extend(read_boxes(GDINO_FULL / f'{stem}.txt', GDINO_SCALE, 'gdino'))
    return class_aware_nms(boxes, NMS_THRESHOLD)


def save_predictions(stems: Sequence[str], out_dir: Path) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for stem in stems:
        boxes = fuse_stem(stem)
        total += len(boxes)
        with (out_dir / f'{stem}.txt').open('w', encoding='utf-8') as f:
            for box in boxes:
                f.write(f'{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {box.score:.6f}\n')
    return total


def convert_gt_dirs() -> None:
    GT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        ann_dir = DATA_ROOT / f'VisDrone2019-DET-{split}' / 'annotations'
        img_dir = DATA_ROOT / f'VisDrone2019-DET-{split}' / 'images'
        convert_visdrone_gt_to_xyxy(ann_dir, img_dir, GT_ROOT / split)
    for out_name, splits in [('trainval', ['train', 'val']), ('full', ['train', 'val', 'test'])]:
        out_dir = GT_ROOT / out_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            for path in (GT_ROOT / split).glob('*.txt'):
                shutil.copy2(path, out_dir / path.name)


def write_summary(path: Path, title: str, metrics: Dict[str, float]) -> None:
    lines = [
        f'# {title}',
        '',
        f"- 文件数: {metrics.get('file_count', 0)}",
        f"- GT框数: {metrics.get('gt_box_count', 0)}",
        f"- 预测框数: {metrics.get('prediction_box_count', 0)}",
        f"- Acc@0.5: {metrics.get('acc_05', 0.0):.4f}",
        f"- Acc@0.75: {metrics.get('acc_075', 0.0):.4f}",
        f"- mAP@0.5: {metrics.get('map_05', 0.0):.4f}",
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    if not EXP18_FULL.exists():
        raise RuntimeError(f'Missing Exp18 predictions: {EXP18_FULL}')
    if not GDINO_FULL.exists():
        raise RuntimeError(f'Missing GroundingDINO predictions: {GDINO_FULL}')

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')
    split_map = read_split_map()
    trainval_stems = split_map['train'] + split_map['val']
    full_stems = trainval_stems + split_map['test']

    log('[Start] Exp23 metric-driven fusion restore')
    log(f'[Config] threshold={THRESHOLD:.2f} nms={NMS_THRESHOLD:.2f} sources=exp18+gdino')
    convert_gt_dirs()

    trainval_pred_count = save_predictions(trainval_stems, PRED_ROOT / 'trainval')
    test_pred_count = save_predictions(split_map['test'], PRED_ROOT / 'test')
    full_pred_count = save_predictions(full_stems, PRED_ROOT / 'full')

    trainval_metrics = evaluate_detections(GT_ROOT / 'trainval', PRED_ROOT / 'trainval', iou_thr=0.5)
    test_metrics = evaluate_detections(GT_ROOT / 'test', PRED_ROOT / 'test', iou_thr=0.5)
    full_metrics = evaluate_detections(GT_ROOT / 'full', PRED_ROOT / 'full', iou_thr=0.5)

    write_summary(SUMMARY_TRAINVAL_MD, 'Exp23 TrainVal Fusion Summary', trainval_metrics)
    write_summary(SUMMARY_TEST_MD, 'Exp23 Test Fusion Summary', test_metrics)
    write_summary(SUMMARY_FULL_MD, 'Exp23 Full Fusion Summary', full_metrics)

    with SEARCH_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sources', 'exp18_scale', 'gdino_scale', 'threshold', 'nms', 'val_acc_05', 'val_map_05'])
        writer.writerow(['exp18+gdino', EXP18_SCALE, GDINO_SCALE, THRESHOLD, NMS_THRESHOLD, trainval_metrics['acc_05'], trainval_metrics['map_05']])

    summary = {
        'seed': 42,
        'data_root': str(DATA_ROOT),
        'sources': {
            'exp18': str(EXP18_FULL),
            'gdino': str(GDINO_FULL),
        },
        'best_config': {
            'sources': 'exp18+gdino',
            'exp18_scale': EXP18_SCALE,
            'gdino_scale': GDINO_SCALE,
            'threshold': THRESHOLD,
            'nms': NMS_THRESHOLD,
            'topk': 'all',
        },
        'prediction_counts_from_loop': {
            'trainval': trainval_pred_count,
            'test': test_pred_count,
            'full': full_pred_count,
        },
        'trainval_metrics': trainval_metrics,
        'test_metrics': test_metrics,
        'full_metrics': full_metrics,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log(f"[Done] full acc05={full_metrics['acc_05']:.4f} acc075={full_metrics['acc_075']:.4f} map05={full_metrics['map_05']:.4f} pred={int(full_metrics['prediction_box_count'])}")


if __name__ == '__main__':
    main()
