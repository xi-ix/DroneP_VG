import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import nms

ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/baseline/groundingdino_base_refdrone100_recovered_20260421'
LOG_DIR = EXP_ROOT / 'log'

DATA_ROOT = ROOT / 'visdrone_test_100'
DATA_IMAGE_DIR = DATA_ROOT / 'images'
DATA_GT_DIR = DATA_ROOT / 'annotations'
MANIFEST = DATA_ROOT / 'manifest.tsv'

GROUNDINGDINO_ROOT = Path('/home/wangzhe/GroundingDINO')
CONFIG_PATH = GROUNDINGDINO_ROOT / 'groundingdino/config/GroundingDINO_SwinT_OGC.py'
CHECKPOINT_PATH = GROUNDINGDINO_ROOT / 'weights/groundingdino_swint_ogc.pth'

BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.20
NMS_THRESHOLD = 0.40

CLASS_NAMES = [
    'pedestrian',
    'people',
    'bicycle',
    'car',
    'van',
    'truck',
    'tricycle',
    'awning tricycle',
    'bus',
    'motor',
]

OUTPUT_PRED_DIR = LOG_DIR / 'predictions'
OUTPUT_GT_XYXY_DIR = LOG_DIR / 'gt_xyxy'
RUN_LOG = LOG_DIR / 'run_log.txt'
SUMMARY_MD = LOG_DIR / 'evaluation_summary_groundingdino_base_refdrone100_recovered_20260421_class_aware.md'
SUMMARY_JSON = LOG_DIR / 'groundingdino_base_refdrone100_recovered_20260421_summary.json'


def ensure_importable() -> None:
    root_str = str(GROUNDINGDINO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(msg + '\n')


def read_manifest_stems() -> List[str]:
    stems: List[str] = []
    with MANIFEST.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row:
                continue
            stems.append(row[0].strip())
    return sorted(set(stems))


def write_prediction_file(output_path: Path, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0.0, float(x1))
            y1 = max(0.0, float(y1))
            x2 = max(x1 + 1e-6, float(x2))
            y2 = max(y1 + 1e-6, float(y2))
            f.write(f'{int(class_id)} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {float(score):.6f}\n')


def convert_visdrone_gt_to_xyxy(gt_dir: Path, image_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for label_path in sorted(gt_dir.glob('*.txt')):
        image_path = image_dir / f'{label_path.stem}.jpg'
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
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
                x1 = max(0.0, min(float(w) - 1.0, x))
                y1 = max(0.0, min(float(h) - 1.0, y))
                x2 = max(x1 + 1e-6, min(float(w), x + bw))
                y2 = max(y1 + 1e-6, min(float(h), y + bh))
                rows.append(f'{class_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n')
        with (out_dir / f'{label_path.stem}.txt').open('w', encoding='utf-8') as out_f:
            out_f.writelines(rows)


def _parse_det_line(line: str, is_prediction: bool) -> Optional[Tuple[int, float, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) not in (5, 6):
        return None
    class_id = int(float(parts[0]))
    x1 = float(parts[1])
    y1 = float(parts[2])
    x2 = float(parts[3])
    y2 = float(parts[4])
    score = float(parts[5]) if is_prediction and len(parts) == 6 else 1.0
    if x2 <= x1 or y2 <= y1:
        return None
    return class_id, x1, y1, x2, y2, score


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
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


def evaluate_detections(gt_dir: Path, pred_dir: Path, class_aware: bool = True, iou_thr: float = 0.5) -> Dict[str, float]:
    gt_files = sorted(gt_dir.glob('*.txt'))

    total_gt = 0
    total_pred = 0
    tp_05 = 0
    tp_075 = 0

    cls_gt_count = {c: 0 for c in range(1, 11)}

    for gt_path in gt_files:
        stem = gt_path.stem
        pred_path = pred_dir / f'{stem}.txt'

        gt_objs: List[Tuple[int, Tuple[float, float, float, float]]] = []
        with gt_path.open('r', encoding='utf-8') as f:
            for line in f:
                parsed = _parse_det_line(line, is_prediction=False)
                if parsed is None:
                    continue
                cls, x1, y1, x2, y2, _ = parsed
                gt_objs.append((cls, (x1, y1, x2, y2)))

        pred_objs: List[Tuple[int, Tuple[float, float, float, float], float]] = []
        if pred_path.exists():
            with pred_path.open('r', encoding='utf-8') as f:
                for line in f:
                    parsed = _parse_det_line(line, is_prediction=True)
                    if parsed is None:
                        continue
                    cls, x1, y1, x2, y2, score = parsed
                    pred_objs.append((cls, (x1, y1, x2, y2), score))

        total_gt += len(gt_objs)
        total_pred += len(pred_objs)

        for cls, _ in gt_objs:
            if cls in cls_gt_count:
                cls_gt_count[cls] += 1

        pred_objs = sorted(pred_objs, key=lambda x: x[2], reverse=True)

        # Single used mask: acc_075 is a subset of acc_05 matches
        used = [False] * len(gt_objs)

        for cls, box, score in pred_objs:
            best_iou = 0.0
            best_idx = -1
            for idx, (gt_cls, gt_box) in enumerate(gt_objs):
                if class_aware and gt_cls != cls:
                    continue
                iou = _iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= 0.5 and not used[best_idx]:
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

        # Rebuild class predictions with fresh matching to avoid cross-class leakage.
        for gt_path in gt_files:
            stem = gt_path.stem
            pred_path = pred_dir / f'{stem}.txt'

            gt_cls_boxes: List[Tuple[float, float, float, float]] = []
            with gt_path.open('r', encoding='utf-8') as f:
                for line in f:
                    parsed = _parse_det_line(line, is_prediction=False)
                    if parsed is None:
                        continue
                    c, x1, y1, x2, y2, _ = parsed
                    if c == cls:
                        gt_cls_boxes.append((x1, y1, x2, y2))

            pred_cls_boxes: List[Tuple[float, Tuple[float, float, float, float]]] = []
            if pred_path.exists():
                with pred_path.open('r', encoding='utf-8') as f:
                    for line in f:
                        parsed = _parse_det_line(line, is_prediction=True)
                        if parsed is None:
                            continue
                        c, x1, y1, x2, y2, score = parsed
                        if c == cls:
                            pred_cls_boxes.append((score, (x1, y1, x2, y2)))

            pred_cls_boxes.sort(key=lambda x: x[0], reverse=True)
            used = [False] * len(gt_cls_boxes)
            for score, box in pred_cls_boxes:
                best_iou = 0.0
                best_idx = -1
                for i, gt_box in enumerate(gt_cls_boxes):
                    iou = _iou(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_idx >= 0 and best_iou >= iou_thr and not used[best_idx]:
                    used[best_idx] = True
                    per_cls_preds.append((score, 1))
                else:
                    per_cls_preds.append((score, 0))

        if not per_cls_preds:
            aps.append(0.0)
            continue

        per_cls_preds.sort(key=lambda x: x[0], reverse=True)
        tp = np.array([p[1] for p in per_cls_preds], dtype=np.float32)
        fp = 1.0 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(1, gt_count)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        aps.append(ap)

    map_05 = float(np.mean(aps)) if aps else 0.0
    return {
        'file_count': float(len(gt_files)),
        'gt_box_count': float(total_gt),
        'prediction_box_count': float(total_pred),
        'acc_05': float(tp_05 / total_gt) if total_gt > 0 else 0.0,
        'acc_075': float(tp_075 / total_gt) if total_gt > 0 else 0.0,
        'map_05': map_05,
        'tp_05': float(tp_05),
        'tp_075': float(tp_075),
    }


def run_groundingdino_inference() -> int:
    ensure_importable()
    from groundingdino.util.inference import Model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(
        model_config_path=str(CONFIG_PATH),
        model_checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
    )

    stems = read_manifest_stems()
    total_boxes = 0
    for idx, stem in enumerate(stems, start=1):
        image_path = DATA_IMAGE_DIR / f'{stem}.jpg'
        image = cv2.imread(str(image_path))
        if image is None:
            write_prediction_file(OUTPUT_PRED_DIR / f'{stem}.txt', np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int32))
            continue

        detections = model.predict_with_classes(
            image=image,
            classes=CLASS_NAMES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        boxes = np.asarray(detections.xyxy)
        scores = np.asarray(detections.confidence)
        raw_class_ids = np.asarray(detections.class_id, dtype=object)

        valid_indices: List[int] = []
        class_ids_list: List[int] = []
        for i, raw_cid in enumerate(raw_class_ids.tolist()):
            if raw_cid is None:
                continue
            try:
                cid = int(raw_cid)
            except (TypeError, ValueError):
                continue
            if 0 <= cid < len(CLASS_NAMES):
                valid_indices.append(i)
                class_ids_list.append(cid)

        if valid_indices:
            valid_idx = np.asarray(valid_indices, dtype=np.int64)
            boxes = boxes[valid_idx]
            scores = scores[valid_idx]
            class_ids = np.asarray(class_ids_list, dtype=np.int32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            class_ids = np.zeros((0,), dtype=np.int32)

        if len(boxes) > 0:
            keep_idx_list = []
            for cls in range(len(CLASS_NAMES)):
                cls_idx = np.where(class_ids == cls)[0]
                if len(cls_idx) == 0:
                    continue
                keep_local = nms(
                    torch.tensor(boxes[cls_idx], dtype=torch.float32),
                    torch.tensor(scores[cls_idx], dtype=torch.float32),
                    NMS_THRESHOLD,
                ).cpu().numpy()
                keep_idx_list.extend(cls_idx[keep_local].tolist())
            keep_idx = np.array(sorted(keep_idx_list), dtype=np.int64)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            class_ids = class_ids[keep_idx] + 1

        write_prediction_file(OUTPUT_PRED_DIR / f'{stem}.txt', boxes, scores, class_ids.astype(np.int32))
        total_boxes += int(len(boxes))

        if idx % 10 == 0 or idx == len(stems):
            log(f'[Progress] {idx}/{len(stems)} {stem}.jpg boxes={len(boxes)}')

    log(f'[Done] images={len(stems)} total_pred_boxes={total_boxes}')
    return total_boxes


def main() -> None:
    t0 = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')

    log('[Start] GroundingDINO baseline rerun on visdrone_test_100')
    convert_visdrone_gt_to_xyxy(DATA_GT_DIR, DATA_IMAGE_DIR, OUTPUT_GT_XYXY_DIR)
    log('[Step] GT conversion completed')

    pred_count = run_groundingdino_inference()
    metrics = evaluate_detections(OUTPUT_GT_XYXY_DIR, OUTPUT_PRED_DIR, class_aware=True, iou_thr=0.5)

    elapsed = time.time() - t0
    lines = [
        '# GroundingDINO Baseline Summary',
        '',
        '## Config',
        f'- Dataset: {DATA_ROOT}',
        f'- Checkpoint: {CHECKPOINT_PATH}',
        f'- Config: {CONFIG_PATH}',
        f'- box_threshold={BOX_THRESHOLD:.2f}',
        f'- text_threshold={TEXT_THRESHOLD:.2f}',
        f'- nms_threshold={NMS_THRESHOLD:.2f}',
        '- eval_mode=class-aware',
        '',
        '## Metrics',
        f"- Label files considered: {int(metrics['file_count'])}",
        f"- GT boxes: {int(metrics['gt_box_count'])}",
        f"- Prediction boxes: {int(metrics['prediction_box_count'])}",
        f"- Acc@0.5: {metrics['acc_05']:.4f} ({int(metrics['tp_05'])}/{int(metrics['gt_box_count'])})",
        f"- Acc@0.75: {metrics['acc_075']:.4f} ({int(metrics['tp_075'])}/{int(metrics['gt_box_count'])})",
        f"- mAP@0.5: {metrics['map_05']:.4f}",
        f'- elapsed_sec: {elapsed:.2f}',
    ]
    SUMMARY_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    summary = {
        'config': {
            'dataset': str(DATA_ROOT),
            'checkpoint': str(CHECKPOINT_PATH),
            'model_config': str(CONFIG_PATH),
            'box_threshold': BOX_THRESHOLD,
            'text_threshold': TEXT_THRESHOLD,
            'nms_threshold': NMS_THRESHOLD,
        },
        'metrics': metrics,
        'pred_count_from_loop': pred_count,
        'elapsed_sec': elapsed,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    log(f'[Done] summary_md={SUMMARY_MD}')
    log(f'[Done] summary_json={SUMMARY_JSON}')
    log(
        '[Result] '
        f"acc05={metrics['acc_05']:.4f} "
        f"acc075={metrics['acc_075']:.4f} "
        f"map05={metrics['map_05']:.4f} "
        f"pred={int(metrics['prediction_box_count'])}"
    )


if __name__ == '__main__':
    main()
