import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import nms

ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/baseline/groundingdino_base_visdrone_split1000guarded_test_only_20260429'
LOG_DIR = EXP_ROOT / 'log'

DATA_ROOT = ROOT / 'dataset' / 'VisDroneSplit1000Guarded'
TEST_IMAGE_DIR = DATA_ROOT / 'VisDrone2019-DET-test' / 'images'
TEST_GT_DIR = DATA_ROOT / 'VisDrone2019-DET-test' / 'annotations'

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
SUMMARY_MD = LOG_DIR / 'evaluation_summary_groundingdino_test_only_20260429_class_aware.md'
SUMMARY_JSON = LOG_DIR / 'groundingdino_test_only_20260429_summary.json'
REFERENCE_SCRIPT = ROOT / 'experiment/baseline/groundingdino_base_refdrone100_recovered_20260421/scripts/run_groundingdino_base_refdrone100_recovered_20260421.py'


def load_reference_module():
    spec = importlib.util.spec_from_file_location('groundingdino_baseline_reference', str(REFERENCE_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load reference baseline script: {REFERENCE_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = load_reference_module()


def ensure_importable() -> None:
    root_str = str(GROUNDINGDINO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as file:
        file.write(msg + '\n')


def build_test_index() -> Dict[str, Tuple[Path, Path]]:
    index: Dict[str, Tuple[Path, Path]] = {}
    if not TEST_IMAGE_DIR.exists() or not TEST_GT_DIR.exists():
        return index
    for ann_path in sorted(TEST_GT_DIR.glob('*.txt')):
        stem = ann_path.stem
        image_path = TEST_IMAGE_DIR / f'{stem}.jpg'
        if image_path.exists():
            index[stem] = (image_path, ann_path)
    return index


def write_prediction_file(output_path: Path, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> None:
    REF.write_prediction_file(output_path, boxes, scores, class_ids)


def convert_visdrone_gt_to_xyxy(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    REF.convert_visdrone_gt_to_xyxy(TEST_GT_DIR, TEST_IMAGE_DIR, out_dir)


def run_groundingdino_inference() -> int:
    ensure_importable()
    from groundingdino.util.inference import Model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(
        model_config_path=str(CONFIG_PATH),
        model_checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
    )

    index = build_test_index()
    stems = sorted(index.keys())
    total_boxes = 0
    for idx, stem in enumerate(stems, start=1):
        image_path, _ = index[stem]
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

    log('[Start] GroundingDINO baseline rerun on VisDroneSplit1000Guarded test split only')
    convert_visdrone_gt_to_xyxy(OUTPUT_GT_XYXY_DIR)
    log('[Step] GT conversion completed')

    pred_count = run_groundingdino_inference()
    metrics = REF.evaluate_detections(OUTPUT_GT_XYXY_DIR, OUTPUT_PRED_DIR, class_aware=True, iou_thr=0.5)

    elapsed = time.time() - t0
    lines = [
        '# GroundingDINO Test-Only Baseline Summary',
        '',
        '## Config',
        f'- Dataset: {DATA_ROOT}',
        f'- Split: test only',
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
            'split': 'test',
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
    log(f'[Done] elapsed_sec={elapsed:.2f}')


if __name__ == '__main__':
    main()
