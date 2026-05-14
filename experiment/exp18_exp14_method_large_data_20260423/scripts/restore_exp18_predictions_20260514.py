import csv
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import torch
import torch.nn as nn
from torchvision.ops import nms


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp18_exp14_method_large_data_20260423'
LOG_DIR = EXP_ROOT / 'log'
DATA_ROOT = ROOT / 'dataset/VisDroneSplit1000Guarded'
EXP17_SPLIT_MANIFEST = ROOT / 'experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv'
EXP13_PRED_DIR = ROOT / 'experiment/exp13_full_rebuild_20260421/log/predictions'

RUN_LOG = LOG_DIR / 'restore_exp18_predictions_20260514.log'
BEST_CKPT = LOG_DIR / 'exp18_exp14_method_large_data_20260423_best.pt'
SUMMARY_JSON = LOG_DIR / 'exp18_exp14_method_large_data_20260423_summary.json'
PRED_ROOT = LOG_DIR / 'predictions'
GT_ROOT = LOG_DIR / 'gt_xyxy'
SUMMARY_TRAINVAL_MD = LOG_DIR / 'evaluation_summary_exp18_exp14_method_large_data_trainval_class_aware.md'
SUMMARY_TEST_MD = LOG_DIR / 'evaluation_summary_exp18_exp14_method_large_data_test_class_aware.md'
SUMMARY_FULL_MD = LOG_DIR / 'evaluation_summary_exp18_exp14_method_large_data_full_class_aware.md'

THRESHOLD = 0.02
NMS_THRESHOLD = 0.40


@dataclass
class DetBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: int = 0


class HybridExternalCalibrator(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, head_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_baseline_module():
    script = ROOT / 'experiment/baseline/groundingdino_base_refdrone100_recovered_20260421/scripts/run_groundingdino_base_refdrone100_recovered_20260421.py'
    spec = importlib.util.spec_from_file_location('baseline_eval_helpers', str(script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import baseline helpers from {script}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['baseline_eval_helpers'] = module
    spec.loader.exec_module(module)
    return module


BASE = load_baseline_module()


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


def image_path(split: str, stem: str) -> Path:
    return DATA_ROOT / f'VisDrone2019-DET-{split}' / 'images' / f'{stem}.jpg'


def annotation_dir(split: str) -> Path:
    return DATA_ROOT / f'VisDrone2019-DET-{split}' / 'annotations'


def image_dir(split: str) -> Path:
    return DATA_ROOT / f'VisDrone2019-DET-{split}' / 'images'


def read_xyxy_labels(path: Path, is_prediction: bool) -> List[DetBox]:
    boxes: List[DetBox] = []
    if not path.exists():
        return boxes
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) not in (5, 6):
                continue
            class_id = int(float(parts[0]))
            x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
            if x2 <= x1 or y2 <= y1:
                continue
            score = float(parts[5]) if is_prediction and len(parts) == 6 else 1.0
            boxes.append(DetBox(class_id, x1, y1, x2, y2, score))
    return boxes


def convert_gt_dirs() -> None:
    GT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        BASE.convert_visdrone_gt_to_xyxy(annotation_dir(split), image_dir(split), GT_ROOT / split)
    trainval_dir = GT_ROOT / 'trainval'
    full_dir = GT_ROOT / 'full'
    for out_dir, splits in [(trainval_dir, ['train', 'val']), (full_dir, ['train', 'val', 'test'])]:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            for path in (GT_ROOT / split).glob('*.txt'):
                shutil.copy2(path, out_dir / path.name)


def feature_from_box(box: DetBox, width: int, height: int) -> List[float]:
    bw = max(1e-6, box.x2 - box.x1)
    bh = max(1e-6, box.y2 - box.y1)
    cx = ((box.x1 + box.x2) / 2.0) / width
    cy = ((box.y1 + box.y2) / 2.0) / height
    nw = bw / width
    nh = bh / height
    area = nw * nh
    aspect = bw / bh
    return [cx, cy, nw, nh, area, math.log(max(aspect, 1e-6)), float(box.score), float(box.class_id) / 10.0, float(box.source)]


def load_records(split: str, stems: Sequence[str]):
    records = []
    for stem in stems:
        image = cv2.imread(str(image_path(split, stem)))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = read_xyxy_labels(EXP13_PRED_DIR / f'{stem}.txt', is_prediction=True)
        records.append({
            'split': split,
            'stem': stem,
            'boxes': boxes,
            'features': [feature_from_box(box, width, height) for box in boxes],
        })
    return records


def flatten_features(records) -> torch.Tensor:
    feats: List[List[float]] = []
    for record in records:
        feats.extend(record['features'])
    if not feats:
        return torch.zeros((0, 9), dtype=torch.float32)
    return torch.tensor(feats, dtype=torch.float32)


def class_aware_nms(boxes: List[DetBox], thr: float) -> List[DetBox]:
    selected: List[DetBox] = []
    by_class: Dict[int, List[DetBox]] = {}
    for box in boxes:
        by_class.setdefault(box.class_id, []).append(box)
    for items in by_class.values():
        if not items:
            continue
        t_boxes = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in items], dtype=torch.float32)
        t_scores = torch.tensor([b.score for b in items], dtype=torch.float32)
        keep = nms(t_boxes, t_scores, thr).cpu().tolist()
        selected.extend(items[idx] for idx in keep)
    return sorted(selected, key=lambda b: b.score, reverse=True)


def apply_model(model: nn.Module, records, mu: torch.Tensor, sigma: torch.Tensor) -> Dict[str, List[DetBox]]:
    output: Dict[str, List[DetBox]] = {}
    model.eval()
    with torch.no_grad():
        for record in records:
            if not record['boxes']:
                output[record['stem']] = []
                continue
            feats = torch.tensor(record['features'], dtype=torch.float32)
            probs = torch.sigmoid(model((feats - mu) / sigma)).squeeze(1).cpu().tolist()
            selected = [
                DetBox(box.class_id, box.x1, box.y1, box.x2, box.y2, float(prob), box.source)
                for box, prob in zip(record['boxes'], probs)
                if prob >= THRESHOLD
            ]
            output[record['stem']] = class_aware_nms(selected, NMS_THRESHOLD)
    return output


def save_predictions(pred_by_stem: Dict[str, List[DetBox]], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, boxes in pred_by_stem.items():
        with (out_dir / f'{stem}.txt').open('w', encoding='utf-8') as f:
            for box in boxes:
                f.write(f'{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {box.score:.6f}\n')


def write_summary_md(path: Path, title: str, metrics: Dict[str, float]) -> None:
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
    if not DATA_ROOT.exists():
        raise RuntimeError(f'Missing dataset: {DATA_ROOT}')
    if not EXP13_PRED_DIR.exists():
        raise RuntimeError(f'Missing candidate predictions: {EXP13_PRED_DIR}')
    if not BEST_CKPT.exists():
        raise RuntimeError(f'Missing checkpoint: {BEST_CKPT}')

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')
    log('[Start] Restore Exp18 predictions from checkpoint and Exp13 candidates')
    log(f'[Config] dataset={DATA_ROOT}')
    log(f'[Config] checkpoint={BEST_CKPT}')
    log(f'[Config] threshold={THRESHOLD:.2f} nms={NMS_THRESHOLD:.2f}')

    split_map = read_split_map()
    convert_gt_dirs()

    train_records = load_records('train', split_map['train'])
    val_records = load_records('val', split_map['val'])
    test_records = load_records('test', split_map['test'])
    trainval_records = train_records + val_records
    full_records = train_records + val_records + test_records

    x_train = flatten_features(train_records)
    if x_train.numel() == 0:
        raise RuntimeError('No train candidates found.')
    mu = x_train.mean(dim=0, keepdim=True)
    sigma = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)

    model = HybridExternalCalibrator()
    ckpt = torch.load(BEST_CKPT, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)

    pred_trainval = apply_model(model, trainval_records, mu, sigma)
    pred_test = apply_model(model, test_records, mu, sigma)
    pred_full = apply_model(model, full_records, mu, sigma)

    save_predictions(pred_trainval, PRED_ROOT / 'trainval')
    save_predictions(pred_test, PRED_ROOT / 'test')
    save_predictions(pred_full, PRED_ROOT / 'full')

    trainval_metrics = BASE.evaluate_detections(GT_ROOT / 'trainval', PRED_ROOT / 'trainval', class_aware=True, iou_thr=0.5)
    test_metrics = BASE.evaluate_detections(GT_ROOT / 'test', PRED_ROOT / 'test', class_aware=True, iou_thr=0.5)
    full_metrics = BASE.evaluate_detections(GT_ROOT / 'full', PRED_ROOT / 'full', class_aware=True, iou_thr=0.5)

    write_summary_md(SUMMARY_TRAINVAL_MD, 'Exp18 TrainVal Restored Summary', trainval_metrics)
    write_summary_md(SUMMARY_TEST_MD, 'Exp18 Test Restored Summary', test_metrics)
    write_summary_md(SUMMARY_FULL_MD, 'Exp18 Full Restored Summary', full_metrics)

    summary = {
        'seed': 42,
        'data_root': str(DATA_ROOT),
        'exp13_predictions': str(EXP13_PRED_DIR),
        'checkpoint': str(BEST_CKPT),
        'restore_script': str(Path(__file__).resolve()),
        'best_threshold': THRESHOLD,
        'best_nms': NMS_THRESHOLD,
        'split_counts': {k: len(v) for k, v in split_map.items()},
        'trainval_metrics': trainval_metrics,
        'test_metrics': test_metrics,
        'full_metrics': full_metrics,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log(f"[Done] full acc05={full_metrics['acc_05']:.4f} acc075={full_metrics['acc_075']:.4f} map05={full_metrics['map_05']:.4f}")


if __name__ == '__main__':
    main()
