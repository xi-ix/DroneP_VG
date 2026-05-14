import csv
import importlib.util
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import nms


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp24_teacher_distill_exp23_to_exp22_20260513'
LOG_DIR = EXP_ROOT / 'log'
DATA_ROOT = ROOT / 'dataset/VisDroneSplit1000Guarded'
EXP17_SPLIT_MANIFEST = ROOT / 'experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv'
EXP18_FULL = ROOT / 'experiment/exp18_exp14_method_large_data_20260423/log/predictions/full'
GDINO_FULL = ROOT / 'experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions'
TEACHER_FULL = ROOT / 'experiment/exp23_metric_driven_fusion_20260512/log/predictions/full'

EXP23_SCRIPT = ROOT / 'experiment/exp23_metric_driven_fusion_20260512/scripts/run_exp23_metric_driven_fusion_20260512.py'

RUN_LOG = LOG_DIR / 'run_log.txt'
PRED_ROOT = LOG_DIR / 'predictions'
GT_ROOT = LOG_DIR / 'gt_xyxy'
BEST_CKPT = LOG_DIR / 'exp24_teacher_distill_exp23_to_exp22_20260513_best.pt'
SUMMARY_JSON = LOG_DIR / 'exp24_teacher_distill_exp23_to_exp22_20260513_summary.json'
SUMMARY_TRAINVAL_MD = LOG_DIR / 'evaluation_summary_exp24_teacher_distill_trainval_class_aware.md'
SUMMARY_VAL_MD = LOG_DIR / 'evaluation_summary_exp24_teacher_distill_val_class_aware.md'
SUMMARY_TEST_MD = LOG_DIR / 'evaluation_summary_exp24_teacher_distill_test_class_aware.md'
SUMMARY_FULL_MD = LOG_DIR / 'evaluation_summary_exp24_teacher_distill_full_class_aware.md'

SEED = 42
EPOCHS = 24
BATCH_SIZE = 4096
LR = 0.001
ALPHA_TEACHER = 0.7
THRESHOLD = 0.05
NMS_THRESHOLD = 0.50


@dataclass
class DetBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: int


def load_exp23_helpers():
    spec = importlib.util.spec_from_file_location('exp23_helpers_for_exp24', str(EXP23_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import Exp23 helpers from {EXP23_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp23_helpers_for_exp24'] = module
    spec.loader.exec_module(module)
    return module


EXP23 = load_exp23_helpers()


class DistillScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


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


def read_boxes(path: Path, source: int) -> List[DetBox]:
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
            cls = int(float(parts[0]))
            x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
            score = float(parts[5])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(DetBox(cls, x1, y1, x2, y2, score, source))
    return boxes


def image_size(split: str, stem: str) -> Tuple[int, int]:
    path = DATA_ROOT / f'VisDrone2019-DET-{split}' / 'images' / f'{stem}.jpg'
    with Image.open(path) as image:
        return image.size


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
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


def box_tuple(box: DetBox) -> Tuple[float, float, float, float]:
    return box.x1, box.y1, box.x2, box.y2


def feature_from_box(box: DetBox, width: int, height: int) -> List[float]:
    bw = max(1e-6, box.x2 - box.x1)
    bh = max(1e-6, box.y2 - box.y1)
    cx = ((box.x1 + box.x2) / 2.0) / width
    cy = ((box.y1 + box.y2) / 2.0) / height
    nw = bw / width
    nh = bh / height
    area = nw * nh
    aspect = bw / bh
    return [
        cx,
        cy,
        nw,
        nh,
        area,
        math.log(max(aspect, 1e-6)),
        float(box.score),
        float(box.class_id) / 10.0,
        float(box.source),
        1.0 if box.source == 0 else 0.0,
        1.0 if box.source == 1 else 0.0,
    ]


def best_match_label(candidate: DetBox, targets: Sequence[DetBox], threshold: float) -> float:
    best = 0.0
    for target in targets:
        if target.class_id != candidate.class_id:
            continue
        best = max(best, iou(box_tuple(candidate), box_tuple(target)))
    return 1.0 if best >= threshold else 0.0


def teacher_soft_label(candidate: DetBox, teacher_boxes: Sequence[DetBox]) -> float:
    best_iou = 0.0
    best_score = 0.0
    for teacher in teacher_boxes:
        if teacher.class_id != candidate.class_id:
            continue
        cur_iou = iou(box_tuple(candidate), box_tuple(teacher))
        if cur_iou > best_iou:
            best_iou = cur_iou
            best_score = teacher.score
    if best_iou >= 0.7:
        return max(0.5, min(1.0, best_score))
    if best_iou >= 0.5:
        return max(0.05, min(0.8, best_score * ((best_iou - 0.5) / 0.2)))
    return 0.0


def load_gt_xyxy(split: str, stem: str) -> List[DetBox]:
    path = GT_ROOT / split / f'{stem}.txt'
    return read_boxes(path, source=0)


def load_record(split: str, stem: str):
    width, height = image_size(split, stem)
    candidates = read_boxes(EXP18_FULL / f'{stem}.txt', source=0) + read_boxes(GDINO_FULL / f'{stem}.txt', source=1)
    teacher = read_boxes(TEACHER_FULL / f'{stem}.txt', source=2)
    gt = load_gt_xyxy(split, stem)
    feats = [feature_from_box(box, width, height) for box in candidates]
    gt_labels = [best_match_label(box, gt, 0.5) for box in candidates]
    soft_labels = [teacher_soft_label(box, teacher) for box in candidates]
    targets = [max(g, s) for g, s in zip(gt_labels, soft_labels)]
    return {
        'split': split,
        'stem': stem,
        'boxes': candidates,
        'features': feats,
        'gt_labels': gt_labels,
        'soft_labels': soft_labels,
        'targets': targets,
    }


def flatten(records):
    features: List[List[float]] = []
    gt_labels: List[float] = []
    soft_labels: List[float] = []
    targets: List[float] = []
    for record in records:
        features.extend(record['features'])
        gt_labels.extend(record['gt_labels'])
        soft_labels.extend(record['soft_labels'])
        targets.extend(record['targets'])
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(gt_labels, dtype=torch.float32),
        torch.tensor(soft_labels, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


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
    model.eval()
    pred_by_stem: Dict[str, List[DetBox]] = {}
    with torch.no_grad():
        for record in records:
            if not record['boxes']:
                pred_by_stem[record['stem']] = []
                continue
            x = torch.tensor(record['features'], dtype=torch.float32)
            probs = torch.sigmoid(model((x - mu) / sigma)).cpu().tolist()
            selected = [
                DetBox(box.class_id, box.x1, box.y1, box.x2, box.y2, float(prob), box.source)
                for box, prob in zip(record['boxes'], probs)
                if prob >= THRESHOLD
            ]
            pred_by_stem[record['stem']] = class_aware_nms(selected, NMS_THRESHOLD)
    return pred_by_stem


def save_predictions(pred_by_stem: Dict[str, List[DetBox]], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, boxes in pred_by_stem.items():
        with (out_dir / f'{stem}.txt').open('w', encoding='utf-8') as f:
            for box in boxes:
                f.write(f'{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {box.score:.6f}\n')


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


def build_gt_dirs() -> None:
    EXP23.convert_gt_dirs()
    src_root = ROOT / 'experiment/exp23_metric_driven_fusion_20260512/log/gt_xyxy'
    if GT_ROOT.exists():
        shutil.rmtree(GT_ROOT)
    shutil.copytree(src_root, GT_ROOT)


def train_model(train_records, val_records):
    x_train_raw, gt_train, soft_train, target_train = flatten(train_records)
    x_val_raw, _, _, _ = flatten(val_records)
    mu = x_train_raw.mean(dim=0, keepdim=True)
    sigma = x_train_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train = (x_train_raw - mu) / sigma

    model = DistillScorer(input_dim=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion_gt = nn.BCEWithLogitsLoss()
    criterion_soft = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(x_train, gt_train, soft_train, target_train), batch_size=BATCH_SIZE, shuffle=True)

    best = {'epoch': 0, 'val_objective': -1.0, 'val_acc_05': 0.0, 'val_map_05': 0.0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, y_gt, y_soft, y_target in loader:
            logits = model(xb)
            loss = criterion_gt(logits, y_gt) + ALPHA_TEACHER * criterion_soft(logits, y_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_pred = apply_model(model, val_records, mu, sigma)
        save_predictions(val_pred, PRED_ROOT / 'val')
        val_metrics = EXP23.evaluate_detections(GT_ROOT / 'val', PRED_ROOT / 'val', iou_thr=0.5)
        val_objective = 0.5 * val_metrics['acc_05'] + 0.5 * val_metrics['map_05']
        log(
            f"[Epoch] {epoch:03d}/{EPOCHS} "
            f"loss={sum(losses)/max(1, len(losses)):.6f} "
            f"val_acc05={val_metrics['acc_05']:.4f} "
            f"val_map05={val_metrics['map_05']:.4f} "
            f"objective={val_objective:.4f}"
        )
        if val_objective > best['val_objective']:
            best = {
                'epoch': epoch,
                'val_objective': val_objective,
                'val_acc_05': val_metrics['acc_05'],
                'val_map_05': val_metrics['map_05'],
            }
            torch.save({'model': model.state_dict(), 'mu': mu, 'sigma': sigma, 'best': best}, BEST_CKPT)

    ckpt = torch.load(BEST_CKPT, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    return model, ckpt['mu'], ckpt['sigma'], best


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    for path in [DATA_ROOT, EXP18_FULL, GDINO_FULL, TEACHER_FULL]:
        if not path.exists():
            raise RuntimeError(f'Missing required path: {path}')

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')
    log('[Start] Exp24 teacher distillation restore')
    log(f'[Config] threshold={THRESHOLD:.2f} nms={NMS_THRESHOLD:.2f} epochs={EPOCHS}')

    build_gt_dirs()
    split_map = read_split_map()
    train_records = [load_record('train', stem) for stem in split_map['train']]
    val_records = [load_record('val', stem) for stem in split_map['val']]
    test_records = [load_record('test', stem) for stem in split_map['test']]
    trainval_records = train_records + val_records
    full_records = trainval_records + test_records

    model, mu, sigma, best = train_model(train_records, val_records)

    pred_trainval = apply_model(model, trainval_records, mu, sigma)
    pred_test = apply_model(model, test_records, mu, sigma)
    pred_full = apply_model(model, full_records, mu, sigma)
    save_predictions(pred_trainval, PRED_ROOT / 'trainval')
    save_predictions(pred_test, PRED_ROOT / 'test')
    save_predictions(pred_full, PRED_ROOT / 'full')

    trainval_metrics = EXP23.evaluate_detections(GT_ROOT / 'trainval', PRED_ROOT / 'trainval', iou_thr=0.5)
    val_metrics = EXP23.evaluate_detections(GT_ROOT / 'val', PRED_ROOT / 'val', iou_thr=0.5)
    test_metrics = EXP23.evaluate_detections(GT_ROOT / 'test', PRED_ROOT / 'test', iou_thr=0.5)
    full_metrics = EXP23.evaluate_detections(GT_ROOT / 'full', PRED_ROOT / 'full', iou_thr=0.5)

    write_summary(SUMMARY_TRAINVAL_MD, 'Exp24 TrainVal Distill Summary', trainval_metrics)
    write_summary(SUMMARY_VAL_MD, 'Exp24 Val Distill Summary', val_metrics)
    write_summary(SUMMARY_TEST_MD, 'Exp24 Test Distill Summary', test_metrics)
    write_summary(SUMMARY_FULL_MD, 'Exp24 Full Distill Summary', full_metrics)

    summary = {
        'seed': SEED,
        'data_root': str(DATA_ROOT),
        'sources': {
            'exp18': str(EXP18_FULL),
            'gdino': str(GDINO_FULL),
            'teacher': str(TEACHER_FULL),
        },
        'config': {
            'epochs': EPOCHS,
            'threshold': THRESHOLD,
            'nms': NMS_THRESHOLD,
            'alpha_teacher': ALPHA_TEACHER,
        },
        'best': best,
        'trainval_metrics': trainval_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'full_metrics': full_metrics,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log(f"[Done] full acc05={full_metrics['acc_05']:.4f} acc075={full_metrics['acc_075']:.4f} map05={full_metrics['map_05']:.4f} pred={int(full_metrics['prediction_box_count'])}")


if __name__ == '__main__':
    main()
