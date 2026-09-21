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


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp37_online_fusion_calibrator_20260712'
LOG_DIR = EXP_ROOT / 'log'
RUN_LOG = LOG_DIR / 'run_log.txt'
SUMMARY_JSON = LOG_DIR / 'exp37_online_fusion_calibrator_20260712_summary.json'
PRED_ROOT = LOG_DIR / 'predictions'
BEST_CKPT = LOG_DIR / 'exp37_online_fusion_calibrator_20260712_best.pt'

DATA_ROOT = ROOT / 'dataset/VisDroneSplit1000Guarded'
EXP17_SPLIT_MANIFEST = ROOT / 'experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv'
EXP23_SCRIPT = ROOT / 'experiment/exp23_metric_driven_fusion_20260512/scripts/run_exp23_metric_driven_fusion_20260512.py'
GT_ROOT = ROOT / 'experiment/exp30_real_query_focal_rerank_20260711/log/gt_xyxy'
EXP35_PRED_ROOT = ROOT / 'experiment/exp35_online_candidate_compression_20260711/log/predictions'
EXP36_SCRIPT = ROOT / 'experiment/exp36_small_target_alias_rerank_20260711/scripts/run_exp36_small_target_alias_rerank_20260711.py'
EXP36_CKPT = ROOT / 'experiment/exp36_small_target_alias_rerank_20260711/log/exp36_small_target_alias_rerank_20260711_best.pt'
EXP36_PRED_TEST = ROOT / 'experiment/exp36_small_target_alias_rerank_20260711/log/predictions/test'

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
BATCH_SIZE = 4096
LR = 0.0008
SMALL_CLASSES = {1, 2, 3, 7, 8, 10}
WEAK_SMALL_CLASSES = {2, 7, 8, 10}
PERSON_CLASSES = {1, 2}
SMALL_VEHICLE_CLASSES = {3, 7, 8, 10}
LARGE_VEHICLE_CLASSES = {4, 5, 6, 9}
VEHICLE_CLASSES = SMALL_VEHICLE_CLASSES | LARGE_VEHICLE_CLASSES
CONFUSION_GROUPS = {
    1: {2},
    2: {1},
    3: {7, 10},
    7: {3, 8, 10},
    8: {7, 10},
    10: {3, 7, 8},
}
AREA_PRIOR = {
    1: 0.0019,
    2: 0.0011,
    3: 0.0020,
    4: 0.0200,
    5: 0.0180,
    6: 0.0250,
    7: 0.0062,
    8: 0.0037,
    9: 0.0300,
    10: 0.0018,
}
CLASS_NAMES = {
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning tricycle',
    9: 'bus',
    10: 'motor',
}


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


class FusionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def log(message: str) -> None:
    print(message, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(message + '\n')


def load_exp23_helpers():
    spec = importlib.util.spec_from_file_location('exp23_helpers_for_exp37', str(EXP23_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import Exp23 helpers from {EXP23_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp23_helpers_for_exp37'] = module
    spec.loader.exec_module(module)
    return module


def load_exp36_module():
    spec = importlib.util.spec_from_file_location('exp36_for_exp37', str(EXP36_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import Exp36 from {EXP36_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp36_for_exp37'] = module
    spec.loader.exec_module(module)
    return module


EXP23 = load_exp23_helpers()
EXP36 = load_exp36_module()


def read_split_map() -> Dict[str, List[str]]:
    split_map: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}
    with EXP17_SPLIT_MANIFEST.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            split = row['split']
            if split in split_map:
                split_map[split].append(row['stem'])
    return {split: sorted(stems) for split, stems in split_map.items()}


def image_path(split: str, stem: str) -> Path:
    return DATA_ROOT / f'VisDrone2019-DET-{split}' / 'images' / f'{stem}.jpg'


def image_size(split: str, stem: str) -> Tuple[int, int]:
    with Image.open(image_path(split, stem)) as image:
        return image.size


def read_pred(path: Path) -> List[Box]:
    boxes: List[Box] = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding='utf-8').splitlines():
        parts = raw.split()
        if len(parts) != 6:
            continue
        class_id = int(float(parts[0]))
        x1, y1, x2, y2 = [float(value) for value in parts[1:5]]
        score = float(parts[5])
        if x2 > x1 and y2 > y1 and class_id in CLASS_NAMES:
            boxes.append(Box(class_id, x1, y1, x2, y2, score))
    return boxes


def read_gt(path: Path) -> List[Box]:
    boxes: List[Box] = []
    if not path.exists():
        return boxes
    for raw in path.read_text(encoding='utf-8').splitlines():
        parts = raw.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        x1, y1, x2, y2 = [float(value) for value in parts[1:5]]
        if x2 > x1 and y2 > y1 and class_id in CLASS_NAMES:
            boxes.append(Box(class_id, x1, y1, x2, y2, 1.0))
    return boxes


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / max(area_a + area_b - inter, 1e-9)


def best_iou(box: Box, gt_boxes: Sequence[Box], same_class_only: bool) -> float:
    best = 0.0
    for gt in gt_boxes:
        if same_class_only and gt.class_id != box.class_id:
            continue
        best = max(best, iou(box, gt))
    return best


def confusion_iou(box: Box, gt_boxes: Sequence[Box]) -> float:
    confusable = CONFUSION_GROUPS.get(box.class_id, set())
    return max([iou(box, gt) for gt in gt_boxes if gt.class_id in confusable] or [0.0])


def hierarchy_features(class_id: int) -> List[float]:
    return [
        1.0 if class_id in SMALL_CLASSES else 0.0,
        1.0 if class_id in WEAK_SMALL_CLASSES else 0.0,
        1.0 if class_id in PERSON_CLASSES else 0.0,
        1.0 if class_id in VEHICLE_CLASSES else 0.0,
        1.0 if class_id in SMALL_VEHICLE_CLASSES else 0.0,
        1.0 if class_id in LARGE_VEHICLE_CLASSES else 0.0,
        1.0 if class_id == 10 else 0.0,
        1.0 if class_id in {7, 8} else 0.0,
        1.0 if class_id == 8 else 0.0,
        1.0 if class_id in {3, 7, 8, 10} else 0.0,
    ]


def geometry_features(box: Box, width: int, height: int) -> List[float]:
    bw = max(1.0, box.x2 - box.x1)
    bh = max(1.0, box.y2 - box.y1)
    area = (bw * bh) / max(1.0, width * height)
    prior = AREA_PRIOR.get(box.class_id, 0.01)
    scale_fit = math.exp(-abs(math.log(max(area, 1e-8) / max(prior, 1e-8))))
    return [
        math.sqrt(area),
        math.log(max(area, 1e-8)),
        bw / width,
        bh / height,
        math.log(max(bw / bh, 1e-6)),
        ((box.x1 + box.x2) / 2.0) / width,
        ((box.y1 + box.y2) / 2.0) / height,
        box.class_id / 10.0,
        scale_fit,
        min(1.0, area / max(prior, 1e-8)),
    ]


def exp36_scores_for_boxes(exp36_model, exp36_mu: torch.Tensor, exp36_sigma: torch.Tensor, boxes: Sequence[Box], width: int, height: int) -> List[float]:
    if not boxes:
        return []
    exp36_boxes = [EXP36.Box(box.class_id, box.x1, box.y1, box.x2, box.y2, box.score) for box in boxes]
    features = [EXP36.box_features(box, width, height, ()) for box in exp36_boxes]
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        x = ((x - exp36_mu.cpu()) / exp36_sigma.cpu()).to(DEVICE)
        return torch.sigmoid(exp36_model(x)).cpu().tolist()


def fusion_features(box: Box, exp36_score: float, width: int, height: int) -> List[float]:
    base_score = min(max(box.score, 1e-6), 1.0 - 1e-6)
    alias_score = min(max(exp36_score, 1e-6), 1.0 - 1e-6)
    score_features = [
        base_score,
        alias_score,
        base_score - alias_score,
        base_score * alias_score,
        max(base_score, alias_score),
        min(base_score, alias_score),
        math.log(base_score / (1.0 - base_score)),
        math.log(alias_score / (1.0 - alias_score)),
    ]
    one_hot = [1.0 if box.class_id == class_id else 0.0 for class_id in range(1, 11)]
    return score_features + geometry_features(box, width, height) + hierarchy_features(box.class_id) + one_hot


def label_and_weight(box: Box, gt_boxes: Sequence[Box]) -> Tuple[float, float]:
    same_iou = best_iou(box, gt_boxes, same_class_only=True)
    conf_iou = confusion_iou(box, gt_boxes)
    label = 1.0 if same_iou >= 0.5 else 0.0
    weight = 1.0
    if box.class_id in SMALL_CLASSES:
        weight *= 1.8
    if box.class_id in WEAK_SMALL_CLASSES:
        weight *= 2.2
    if label > 0.5:
        weight *= 2.0
    elif box.score >= 0.30 or conf_iou >= 0.30:
        weight *= 2.8
    return label, weight


def load_exp36_model():
    ckpt = torch.load(EXP36_CKPT, map_location='cpu')
    model = EXP36.AliasReranker(input_dim=ckpt['mu'].shape[1]).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt['mu'].float(), ckpt['sigma'].float()


def build_dataset(split: str, stems: Sequence[str], exp36_model, exp36_mu: torch.Tensor, exp36_sigma: torch.Tensor):
    features: List[List[float]] = []
    labels: List[float] = []
    weights: List[float] = []
    for idx, stem in enumerate(stems, start=1):
        width, height = image_size(split, stem)
        boxes = read_pred(EXP35_PRED_ROOT / split / f'{stem}.txt')
        gt_boxes = read_gt(GT_ROOT / split / f'{stem}.txt')
        exp36_scores = exp36_scores_for_boxes(exp36_model, exp36_mu, exp36_sigma, boxes, width, height)
        for box, exp36_score in zip(boxes, exp36_scores):
            label, weight = label_and_weight(box, gt_boxes)
            labels.append(label)
            weights.append(weight)
            features.append(fusion_features(box, exp36_score, width, height))
        if idx == 1 or idx % 50 == 0 or idx == len(stems):
            log(f'[BuildDataset] split={split} processed={idx}/{len(stems)} samples={len(labels)}')
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def train_model(x_train_raw: torch.Tensor, y_train: torch.Tensor, w_train: torch.Tensor, x_hold_raw: torch.Tensor, y_hold: torch.Tensor):
    mu = x_train_raw.mean(dim=0, keepdim=True)
    sigma = x_train_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train = (x_train_raw - mu) / sigma
    x_hold = (x_hold_raw - mu) / sigma
    model = FusionModel(input_dim=x_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(x_train, y_train, w_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == 'cuda'),
    )
    best = {'epoch': 0, 'hold_loss': float('inf')}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb, wb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss_raw = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
            loss = (loss_raw * wb).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            hold_loss = float(nn.functional.binary_cross_entropy_with_logits(model(x_hold.to(DEVICE)).cpu(), y_hold).item())
        if hold_loss < best['hold_loss']:
            best = {'epoch': epoch, 'hold_loss': hold_loss, 'train_loss': sum(losses) / max(1, len(losses))}
            torch.save(
                {
                    'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    'mu': mu.cpu(),
                    'sigma': sigma.cpu(),
                    'best': best,
                },
                BEST_CKPT,
            )
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            log(f"[Epoch] {epoch:03d}/{EPOCHS} train_loss={sum(losses)/max(1, len(losses)):.6f} hold_loss={hold_loss:.6f}")
    ckpt = torch.load(BEST_CKPT, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE)
    model.eval()
    return model, ckpt['mu'].float(), ckpt['sigma'].float(), ckpt['best']


def write_predictions(model, mu: torch.Tensor, sigma: torch.Tensor, exp36_model, exp36_mu: torch.Tensor, exp36_sigma: torch.Tensor, split: str, stems: Sequence[str], alpha: float, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        width, height = image_size(split, stem)
        boxes = read_pred(EXP35_PRED_ROOT / split / f'{stem}.txt')
        exp36_scores = exp36_scores_for_boxes(exp36_model, exp36_mu, exp36_sigma, boxes, width, height)
        features = [fusion_features(box, exp36_score, width, height) for box, exp36_score in zip(boxes, exp36_scores)]
        if features:
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32)
                x = ((x - mu.cpu()) / sigma.cpu()).to(DEVICE)
                fusion_scores = torch.sigmoid(model(x)).cpu().tolist()
        else:
            fusion_scores = []
        with (out_dir / f'{stem}.txt').open('w', encoding='utf-8') as f:
            for box, exp36_score, fusion_score in zip(boxes, exp36_scores, fusion_scores):
                exp36_blend = 0.5 * box.score + 0.5 * exp36_score
                score = alpha * exp36_blend + (1.0 - alpha) * float(fusion_score)
                f.write(f'{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {score:.6f}\n')


def subset_gt_dir(split: str, stems: Sequence[str], name: str) -> Path:
    out_dir = LOG_DIR / 'gt_subset' / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = GT_ROOT / split / f'{stem}.txt'
        if src.exists():
            (out_dir / f'{stem}.txt').write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
    return out_dir


def evaluate_by_class(gt_dir: Path, pred_dir: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for class_id in range(1, 11):
        gt_count = 0
        pred_count = 0
        tp = 0
        for gt_path in sorted(gt_dir.glob('*.txt')):
            gt_boxes = [box for box in read_gt(gt_path) if box.class_id == class_id]
            pred_boxes = [box for box in read_pred(pred_dir / f'{gt_path.stem}.txt') if box.class_id == class_id]
            gt_count += len(gt_boxes)
            pred_count += len(pred_boxes)
            used = [False] * len(gt_boxes)
            for pred in sorted(pred_boxes, key=lambda item: item.score, reverse=True):
                best_idx = -1
                best_value = 0.0
                for idx, gt in enumerate(gt_boxes):
                    if used[idx]:
                        continue
                    cur = iou(pred, gt)
                    if cur > best_value:
                        best_value = cur
                        best_idx = idx
                if best_idx >= 0 and best_value >= 0.5:
                    used[best_idx] = True
                    tp += 1
        out[str(class_id)] = {
            'name': CLASS_NAMES[class_id],
            'gt': float(gt_count),
            'pred': float(pred_count),
            'tp_05': float(tp),
            'recall_05': tp / gt_count if gt_count else 0.0,
            'precision_05': tp / pred_count if pred_count else 0.0,
        }
    return out


def ensure_inputs() -> None:
    missing = []
    for path in [EXP36_CKPT, EXP35_PRED_ROOT / 'val', EXP35_PRED_ROOT / 'test']:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise RuntimeError('Missing required inputs: ' + ', '.join(missing))


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')
    ensure_inputs()
    split_map = read_split_map()
    val_stems = split_map['val']
    test_stems = split_map['test']
    random.Random(SEED).shuffle(val_stems)
    holdout_count = max(20, int(len(val_stems) * 0.2))
    hold_stems = sorted(val_stems[:holdout_count])
    train_stems = sorted(val_stems[holdout_count:])
    log(f'[Start] Exp37 online fusion calibrator device={DEVICE}')
    log(f'[Split] val_train={len(train_stems)} val_holdout={len(hold_stems)} test={len(test_stems)}')

    exp36_model, exp36_mu, exp36_sigma = load_exp36_model()
    x_train, y_train, w_train = build_dataset('val', train_stems, exp36_model, exp36_mu, exp36_sigma)
    x_hold, y_hold, _ = build_dataset('val', hold_stems, exp36_model, exp36_mu, exp36_sigma)
    log(
        f'[Dataset] train_samples={len(y_train)} train_pos={int((y_train >= 0.5).sum())} '
        f'hold_samples={len(y_hold)} hold_pos={int((y_hold >= 0.5).sum())} feature_dim={x_train.shape[1]}'
    )
    model, mu, sigma, best = train_model(x_train, y_train, w_train, x_hold, y_hold)

    hold_gt_dir = subset_gt_dir('val', hold_stems, 'val_holdout')
    rows = []
    best_alpha = 1.0
    best_metrics = None
    for alpha in [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0]:
        out_dir = PRED_ROOT / 'val_holdout_search' / f'alpha_{alpha:.2f}'
        write_predictions(model, mu, sigma, exp36_model, exp36_mu, exp36_sigma, 'val', hold_stems, alpha, out_dir)
        metrics = EXP23.evaluate_detections(hold_gt_dir, out_dir, iou_thr=0.5)
        objective = 0.70 * metrics['map_05'] + 0.30 * metrics['acc_05']
        rows.append({'alpha': alpha, 'objective': objective, **metrics})
        log(f"[Search] alpha={alpha:.2f} acc05={metrics['acc_05']:.4f} map05={metrics['map_05']:.4f} pred={int(metrics['prediction_box_count'])} objective={objective:.4f}")
        if best_metrics is None or objective > 0.70 * best_metrics['map_05'] + 0.30 * best_metrics['acc_05']:
            best_alpha = alpha
            best_metrics = metrics

    test_out = PRED_ROOT / 'test'
    write_predictions(model, mu, sigma, exp36_model, exp36_mu, exp36_sigma, 'test', test_stems, best_alpha, test_out)
    test_metrics = EXP23.evaluate_detections(GT_ROOT / 'test', test_out, iou_thr=0.5)
    exp35_metrics = EXP23.evaluate_detections(GT_ROOT / 'test', EXP35_PRED_ROOT / 'test', iou_thr=0.5)
    exp36_metrics = EXP23.evaluate_detections(GT_ROOT / 'test', EXP36_PRED_TEST, iou_thr=0.5) if EXP36_PRED_TEST.exists() else {}
    by_class = evaluate_by_class(GT_ROOT / 'test', test_out)
    summary = {
        'seed': SEED,
        'method': 'online fusion calibrator over Exp35 score, Exp36 alias score, geometry, hierarchy, and class priors',
        'source_predictions': str(EXP35_PRED_ROOT),
        'exp36_checkpoint': str(EXP36_CKPT),
        'train_split': {
            'val_train_count': len(train_stems),
            'val_holdout_count': len(hold_stems),
            'test_count': len(test_stems),
        },
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'best': best,
            'best_alpha': best_alpha,
            'small_classes': sorted(SMALL_CLASSES),
            'weak_small_classes': sorted(WEAK_SMALL_CLASSES),
            'confusion_groups': {str(k): sorted(v) for k, v in CONFUSION_GROUPS.items()},
            'area_prior': AREA_PRIOR,
        },
        'alpha_search': rows,
        'val_holdout_best_metrics': best_metrics,
        'exp35_test_metrics': exp35_metrics,
        'exp36_test_metrics': exp36_metrics,
        'test_metrics': test_metrics,
        'test_by_class': by_class,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log(
        f"[Done] alpha={best_alpha:.2f} "
        f"test_acc05={test_metrics['acc_05']:.4f} "
        f"test_acc075={test_metrics['acc_075']:.4f} "
        f"test_map05={test_metrics['map_05']:.4f} "
        f"pred={int(test_metrics['prediction_box_count'])}"
    )


if __name__ == '__main__':
    main()
