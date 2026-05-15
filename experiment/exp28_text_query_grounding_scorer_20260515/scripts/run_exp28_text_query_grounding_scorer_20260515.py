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

from PIL import Image, ImageStat
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import nms


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp28_text_query_grounding_scorer_20260515'
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
BEST_CKPT = LOG_DIR / 'exp28_text_query_grounding_scorer_20260515_best.pt'
SUMMARY_JSON = LOG_DIR / 'exp28_text_query_grounding_scorer_20260515_summary.json'
SUMMARY_TRAINVAL_MD = LOG_DIR / 'evaluation_summary_exp28_text_query_grounding_trainval_class_aware.md'
SUMMARY_VAL_MD = LOG_DIR / 'evaluation_summary_exp28_text_query_grounding_val_class_aware.md'
SUMMARY_TEST_MD = LOG_DIR / 'evaluation_summary_exp28_text_query_grounding_test_class_aware.md'
SUMMARY_FULL_MD = LOG_DIR / 'evaluation_summary_exp28_text_query_grounding_full_class_aware.md'
SEARCH_CSV = LOG_DIR / 'postprocess_search_exp28_text_query_grounding.csv'

SEED = 42
EPOCHS = 24
BATCH_SIZE = 8192
LR = 0.001
ALPHA_TEACHER = 0.7
THRESHOLD = 0.05
NMS_THRESHOLD = 0.50
CONTEXT_SCALE = 2.0
REFINE_SCALE = 0.70
THRESHOLD_GRID = [0.03, 0.04, 0.05, 0.06, 0.08]
NMS_GRID = [0.45, 0.50, 0.55, 0.60]

CLASS_PROMPTS = {
    0: 'ignored region',
    1: 'pedestrian person walking on road',
    2: 'people group or standing person',
    3: 'bicycle small rider vehicle',
    4: 'car compact road vehicle',
    5: 'van medium road vehicle',
    6: 'truck large road vehicle',
    7: 'tricycle three wheel vehicle',
    8: 'awning tricycle covered three wheel vehicle',
    9: 'bus large passenger vehicle',
    10: 'motor motorcycle small fast vehicle',
    11: 'other object',
}

SMALL_TARGET_CLASSES = {1, 2, 3, 7, 8, 10}
VEHICLE_CLASSES = {3, 4, 5, 6, 7, 8, 9, 10}
PERSON_CLASSES = {1, 2}
TEXT_HASH_DIM = 24
QUERY_CLASSES = list(range(1, 11))
QUERY_TEMPLATES = {
    1: ['pedestrian', 'person', 'small person in drone image'],
    2: ['people', 'standing person', 'group of people'],
    3: ['bicycle', 'small bicycle', 'bicycle rider'],
    4: ['car', 'small car', 'vehicle on road'],
    5: ['van', 'medium vehicle', 'white van'],
    6: ['truck', 'large vehicle', 'cargo truck'],
    7: ['tricycle', 'three wheel vehicle', 'small tricycle'],
    8: ['awning tricycle', 'covered tricycle', 'covered three wheel vehicle'],
    9: ['bus', 'large passenger vehicle', 'bus on road'],
    10: ['motor', 'motorcycle', 'small motor vehicle'],
}
CANONICAL_QUERY = {class_id: texts[0] for class_id, texts in QUERY_TEMPLATES.items()}


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
    spec = importlib.util.spec_from_file_location('exp23_helpers_for_exp28', str(EXP23_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import Exp23 helpers from {EXP23_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp23_helpers_for_exp28'] = module
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


def image_path(split: str, stem: str) -> Path:
    return DATA_ROOT / f'VisDrone2019-DET-{split}' / 'images' / f'{stem}.jpg'


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


def scaled_region(box: DetBox, scale: float, width: int, height: int) -> Tuple[int, int, int, int]:
    bw = max(1.0, box.x2 - box.x1)
    bh = max(1.0, box.y2 - box.y1)
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    sw = bw * scale
    sh = bh * scale
    x1 = max(0, int(math.floor(cx - sw / 2.0)))
    y1 = max(0, int(math.floor(cy - sh / 2.0)))
    x2 = min(width, int(math.ceil(cx + sw / 2.0)))
    y2 = min(height, int(math.ceil(cy + sh / 2.0)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def region_stats(image: Image.Image, region: Tuple[int, int, int, int]) -> List[float]:
    crop = image.crop(region)
    stat = ImageStat.Stat(crop)
    means = [(value / 255.0) for value in stat.mean[:3]]
    stds = [(value / 255.0) for value in stat.stddev[:3]]
    brightness = sum(means) / 3.0
    contrast = sum(stds) / 3.0
    return means + stds + [brightness, contrast]


def hashed_text_embedding(text: str) -> List[float]:
    values = [0.0] * TEXT_HASH_DIM
    for token in text.lower().replace('-', ' ').split():
        bucket = sum(ord(ch) for ch in token) % TEXT_HASH_DIM
        sign = 1.0 if (sum(ord(ch) * (idx + 1) for idx, ch in enumerate(token)) % 2 == 0) else -1.0
        values[bucket] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm > 0:
        values = [value / norm for value in values]
    return values


def text_query_features(query: str, area: float, context_contrast: float, refine_contrast: float) -> List[float]:
    tokens = set(query.lower().replace('-', ' ').split())
    is_small_target = 1.0 if {'small', 'tiny', 'pedestrian', 'person', 'people', 'bicycle', 'motorcycle', 'motor', 'tricycle'} & tokens else 0.0
    is_vehicle = 1.0 if {'vehicle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor', 'motorcycle', 'bicycle'} & tokens else 0.0
    is_person = 1.0 if {'pedestrian', 'person', 'people'} & tokens else 0.0
    compactness_prior = is_small_target * max(0.0, min(1.0, 1.0 - area * 250.0))
    contrast_gain = refine_contrast - context_contrast
    return [
        is_small_target,
        is_vehicle,
        is_person,
        compactness_prior,
        contrast_gain * is_small_target,
        context_contrast * is_vehicle,
        refine_contrast * is_person,
    ] + hashed_text_embedding(query)


def box_visual_feature(box: DetBox, width: int, height: int, image: Image.Image) -> List[float]:
    bw = max(1e-6, box.x2 - box.x1)
    bh = max(1e-6, box.y2 - box.y1)
    cx = ((box.x1 + box.x2) / 2.0) / width
    cy = ((box.y1 + box.y2) / 2.0) / height
    nw = bw / width
    nh = bh / height
    area = nw * nh
    aspect = bw / bh
    base = [
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
    context_region = scaled_region(box, CONTEXT_SCALE, width, height)
    refine_region = scaled_region(box, REFINE_SCALE, width, height)
    context = region_stats(image, context_region)
    refine = region_stats(image, refine_region)
    delta = [b - a for a, b in zip(context, refine)]
    ratio = [
        ((refine_region[2] - refine_region[0]) * (refine_region[3] - refine_region[1])) / max(1.0, width * height),
        ((context_region[2] - context_region[0]) * (context_region[3] - context_region[1])) / max(1.0, width * height),
    ]
    return base + context + refine + delta + ratio


def query_box_feature(visual_feature: List[float], query: str) -> List[float]:
    area = visual_feature[4]
    context_contrast = visual_feature[18]
    refine_contrast = visual_feature[26]
    return visual_feature + text_query_features(query, area, context_contrast, refine_contrast)


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


def query_matches_class(query_class: int, candidate_class: int) -> bool:
    return query_class == candidate_class


def query_hard_label(candidate: DetBox, targets: Sequence[DetBox], query_class: int, threshold: float) -> float:
    if not query_matches_class(query_class, candidate.class_id):
        return 0.0
    return best_match_label(candidate, targets, threshold)


def query_soft_label(candidate: DetBox, teacher_boxes: Sequence[DetBox], query_class: int) -> float:
    if not query_matches_class(query_class, candidate.class_id):
        return 0.0
    return teacher_soft_label(candidate, teacher_boxes)


def load_gt_xyxy(split: str, stem: str) -> List[DetBox]:
    path = GT_ROOT / split / f'{stem}.txt'
    return read_boxes(path, source=0)


def load_record(split: str, stem: str):
    with Image.open(image_path(split, stem)) as raw_image:
        image = raw_image.convert('RGB')
        width, height = image.size
        candidates = read_boxes(EXP18_FULL / f'{stem}.txt', source=0) + read_boxes(GDINO_FULL / f'{stem}.txt', source=1)
        visual_features = [box_visual_feature(box, width, height, image) for box in candidates]
    teacher = read_boxes(TEACHER_FULL / f'{stem}.txt', source=2)
    gt = load_gt_xyxy(split, stem)
    return {
        'split': split,
        'stem': stem,
        'boxes': candidates,
        'visual_features': visual_features,
        'gt': gt,
        'teacher': teacher,
    }


def flatten(records):
    features: List[List[float]] = []
    gt_labels: List[float] = []
    soft_labels: List[float] = []
    targets: List[float] = []
    for record in records:
        for query_class in QUERY_CLASSES:
            for query in QUERY_TEMPLATES[query_class]:
                for box, visual_feature in zip(record['boxes'], record['visual_features']):
                    gt_label = query_hard_label(box, record['gt'], query_class, 0.5)
                    soft_label = query_soft_label(box, record['teacher'], query_class)
                    features.append(query_box_feature(visual_feature, query))
                    gt_labels.append(gt_label)
                    soft_labels.append(soft_label)
                    targets.append(max(gt_label, soft_label))
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


def score_query(
    model: nn.Module,
    record,
    query: str,
    output_class: int,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    threshold: float,
    nms_threshold: float,
) -> List[DetBox]:
    if not record['boxes']:
        return []
    features = [query_box_feature(visual_feature, query) for visual_feature in record['visual_features']]
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        probs = torch.sigmoid(model((x - mu) / sigma)).cpu().tolist()
    selected = [
        DetBox(output_class, box.x1, box.y1, box.x2, box.y2, float(prob), box.source)
        for box, prob in zip(record['boxes'], probs)
        if prob >= threshold
    ]
    return class_aware_nms(selected, nms_threshold)


def apply_model(
    model: nn.Module,
    records,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    threshold: float = THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD,
) -> Dict[str, List[DetBox]]:
    model.eval()
    pred_by_stem: Dict[str, List[DetBox]] = {}
    with torch.no_grad():
        for record in records:
            selected: List[DetBox] = []
            for query_class, query in CANONICAL_QUERY.items():
                selected.extend(score_query(model, record, query, query_class, mu, sigma, threshold, nms_threshold))
            pred_by_stem[record['stem']] = class_aware_nms(selected, nms_threshold)
    return pred_by_stem


def search_postprocess(model: nn.Module, val_records, mu: torch.Tensor, sigma: torch.Tensor) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    best = {
        'threshold': THRESHOLD,
        'nms': NMS_THRESHOLD,
        'objective': -1.0,
        'acc_05': 0.0,
        'map_05': 0.0,
        'acc_075': 0.0,
        'prediction_box_count': 0.0,
    }
    search_root = PRED_ROOT / 'val_postprocess_search'
    for threshold in THRESHOLD_GRID:
        for nms_threshold in NMS_GRID:
            out_dir = search_root / f'thr_{threshold:.2f}_nms_{nms_threshold:.2f}'
            val_pred = apply_model(model, val_records, mu, sigma, threshold=threshold, nms_threshold=nms_threshold)
            save_predictions(val_pred, out_dir)
            metrics = EXP23.evaluate_detections(GT_ROOT / 'val', out_dir, iou_thr=0.5)
            objective = 0.5 * metrics['acc_05'] + 0.5 * metrics['map_05']
            row = {
                'threshold': threshold,
                'nms': nms_threshold,
                'objective': objective,
                'acc_05': metrics['acc_05'],
                'acc_075': metrics['acc_075'],
                'map_05': metrics['map_05'],
                'prediction_box_count': metrics['prediction_box_count'],
            }
            rows.append(row)
            if objective > best['objective']:
                best = row.copy()

    SEARCH_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SEARCH_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['threshold', 'nms', 'objective', 'acc_05', 'acc_075', 'map_05', 'prediction_box_count'],
        )
        writer.writeheader()
        writer.writerows(rows)
    return best


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
    log('[Start] Exp28 text-query grounding scorer')
    log(
        f'[Config] threshold={THRESHOLD:.2f} nms={NMS_THRESHOLD:.2f} epochs={EPOCHS} '
        f'context_scale={CONTEXT_SCALE:.2f} refine_scale={REFINE_SCALE:.2f} '
        f'text_hash_dim={TEXT_HASH_DIM} query_classes={len(QUERY_CLASSES)}'
    )

    build_gt_dirs()
    split_map = read_split_map()
    train_records = [load_record('train', stem) for stem in split_map['train']]
    val_records = [load_record('val', stem) for stem in split_map['val']]
    test_records = [load_record('test', stem) for stem in split_map['test']]
    trainval_records = train_records + val_records
    full_records = trainval_records + test_records

    model, mu, sigma, best = train_model(train_records, val_records)
    best_postprocess = search_postprocess(model, val_records, mu, sigma)
    log(
        f"[Postprocess] threshold={best_postprocess['threshold']:.2f} "
        f"nms={best_postprocess['nms']:.2f} "
        f"val_acc05={best_postprocess['acc_05']:.4f} "
        f"val_map05={best_postprocess['map_05']:.4f} "
        f"objective={best_postprocess['objective']:.4f}"
    )
    final_threshold = float(best_postprocess['threshold'])
    final_nms = float(best_postprocess['nms'])

    pred_trainval = apply_model(model, trainval_records, mu, sigma, threshold=final_threshold, nms_threshold=final_nms)
    pred_test = apply_model(model, test_records, mu, sigma, threshold=final_threshold, nms_threshold=final_nms)
    pred_full = apply_model(model, full_records, mu, sigma, threshold=final_threshold, nms_threshold=final_nms)
    save_predictions(pred_trainval, PRED_ROOT / 'trainval')
    save_predictions(pred_test, PRED_ROOT / 'test')
    save_predictions(pred_full, PRED_ROOT / 'full')

    trainval_metrics = EXP23.evaluate_detections(GT_ROOT / 'trainval', PRED_ROOT / 'trainval', iou_thr=0.5)
    val_metrics = EXP23.evaluate_detections(GT_ROOT / 'val', PRED_ROOT / 'val', iou_thr=0.5)
    test_metrics = EXP23.evaluate_detections(GT_ROOT / 'test', PRED_ROOT / 'test', iou_thr=0.5)
    full_metrics = EXP23.evaluate_detections(GT_ROOT / 'full', PRED_ROOT / 'full', iou_thr=0.5)

    write_summary(SUMMARY_TRAINVAL_MD, 'Exp28 TrainVal Text-Query Grounding Summary', trainval_metrics)
    write_summary(SUMMARY_VAL_MD, 'Exp28 Val Text-Query Grounding Summary', val_metrics)
    write_summary(SUMMARY_TEST_MD, 'Exp28 Test Text-Query Grounding Summary', test_metrics)
    write_summary(SUMMARY_FULL_MD, 'Exp28 Full Text-Query Grounding Summary', full_metrics)

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
            'context_scale': CONTEXT_SCALE,
            'refine_scale': REFINE_SCALE,
            'threshold_grid': THRESHOLD_GRID,
            'nms_grid': NMS_GRID,
            'text_hash_dim': TEXT_HASH_DIM,
            'query_classes': QUERY_CLASSES,
            'query_templates': QUERY_TEMPLATES,
            'canonical_query': CANONICAL_QUERY,
            'feature_design': 'box geometry + A context RGB stats + B refined RGB stats + B-A deltas + dynamic text query hash + query semantic priors',
        },
        'best_postprocess': best_postprocess,
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
