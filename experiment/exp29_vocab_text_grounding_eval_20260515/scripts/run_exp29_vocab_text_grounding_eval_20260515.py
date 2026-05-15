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
EXP_ROOT = ROOT / 'experiment/exp29_vocab_text_grounding_eval_20260515'
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
BEST_CKPT = LOG_DIR / 'exp29_vocab_text_grounding_eval_20260515_best.pt'
SUMMARY_JSON = LOG_DIR / 'exp29_vocab_text_grounding_eval_20260515_summary.json'
SUMMARY_TRAINVAL_MD = LOG_DIR / 'evaluation_summary_exp29_vocab_text_grounding_trainval_class_aware.md'
SUMMARY_VAL_MD = LOG_DIR / 'evaluation_summary_exp29_vocab_text_grounding_val_class_aware.md'
SUMMARY_TEST_MD = LOG_DIR / 'evaluation_summary_exp29_vocab_text_grounding_test_class_aware.md'
SUMMARY_FULL_MD = LOG_DIR / 'evaluation_summary_exp29_vocab_text_grounding_full_class_aware.md'
SEARCH_CSV = LOG_DIR / 'postprocess_search_exp29_vocab_text_grounding.csv'
GROUNDING_VAL_JSON = LOG_DIR / 'grounding_eval_val_topk.json'
GROUNDING_TEST_JSON = LOG_DIR / 'grounding_eval_test_topk.json'
GROUNDING_FULL_JSON = LOG_DIR / 'grounding_eval_full_topk.json'
GROUNDING_FULL_MD = LOG_DIR / 'grounding_eval_full_topk.md'

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
TEXT_EMBED_DIM = 32
TEXT_HIDDEN_DIM = 32
MAX_QUERY_LEN = 6
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
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
    spec = importlib.util.spec_from_file_location('exp23_helpers_for_exp29', str(EXP23_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import Exp23 helpers from {EXP23_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp23_helpers_for_exp29'] = module
    spec.loader.exec_module(module)
    return module


EXP23 = load_exp23_helpers()


class VocabTextScorer(nn.Module):
    def __init__(self, numeric_dim: int, vocab_size: int, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, TEXT_EMBED_DIM, padding_idx=0)
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_EMBED_DIM, TEXT_HIDDEN_DIM),
            nn.LayerNorm(TEXT_HIDDEN_DIM),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(numeric_dim + TEXT_HIDDEN_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, numeric: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        mask = (token_ids != 0).float().unsqueeze(-1)
        embedded = self.embedding(token_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        text_feature = self.text_proj(pooled)
        return self.net(torch.cat([numeric, text_feature], dim=1)).squeeze(1)


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


def read_gt_boxes(path: Path) -> List[DetBox]:
    boxes: List[DetBox] = []
    if not path.exists():
        return boxes
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(DetBox(cls, x1, y1, x2, y2, 1.0, source=0))
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


def tokenize(text: str) -> List[str]:
    return text.lower().replace('-', ' ').split()


def build_vocab() -> Dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for templates in QUERY_TEMPLATES.values():
        for query in templates:
            for token in tokenize(query):
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def encode_query(query: str, vocab: Dict[str, int]) -> List[int]:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(query)[:MAX_QUERY_LEN]]
    ids.extend([vocab[PAD_TOKEN]] * (MAX_QUERY_LEN - len(ids)))
    return ids


def text_query_features(query: str, area: float, context_contrast: float, refine_contrast: float) -> List[float]:
    tokens = set(tokenize(query))
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
    ]


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


def query_box_numeric_feature(visual_feature: List[float], query: str) -> List[float]:
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
    return read_gt_boxes(path)


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


def flatten(records, vocab: Dict[str, int]):
    features: List[List[float]] = []
    token_ids: List[List[int]] = []
    gt_labels: List[float] = []
    soft_labels: List[float] = []
    targets: List[float] = []
    for record in records:
        for query_class in QUERY_CLASSES:
            for query in QUERY_TEMPLATES[query_class]:
                for box, visual_feature in zip(record['boxes'], record['visual_features']):
                    gt_label = query_hard_label(box, record['gt'], query_class, 0.5)
                    soft_label = query_soft_label(box, record['teacher'], query_class)
                    features.append(query_box_numeric_feature(visual_feature, query))
                    token_ids.append(encode_query(query, vocab))
                    gt_labels.append(gt_label)
                    soft_labels.append(soft_label)
                    targets.append(max(gt_label, soft_label))
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(token_ids, dtype=torch.long),
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
    vocab: Dict[str, int],
    threshold: float,
    nms_threshold: float,
) -> List[DetBox]:
    if not record['boxes']:
        return []
    features = [query_box_numeric_feature(visual_feature, query) for visual_feature in record['visual_features']]
    token_ids = [encode_query(query, vocab) for _ in features]
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        probs = torch.sigmoid(model((x - mu) / sigma, tokens)).cpu().tolist()
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
    vocab: Dict[str, int],
    threshold: float = THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD,
) -> Dict[str, List[DetBox]]:
    model.eval()
    pred_by_stem: Dict[str, List[DetBox]] = {}
    with torch.no_grad():
        for record in records:
            selected: List[DetBox] = []
            for query_class, query in CANONICAL_QUERY.items():
                selected.extend(score_query(model, record, query, query_class, mu, sigma, vocab, threshold, nms_threshold))
            pred_by_stem[record['stem']] = class_aware_nms(selected, nms_threshold)
    return pred_by_stem


def search_postprocess(model: nn.Module, val_records, mu: torch.Tensor, sigma: torch.Tensor, vocab: Dict[str, int]) -> Dict[str, float]:
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
            val_pred = apply_model(model, val_records, mu, sigma, vocab, threshold=threshold, nms_threshold=nms_threshold)
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


def has_gt_class(record, class_id: int) -> bool:
    return any(box.class_id == class_id for box in record['gt'])


def query_hit_at_k(scored_boxes: Sequence[DetBox], gt_boxes: Sequence[DetBox], k: int) -> bool:
    for pred in scored_boxes[:k]:
        for gt in gt_boxes:
            if iou(box_tuple(pred), box_tuple(gt)) >= 0.5:
                return True
    return False


def evaluate_grounding_topk(model: nn.Module, records, mu: torch.Tensor, sigma: torch.Tensor, vocab: Dict[str, int]) -> Dict[str, object]:
    total_queries = 0
    hit_counts = {1: 0, 5: 0, 10: 0}
    rank_sum = 0.0
    found_count = 0
    by_class: Dict[str, Dict[str, float]] = {
        str(class_id): {'queries': 0.0, 'recall_at_1': 0.0, 'recall_at_5': 0.0, 'recall_at_10': 0.0}
        for class_id in QUERY_CLASSES
    }

    model.eval()
    for record in records:
        for query_class, query in CANONICAL_QUERY.items():
            gt_boxes = [box for box in record['gt'] if box.class_id == query_class]
            if not gt_boxes:
                continue
            total_queries += 1
            class_key = str(query_class)
            by_class[class_key]['queries'] += 1.0
            scored = score_query(
                model,
                record,
                query,
                query_class,
                mu,
                sigma,
                vocab,
                threshold=-1.0,
                nms_threshold=1.0,
            )
            scored = sorted(scored, key=lambda box: box.score, reverse=True)
            hit_by_k = {k: query_hit_at_k(scored, gt_boxes, k) for k in hit_counts}
            for k, hit in hit_by_k.items():
                if hit:
                    hit_counts[k] += 1
                    by_class[class_key][f'recall_at_{k}'] += 1.0

            first_rank = None
            for idx, pred in enumerate(scored, start=1):
                if query_hit_at_k([pred], gt_boxes, 1):
                    first_rank = idx
                    break
            if first_rank is not None:
                rank_sum += first_rank
                found_count += 1

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
        'found_query_count': found_count,
        'by_class': by_class,
    }


def write_grounding_summary(path: Path, metrics: Dict[str, object]) -> None:
    lines = [
        '# Exp29 Grounding TopK Evaluation',
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


def build_gt_dirs() -> None:
    EXP23.convert_gt_dirs()
    src_root = ROOT / 'experiment/exp23_metric_driven_fusion_20260512/log/gt_xyxy'
    if GT_ROOT.exists():
        shutil.rmtree(GT_ROOT)
    shutil.copytree(src_root, GT_ROOT)


def train_model(train_records, val_records, vocab: Dict[str, int]):
    x_train_raw, tok_train, gt_train, soft_train, target_train = flatten(train_records, vocab)
    x_val_raw, _, _, _, _ = flatten(val_records, vocab)
    mu = x_train_raw.mean(dim=0, keepdim=True)
    sigma = x_train_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train = (x_train_raw - mu) / sigma

    model = VocabTextScorer(numeric_dim=x_train.shape[1], vocab_size=len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion_gt = nn.BCEWithLogitsLoss()
    criterion_soft = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(x_train, tok_train, gt_train, soft_train, target_train), batch_size=BATCH_SIZE, shuffle=True)

    best = {'epoch': 0, 'val_objective': -1.0, 'val_acc_05': 0.0, 'val_map_05': 0.0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, token_ids, y_gt, y_soft, y_target in loader:
            logits = model(xb, token_ids)
            loss = criterion_gt(logits, y_gt) + ALPHA_TEACHER * criterion_soft(logits, y_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_pred = apply_model(model, val_records, mu, sigma, vocab)
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
            torch.save({'model': model.state_dict(), 'mu': mu, 'sigma': sigma, 'best': best, 'vocab': vocab}, BEST_CKPT)

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
    log('[Start] Exp29 vocab text grounding scorer with TopK evaluation')
    log(
        f'[Config] threshold={THRESHOLD:.2f} nms={NMS_THRESHOLD:.2f} epochs={EPOCHS} '
        f'context_scale={CONTEXT_SCALE:.2f} refine_scale={REFINE_SCALE:.2f} '
        f'text_embed_dim={TEXT_EMBED_DIM} query_classes={len(QUERY_CLASSES)}'
    )

    build_gt_dirs()
    split_map = read_split_map()
    train_records = [load_record('train', stem) for stem in split_map['train']]
    val_records = [load_record('val', stem) for stem in split_map['val']]
    test_records = [load_record('test', stem) for stem in split_map['test']]
    trainval_records = train_records + val_records
    full_records = trainval_records + test_records
    vocab = build_vocab()
    log(f'[Vocab] size={len(vocab)} tokens={",".join(sorted(vocab.keys()))}')

    model, mu, sigma, best = train_model(train_records, val_records, vocab)
    best_postprocess = search_postprocess(model, val_records, mu, sigma, vocab)
    log(
        f"[Postprocess] threshold={best_postprocess['threshold']:.2f} "
        f"nms={best_postprocess['nms']:.2f} "
        f"val_acc05={best_postprocess['acc_05']:.4f} "
        f"val_map05={best_postprocess['map_05']:.4f} "
        f"objective={best_postprocess['objective']:.4f}"
    )
    final_threshold = float(best_postprocess['threshold'])
    final_nms = float(best_postprocess['nms'])

    pred_trainval = apply_model(model, trainval_records, mu, sigma, vocab, threshold=final_threshold, nms_threshold=final_nms)
    pred_test = apply_model(model, test_records, mu, sigma, vocab, threshold=final_threshold, nms_threshold=final_nms)
    pred_full = apply_model(model, full_records, mu, sigma, vocab, threshold=final_threshold, nms_threshold=final_nms)
    save_predictions(pred_trainval, PRED_ROOT / 'trainval')
    save_predictions(pred_test, PRED_ROOT / 'test')
    save_predictions(pred_full, PRED_ROOT / 'full')

    trainval_metrics = EXP23.evaluate_detections(GT_ROOT / 'trainval', PRED_ROOT / 'trainval', iou_thr=0.5)
    val_metrics = EXP23.evaluate_detections(GT_ROOT / 'val', PRED_ROOT / 'val', iou_thr=0.5)
    test_metrics = EXP23.evaluate_detections(GT_ROOT / 'test', PRED_ROOT / 'test', iou_thr=0.5)
    full_metrics = EXP23.evaluate_detections(GT_ROOT / 'full', PRED_ROOT / 'full', iou_thr=0.5)

    grounding_val = evaluate_grounding_topk(model, val_records, mu, sigma, vocab)
    grounding_test = evaluate_grounding_topk(model, test_records, mu, sigma, vocab)
    grounding_full = evaluate_grounding_topk(model, full_records, mu, sigma, vocab)
    GROUNDING_VAL_JSON.write_text(json.dumps(grounding_val, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    GROUNDING_TEST_JSON.write_text(json.dumps(grounding_test, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    GROUNDING_FULL_JSON.write_text(json.dumps(grounding_full, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_grounding_summary(GROUNDING_FULL_MD, grounding_full)

    write_summary(SUMMARY_TRAINVAL_MD, 'Exp29 TrainVal Vocab Text Grounding Summary', trainval_metrics)
    write_summary(SUMMARY_VAL_MD, 'Exp29 Val Vocab Text Grounding Summary', val_metrics)
    write_summary(SUMMARY_TEST_MD, 'Exp29 Test Vocab Text Grounding Summary', test_metrics)
    write_summary(SUMMARY_FULL_MD, 'Exp29 Full Vocab Text Grounding Summary', full_metrics)

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
            'text_embed_dim': TEXT_EMBED_DIM,
            'text_hidden_dim': TEXT_HIDDEN_DIM,
            'max_query_len': MAX_QUERY_LEN,
            'vocab': vocab,
            'query_classes': QUERY_CLASSES,
            'query_templates': QUERY_TEMPLATES,
            'canonical_query': CANONICAL_QUERY,
            'feature_design': 'box geometry + A context RGB stats + B refined RGB stats + B-A deltas + query semantic numeric priors + learnable vocab text embedding',
        },
        'best_postprocess': best_postprocess,
        'best': best,
        'trainval_metrics': trainval_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'full_metrics': full_metrics,
        'grounding_val': grounding_val,
        'grounding_test': grounding_test,
        'grounding_full': grounding_full,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log(
        f"[Grounding] full R@1={grounding_full['recall_at_1']:.4f} "
        f"R@5={grounding_full['recall_at_5']:.4f} R@10={grounding_full['recall_at_10']:.4f} "
        f"mean_rank={grounding_full['mean_first_hit_rank']:.2f}"
    )
    log(f"[Done] full acc05={full_metrics['acc_05']:.4f} acc075={full_metrics['acc_075']:.4f} map05={full_metrics['map_05']:.4f} pred={int(full_metrics['prediction_box_count'])}")


if __name__ == '__main__':
    main()
