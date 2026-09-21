import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageStat
import torch
import torch.nn as nn
from torchvision.ops import nms


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp35_online_candidate_compression_20260711'
LOG_DIR = EXP_ROOT / 'log'
RUN_LOG = LOG_DIR / 'run_log.txt'
PRED_ROOT = LOG_DIR / 'predictions'
GT_ROOT = ROOT / 'experiment/exp30_real_query_focal_rerank_20260711/log/gt_xyxy'
SUMMARY_JSON = LOG_DIR / 'exp35_online_candidate_compression_20260711_summary.json'

DATA_ROOT = ROOT / 'dataset/VisDroneSplit1000Guarded'
EXP17_SPLIT_MANIFEST = ROOT / 'experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv'
EXP23_SCRIPT = ROOT / 'experiment/exp23_metric_driven_fusion_20260512/scripts/run_exp23_metric_driven_fusion_20260512.py'

EXP30_CKPT = ROOT / 'experiment/exp30_real_query_focal_rerank_20260711/log/exp30_real_query_focal_rerank_20260711_best.pt'
GROUNDINGDINO_ROOT = Path('/home/wangzhe/GroundingDINO')
GDINO_CONFIG = GROUNDINGDINO_ROOT / 'groundingdino/config/GroundingDINO_SwinT_OGC.py'
GDINO_CHECKPOINT = GROUNDINGDINO_ROOT / 'weights/groundingdino_swint_ogc.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT_EMBED_DIM = 32
TEXT_HIDDEN_DIM = 32
MAX_QUERY_LEN = 6
CONTEXT_SCALE = 2.0
REFINE_SCALE = 0.70
OUTER_CONTEXT_SCALE = 2.8
INNER_REFINE_SCALE = 0.50
DEFAULT_SCORE_THRESHOLD = 0.02
DEFAULT_NMS_THRESHOLD = 0.65

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

CLASS_KEYWORDS = {
    1: ['pedestrian', 'person', 'man', 'woman', 'walker'],
    2: ['people', 'persons', 'crowd', 'pedestrians', 'group of people'],
    3: ['bicycle', 'bike', 'cyclist'],
    4: ['car', 'sedan', 'suv', 'automobile', 'vehicle'],
    5: ['van', 'minivan'],
    6: ['truck', 'pickup', 'lorry'],
    7: ['tricycle', 'three wheel'],
    8: ['awning tricycle', 'covered tricycle', 'canopy tricycle'],
    9: ['bus', 'passenger vehicle'],
    10: ['motor', 'motorcycle', 'motorbike', 'scooter'],
}

CANONICAL_QUERY = {
    1: 'pedestrian',
    2: 'group of people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'small tricycle',
    8: 'covered tricycle',
    9: 'bus',
    10: 'small motorcycle',
}

BASE_GDINO_CLASSES = [
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

SMALL_ALIAS_PROMPTS = [
    (2, 'person'),
    (2, 'people'),
    (2, 'group of people'),
    (7, 'tricycle'),
    (8, 'covered tricycle'),
    (8, 'awning tricycle'),
    (10, 'motorcycle'),
    (10, 'motorbike'),
    (10, 'scooter'),
    (10, 'bicycle'),
    (10, 'tricycle'),
]

AREA_THRESHOLDS = {
    1: 0.0035,
    2: 0.0040,
    3: 0.0030,
    7: 0.0045,
    8: 0.0060,
    10: 0.0035,
}

BASE_CLASS_TOPK = {
    1: 120,
    2: 120,
    3: 100,
    4: 140,
    5: 80,
    6: 80,
    7: 100,
    8: 100,
    9: 80,
    10: 140,
}

ALIAS_PROMPT_TOPK = {
    'person': 80,
    'people': 80,
    'group of people': 80,
    'tricycle': 80,
    'covered tricycle': 80,
    'awning tricycle': 80,
    'motorcycle': 100,
    'motorbike': 100,
    'scooter': 100,
    'bicycle': 80,
}

ALIAS_CLASS_TOPK = {
    2: 120,
    7: 100,
    8: 100,
    10: 140,
}

ALIAS_AREA_LIMITS = {
    2: (0.0, 0.0040),
    7: (0.0, 0.0100),
    8: (0.0, 0.0100),
    10: (0.0, 0.0040),
}

CANDIDATE_COMPATIBILITY = {
    1: set(range(1, 11)),
    2: set(range(1, 11)),
    3: set(range(1, 11)),
    4: set(range(1, 11)),
    5: set(range(1, 11)),
    6: set(range(1, 11)),
    7: set(range(1, 11)),
    8: set(range(1, 11)),
    9: set(range(1, 11)),
    10: set(range(1, 11)),
}

FINAL_CLASS_TOPK = {
    1: 160,
    2: 160,
    3: 140,
    4: 120,
    5: 80,
    6: 80,
    7: 160,
    8: 160,
    9: 60,
    10: 200,
}


@dataclass
class Candidate:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: int
    source_name: str


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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(message + '\n')


def ensure_groundingdino_importable() -> None:
    root = str(GROUNDINGDINO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_exp23_helpers():
    spec = importlib.util.spec_from_file_location('exp23_helpers_for_exp34', str(EXP23_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import Exp23 helpers from {EXP23_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp23_helpers_for_exp34'] = module
    spec.loader.exec_module(module)
    return module


EXP23 = load_exp23_helpers()


def tokenize(text: str) -> List[str]:
    return re.sub(r'[^a-z0-9 ]+', ' ', text.lower().replace('-', ' ')).split()


def infer_query_class(text: str) -> int:
    normalized = f" {' '.join(tokenize(text))} "
    ordered = sorted(CLASS_KEYWORDS.items(), key=lambda item: max(len(k) for k in item[1]), reverse=True)
    for class_id, keywords in ordered:
        for keyword in keywords:
            if f" {keyword} " in normalized:
                return class_id
    return 0


def encode_query(query: str, vocab: Dict[str, int]) -> List[int]:
    pad = vocab.get('<pad>', 0)
    unk = vocab.get('<unk>', 1)
    ids = [vocab.get(token, unk) for token in tokenize(query)[:MAX_QUERY_LEN]]
    ids.extend([pad] * (MAX_QUERY_LEN - len(ids)))
    return ids


def load_scorer():
    ckpt = torch.load(EXP30_CKPT, map_location='cpu')
    vocab = ckpt['vocab']
    mu = ckpt['mu'].float()
    sigma = ckpt['sigma'].float()
    model = VocabTextScorer(numeric_dim=mu.shape[1], vocab_size=len(vocab))
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE)
    model.eval()
    return model, mu, sigma, vocab


def load_groundingdino_model():
    ensure_groundingdino_importable()
    from groundingdino.util.inference import Model

    return Model(
        model_config_path=str(GDINO_CONFIG),
        model_checkpoint_path=str(GDINO_CHECKPOINT),
        device=str(DEVICE),
    )


def scaled_region(box: Candidate, scale: float, width: int, height: int) -> Tuple[int, int, int, int]:
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
    means = [value / 255.0 for value in stat.mean[:3]]
    stds = [value / 255.0 for value in stat.stddev[:3]]
    brightness = sum(means) / 3.0
    contrast = sum(stds) / 3.0
    gray = crop.convert('L')
    gstat = ImageStat.Stat(gray)
    return means + stds + [brightness, contrast, gstat.mean[0] / 255.0, gstat.stddev[0] / 255.0]


def text_query_features(query: str, area: float, context_contrast: float, refine_contrast: float) -> List[float]:
    tokens = set(tokenize(query))
    is_small_target = 1.0 if {'small', 'tiny', 'pedestrian', 'person', 'people', 'bicycle', 'motorcycle', 'motor', 'tricycle', 'scooter'} & tokens else 0.0
    is_vehicle = 1.0 if {'vehicle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor', 'motorcycle', 'bicycle', 'scooter'} & tokens else 0.0
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


def box_visual_feature(box: Candidate, width: int, height: int, image: Image.Image) -> List[float]:
    bw = max(1e-6, box.x2 - box.x1)
    bh = max(1e-6, box.y2 - box.y1)
    cx = ((box.x1 + box.x2) / 2.0) / width
    cy = ((box.y1 + box.y2) / 2.0) / height
    nw = bw / width
    nh = bh / height
    area = nw * nh
    aspect = bw / bh
    source_flag = 1.0 if box.source == 1 else 0.0
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
        source_flag,
    ]
    context_region = scaled_region(box, CONTEXT_SCALE, width, height)
    refine_region = scaled_region(box, REFINE_SCALE, width, height)
    outer_context_region = scaled_region(box, OUTER_CONTEXT_SCALE, width, height)
    inner_refine_region = scaled_region(box, INNER_REFINE_SCALE, width, height)
    context = region_stats(image, context_region)
    refine = region_stats(image, refine_region)
    outer_context = region_stats(image, outer_context_region)
    inner_refine = region_stats(image, inner_refine_region)
    delta = [b - a for a, b in zip(context, refine)]
    multiscale_delta = [b - a for a, b in zip(outer_context, inner_refine)]
    ratio = [
        ((refine_region[2] - refine_region[0]) * (refine_region[3] - refine_region[1])) / max(1.0, width * height),
        ((context_region[2] - context_region[0]) * (context_region[3] - context_region[1])) / max(1.0, width * height),
        ((inner_refine_region[2] - inner_refine_region[0]) * (inner_refine_region[3] - inner_refine_region[1])) / max(1.0, width * height),
        ((outer_context_region[2] - outer_context_region[0]) * (outer_context_region[3] - outer_context_region[1])) / max(1.0, width * height),
    ]
    return base + context + refine + delta + outer_context + inner_refine + multiscale_delta + ratio


def query_box_numeric_feature(visual_feature: List[float], query: str) -> List[float]:
    area = visual_feature[4]
    context_contrast = visual_feature[18]
    refine_contrast = visual_feature[28]
    return visual_feature + text_query_features(query, area, context_contrast, refine_contrast)


def small_area_prior(box: Candidate, width: int, height: int) -> float:
    threshold = AREA_THRESHOLDS.get(box.class_id, 0.0)
    if threshold <= 0:
        return 0.0
    area_ratio = ((box.x2 - box.x1) * (box.y2 - box.y1)) / max(1.0, width * height)
    return max(0.0, min(1.0, (threshold - area_ratio) / threshold))


def box_area_ratio(box: Candidate, width: int, height: int) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1) / max(1.0, width * height)


def candidate_iou(a: Candidate, b: Candidate) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / max(area_a + area_b - inter, 1e-9)


def deduplicate_candidates(candidates: Sequence[Candidate], iou_threshold: float = 0.82) -> List[Candidate]:
    selected: List[Candidate] = []
    for box in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(candidate_iou(box, kept) < iou_threshold for kept in selected):
            selected.append(box)
    return selected


def limit_by_class(candidates: Sequence[Candidate], topk_by_class: Dict[int, int]) -> List[Candidate]:
    limited: List[Candidate] = []
    for class_id in sorted({box.class_id for box in candidates}):
        items = [box for box in candidates if box.class_id == class_id]
        topk = topk_by_class.get(class_id, 999999)
        limited.extend(sorted(items, key=lambda item: item.score, reverse=True)[:topk])
    return limited


def candidate_pool_for_class(candidates: Sequence[Candidate], output_class: int) -> List[Candidate]:
    compatible = CANDIDATE_COMPATIBILITY.get(output_class, {output_class})
    return [box for box in candidates if box.class_id in compatible]


def gdino_predict(gdino_model, image_bgr: np.ndarray, classes: Sequence[str], box_threshold: float, text_threshold: float):
    detections = gdino_model.predict_with_classes(
        image=image_bgr,
        classes=list(classes),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    boxes = np.asarray(detections.xyxy)
    scores = np.asarray(detections.confidence)
    class_ids = np.asarray(detections.class_id, dtype=object)
    return boxes, scores, class_ids


def generate_online_candidates(gdino_model, image_path: Path) -> List[Candidate]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f'Cannot read image: {image_path}')
    candidates: List[Candidate] = []

    boxes, scores, raw_class_ids = gdino_predict(gdino_model, image_bgr, BASE_GDINO_CLASSES, 0.20, 0.20)
    for box, score, raw_cid in zip(boxes, scores, raw_class_ids):
        if raw_cid is None:
            continue
        class_id = int(raw_cid) + 1
        if class_id not in CLASS_NAMES:
            continue
        candidates.append(Candidate(class_id, float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score), 1, 'gdino_base'))

    for target_class, prompt in SMALL_ALIAS_PROMPTS:
        boxes, scores, raw_class_ids = gdino_predict(gdino_model, image_bgr, [prompt], 0.03, 0.15)
        for box, score, raw_cid in zip(boxes, scores, raw_class_ids):
            if raw_cid is None:
                continue
            candidates.append(
                Candidate(
                    target_class,
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(score),
                    1,
                    f'gdino_alias:{prompt}',
                )
            )
    return candidates


def score_candidates(
    model: nn.Module,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    vocab: Dict[str, int],
    image_path: Path,
    candidates: Sequence[Candidate],
    query: str,
    output_class: int,
) -> List[Candidate]:
    if not candidates:
        return []
    with Image.open(image_path) as raw:
        image = raw.convert('RGB')
        width, height = image.size
        visual_features = [box_visual_feature(box, width, height, image) for box in candidates]
    features = [query_box_numeric_feature(feature, query) for feature in visual_features]
    tokens = [encode_query(query, vocab) for _ in features]
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        token_ids = torch.tensor(tokens, dtype=torch.long)
        x = ((x - mu.cpu()) / sigma.cpu()).to(DEVICE)
        token_ids = token_ids.to(DEVICE)
        probs = torch.sigmoid(model(x, token_ids)).cpu().tolist()
    scored: List[Candidate] = []
    for box, prob in zip(candidates, probs):
        score = float(prob)
        if output_class in AREA_THRESHOLDS:
            with Image.open(image_path) as raw:
                width, height = raw.size
            score += 0.05 * small_area_prior(Candidate(output_class, box.x1, box.y1, box.x2, box.y2, box.score, box.source, box.source_name), width, height)
        scored.append(Candidate(output_class, box.x1, box.y1, box.x2, box.y2, score, box.source, box.source_name))
    return scored


def class_aware_nms(boxes: List[Candidate], threshold: float) -> List[Candidate]:
    selected: List[Candidate] = []
    for class_id in sorted({box.class_id for box in boxes}):
        items = [box for box in boxes if box.class_id == class_id]
        if not items:
            continue
        t_boxes = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in items], dtype=torch.float32)
        t_scores = torch.tensor([b.score for b in items], dtype=torch.float32)
        keep = nms(t_boxes, t_scores, threshold).cpu().tolist()
        selected.extend(items[idx] for idx in keep)
    return sorted(selected, key=lambda item: item.score, reverse=True)


def final_class_topk(boxes: Sequence[Candidate]) -> List[Candidate]:
    selected: List[Candidate] = []
    for class_id in sorted({box.class_id for box in boxes}):
        items = [box for box in boxes if box.class_id == class_id]
        topk = FINAL_CLASS_TOPK.get(class_id, 999999)
        selected.extend(sorted(items, key=lambda item: item.score, reverse=True)[:topk])
    return sorted(selected, key=lambda item: item.score, reverse=True)


def predict_image(
    image_path: Path,
    query: str,
    gdino_model,
    scorer_model,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    vocab: Dict[str, int],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
) -> List[Candidate]:
    candidates = generate_online_candidates(gdino_model, image_path)
    return predict_image_from_candidates(
        image_path=image_path,
        query=query,
        candidates=candidates,
        scorer_model=scorer_model,
        mu=mu,
        sigma=sigma,
        vocab=vocab,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )


def predict_image_from_candidates(
    image_path: Path,
    query: str,
    candidates: Sequence[Candidate],
    scorer_model,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    vocab: Dict[str, int],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
) -> List[Candidate]:
    target_class = infer_query_class(query)
    scored: List[Candidate] = []
    if target_class:
        class_candidates = candidate_pool_for_class(candidates, target_class)
        scored.extend(score_candidates(scorer_model, mu, sigma, vocab, image_path, class_candidates, query, target_class))
    else:
        for class_id, class_query in CANONICAL_QUERY.items():
            class_candidates = candidate_pool_for_class(candidates, class_id)
            scored.extend(score_candidates(scorer_model, mu, sigma, vocab, image_path, class_candidates, class_query, class_id))
    filtered = [box for box in scored if box.score >= score_threshold]
    return final_class_topk(class_aware_nms(filtered, nms_threshold))


def write_prediction_txt(path: Path, boxes: Sequence[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for box in boxes:
            f.write(f'{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {box.score:.6f}\n')


def write_prediction_json(path: Path, image_path: Path, query: str, boxes: Sequence[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'image': str(image_path),
        'query': query,
        'predictions': [
            {
                'class_id': box.class_id,
                'class_name': CLASS_NAMES.get(box.class_id, 'unknown'),
                'box_xyxy': [box.x1, box.y1, box.x2, box.y2],
                'score': box.score,
                'source': box.source_name,
            }
            for box in boxes
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_split_map() -> Dict[str, List[str]]:
    split_map = {'train': [], 'val': [], 'test': []}
    with EXP17_SPLIT_MANIFEST.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            split = row['split']
            if split in split_map:
                split_map[split].append(row['stem'])
    return {split: sorted(stems) for split, stems in split_map.items()}


def image_for_split(split: str, stem: str) -> Path:
    return DATA_ROOT / f'VisDrone2019-DET-{split}' / 'images' / f'{stem}.jpg'


def build_eval_gt_dir(split: str, stems: Sequence[Tuple[str, str]]) -> Path:
    if split != 'full' and len(stems) == len(read_split_map()[split]):
        return GT_ROOT / split
    out_dir = LOG_DIR / 'gt_subset' / split
    if out_dir.exists():
        for old in out_dir.glob('*.txt'):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    for part, stem in stems:
        src = GT_ROOT / part / f'{stem}.txt'
        if src.exists():
            (out_dir / f'{stem}.txt').write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
    return out_dir


def run_dataset(split: str, limit: Optional[int]) -> None:
    RUN_LOG.write_text('', encoding='utf-8')
    log(f'[Start] Exp35 online candidate compression split={split} limit={limit}')
    gdino_model = load_groundingdino_model()
    scorer_model, mu, sigma, vocab = load_scorer()
    split_map = read_split_map()
    stems = []
    if split == 'full':
        for part in ['train', 'val', 'test']:
            stems.extend((part, stem) for stem in split_map[part])
    else:
        stems = [(split, stem) for stem in split_map[split]]
    if limit:
        stems = stems[:limit]
    out_dir = PRED_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (part, stem) in enumerate(stems, start=1):
        image_path = image_for_split(part, stem)
        candidates = generate_online_candidates(gdino_model, image_path)
        boxes: List[Candidate] = []
        for class_id, query in CANONICAL_QUERY.items():
            boxes.extend(
                predict_image_from_candidates(
                    image_path=image_path,
                    query=query,
                    candidates=candidates,
                    scorer_model=scorer_model,
                    mu=mu,
                    sigma=sigma,
                    vocab=vocab,
                )
            )
        boxes = class_aware_nms(boxes, DEFAULT_NMS_THRESHOLD)
        write_prediction_txt(out_dir / f'{stem}.txt', boxes)
        if idx % 10 == 0 or idx == len(stems):
            log(f'[Progress] {idx}/{len(stems)} stem={stem} boxes={len(boxes)}')
    eval_gt_dir = build_eval_gt_dir(split, stems)
    if eval_gt_dir.exists():
        metrics = EXP23.evaluate_detections(eval_gt_dir, out_dir, iou_thr=0.5)
        summary = {
            'split': split,
            'limit': limit,
            'metrics': metrics,
            'config': {
                'score_threshold': DEFAULT_SCORE_THRESHOLD,
                'nms_threshold': DEFAULT_NMS_THRESHOLD,
                'final_class_topk': FINAL_CLASS_TOPK,
                'candidate_compatibility': {k: sorted(v) for k, v in CANDIDATE_COMPATIBILITY.items()},
            },
        }
        SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        log(f"[Done] acc05={metrics['acc_05']:.4f} acc075={metrics['acc_075']:.4f} map05={metrics['map_05']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Exp35 online candidate-compressed small-object grounding inference')
    parser.add_argument('--image', type=Path, help='input image path')
    parser.add_argument('--query', type=str, default='', help='text query')
    parser.add_argument('--output-json', type=Path, default=LOG_DIR / 'single_prediction.json')
    parser.add_argument('--output-txt', type=Path, default=LOG_DIR / 'single_prediction.txt')
    parser.add_argument('--dataset-split', choices=['train', 'val', 'test', 'full'], help='optional online batch inference split')
    parser.add_argument('--limit', type=int, default=None, help='limit images for dataset inference')
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    if args.dataset_split:
        run_dataset(args.dataset_split, args.limit)
        return

    if args.image is None or not args.query:
        raise SystemExit('Either provide --dataset-split or provide both --image and --query')

    RUN_LOG.write_text('', encoding='utf-8')
    log('[Start] Exp35 online single-image inference')
    gdino_model = load_groundingdino_model()
    scorer_model, mu, sigma, vocab = load_scorer()
    boxes = predict_image(args.image, args.query, gdino_model, scorer_model, mu, sigma, vocab)
    write_prediction_txt(args.output_txt, boxes)
    write_prediction_json(args.output_json, args.image, args.query, boxes)
    log(f'[Done] image={args.image} query="{args.query}" boxes={len(boxes)} json={args.output_json}')


if __name__ == '__main__':
    main()
