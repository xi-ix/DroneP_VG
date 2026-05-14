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

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import nms


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp16_attn_bigmlp_20260422'
LOG_DIR = EXP_ROOT / 'log'

DATA_SOURCE = ROOT / 'visdrone_test_100'
MANIFEST = DATA_SOURCE / 'manifest.tsv'
DATA_ROOT = EXP_ROOT / 'data' / 'VisDroneSplit100'

EXP13_PRED_DIR = ROOT / 'experiment/exp13_full_rebuild_20260421/log/predictions'
LAB9_SW_ROOT = ROOT / 'experiment/lab9_reproduce_20260421/log/predictions'

RUN_LOG = LOG_DIR / 'run_log.txt'
SUMMARY_JSON = LOG_DIR / 'exp16_attn_bigmlp_20260422_summary.json'
BEST_CKPT = LOG_DIR / 'exp16_attn_bigmlp_20260422_best.pt'
PRED_ROOT = LOG_DIR / 'predictions'
GT_ROOT = LOG_DIR / 'gt_xyxy'
SPLIT_PLAN_JSON = LOG_DIR / 'split_plan.json'
SPLIT_MANIFEST_TSV = LOG_DIR / 'split_manifest.tsv'
SUMMARY_TRAINVAL_MD = LOG_DIR / 'evaluation_summary_exp16_attn_bigmlp_trainval_class_aware.md'
SUMMARY_TEST_MD = LOG_DIR / 'evaluation_summary_exp16_attn_bigmlp_test_class_aware.md'
SUMMARY_FULL_MD = LOG_DIR / 'evaluation_summary_exp16_attn_bigmlp_full_class_aware.md'

LANGUAGE_PROMPT = 'aerial traffic scene with roads, vehicles, and drones in sky'
PROMPT_VOCAB_SIZE = 2048
PROMPT_MAX_TOKENS = 24
EXP14_BEST_THRESHOLD = 0.08
EXP14_BEST_NMS = 0.66


@dataclass
class DetBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: int


def tokenize_prompt(prompt: str, vocab_size: int = PROMPT_VOCAB_SIZE, max_tokens: int = PROMPT_MAX_TOKENS) -> torch.Tensor:
    tokens = prompt.lower().replace(',', ' ').replace('.', ' ').split()
    if not tokens:
        tokens = ['generic']
    ids = [abs(hash(tok)) % vocab_size for tok in tokens[:max_tokens]]
    return torch.tensor(ids, dtype=torch.long)


def load_baseline_module():
    baseline_script = ROOT / 'experiment/baseline/groundingdino_base_refdrone100_recovered_20260421/scripts/run_groundingdino_base_refdrone100_recovered_20260421.py'
    spec = importlib.util.spec_from_file_location('exp14_baseline', str(baseline_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import baseline helpers from {baseline_script}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp14_baseline'] = module
    spec.loader.exec_module(module)
    return module


BASE = load_baseline_module()


def log(message: str) -> None:
    print(message, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(message + '\n')


def list_stems() -> List[str]:
    stems = []
    with MANIFEST.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row:
                stems.append(row[0].strip())
    return sorted(stems)


def ensure_split_dataset() -> Dict[str, List[str]]:
    source_images = DATA_SOURCE / 'images'
    source_annotations = DATA_SOURCE / 'annotations'
    split_dirs = {
        'train': DATA_ROOT / 'VisDrone2019-DET-train',
        'val': DATA_ROOT / 'VisDrone2019-DET-val',
        'test': DATA_ROOT / 'VisDrone2019-DET-test',
    }
    for split_dir in split_dirs.values():
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'annotations').mkdir(parents=True, exist_ok=True)

    stems = list_stems()
    rng = random.Random(42)
    rng.shuffle(stems)
    split_map = {
        'train': sorted(stems[:70]),
        'val': sorted(stems[70:85]),
        'test': sorted(stems[85:]),
    }
    SPLIT_PLAN_JSON.write_text(json.dumps(split_map, ensure_ascii=False, indent=2), encoding='utf-8')
    with SPLIT_MANIFEST_TSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['split', 'stem', 'image_path', 'annotation_path'])
        for split, items in split_map.items():
            for stem in items:
                image_src = source_images / f'{stem}.jpg'
                ann_src = source_annotations / f'{stem}.txt'
                image_dst = split_dirs[split] / 'images' / f'{stem}.jpg'
                ann_dst = split_dirs[split] / 'annotations' / f'{stem}.txt'
                if image_dst.exists() or image_dst.is_symlink():
                    image_dst.unlink()
                if ann_dst.exists() or ann_dst.is_symlink():
                    ann_dst.unlink()
                image_dst.symlink_to(image_src)
                ann_dst.symlink_to(ann_src)
                writer.writerow([split, stem, str(image_dst), str(ann_dst)])
    return split_map


def get_image_path(split: str, stem: str) -> Path:
    return DATA_ROOT / f'VisDrone2019-DET-{split}' / 'images' / f'{stem}.jpg'


def read_xyxy_labels(path: Path, is_prediction: bool, source: int) -> List[DetBox]:
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
            x1 = float(parts[1])
            y1 = float(parts[2])
            x2 = float(parts[3])
            y2 = float(parts[4])
            if x2 <= x1 or y2 <= y1:
                continue
            score = float(parts[5]) if is_prediction and len(parts) == 6 else 1.0
            boxes.append(DetBox(class_id, x1, y1, x2, y2, score, source))
    return boxes


def convert_gt_dirs() -> None:
    for split in ['train', 'val', 'test']:
        src_gt = DATA_ROOT / f'VisDrone2019-DET-{split}' / 'annotations'
        src_img = DATA_ROOT / f'VisDrone2019-DET-{split}' / 'images'
        BASE.convert_visdrone_gt_to_xyxy(src_gt, src_img, GT_ROOT / split)


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


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
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


def greedy_match_labels(candidates: List[DetBox], gt_boxes: List[DetBox]) -> List[int]:
    labels = [0] * len(candidates)
    used = set()
    order = sorted(range(len(candidates)), key=lambda i: candidates[i].score, reverse=True)
    for idx in order:
        pred = candidates[idx]
        best_gt = -1
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in used or gt.class_id != pred.class_id:
                continue
            cur_iou = iou_xyxy((pred.x1, pred.y1, pred.x2, pred.y2), (gt.x1, gt.y1, gt.x2, gt.y2))
            if cur_iou >= 0.5 and cur_iou > best_iou:
                best_iou = cur_iou
                best_gt = gt_idx
        if best_gt >= 0:
            used.add(best_gt)
            labels[idx] = 1
    return labels


def load_candidate_records(split: str, stems: Sequence[str]):
    records = []
    sw_dir = LAB9_SW_ROOT / f'{split}_sw'
    for stem in stems:
        image = cv2.imread(str(get_image_path(split, stem)))
        if image is None:
            continue
        height, width = image.shape[:2]
        exp13_boxes = read_xyxy_labels(EXP13_PRED_DIR / f'{stem}.txt', is_prediction=True, source=0)
        sw_boxes = read_xyxy_labels(sw_dir / f'{stem}.txt', is_prediction=True, source=1) if sw_dir.exists() else []
        gt_boxes = read_xyxy_labels(GT_ROOT / split / f'{stem}.txt', is_prediction=False, source=0)
        merged = exp13_boxes + sw_boxes
        labels = greedy_match_labels(merged, gt_boxes)
        feats = [feature_from_box(box, width, height) for box in merged]
        records.append({'stem': stem, 'features': feats, 'labels': labels, 'boxes': merged, 'gt_boxes': gt_boxes})
    return records


def flatten_records(records):
    feats: List[List[float]] = []
    labels: List[int] = []
    for record in records:
        for feat, label in zip(record['features'], record['labels']):
            feats.append(feat)
            labels.append(label)
    if not feats:
        return torch.zeros((0, 9), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)
    return torch.tensor(feats, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


class AttnBigMLPScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, backbone_dim: int = 64, prompt_vocab_size: int = PROMPT_VOCAB_SIZE, prompt_dim: int = 32, prompt_hidden_dim: int = 32, attn_scale: float = 0.15):
        super().__init__()
        self.image_preprocess = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
        )
        self.prompt_embedding = nn.Embedding(prompt_vocab_size, prompt_dim)
        self.text_preprocess = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.LayerNorm(prompt_dim),
            nn.ReLU(),
        )
        self.null_prompt = nn.Parameter(torch.zeros(prompt_dim, dtype=torch.float32))
        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, prompt_hidden_dim),
            nn.ReLU(),
            nn.Linear(prompt_hidden_dim, input_dim),
        )
        self.attn_scale = attn_scale
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, backbone_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.ReLU(),
            nn.Linear(backbone_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
        x = self.image_preprocess(x)
        if prompt_ids.numel() == 0:
            prompt_vec = self.text_preprocess(self.null_prompt.to(x.device).unsqueeze(0)).squeeze(0)
        else:
            prompt_tokens = self.prompt_embedding(prompt_ids.to(x.device))
            prompt_tokens = self.text_preprocess(prompt_tokens)
            prompt_vec = prompt_tokens.mean(dim=0)
        gate = torch.sigmoid(self.prompt_proj(prompt_vec)).unsqueeze(0).expand(x.shape[0], -1)
        x = x * (1.0 + self.attn_scale * gate)
        features = self.backbone(x)
        return self.score_head(features)


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
        for idx in keep:
            selected.append(items[idx])
    return sorted(selected, key=lambda b: b.score, reverse=True)


def apply_model_to_records(model: nn.Module, records, mu: torch.Tensor, sigma: torch.Tensor, threshold: float, nms_thr: float, prompt_ids: torch.Tensor) -> Dict[str, List[DetBox]]:
    model.eval()
    out: Dict[str, List[DetBox]] = {}
    with torch.no_grad():
        for record in records:
            if not record['boxes']:
                out[record['stem']] = []
                continue
            feats = torch.tensor(record['features'], dtype=torch.float32)
            feats = (feats - mu) / sigma
            probs = torch.sigmoid(model(feats, prompt_ids)).squeeze(1).cpu().tolist()
            selected = []
            for box, prob in zip(record['boxes'], probs):
                if prob >= threshold:
                    selected.append(DetBox(box.class_id, box.x1, box.y1, box.x2, box.y2, float(prob), box.source))
            out[record['stem']] = class_aware_nms(selected, nms_thr)
    return out


def save_predictions(pred_by_stem: Dict[str, List[DetBox]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, boxes in pred_by_stem.items():
        with (out_dir / f'{stem}.txt').open('w', encoding='utf-8') as f:
            for box in boxes:
                f.write(f'{box.class_id} {box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} {box.score:.6f}\n')


def evaluate_from_records(pred_by_stem: Dict[str, List[DetBox]], split: str) -> Dict[str, float]:
    pred_dir = PRED_ROOT / split
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    save_predictions(pred_by_stem, pred_dir)
    return BASE.evaluate_detections(GT_ROOT / split, pred_dir, class_aware=True, iou_thr=0.5)


def record_summary(metrics: Dict[str, float]) -> List[str]:
    return [
        '# Exp16 Attention + Big MLP + Scoring Head Summary',
        '',
        f"- 文件数: {metrics.get('file_count', 0)}",
        f"- GT框数: {metrics.get('gt_box_count', 0)}",
        f"- 预测框数: {metrics.get('prediction_box_count', 0)}",
        f"- Acc@0.5: {metrics.get('acc_05', 0.0):.4f}",
        f"- Acc@0.75: {metrics.get('acc_075', 0.0):.4f}",
        f"- mAP@0.5: {metrics.get('map_05', 0.0):.4f}",
        f"- 空图误报率: {metrics.get('empty_fp_rate', 0.0):.4f}",
    ]


def search_thresholds(model: nn.Module, val_records, mu: torch.Tensor, sigma: torch.Tensor, prompt_ids: torch.Tensor) -> Tuple[float, float, Dict[str, float]]:
    best_score = -1.0
    best_threshold = 0.20
    best_nms = 0.40
    best_metrics: Dict[str, float] = {}
    thresholds = [0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48]
    nms_values = [0.40, 0.50, 0.60, 0.70]
    for threshold in thresholds:
        for nms_thr in nms_values:
            pred = apply_model_to_records(model, val_records, mu, sigma, threshold, nms_thr, prompt_ids)
            temp_dir = LOG_DIR / 'threshold_search_tmp'
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            save_predictions(pred, temp_dir)
            metrics = BASE.evaluate_detections(GT_ROOT / 'val', temp_dir, class_aware=True, iou_thr=0.5)
            score = 0.7 * metrics['acc_05'] + 0.3 * metrics['map_05']
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_nms = nms_thr
                best_metrics = metrics
    return best_threshold, best_nms, best_metrics


def train_supervised(model: AttnBigMLPScorer, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, prompt_ids: torch.Tensor) -> Dict[str, float]:
    pos = float(y_train.sum().item())
    neg = float(y_train.numel() - pos)
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=1024, shuffle=True)
    best = {'val_acc': -1.0, 'epoch': 0}
    for epoch in range(1, 61):
        model.train()
        losses = []
        for xb, yb in loader:
            logits = model(xb, prompt_ids).squeeze(1)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        with torch.no_grad():
            train_acc = float(((torch.sigmoid(model(x_train, prompt_ids)).squeeze(1) > 0.5).float() == y_train).float().mean().item())
            val_logits = model(x_val, prompt_ids).squeeze(1)
            val_acc = float(((torch.sigmoid(val_logits) > 0.5).float() == y_val).float().mean().item()) if y_val.numel() else 0.0
            val_loss = float(criterion(val_logits, y_val).item()) if y_val.numel() else 0.0
        log(f"[SL] epoch={epoch:03d}/60 loss={sum(losses)/max(len(losses),1):.6f} train_acc={train_acc:.4f} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
        if val_acc > best['val_acc']:
            best = {'val_acc': val_acc, 'epoch': epoch}
            torch.save({'model': model.state_dict(), 'stage': 'sl'}, BEST_CKPT)
    return best


def train_reinforcement(model: AttnBigMLPScorer, train_records, val_records, mu: torch.Tensor, sigma: torch.Tensor, prompt_ids: torch.Tensor) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    entropy_coef = 0.001
    baseline = 0.0
    best = {'val_objective': -1e9, 'epoch': 0, 'val_acc05': 0.0, 'val_map_proxy': 0.0}
    for epoch in range(1, 41):
        model.train()
        reward_hist = []
        loss_hist = []
        for record in train_records:
            if not record['boxes']:
                continue
            feats = torch.tensor(record['features'], dtype=torch.float32)
            feats = (feats - mu) / sigma
            logits = model(feats, prompt_ids).squeeze(1)
            probs = torch.sigmoid(logits)
            actions = (probs > 0.5).float()
            chosen = []
            for action, box, prob in zip(actions.tolist(), record['boxes'], probs.tolist()):
                if action > 0.5:
                    chosen.append(DetBox(box.class_id, box.x1, box.y1, box.x2, box.y2, float(prob), box.source))
            chosen = class_aware_nms(chosen, 0.40)
            matched = 0
            used = set()
            for pred in sorted(chosen, key=lambda b: b.score, reverse=True):
                best_idx = -1
                best_iou = 0.0
                for idx, gt in enumerate(record['gt_boxes']):
                    if idx in used or gt.class_id != pred.class_id:
                        continue
                    cur_iou = iou_xyxy((pred.x1, pred.y1, pred.x2, pred.y2), (gt.x1, gt.y1, gt.x2, gt.y2))
                    if cur_iou >= 0.5 and cur_iou > best_iou:
                        best_iou = cur_iou
                        best_idx = idx
                if best_idx >= 0:
                    used.add(best_idx)
                    matched += 1
            gt_count = len(record['gt_boxes'])
            precision = matched / len(chosen) if chosen else 0.0
            recall = matched / gt_count if gt_count > 0 else 0.0
            reward = 0.7 * recall + 0.3 * precision
            baseline = 0.95 * baseline + 0.05 * reward
            advantage = reward - baseline
            log_prob = actions * torch.log(probs.clamp(1e-6, 1 - 1e-6)) + (1 - actions) * torch.log((1 - probs).clamp(1e-6, 1 - 1e-6))
            entropy = -(probs * torch.log(probs.clamp(1e-6, 1 - 1e-6)) + (1 - probs) * torch.log((1 - probs).clamp(1e-6, 1 - 1e-6)))
            loss = -(advantage * log_prob.mean()) - entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            reward_hist.append(reward)
            loss_hist.append(float(loss.item()))

        model.eval()
        val_rewards = []
        val_accs = []
        val_precs = []
        with torch.no_grad():
            for record in val_records:
                if not record['boxes']:
                    continue
                feats = torch.tensor(record['features'], dtype=torch.float32)
                feats = (feats - mu) / sigma
                probs = torch.sigmoid(model(feats, prompt_ids).squeeze(1)).tolist()
                chosen = []
                for box, prob in zip(record['boxes'], probs):
                    if prob > 0.5:
                        chosen.append(DetBox(box.class_id, box.x1, box.y1, box.x2, box.y2, float(prob), box.source))
                chosen = class_aware_nms(chosen, 0.40)
                matched = 0
                used = set()
                for pred in sorted(chosen, key=lambda b: b.score, reverse=True):
                    best_idx = -1
                    best_iou = 0.0
                    for idx, gt in enumerate(record['gt_boxes']):
                        if idx in used or gt.class_id != pred.class_id:
                            continue
                        cur_iou = iou_xyxy((pred.x1, pred.y1, pred.x2, pred.y2), (gt.x1, gt.y1, gt.x2, gt.y2))
                        if cur_iou >= 0.5 and cur_iou > best_iou:
                            best_iou = cur_iou
                            best_idx = idx
                    if best_idx >= 0:
                        used.add(best_idx)
                        matched += 1
                gt_count = len(record['gt_boxes'])
                precision = matched / len(chosen) if chosen else 0.0
                recall = matched / gt_count if gt_count > 0 else 0.0
                reward = 0.7 * recall + 0.3 * precision
                val_rewards.append(reward)
                val_accs.append(recall)
                val_precs.append(precision)
        val_objective = float(np.mean(val_rewards)) if val_rewards else 0.0
        val_acc05 = float(np.mean(val_accs)) if val_accs else 0.0
        val_map_proxy = float(np.mean(val_precs)) if val_precs else 0.0
        log(f"[RL] epoch={epoch:03d}/40 loss={sum(loss_hist)/max(len(loss_hist),1):.6f} reward={np.mean(reward_hist) if reward_hist else 0.0:.4f} val_reward={val_objective:.4f} val_acc05={val_acc05:.4f}")
        if val_objective > best['val_objective'] + 1e-6:
            best = {'val_objective': val_objective, 'epoch': epoch, 'val_acc05': val_acc05, 'val_map_proxy': val_map_proxy}
            torch.save({'model': model.state_dict(), 'stage': 'rl'}, BEST_CKPT)
    return best


def main() -> None:
    if not EXP13_PRED_DIR.exists():
        raise RuntimeError(f'Missing Exp13 predictions: {EXP13_PRED_DIR}')

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    GT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    split_map = ensure_split_dataset()
    convert_gt_dirs()

    prompt_ids = tokenize_prompt(LANGUAGE_PROMPT)

    log('[Start] Exp16 attention + big-MLP scoring rebuild based on Exp14 pipeline')
    log(f'[Config] data_root={DATA_SOURCE}')
    log(f'[Config] exp13_pred_dir={EXP13_PRED_DIR}')
    log(f'[Config] sw_pred_root={LAB9_SW_ROOT}')
    log(f'[Config] language_prompt={LANGUAGE_PROMPT}')
    log('[Config] split=70/15/15, SL=60 epochs, RL=40 epochs')
    log(f'[Config] aligned_postprocess threshold={EXP14_BEST_THRESHOLD:.2f} nms={EXP14_BEST_NMS:.2f}')

    train_records = load_candidate_records('train', split_map['train'])
    val_records = load_candidate_records('val', split_map['val'])
    test_records = load_candidate_records('test', split_map['test'])
    full_records = train_records + val_records + test_records

    x_train_raw, y_train = flatten_records(train_records)
    x_val_raw, y_val = flatten_records(val_records)

    if x_train_raw.numel() == 0:
        raise RuntimeError('No training candidates were generated.')

    mu = x_train_raw.mean(dim=0, keepdim=True)
    sigma = x_train_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train = (x_train_raw - mu) / sigma
    x_val = (x_val_raw - mu) / sigma if x_val_raw.numel() else x_val_raw

    calibrator = AttnBigMLPScorer(input_dim=x_train.shape[1], hidden_dim=128, backbone_dim=64)
    sl_best = train_supervised(calibrator, x_train, y_train, x_val, y_val, prompt_ids)
    rl_best = train_reinforcement(calibrator, train_records, val_records, mu, sigma, prompt_ids)

    ckpt = torch.load(BEST_CKPT, map_location='cpu')
    calibrator.load_state_dict(ckpt['model'], strict=False)

    best_threshold, best_nms = EXP14_BEST_THRESHOLD, EXP14_BEST_NMS
    log(f'[AlignedConfig] fixed threshold={best_threshold:.2f} nms={best_nms:.2f} (same as Exp14 local refine best)')

    trainval_pred = apply_model_to_records(calibrator, val_records, mu, sigma, best_threshold, best_nms, prompt_ids)
    test_pred = apply_model_to_records(calibrator, test_records, mu, sigma, best_threshold, best_nms, prompt_ids)
    full_pred = apply_model_to_records(calibrator, full_records, mu, sigma, best_threshold, best_nms, prompt_ids)

    trainval_metrics = evaluate_from_records(trainval_pred, 'val')
    test_metrics = evaluate_from_records(test_pred, 'test')

    full_pred_dir = PRED_ROOT / 'full'
    if full_pred_dir.exists():
        shutil.rmtree(full_pred_dir)
    save_predictions(full_pred, full_pred_dir)

    merged_gt_dir = GT_ROOT / 'full'
    if merged_gt_dir.exists():
        shutil.rmtree(merged_gt_dir)
    merged_gt_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        for gt_file in (GT_ROOT / split).glob('*.txt'):
            shutil.copy2(gt_file, merged_gt_dir / gt_file.name)
    full_metrics = BASE.evaluate_detections(merged_gt_dir, full_pred_dir, class_aware=True, iou_thr=0.5)

    SUMMARY_TRAINVAL_MD.write_text('\n'.join(record_summary(trainval_metrics)) + '\n', encoding='utf-8')
    SUMMARY_TEST_MD.write_text('\n'.join(record_summary(test_metrics)) + '\n', encoding='utf-8')
    SUMMARY_FULL_MD.write_text('\n'.join(record_summary(full_metrics)) + '\n', encoding='utf-8')

    summary = {
        'seed': 42,
        'data_root': str(DATA_SOURCE),
        'exp13_predictions': str(EXP13_PRED_DIR),
        'sw_predictions': str(LAB9_SW_ROOT),
        'language_prompt': LANGUAGE_PROMPT,
        'split_counts': {k: len(v) for k, v in split_map.items()},
        'sl_best': sl_best,
        'rl_best': rl_best,
        'best_threshold': best_threshold,
        'best_nms': best_nms,
        'trainval_metrics': trainval_metrics,
        'test_metrics': test_metrics,
        'full_metrics': full_metrics,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'[Done] trainval={SUMMARY_TRAINVAL_MD}')
    log(f'[Done] test={SUMMARY_TEST_MD}')
    log(f'[Done] full={SUMMARY_FULL_MD}')
    log(f'[Done] summary={SUMMARY_JSON}')


if __name__ == '__main__':
    main()
