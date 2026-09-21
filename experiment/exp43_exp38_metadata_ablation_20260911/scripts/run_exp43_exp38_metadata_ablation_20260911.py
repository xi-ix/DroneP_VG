import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp43_exp38_metadata_ablation_20260911'
LOG_DIR = EXP_ROOT / 'log'
RUN_LOG = LOG_DIR / 'run_log.txt'
SUMMARY_JSON = LOG_DIR / 'exp43_exp38_metadata_ablation_20260911_summary.json'
REPORT_MD = EXP_ROOT / 'Exp38元数据特征消融结果.md'
EXP38_SCRIPT = ROOT / 'experiment/exp38_online_rich_metadata_fusion_20260712/scripts/run_exp38_rich_metadata_fusion_20260712.py'

BATCH_SIZE = 4096
LR = 0.0007
ALPHAS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]


def load_exp38_module():
    spec = importlib.util.spec_from_file_location('exp38_for_exp43', str(EXP38_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import Exp38 from {EXP38_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp38_for_exp43'] = module
    spec.loader.exec_module(module)
    return module


EXP38 = load_exp38_module()
DEVICE = EXP38.DEVICE


# Names follow the exact order returned by Exp38.rich_features. Keeping all 68
# slots in every run makes model capacity identical; disabled slots are zeroed.
FEATURE_NAMES = [
    'final_score', 'gdino_score', 'query_match_score', 'exp36_score',
    'small_area_bonus', 'final_minus_gdino', 'final_minus_query',
    'query_minus_gdino', 'exp36_minus_final', 'final_times_exp36',
    'final_logit', 'exp36_logit',
    'sqrt_area_ratio', 'log_area_ratio', 'box_width_ratio', 'box_height_ratio',
    'log_aspect_ratio', 'center_x', 'center_y', 'area_prior_similarity',
    'area_prior_ratio', 'log_rank',
    'source_is_alias', 'source_is_base', 'source_class_mismatch',
    'source_class_scaled', 'class_scaled',
] + [f'hierarchy_{idx}' for idx in range(10)] \
  + [f'class_{idx}' for idx in range(1, 11)] \
  + [f'source_class_{idx}' for idx in range(1, 11)] \
  + [f'prompt_{prompt}' for prompt in EXP38.PROMPT_VOCAB]

RAW = {'final_score'}
GEOMETRY = {
    'sqrt_area_ratio', 'log_area_ratio', 'box_width_ratio', 'box_height_ratio',
    'log_aspect_ratio', 'center_x', 'center_y', 'log_rank',
}
QUERY = {'query_match_score'}
SOURCE = {
    'source_is_alias', 'source_is_base', 'source_class_mismatch',
    'source_class_scaled', 'class_scaled',
} | {f'source_class_{idx}' for idx in range(1, 11)} \
  | {f'prompt_{prompt}' for prompt in EXP38.PROMPT_VOCAB}
SMALL_PRIOR = {
    'small_area_bonus', 'area_prior_similarity', 'area_prior_ratio',
} | {f'hierarchy_{idx}' for idx in range(10)} \
  | {f'class_{idx}' for idx in range(1, 11)}

FEATURE_CONFIGS = [
    ('raw_score', 'Raw score', RAW),
    ('geometry', '+ geometry', RAW | GEOMETRY),
    ('query', '+ query', RAW | GEOMETRY | QUERY),
    ('source', '+ source', RAW | GEOMETRY | QUERY | SOURCE),
    ('small_prior', '+ small prior', RAW | GEOMETRY | QUERY | SOURCE | SMALL_PRIOR),
    ('full_exp38', 'Full Exp38', set(FEATURE_NAMES)),
]


def log(message: str) -> None:
    print(message, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(message + '\n')


def feature_mask(enabled: set) -> torch.Tensor:
    unknown = enabled - set(FEATURE_NAMES)
    if unknown:
        raise RuntimeError(f'Unknown features: {sorted(unknown)}')
    return torch.tensor([[1.0 if name in enabled else 0.0 for name in FEATURE_NAMES]], dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    x_train_raw: torch.Tensor,
    y_train: torch.Tensor,
    w_train: torch.Tensor,
    x_hold_raw: torch.Tensor,
    y_hold: torch.Tensor,
    mask: torch.Tensor,
    seed: int,
    epochs: int,
):
    set_seed(seed)
    mu = x_train_raw.mean(dim=0, keepdim=True)
    sigma = x_train_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train = ((x_train_raw - mu) / sigma) * mask
    x_hold = ((x_hold_raw - mu) / sigma) * mask
    model = EXP38.RichFusionModel(input_dim=len(FEATURE_NAMES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train, w_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        pin_memory=(DEVICE.type == 'cuda'),
    )
    best = {'epoch': 0, 'hold_loss': float('inf'), 'state': None}
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb, wb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none') * wb).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            hold_logits = model(x_hold.to(DEVICE)).cpu()
            hold_loss = float(nn.functional.binary_cross_entropy_with_logits(hold_logits, y_hold).item())
        if hold_loss < best['hold_loss']:
            best = {
                'epoch': epoch,
                'hold_loss': hold_loss,
                'train_loss': sum(losses) / max(1, len(losses)),
                'state': {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
    model.load_state_dict(best.pop('state'))
    model.eval()
    return model, mu, sigma, best


def write_predictions(
    model,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
    split: str,
    stems: Sequence[str],
    exp36_model,
    exp36_mu: torch.Tensor,
    exp36_sigma: torch.Tensor,
    alpha: float,
    out_dir: Path,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        rows = EXP38.read_metadata(EXP38.META_ROOT / split / f'{stem}.jsonl')
        exp36_scores = EXP38.exp36_scores_for_meta(exp36_model, exp36_mu, exp36_sigma, rows)
        features = [EXP38.rich_features(row, score) for row, score in zip(rows, exp36_scores)]
        if features:
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32)
                x = (((x - mu) / sigma) * mask).to(DEVICE)
                fusion_scores = torch.sigmoid(model(x)).cpu().tolist()
        else:
            fusion_scores = []
        with (out_dir / f'{stem}.txt').open('w', encoding='utf-8') as f:
            for row, fusion_score in zip(rows, fusion_scores):
                # The common residual is final_score only. No hidden Exp36 feature
                # leaks into score-only or partial-metadata controls.
                score = alpha * float(row['final_score']) + (1.0 - alpha) * float(fusion_score)
                x1, y1, x2, y2 = [float(value) for value in row['box_xyxy']]
                f.write(f"{int(row['class_id'])} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.6f}\n")


def candidate_digest(split: str, stems: Sequence[str]) -> Tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for stem in stems:
        path = EXP38.META_ROOT / split / f'{stem}.jsonl'
        data = path.read_bytes()
        digest.update(stem.encode('utf-8'))
        digest.update(b'\0')
        digest.update(data)
        count += sum(1 for line in data.splitlines() if line.strip())
    return count, digest.hexdigest()


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean()), float(tensor.std(unbiased=True)) if len(values) > 1 else 0.0


def aggregate(config_runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {'runs': list(config_runs)}
    for key in ['acc_05', 'acc_075', 'map_05', 'prediction_box_count']:
        mean, std = mean_std([float(run['test_metrics'][key]) for run in config_runs])
        result[f'{key}_mean'] = mean
        result[f'{key}_std'] = std
    result['best_alpha_mean'], result['best_alpha_std'] = mean_std([float(run['best_alpha']) for run in config_runs])
    return result


def render_report(summary: Dict[str, object]) -> str:
    lines = [
        '# Exp38 元数据特征消融结果', '',
        '## 实验设置', '',
        f"- 固定候选集合：test 共 {summary['candidate_count']} 个候选框，SHA256 `{summary['candidate_sha256']}`。",
        f"- 随机种子：{', '.join(str(seed) for seed in summary['seeds'])}；每组 {summary['epochs']} epochs。",
        f"- 模型结构固定：68 维输入槽位，参数量 {summary['model_parameter_count']}。",
        '- 控制变量：候选坐标、类别、训练/holdout/test 划分、标签、样本权重、MLP 结构和 alpha 搜索均保持不变。',
        '- 所有配置均保留相同的 68 维输入槽位，未启用特征置零，因此参数量完全相同。',
        '- alpha 只在 val holdout 上选择；test 仅用于最终报告。', '',
        '## 结果', '',
        '| 配置 | 启用维数 | Acc@0.5 | Acc@0.75 | mAP@0.5 | 逐步 ΔmAP | 相对 Raw ΔmAP | 候选数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    raw_map = float(summary['configs'][0]['map_05_mean'])
    previous_map = raw_map
    for config in summary['configs']:
        delta = float(config['map_05_mean']) - raw_map
        step_delta = float(config['map_05_mean']) - previous_map
        lines.append(
            f"| {config['label']} | {config['enabled_dim']} | "
            f"{config['acc_05_mean']:.4f} +/- {config['acc_05_std']:.4f} | "
            f"{config['acc_075_mean']:.4f} +/- {config['acc_075_std']:.4f} | "
            f"{config['map_05_mean']:.4f} +/- {config['map_05_std']:.4f} | "
            f"{step_delta:+.4f} | {delta:+.4f} | {config['prediction_box_count_mean']:.0f} |"
        )
        previous_map = float(config['map_05_mean'])
    full = summary['configs'][-1]
    lines += [
        '', '## 结论', '',
        f"Full Exp38 相对 score-only MLP 的 mAP@0.5 变化为 {float(full['map_05_mean']) - raw_map:+.4f}，相对提升 {(float(full['map_05_mean']) / raw_map - 1.0) * 100:.1f}%。",
        '由于各组候选数量和模型参数量一致，差异只来自被启用的元数据字段；Acc@0.5 基本不变时，mAP 的变化直接反映置信度排序质量。',
        '逐级结果应按实际数值表述；若某一步下降，不能宣称该字段单独带来正收益，只能结合后续字段的互补效果讨论。', '',
    ]
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Controlled Exp38 metadata feature ablation.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(FEATURE_NAMES) != 68:
        raise RuntimeError(f'Expected 68 Exp38 features, got {len(FEATURE_NAMES)}')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')
    EXP38.ensure_inputs()
    EXP38.RUN_LOG = RUN_LOG
    split_map = EXP38.read_split_map()
    val_stems = list(split_map['val'])
    test_stems = list(split_map['test'])
    random.Random(EXP38.SEED).shuffle(val_stems)
    holdout_count = max(20, int(len(val_stems) * 0.2))
    hold_stems = sorted(val_stems[:holdout_count])
    train_stems = sorted(val_stems[holdout_count:])
    hold_gt_dir = EXP38.subset_gt_dir('val', hold_stems, 'exp43_val_holdout')
    candidate_count, candidate_sha256 = candidate_digest('test', test_stems)
    log(f'[Start] Exp43 device={DEVICE} seeds={args.seeds} epochs={args.epochs}')
    log(f'[Control] train={len(train_stems)} holdout={len(hold_stems)} test={len(test_stems)} candidates={candidate_count} sha256={candidate_sha256}')

    exp36_model, exp36_mu, exp36_sigma = EXP38.load_exp36_model()
    x_train, y_train, w_train = EXP38.build_dataset('val', train_stems, exp36_model, exp36_mu, exp36_sigma)
    x_hold, y_hold, _ = EXP38.build_dataset('val', hold_stems, exp36_model, exp36_mu, exp36_sigma)
    if x_train.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f'Exp38 feature dimension changed: expected {len(FEATURE_NAMES)}, got {x_train.shape[1]}')
    model_parameter_count = sum(parameter.numel() for parameter in EXP38.RichFusionModel(len(FEATURE_NAMES)).parameters())
    config_summaries: List[Dict[str, object]] = []

    for config_key, label, enabled in FEATURE_CONFIGS:
        mask = feature_mask(enabled)
        runs = []
        for seed in args.seeds:
            log(f'[Train] config={config_key} enabled_dim={int(mask.sum())} seed={seed}')
            model, mu, sigma, best = train_model(
                x_train, y_train, w_train, x_hold, y_hold, mask, seed, args.epochs
            )
            search_root = LOG_DIR / '_alpha_search' / config_key / f'seed_{seed}'
            best_alpha = ALPHAS[0]
            best_objective = float('-inf')
            best_hold_metrics = None
            for alpha in ALPHAS:
                out_dir = search_root / f'alpha_{alpha:.2f}'
                write_predictions(
                    model, mu, sigma, mask, 'val', hold_stems,
                    exp36_model, exp36_mu, exp36_sigma, alpha, out_dir,
                )
                metrics = EXP38.EXP23.evaluate_detections(hold_gt_dir, out_dir, iou_thr=0.5)
                objective = 0.70 * metrics['map_05'] + 0.30 * metrics['acc_05']
                if objective > best_objective:
                    best_alpha = alpha
                    best_objective = objective
                    best_hold_metrics = metrics
            shutil.rmtree(search_root)
            test_out = LOG_DIR / 'predictions' / config_key / f'seed_{seed}' / 'test'
            write_predictions(
                model, mu, sigma, mask, 'test', test_stems,
                exp36_model, exp36_mu, exp36_sigma, best_alpha, test_out,
            )
            test_metrics = EXP38.EXP23.evaluate_detections(EXP38.GT_ROOT / 'test', test_out, iou_thr=0.5)
            log(
                f"[Result] config={config_key} seed={seed} alpha={best_alpha:.2f} "
                f"acc05={test_metrics['acc_05']:.4f} acc075={test_metrics['acc_075']:.4f} "
                f"map05={test_metrics['map_05']:.4f} pred={int(test_metrics['prediction_box_count'])}"
            )
            runs.append({
                'seed': seed,
                'best_epoch': best['epoch'],
                'best_hold_loss': best['hold_loss'],
                'best_alpha': best_alpha,
                'holdout_metrics': best_hold_metrics,
                'test_metrics': test_metrics,
            })
        config_summary = {
            'key': config_key,
            'label': label,
            'enabled_dim': int(mask.sum()),
            'enabled_features': [name for name in FEATURE_NAMES if name in enabled],
            **aggregate(runs),
        }
        config_summaries.append(config_summary)

    summary = {
        'experiment': 'Exp43 controlled nested feature ablation for Exp38 rich metadata design',
        'seeds': args.seeds,
        'epochs': args.epochs,
        'device': str(DEVICE),
        'split': {'val_train': len(train_stems), 'val_holdout': len(hold_stems), 'test': len(test_stems)},
        'candidate_count': candidate_count,
        'candidate_sha256': candidate_sha256,
        'feature_slot_count': len(FEATURE_NAMES),
        'model_parameter_count': model_parameter_count,
        'common_residual_score': 'alpha * final_score + (1-alpha) * fusion_probability',
        'alpha_grid': ALPHAS,
        'configs': config_summaries,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    REPORT_MD.write_text(render_report(summary), encoding='utf-8')
    log(f'[Done] summary={SUMMARY_JSON} report={REPORT_MD}')


if __name__ == '__main__':
    main()
