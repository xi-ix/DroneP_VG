import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
LOG_DIR = EXP_ROOT / 'log'
BASE_SCRIPT = SCRIPT_DIR / 'run_exp14_full_rebuild_20260421.py'

RESULT_JSON = LOG_DIR / 'restricted_search_result_20260421.json'
RESULT_MD = LOG_DIR / 'restricted_search_result_20260421.md'


def load_exp14_module():
    spec = importlib.util.spec_from_file_location('exp14_rebuild_module', str(BASE_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module from {BASE_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp14_rebuild_module'] = module
    spec.loader.exec_module(module)
    return module


def evaluate_full(module, pred_by_stem: Dict[str, List[object]]) -> Dict[str, float]:
    full_pred_dir = module.PRED_ROOT / 'full_restricted_search'
    if full_pred_dir.exists():
        shutil.rmtree(full_pred_dir)
    module.save_predictions(pred_by_stem, full_pred_dir)

    merged_gt_dir = module.GT_ROOT / 'full_restricted_search'
    if merged_gt_dir.exists():
        shutil.rmtree(merged_gt_dir)
    merged_gt_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        split_dir = module.GT_ROOT / split
        if not split_dir.exists():
            continue
        for gt_file in split_dir.glob('*.txt'):
            shutil.copy2(gt_file, merged_gt_dir / gt_file.name)

    return module.BASE.evaluate_detections(merged_gt_dir, full_pred_dir, class_aware=True, iou_thr=0.5)


def evaluate_val(module, pred_by_stem: Dict[str, List[object]]) -> Dict[str, float]:
    val_pred_dir = module.PRED_ROOT / 'val_restricted_search'
    if val_pred_dir.exists():
        shutil.rmtree(val_pred_dir)
    module.save_predictions(pred_by_stem, val_pred_dir)
    return module.BASE.evaluate_detections(module.GT_ROOT / 'val', val_pred_dir, class_aware=True, iou_thr=0.5)


def main() -> None:
    module = load_exp14_module()

    summary_path = module.SUMMARY_JSON
    if not summary_path.exists():
        raise RuntimeError(f'Missing summary file: {summary_path}')
    baseline_summary = json.loads(summary_path.read_text(encoding='utf-8'))
    baseline_full = baseline_summary.get('full_metrics', {})
    baseline_acc05 = float(baseline_full.get('acc_05', 0.0))

    split_map = json.loads(module.SPLIT_PLAN_JSON.read_text(encoding='utf-8'))

    # Ensure GT directories are in place.
    module.convert_gt_dirs()

    train_records = module.load_candidate_records('train', split_map['train'])
    val_records = module.load_candidate_records('val', split_map['val'])
    test_records = module.load_candidate_records('test', split_map['test'])
    full_records = train_records + val_records + test_records

    x_train_raw, _ = module.flatten_records(train_records)
    if x_train_raw.numel() == 0:
        raise RuntimeError('No train features loaded for restricted search.')
    mu = x_train_raw.mean(dim=0, keepdim=True)
    sigma = x_train_raw.std(dim=0, keepdim=True).clamp(min=1e-6)

    model = module.HybridExternalCalibrator(input_dim=x_train_raw.shape[1], hidden_dim=64, head_dim=32)
    ckpt = torch.load(module.BEST_CKPT, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    thresholds = [round(0.12 + 0.01 * i, 2) for i in range(39)]
    nms_values = [0.40, 0.50, 0.60, 0.70]

    trials = []
    best = None
    for threshold in thresholds:
        for nms_thr in nms_values:
            pred_full = module.apply_model_to_records(model, full_records, mu, sigma, threshold, nms_thr)
            full_metrics = evaluate_full(module, pred_full)

            pred_val = module.apply_model_to_records(model, val_records, mu, sigma, threshold, nms_thr)
            val_metrics = evaluate_val(module, pred_val)

            row = {
                'threshold': float(threshold),
                'nms': float(nms_thr),
                'full_acc_05': float(full_metrics['acc_05']),
                'full_acc_075': float(full_metrics['acc_075']),
                'full_map_05': float(full_metrics['map_05']),
                'full_prediction_box_count': int(full_metrics['prediction_box_count']),
                'val_acc_05': float(val_metrics['acc_05']),
                'val_map_05': float(val_metrics['map_05']),
            }
            trials.append(row)

            if best is None:
                best = row
            else:
                if row['full_acc_05'] > best['full_acc_05']:
                    best = row
                elif row['full_acc_05'] == best['full_acc_05'] and row['full_map_05'] > best['full_map_05']:
                    best = row

    if best is None:
        raise RuntimeError('No trial results produced.')

    top10 = sorted(trials, key=lambda r: (r['full_acc_05'], r['full_map_05']), reverse=True)[:10]
    delta_acc05 = best['full_acc_05'] - baseline_acc05

    output = {
        'base_summary': str(summary_path),
        'baseline_full_acc_05': baseline_acc05,
        'search_space': {
            'threshold_min': 0.12,
            'threshold_max': 0.50,
            'threshold_step': 0.01,
            'nms_values': nms_values,
            'trial_count': len(trials),
        },
        'best': best,
        'delta_full_acc_05': delta_acc05,
        'top10': top10,
    }
    RESULT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# Exp14 重构版受限阈值/NMS 精细复搜结果',
        '',
        f"- 基线全量 Acc@0.5: {baseline_acc05:.4f}",
        f"- 最优 threshold: {best['threshold']:.2f}",
        f"- 最优 nms: {best['nms']:.2f}",
        f"- 最优全量 Acc@0.5: {best['full_acc_05']:.4f}",
        f"- Acc@0.5 变化: {delta_acc05:+.4f}",
        f"- 最优全量 Acc@0.75: {best['full_acc_075']:.4f}",
        f"- 最优全量 mAP@0.5: {best['full_map_05']:.4f}",
        '',
        '## Top-10（按 full Acc@0.5, full mAP@0.5 排序）',
        '',
        '| rank | threshold | nms | full Acc@0.5 | full Acc@0.75 | full mAP@0.5 | val Acc@0.5 | val mAP@0.5 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for idx, row in enumerate(top10, start=1):
        lines.append(
            f"| {idx} | {row['threshold']:.2f} | {row['nms']:.2f} | {row['full_acc_05']:.4f} | {row['full_acc_075']:.4f} | {row['full_map_05']:.4f} | {row['val_acc_05']:.4f} | {row['val_map_05']:.4f} |"
        )
    RESULT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'[Done] result_json={RESULT_JSON}')
    print(f'[Done] result_md={RESULT_MD}')
    print(
        f"[Best] threshold={best['threshold']:.2f} nms={best['nms']:.2f} "
        f"full_acc05={best['full_acc_05']:.4f} delta={delta_acc05:+.4f}"
    )


if __name__ == '__main__':
    main()
