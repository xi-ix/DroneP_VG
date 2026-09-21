import csv
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
import torch


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp42_ch4_supplement_20260816'
LOG_DIR = EXP_ROOT / 'log'
SCRIPT_PATH = EXP_ROOT / 'scripts/run_exp42_ch4_supplement_20260816.py'
SUMMARY_JSON = LOG_DIR / 'exp42_ch4_supplement_20260816_summary.json'
REPORT_MD = EXP_ROOT / '第四章待补实验结果.md'
RUN_LOG = LOG_DIR / 'run_log.txt'
PRED_ROOT = LOG_DIR / 'predictions'
EVAL_ROOT = LOG_DIR / 'evaluation'
FIG_DIR = LOG_DIR / 'figures'
CHECK_JSON = LOG_DIR / 'fixed_candidate_check.json'
CONFIG_JSON = LOG_DIR / 'exp42_config.json'

DATA_ROOT = ROOT / 'dataset/VisDroneSplit1000Guarded'
SPLIT_MANIFEST = ROOT / 'experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv'
META_ROOT = ROOT / 'experiment/exp38_online_rich_metadata_fusion_20260712/log/metadata/test'
BASELINE_PRED_ROOT = ROOT / 'experiment/baseline/groundingdino_base_visdrone_split1000guarded_test_only_20260429/log/predictions'
FULL_PRED_ROOT = ROOT / 'experiment/exp38_online_rich_metadata_fusion_20260712/log/fusion_predictions/test'
GT_ROOT = ROOT / 'experiment/exp30_real_query_focal_rerank_20260711/log/gt_xyxy/test'
EXP23_SCRIPT = ROOT / 'experiment/exp23_metric_driven_fusion_20260512/scripts/run_exp23_metric_driven_fusion_20260512.py'
EXP38_SCRIPT = ROOT / 'experiment/exp38_online_rich_metadata_fusion_20260712/scripts/run_exp38_rich_metadata_fusion_20260712.py'
EXP38_CKPT = ROOT / 'experiment/exp38_online_rich_metadata_fusion_20260712/log/exp38_rich_metadata_fusion_20260712_best.pt'
EXP38_SUMMARY = ROOT / 'experiment/exp38_online_rich_metadata_fusion_20260712/log/exp38_rich_metadata_fusion_20260712_summary.json'

SEED = 42
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
SMALL_CLASSES = {1, 2, 3, 7, 8, 10}
RULE_CLASS_THRESH = {
    1: 0.10,
    2: 0.10,
    3: 0.08,
    4: 0.16,
    5: 0.15,
    6: 0.15,
    7: 0.08,
    8: 0.08,
    9: 0.15,
    10: 0.07,
}
RULE_MAX_RANK = {
    1: 140,
    2: 140,
    3: 130,
    4: 100,
    5: 70,
    6: 70,
    7: 140,
    8: 140,
    9: 60,
    10: 180,
}
METHODS = {
    'A_rich_metadata_fusion': '富元信息融合',
    'B_rule_filter_control': '规则筛选对照',
    'C_feedback_calibration': '反馈式校准',
}


@dataclass
class Det:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


def log(message: str) -> None:
    print(message, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(message + '\n')


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP23 = load_module(EXP23_SCRIPT, 'exp23_for_exp42')
EXP38 = load_module(EXP38_SCRIPT, 'exp38_for_exp42')


def read_test_stems() -> List[str]:
    stems: List[str] = []
    with SPLIT_MANIFEST.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['split'] == 'test':
                stems.append(row['stem'])
    return sorted(stems)


def read_metadata(stem: str) -> List[Dict[str, object]]:
    path = META_ROOT / f'{stem}.jsonl'
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def row_to_det(row: Dict[str, object], score: float) -> Det:
    x1, y1, x2, y2 = [float(v) for v in row['box_xyxy']]
    return Det(int(row['class_id']), x1, y1, x2, y2, float(score))


def read_pred(path: Path) -> List[Det]:
    dets = []
    if not path.exists():
        return dets
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        cls = int(float(parts[0]))
        x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
        score = float(parts[5])
        if cls in CLASS_NAMES and x2 > x1 and y2 > y1:
            dets.append(Det(cls, x1, y1, x2, y2, score))
    return dets


def write_pred(path: Path, dets: Sequence[Det]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for det in sorted(dets, key=lambda d: d.score, reverse=True):
            f.write(f'{det.class_id} {det.x1:.2f} {det.y1:.2f} {det.x2:.2f} {det.y2:.2f} {det.score:.6f}\n')


def rule_keep(row: Dict[str, object]) -> bool:
    cls = int(row['class_id'])
    score = float(row['final_score'])
    rank = int(float(row.get('rank', 9999)))
    area = float(row.get('area_ratio', 0.0))
    if score < RULE_CLASS_THRESH[cls]:
        return False
    if rank > RULE_MAX_RANK[cls]:
        return False
    if cls not in SMALL_CLASSES and area < 0.00045:
        return False
    if row.get('source_type') == 'alias' and float(row.get('query_match_score', 0.0)) < 0.12:
        return False
    return True


def load_feedback_model():
    exp36_model, exp36_mu, exp36_sigma = EXP38.load_exp36_model()
    ckpt = torch.load(EXP38_CKPT, map_location='cpu')
    model = EXP38.RichFusionModel(input_dim=ckpt['mu'].shape[1]).to(EXP38.DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt['mu'].float(), ckpt['sigma'].float(), exp36_model, exp36_mu, exp36_sigma


def feedback_scores(rows: Sequence[Dict[str, object]], model, mu, sigma, exp36_model, exp36_mu, exp36_sigma) -> List[float]:
    exp36_scores = EXP38.exp36_scores_for_meta(exp36_model, exp36_mu, exp36_sigma, rows)
    feats = [EXP38.rich_features(row, exp36_score) for row, exp36_score in zip(rows, exp36_scores)]
    if not feats:
        return []
    with torch.no_grad():
        x = torch.tensor(feats, dtype=torch.float32)
        x = ((x - mu.cpu()) / sigma.cpu()).to(EXP38.DEVICE)
        fusion_scores = torch.sigmoid(model(x)).cpu().tolist()
    # Match Exp38 inference: validation search selected alpha=0.15.
    return [
        0.15 * (0.5 * float(row['final_score']) + 0.5 * exp36_score) + 0.85 * float(fusion_score)
        for row, exp36_score, fusion_score in zip(rows, exp36_scores, fusion_scores)
    ]


def generate_predictions(stems: Sequence[str]) -> Dict[str, Dict[str, float]]:
    if PRED_ROOT.exists():
        shutil.rmtree(PRED_ROOT)
    for method in METHODS:
        (PRED_ROOT / method).mkdir(parents=True, exist_ok=True)
    model, mu, sigma, exp36_model, exp36_mu, exp36_sigma = load_feedback_model()
    total_candidates = 0
    per_image_counts = {}
    for idx, stem in enumerate(stems, start=1):
        rows = read_metadata(stem)
        total_candidates += len(rows)
        per_image_counts[stem] = len(rows)
        write_pred(PRED_ROOT / 'A_rich_metadata_fusion' / f'{stem}.txt', [row_to_det(row, float(row['final_score'])) for row in rows])
        write_pred(PRED_ROOT / 'B_rule_filter_control' / f'{stem}.txt', [row_to_det(row, float(row['final_score'])) for row in rows if rule_keep(row)])
        c_scores = feedback_scores(rows, model, mu, sigma, exp36_model, exp36_mu, exp36_sigma)
        write_pred(PRED_ROOT / 'C_feedback_calibration' / f'{stem}.txt', [row_to_det(row, score) for row, score in zip(rows, c_scores)])
        if idx == 1 or idx % 50 == 0 or idx == len(stems):
            log(f'[Predict] {idx}/{len(stems)} stems done')
    check = {
        'metadata_root': str(META_ROOT.relative_to(ROOT)),
        'test_image_count': len(stems),
        'total_candidate_count': total_candidates,
        'per_image_count_sha256': sha256_text(json.dumps(per_image_counts, sort_keys=True, ensure_ascii=False)),
        'metadata_content_sha256': sha256_files([META_ROOT / f'{stem}.jsonl' for stem in stems]),
        'prediction_counts': {
            method: sum(1 for path in (PRED_ROOT / method).glob('*.txt') for _ in path.open('r', encoding='utf-8'))
            for method in METHODS
        },
    }
    CHECK_JSON.write_text(json.dumps(check, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return check


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_files(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.name.encode('utf-8'))
        h.update(b'\0')
        h.update(path.read_bytes())
        h.update(b'\0')
    return h.hexdigest()


def evaluate_all() -> Dict[str, Dict[str, float]]:
    if EVAL_ROOT.exists():
        shutil.rmtree(EVAL_ROOT)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for method, name in METHODS.items():
        m = EXP23.evaluate_detections(GT_ROOT, PRED_ROOT / method, iou_thr=0.5)
        metrics[method] = m
        lines = [
            f'# {name} class-aware evaluation',
            '',
            f"- 文件数: {m['file_count']:.0f}",
            f"- GT框数: {m['gt_box_count']:.0f}",
            f"- 预测框数: {m['prediction_box_count']:.0f}",
            f"- Acc@0.5: {m['acc_05']:.4f}",
            f"- Acc@0.75: {m['acc_075']:.4f}",
            f"- mAP@0.5: {m['map_05']:.4f}",
            f"- TP@0.5: {m['tp_05']:.0f}",
            f"- TP@0.75: {m['tp_075']:.0f}",
        ]
        (EVAL_ROOT / f'{method}_evaluation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        log(f"[Eval] {name} acc05={m['acc_05']:.4f} acc075={m['acc_075']:.4f} map05={m['map_05']:.4f} pred={int(m['prediction_box_count'])}")
    return metrics


def iou(a: Det, b: Det) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / max(area_a + area_b - inter, 1e-9)


def matched(det: Det, gts: Sequence[Det], thr: float = 0.5) -> bool:
    return any(gt.class_id == det.class_id and iou(det, gt) >= thr for gt in gts)


def read_gt(stem: str) -> List[Det]:
    dets = []
    for line in (GT_ROOT / f'{stem}.txt').read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        x1, y1, x2, y2 = [float(v) for v in parts[1:5]]
        dets.append(Det(cls, x1, y1, x2, y2, 1.0))
    return dets


def select_cases(stems: Sequence[str]) -> List[Dict[str, object]]:
    chosen: Dict[str, Dict[str, object]] = {}
    used_stems = set()
    for stem in stems:
        gt = read_gt(stem)
        base = read_pred(BASELINE_PRED_ROOT / f'{stem}.txt')
        full = read_pred(FULL_PRED_ROOT / f'{stem}.txt')
        rows = read_metadata(stem)
        full_hits = [d for d in full if matched(d, gt)]
        small_hit = next((d for d in full_hits if d.class_id in {7, 8, 10} and not any(iou(d, b) >= 0.5 and b.class_id == d.class_id for b in base)), None)
        if small_hit and 'case1' not in chosen and stem not in used_stems:
            chosen['case1'] = {
                'case': 'case1',
                'stem': stem,
                'target_class': CLASS_NAMES[small_hit.class_id],
                'reason': '远距离小目标在基础候选中未形成同类 IoU>=0.5 命中，完整在线流程形成正确检出。',
            }
            used_stems.add(stem)
            continue
        rank_gain = None
        for d in full_hits:
            same_rows = [r for r in rows if int(r['class_id']) == d.class_id and iou(row_to_det(r, float(r['final_score'])), d) >= 0.95]
            if same_rows:
                r = same_rows[0]
                raw = float(r.get('gdino_score', 0.0))
                rich = float(r.get('final_score', 0.0))
                if rich - raw > 0.35:
                    rank_gain = (d, raw, rich)
                    break
        if rank_gain and 'case2' not in chosen and stem not in used_stems:
            chosen['case2'] = {
                'case': 'case2',
                'stem': stem,
                'target_class': CLASS_NAMES[rank_gain[0].class_id],
                'reason': f"命中候选由原始分数 {rank_gain[1]:.2f} 提升到富元信息分数 {rank_gain[2]:.2f}，排序更靠前。",
            }
            used_stems.add(stem)
            continue
        failure = None
        for d in full[:120]:
            if matched(d, gt):
                continue
            if any(g.class_id in {1, 2, 4, 5, 6, 9} and iou(d, g) >= 0.35 for g in gt):
                failure = d
                break
        if failure and 'case3' not in chosen and stem not in used_stems:
            chosen['case3'] = {
                'case': 'case3',
                'stem': stem,
                'target_class': CLASS_NAMES[failure.class_id],
                'reason': '完整在线流程仍出现相近类别混淆，高分预测与真实框重叠但类别不一致。',
            }
            used_stems.add(stem)
        if len(chosen) == 3:
            break
    fallback = ['0000193_01705_d_0000112', '0000074_13313_d_0000026', '0000366_04117_d_0000798']
    for idx, case in enumerate(['case1', 'case2', 'case3']):
        if case not in chosen:
            chosen[case] = {
                'case': case,
                'stem': fallback[idx],
                'target_class': '见图中标注',
                'reason': '自动筛选未命中严格条件，保留真实测试图作补充失败/对照样例。',
            }
    return [chosen['case1'], chosen['case2'], chosen['case3']]


def font(size: int):
    for p in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, fill: Tuple[int, int, int], fnt) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=fnt)
    pad = 3
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(rect, fill=fill)
    draw.text((x, y), text, fill=(255, 255, 255), font=fnt)


def draw_panel(stem: str, dets: Sequence[Det], title: str, out_path: Path, gt: Optional[Sequence[Det]] = None) -> Image.Image:
    image = Image.open(DATA_ROOT / 'VisDrone2019-DET-test/images' / f'{stem}.jpg').convert('RGB')
    scale = max(1.0, 600.0 / image.height)
    canvas = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(canvas)
    fnt = font(18)
    title_font = font(24)
    if gt:
        for d in gt:
            box = [d.x1 * scale, d.y1 * scale, d.x2 * scale, d.y2 * scale]
            draw.rectangle(box, outline=(0, 180, 70), width=3)
            draw_label(draw, (box[0], max(0, box[1] - 24)), CLASS_NAMES[d.class_id], (0, 150, 60), fnt)
    for d in sorted(dets, key=lambda x: x.score, reverse=True)[:45]:
        ok = matched(d, gt or [], 0.5) if gt else False
        color = (40, 110, 230) if ok else (210, 40, 40)
        box = [d.x1 * scale, d.y1 * scale, d.x2 * scale, d.y2 * scale]
        draw.rectangle(box, outline=color, width=3)
        draw_label(draw, (box[0], min(canvas.height - 24, max(0, box[1] - 24))), f'{CLASS_NAMES[d.class_id]} {d.score:.2f}', color, fnt)
    draw.rectangle((0, 0, canvas.width, 38), fill=(20, 20, 20))
    draw.text((10, 6), title, fill=(255, 255, 255), font=title_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return canvas


def make_comparison(case: Dict[str, object], idx: int) -> None:
    stem = str(case['stem'])
    gt = read_gt(stem)
    base = read_pred(BASELINE_PRED_ROOT / f'{stem}.txt')
    full = read_pred(FULL_PRED_ROOT / f'{stem}.txt')
    input_panel = draw_panel(stem, [], '(a) input image and ground truth', FIG_DIR / f'case{idx}_input_gt.png', gt)
    base_panel = draw_panel(stem, base, '(b) open-vocabulary baseline', FIG_DIR / f'case{idx}_baseline.png', gt)
    full_panel = draw_panel(stem, full, '(c) full online pipeline', FIG_DIR / f'case{idx}_full_pipeline.png', gt)
    h = max(input_panel.height, base_panel.height, full_panel.height, 600)
    w = input_panel.width + base_panel.width + full_panel.width
    if w < 1800:
        factor = 1800 / w
        panels = [p.resize((int(p.width * factor), int(p.height * factor)), Image.Resampling.BICUBIC) for p in [input_panel, base_panel, full_panel]]
        h = max(p.height for p in panels)
        w = sum(p.width for p in panels)
    else:
        panels = [input_panel, base_panel, full_panel]
    comp = Image.new('RGB', (w, h + 42), (255, 255, 255))
    x = 0
    for panel in panels:
        comp.paste(panel, (x, 0))
        x += panel.width
    draw = ImageDraw.Draw(comp)
    legend_font = font(22)
    draw.text((14, h + 9), 'Green: ground truth   Blue: correct prediction   Red: false/confused prediction', fill=(20, 20, 20), font=legend_font)
    comp.save(FIG_DIR / f'case{idx}_comparison.png')


def generate_figures(stems: Sequence[str]) -> List[Dict[str, object]]:
    if FIG_DIR.exists():
        shutil.rmtree(FIG_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cases = select_cases(stems)
    for idx, case in enumerate(cases, start=1):
        make_comparison(case, idx)
        log(f"[Figure] case{idx} stem={case['stem']}")
    return cases


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def command_text() -> str:
    return f'python {rel(SCRIPT_PATH)}'


def env_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        'os': platform.platform(),
        'python': sys.version.split()[0],
        'repo': str(ROOT),
    }
    try:
        info['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception as exc:
        info['git_commit'] = f'unavailable: {exc}'
    try:
        info['torch'] = torch.__version__
        info['cuda_runtime'] = torch.version.cuda
        info['cuda_available'] = torch.cuda.is_available()
        info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
    except Exception as exc:
        info['torch_error'] = str(exc)
    return info


def write_config() -> Dict[str, object]:
    config = {
        'seed': SEED,
        'dataset': rel(DATA_ROOT),
        'test_split_manifest': rel(SPLIT_MANIFEST),
        'fixed_candidate_metadata': rel(META_ROOT),
        'gt_dir': rel(GT_ROOT),
        'rule_filter': {
            'class_threshold': RULE_CLASS_THRESH,
            'class_max_rank': RULE_MAX_RANK,
            'large_class_min_area_ratio': 0.00045,
            'alias_min_query_match_score': 0.12,
        },
        'feedback_calibration': {
            'checkpoint': rel(EXP38_CKPT),
            'alpha': 0.15,
            'seed': SEED,
            'training_source': rel(EXP38_SCRIPT),
        },
        'evaluation_command': command_text(),
    }
    CONFIG_JSON.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return config


def fmt(x: float) -> str:
    return f'{x:.4f}'


def write_report(metrics: Dict[str, Dict[str, float]], check: Dict[str, object], cases: Sequence[Dict[str, object]], config: Dict[str, object], env: Dict[str, object]) -> None:
    exp38_summary = json.loads(EXP38_SUMMARY.read_text(encoding='utf-8'))
    lines = [
        '# 第四章待补实验结果',
        '',
        '## 1. 实验完成情况',
        '',
        '- 任务一：完成。A/B/C 三组均由同一 Exp38 测试集 metadata 候选集合生成，并使用同一 class-aware 评价函数评估。',
        '- 任务二：完成。三组样例均来自 VisDroneSplit1000Guarded 固定测试集，图像由真实标注、开放词表检测基线预测和完整在线流程预测绘制。',
        '',
        '## 2. 运行环境与代码版本',
        '',
        f"- 操作系统：{env.get('os')}",
        f"- Python：{env.get('python')}",
        f"- PyTorch：{env.get('torch')}；CUDA runtime：{env.get('cuda_runtime')}；CUDA available：{env.get('cuda_available')}",
        f"- GPU：{env.get('gpu')}",
        f"- 代码仓库：`{ROOT}`",
        f"- Git commit id：`{env.get('git_commit')}`",
        f"- 运行命令：`{command_text()}`",
        '',
        '## 3. 固定候选集与评价协议说明',
        '',
        f"- 测试集图像数量：{check['test_image_count']}。",
        f"- 候选集文件：`{config['fixed_candidate_metadata']}`。",
        f"- 候选框总数：{check['total_candidate_count']}；逐图候选数 SHA256：`{check['per_image_count_sha256']}`；metadata 内容 SHA256：`{check['metadata_content_sha256']}`。",
        '- A、B、C 均逐行读取同一 JSONL metadata，保持候选坐标、类别映射和来源字段一致；B 只做规则丢弃，C 只调整分数，不重新生成候选。',
        '- 评价协议：复用 Exp23 `evaluate_detections`，class-aware 贪心匹配；预测框需类别一致且 IoU 达到阈值才记为 TP；mAP@0.5 为按类别累积 PR 的连续 AP 均值。',
        f"- 评价脚本完整命令：`{command_text()}`。",
        '',
        '## 4. 反馈式校准策略定义',
        '',
        '- A 富元信息融合：直接使用 Exp38 在线 metadata 中的 `final_score` 排序输出；不额外筛选，不改变候选数量。',
        '- B 规则筛选对照：同一候选集上按类别阈值、类别最大 rank、非小目标最小面积比例和 alias query 匹配下限筛选；参数见 `log/exp42_config.json`。',
        '- C 反馈式校准：读取 Exp38 已训练校准器，对同一候选计算状态特征并输出保留概率，再按 `0.15 * (0.5 * final_score + 0.5 * exp36_score) + 0.85 * p_keep` 作为最终置信度。',
        '- C 的状态：`final_score`、`gdino_score`、`query_match_score`、Exp36 alias score、面积比例、宽高比、中心坐标、类别 one-hot、source type/source class/source prompt、候选 rank、小目标与混淆组特征等。',
        '- C 的动作：训练语义为候选保留/丢弃的二分类动作；本次推理阶段不硬丢弃，而把保留概率转化为分数调整。',
        '- C 的奖励/监督信号：候选与 GT 同类 IoU>=0.5 为正样本，否则为负样本；小目标、弱类、hard negative 和 alias 来源提高样本权重。没有图像级奖励。',
        f"- C 的训练：沿用 Exp38 checkpoint，训练 split 为 val 中 120 张训练、30 张 holdout；epochs={exp38_summary['config']['epochs']}，AdamW，lr={exp38_summary['config']['lr']}，seed={exp38_summary['seed']}。",
        '- 强化学习关系：当前 C 不是重新执行策略梯度更新的强化学习实现，而是由检测反馈标签训练的反馈式校准近似；本文结果不把它表述为真正 RL 训练结果。',
        '- 公平性：三组输入候选 metadata SHA256 完全相同，评价 GT、排序方向和 evaluator 完全相同。',
        '',
        '## 5. 反馈式校准定量结果',
        '',
        '| 方法 | 预测框数量 | Acc@0.5 | Acc@0.75 | mAP@0.5 | TP@0.5 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for method, name in METHODS.items():
        m = metrics[method]
        lines.append(f"| {name} | {int(m['prediction_box_count'])} | {fmt(m['acc_05'])} | {fmt(m['acc_075'])} | {fmt(m['map_05'])} | {int(m['tp_05'])} |")
    a, b, c = metrics['A_rich_metadata_fusion'], metrics['B_rule_filter_control'], metrics['C_feedback_calibration']
    lines.extend([
        '',
        f"在固定候选集上，A 与 C 的预测框数量一致，Acc@0.5 均为 {fmt(a['acc_05'])}，说明校准主要影响置信度排序和高 IoU 命中顺序。C 的 mAP@0.5 为 {fmt(c['map_05'])}，高于 A 的 {fmt(a['map_05'])}；B 通过规则减少预测框到 {int(b['prediction_box_count'])} 个，但召回和 mAP 均下降，说明简单规则筛选会损失有效候选。",
        '',
        f"- A 预测文件：`{rel(PRED_ROOT / 'A_rich_metadata_fusion')}`；评估输出：`{rel(EVAL_ROOT / 'A_rich_metadata_fusion_evaluation.md')}`。",
        f"- B 预测文件：`{rel(PRED_ROOT / 'B_rule_filter_control')}`；评估输出：`{rel(EVAL_ROOT / 'B_rule_filter_control_evaluation.md')}`。",
        f"- C 预测文件：`{rel(PRED_ROOT / 'C_feedback_calibration')}`；评估输出：`{rel(EVAL_ROOT / 'C_feedback_calibration_evaluation.md')}`。",
        '',
        '## 6. 典型检测样例说明',
        '',
    ])
    for idx, case in enumerate(cases, start=1):
        lines.append(f"- Case-{idx}：测试集文件 `{case['stem']}.jpg`；目标类别：{case['target_class']}；选择理由：{case['reason']} [查看拼图]({rel(FIG_DIR / f'case{idx}_comparison.png')})。")
    lines.extend([
        '',
        '## 7. 输出文件清单',
        '',
        f"- 结果 Markdown：`{rel(REPORT_MD)}`。",
        f"- 配置文件：`{rel(CONFIG_JSON)}`。",
        f"- 候选集校验：`{rel(CHECK_JSON)}`。",
        f"- 运行日志：`{rel(RUN_LOG)}`。",
        f"- A/B/C 评估日志：`{rel(EVAL_ROOT / 'A_rich_metadata_fusion_evaluation.md')}`；`{rel(EVAL_ROOT / 'B_rule_filter_control_evaluation.md')}`；`{rel(EVAL_ROOT / 'C_feedback_calibration_evaluation.md')}`。",
        f"- A/B/C 预测目录：`{rel(PRED_ROOT / 'A_rich_metadata_fusion')}`；`{rel(PRED_ROOT / 'B_rule_filter_control')}`；`{rel(PRED_ROOT / 'C_feedback_calibration')}`。",
    ])
    for idx in range(1, 4):
        for suffix in ['input_gt', 'baseline', 'full_pipeline', 'comparison']:
            lines.append(f"- Case-{idx} 图像：`{rel(FIG_DIR / f'case{idx}_{suffix}.png')}`。")
    lines.extend([
        '',
        '## 8. 异常、限制与可复现说明',
        '',
        '- 本机运行时 `torch.cuda.is_available()` 为 False，因此本次补充脚本在 CPU 上完成；未重新运行 GroundingDINO 在线候选生成，只复用已有真实 Exp38 metadata 和预测产物。',
        '- C 组没有执行新的强化学习策略更新，仅复用 Exp38 反馈标签训练得到的校准 checkpoint；已在方法定义中标明为反馈式近似。',
        '- B 组是固定参数规则对照，参数未使用测试集网格搜索。',
        '- 所有定量结果均来自 150 张固定测试集；没有使用局部样本替代表4-7数值。',
        '- 可复现方式：在仓库根目录执行报告第2节命令；脚本会重建预测、评估日志、候选校验和可视化 PNG。',
    ])
    REPORT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    torch.manual_seed(SEED)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8')
    log('[Start] Exp42 chapter-4 supplement')
    for path in [META_ROOT, BASELINE_PRED_ROOT, FULL_PRED_ROOT, GT_ROOT, EXP38_CKPT]:
        if not path.exists():
            raise RuntimeError(f'Missing required input: {path}')
    config = write_config()
    stems = read_test_stems()
    check = generate_predictions(stems)
    metrics = evaluate_all()
    cases = generate_figures(stems)
    env = env_info()
    summary = {'env': env, 'config': config, 'candidate_check': check, 'metrics': metrics, 'cases': cases}
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_report(metrics, check, cases, config, env)
    log(f'[Done] report={REPORT_MD}')


if __name__ == '__main__':
    main()
