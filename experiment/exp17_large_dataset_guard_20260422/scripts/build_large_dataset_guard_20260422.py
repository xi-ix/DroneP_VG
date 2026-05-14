import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP_ROOT = ROOT / 'experiment/exp17_large_dataset_guard_20260422'
LOG_DIR = EXP_ROOT / 'log'

SMALL_SOURCE = ROOT / 'dataset' / 'visdrone_test_100'
SMALL_MANIFEST = SMALL_SOURCE / 'manifest.tsv'
FULL_ROOT = ROOT / 'dataset/RefDrone'

EXP14_SPLIT_PLAN = ROOT / 'experiment/exp14_full_rebuild_20260421/log/split_plan.json'
EXP14_SUMMARY = ROOT / 'experiment/exp14_full_rebuild_20260421/log/exp14_full_rebuild_20260421_summary.json'

TARGET_TOTAL = 1000
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DATA_ROOT = ROOT / 'dataset' / 'VisDroneSplit1000Guarded'
RUN_LOG = LOG_DIR / 'run_log.txt'
SPLIT_PLAN_JSON = LOG_DIR / 'split_plan.json'
SPLIT_MANIFEST_TSV = LOG_DIR / 'split_manifest.tsv'
DATASET_STATS_JSON = LOG_DIR / 'dataset_stats.json'
REGRESSION_GUARD_JSON = LOG_DIR / 'regression_guard.json'


def log(message: str) -> None:
    print(message, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as file:
        file.write(message + '\n')


def read_small_stems() -> List[str]:
    stems: List[str] = []
    with SMALL_MANIFEST.open('r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if row:
                stems.append(row[0].strip())
    return sorted(stems)


def read_locked_split_from_exp14(small_stems: List[str]) -> Dict[str, List[str]]:
    if EXP14_SPLIT_PLAN.exists():
        split_map = json.loads(EXP14_SPLIT_PLAN.read_text(encoding='utf-8'))
        return {
            'train': sorted(split_map.get('train', [])),
            'val': sorted(split_map.get('val', [])),
            'test': sorted(split_map.get('test', [])),
        }

    shuffled = small_stems.copy()
    rng = random.Random(SEED)
    rng.shuffle(shuffled)
    return {
        'train': sorted(shuffled[:70]),
        'val': sorted(shuffled[70:85]),
        'test': sorted(shuffled[85:]),
    }


def collect_full_source_index() -> Dict[str, Tuple[Path, Path, str]]:
    index: Dict[str, Tuple[Path, Path, str]] = {}
    ordered_splits = ['train', 'val', 'test']
    for split in ordered_splits:
        image_dir = FULL_ROOT / f'VisDrone2019-DET-{split}' / 'images'
        ann_dir = FULL_ROOT / f'VisDrone2019-DET-{split}' / 'annotations'
        if not image_dir.exists() or not ann_dir.exists():
            continue
        for image_path in sorted(image_dir.glob('*.jpg')):
            stem = image_path.stem
            ann_path = ann_dir / f'{stem}.txt'
            if not ann_path.exists():
                continue
            if stem not in index:
                index[stem] = (image_path, ann_path, split)
    return index


def build_split_map(
    locked_split: Dict[str, List[str]],
    full_index: Dict[str, Tuple[Path, Path, str]],
    small_set: set,
) -> Dict[str, List[str]]:
    train_target = int(round(TARGET_TOTAL * TRAIN_RATIO))
    val_target = int(round(TARGET_TOTAL * VAL_RATIO))
    test_target = TARGET_TOTAL - train_target - val_target

    train_stems = list(locked_split['train'])
    val_stems = list(locked_split['val'])
    test_stems = list(locked_split['test'])

    candidate_stems = [stem for stem in full_index.keys() if stem not in small_set]
    rng = random.Random(SEED)
    rng.shuffle(candidate_stems)

    need_val = max(0, val_target - len(val_stems))
    need_test = max(0, test_target - len(test_stems))
    need_train = max(0, train_target - len(train_stems))
    total_needed = need_val + need_test + need_train

    if len(candidate_stems) < total_needed:
        raise RuntimeError(
            f'Not enough stems in full source: need={total_needed}, available={len(candidate_stems)}'
        )

    cursor = 0
    val_stems.extend(candidate_stems[cursor:cursor + need_val])
    cursor += need_val
    test_stems.extend(candidate_stems[cursor:cursor + need_test])
    cursor += need_test
    train_stems.extend(candidate_stems[cursor:cursor + need_train])

    return {
        'train': sorted(train_stems),
        'val': sorted(val_stems),
        'test': sorted(test_stems),
    }


def source_paths_for_stem(stem: str, full_index: Dict[str, Tuple[Path, Path, str]]) -> Tuple[Path, Path, str]:
    small_img = SMALL_SOURCE / 'images' / f'{stem}.jpg'
    small_ann = SMALL_SOURCE / 'annotations' / f'{stem}.txt'
    if small_img.exists() and small_ann.exists():
        return small_img, small_ann, 'visdrone_test_100'

    if stem in full_index:
        return full_index[stem]

    raise FileNotFoundError(f'Source files missing for stem={stem}')


def write_split_dataset(split_map: Dict[str, List[str]], full_index: Dict[str, Tuple[Path, Path, str]]) -> None:
    split_dirs = {
        'train': DATA_ROOT / 'VisDrone2019-DET-train',
        'val': DATA_ROOT / 'VisDrone2019-DET-val',
        'test': DATA_ROOT / 'VisDrone2019-DET-test',
    }
    for split_dir in split_dirs.values():
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'annotations').mkdir(parents=True, exist_ok=True)

    SPLIT_PLAN_JSON.write_text(json.dumps(split_map, ensure_ascii=False, indent=2), encoding='utf-8')
    with SPLIT_MANIFEST_TSV.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['split', 'stem', 'image_path', 'annotation_path', 'source_split'])
        for split, stems in split_map.items():
            for stem in stems:
                image_src, ann_src, source_split = source_paths_for_stem(stem, full_index)
                image_dst = split_dirs[split] / 'images' / f'{stem}.jpg'
                ann_dst = split_dirs[split] / 'annotations' / f'{stem}.txt'

                if image_dst.exists() or image_dst.is_symlink():
                    image_dst.unlink()
                if ann_dst.exists() or ann_dst.is_symlink():
                    ann_dst.unlink()

                shutil.copy2(image_src, image_dst)
                shutil.copy2(ann_src, ann_dst)
                writer.writerow([split, stem, str(image_dst), str(ann_dst), source_split])


def write_guard_artifacts(split_map: Dict[str, List[str]], locked_split: Dict[str, List[str]], full_index: Dict[str, Tuple[Path, Path, str]]) -> None:
    baseline_summary = {}
    if EXP14_SUMMARY.exists():
        baseline_summary = json.loads(EXP14_SUMMARY.read_text(encoding='utf-8'))

    stats = {
        'seed': SEED,
        'target_total': TARGET_TOTAL,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'counts': {k: len(v) for k, v in split_map.items()},
        'full_source_unique_stems': len(full_index),
        'contains_locked_train': all(s in split_map['train'] for s in locked_split['train']),
        'contains_locked_val': all(s in split_map['val'] for s in locked_split['val']),
        'contains_locked_test': all(s in split_map['test'] for s in locked_split['test']),
    }
    DATASET_STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')

    guard = {
        'frozen_eval_subset': {
            'val_stems': sorted(locked_split['val']),
            'test_stems': sorted(locked_split['test']),
        },
        'baseline_reference_file': str(EXP14_SUMMARY),
        'baseline_reference_metrics': {
            'test_metrics': baseline_summary.get('test_metrics', {}),
            'full_metrics': baseline_summary.get('full_metrics', {}),
            'best_threshold': baseline_summary.get('best_threshold'),
            'best_nms': baseline_summary.get('best_nms'),
        },
        'rule': 'New experiments on expanded data must also report metrics on frozen_eval_subset to ensure no regression against previous best.',
    }
    REGRESSION_GUARD_JSON.write_text(json.dumps(guard, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    random.seed(SEED)

    small_stems = read_small_stems()
    small_set = set(small_stems)
    locked_split = read_locked_split_from_exp14(small_stems)
    full_index = collect_full_source_index()

    log(f'[Config] target_total={TARGET_TOTAL} seed={SEED} split={int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}')
    log(f'[Config] small_source_count={len(small_stems)} full_source_unique={len(full_index)}')

    split_map = build_split_map(locked_split, full_index, small_set)
    write_split_dataset(split_map, full_index)
    write_guard_artifacts(split_map, locked_split, full_index)

    log(f"[Done] train={len(split_map['train'])} val={len(split_map['val'])} test={len(split_map['test'])}")
    log(f'[Done] split_plan={SPLIT_PLAN_JSON}')
    log(f'[Done] split_manifest={SPLIT_MANIFEST_TSV}')
    log(f'[Done] dataset_stats={DATASET_STATS_JSON}')
    log(f'[Done] regression_guard={REGRESSION_GUARD_JSON}')


if __name__ == '__main__':
    main()
