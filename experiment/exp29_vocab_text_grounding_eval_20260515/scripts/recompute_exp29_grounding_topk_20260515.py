import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path('/home/wangzhe/DroneP_VG')
EXP29_SCRIPT = ROOT / 'experiment/exp29_vocab_text_grounding_eval_20260515/scripts/run_exp29_vocab_text_grounding_eval_20260515.py'


def load_exp29():
    spec = importlib.util.spec_from_file_location('exp29_recompute', str(EXP29_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import Exp29 from {EXP29_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['exp29_recompute'] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    exp29 = load_exp29()
    vocab = exp29.build_vocab()
    ckpt = torch.load(exp29.BEST_CKPT, map_location='cpu')
    model = exp29.VocabTextScorer(numeric_dim=44, vocab_size=len(vocab))
    model.load_state_dict(ckpt['model'])
    mu = ckpt['mu']
    sigma = ckpt['sigma']

    split_map = exp29.read_split_map()
    val_records = [exp29.load_record('val', stem) for stem in split_map['val']]
    test_records = [exp29.load_record('test', stem) for stem in split_map['test']]
    full_records = [exp29.load_record('train', stem) for stem in split_map['train']] + val_records + test_records

    grounding_val = exp29.evaluate_grounding_topk(model, val_records, mu, sigma, vocab)
    grounding_test = exp29.evaluate_grounding_topk(model, test_records, mu, sigma, vocab)
    grounding_full = exp29.evaluate_grounding_topk(model, full_records, mu, sigma, vocab)

    exp29.GROUNDING_VAL_JSON.write_text(json.dumps(grounding_val, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    exp29.GROUNDING_TEST_JSON.write_text(json.dumps(grounding_test, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    exp29.GROUNDING_FULL_JSON.write_text(json.dumps(grounding_full, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    exp29.write_grounding_summary(exp29.GROUNDING_FULL_MD, grounding_full)

    summary = json.loads(exp29.SUMMARY_JSON.read_text(encoding='utf-8'))
    summary['grounding_val'] = grounding_val
    summary['grounding_test'] = grounding_test
    summary['grounding_full'] = grounding_full
    summary['grounding_recomputed_after_gt_reader_fix'] = True
    exp29.SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(exp29.GROUNDING_FULL_MD.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
