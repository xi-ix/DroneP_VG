# Exp35 Online Candidate Compression

## Purpose

Exp34 已经完成在线化，并显著提升了小目标召回，但预测框数量过多：

```text
Exp34 test prediction boxes: 54019
Exp34 test Acc@0.5: 0.5383
Exp34 test mAP@0.5: 0.1461
```

Exp35 的目标是在不回退到离线候选文件的前提下，压缩在线候选池，降低假阳性，同时尽量保留 `motor / tricycle / awning tricycle / people` 的小目标召回。

## Method

在线流程仍然是：

```text
image + text query
  -> GroundingDINO online candidates
  -> alias small-object candidates
  -> Exp30 query-aware scorer
  -> class-aware NMS
  -> final boxes
```

相对 Exp34，新增两处约束：

```text
1. Score-after-recall compression
   - keep Exp34 online candidate recall path
   - run the Exp30 query-aware scorer first
   - apply per-class final TopK after scoring

2. Stronger final NMS
   - NMS threshold changed from 0.70 to 0.65
```

## Why This Is Still Online

Exp35 does not read Exp30/Exp33/Exp34 prediction files as inference input. The final deployable path still generates candidates from the input image at runtime, scores candidates online, then applies deterministic final-output compression.

Existing Exp34 predictions were only used for quick diagnostic threshold analysis.

## Run

```bash
tmux new -s exp35_online_test
CUDA_VISIBLE_DEVICES=2 python experiment/exp35_online_candidate_compression_20260711/scripts/run_exp35_online_candidate_compression_20260711.py \
  --dataset-split test
```

For quick smoke tests:

```bash
CUDA_VISIBLE_DEVICES=2 python experiment/exp35_online_candidate_compression_20260711/scripts/run_exp35_online_candidate_compression_20260711.py \
  --dataset-split test \
  --limit 20
```
