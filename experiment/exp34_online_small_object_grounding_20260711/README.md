# Exp34 Online Small Object Grounding

## Purpose

将前面离线验证有效的流程在线化，形成一个可对新图片使用的推理框架。

核心目标：

```text
image + text query -> online candidates -> small-object recovery -> query-aware scorer -> final boxes
```

本实验不再读取 Exp18/GroundingDINO/Exp30 的离线预测文件作为输入候选，而是在推理时现场调用 GroundingDINO 生成候选。

## Online Architecture

```text
Input image + text query
        |
        v
GroundingDINO online candidate generation
        |
        +-- base prompts: 10 VisDrone classes, threshold 0.20
        |
        +-- small-object alias prompts, threshold 0.03
              person / people / group of people
              tricycle / covered tricycle / awning tricycle
              motorcycle / motorbike / scooter / bicycle
        |
        v
Candidate alias recovery
        |
        v
Exp30 vocab text scorer
        |
        v
score threshold + class-aware NMS
        |
        v
Final prediction
```

## Components

- Online candidate generator: GroundingDINO Swin-T OGC
- Query-aware reranker: Exp30 checkpoint
- Small-object recovery:
  - low threshold prompts
  - alias mapping
  - small-area prior

## Single Image Usage

```bash
python experiment/exp34_online_small_object_grounding_20260711/scripts/run_exp34_online_small_object_grounding_20260711.py \
  --image dataset/VisDroneSplit1000Guarded/VisDrone2019-DET-test/images/0000000.jpg \
  --query "small motorcycle" \
  --output-json experiment/exp34_online_small_object_grounding_20260711/log/single_prediction.json \
  --output-txt experiment/exp34_online_small_object_grounding_20260711/log/single_prediction.txt
```

## Batch Evaluation

```bash
tmux new -s exp34_online_test
CUDA_VISIBLE_DEVICES=2 python experiment/exp34_online_small_object_grounding_20260711/scripts/run_exp34_online_small_object_grounding_20260711.py \
  --dataset-split test \
  --limit 20
```

Remove `--limit` to run the full split.

## Notes

Exp33 remains a diagnostic/offline validation experiment proving that candidate recall is the small-object bottleneck. Exp34 is the online version intended for generalization to new images.
