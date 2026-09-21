# Exp36 Small Target Alias Rerank

## Purpose

Exp34/Exp35 已经证明在线候选召回可以明显提升小目标召回，但 `motor / awning tricycle / tricycle / people` 的 TP 与 FP 分数分布非常接近，单纯压预测框或调阈值很难继续提升。

Exp36 聚焦两个方向：

```text
1. 小目标排序优化
   - 从 Exp35 在线预测中构造 TP / FP / hard negative
   - 对小目标正样本和混淆负样本加权
   - 学习一个轻量 score calibrator

2. 多级别名/层级类别特征
   - person: pedestrian / people
   - small vehicle: bicycle / motor / tricycle / awning tricycle
   - large vehicle: car / van / truck / bus
   - weak small classes: people / tricycle / awning tricycle / motor
```

## Method

训练阶段只使用 `val`：

```text
Exp35 online val predictions + val GT
  -> label each predicted box by IoU
  -> build geometry + score + hierarchy alias features
  -> train hard-negative calibrator
  -> search score blending on val holdout
```

测试阶段应用到 `test`：

```text
Exp35 online test predictions
  -> calibrator score
  -> blended final score
  -> evaluation
```

注意：Exp36 不用 test 标签训练；test 标签只用于最终评估。

## Run

先确保 Exp35 已经生成 val/test 在线预测：

```bash
CUDA_VISIBLE_DEVICES=2 python experiment/exp35_online_candidate_compression_20260711/scripts/run_exp35_online_candidate_compression_20260711.py \
  --dataset-split val
```

再运行 Exp36：

```bash
python experiment/exp36_small_target_alias_rerank_20260711/scripts/run_exp36_small_target_alias_rerank_20260711.py
```

## Next Experiment

Exp36 完成后，下一个实验建议做在线融合/校准模块：把 GroundingDINO score、Exp30 scorer score、prompt source、box geometry 和 calibrator score 统一训练成一个最终融合器。

## Result

Full test:

```text
Exp35:
  prediction boxes: 52872
  Acc@0.5:  0.5374
  Acc@0.75: 0.3463
  mAP@0.5:  0.1464

Exp36:
  prediction boxes: 52872
  Acc@0.5:  0.5374
  Acc@0.75: 0.3472
  mAP@0.5:  0.1546
```

Exp36 不改变候选框数量，主要通过重排预测置信度提升 mAP。
