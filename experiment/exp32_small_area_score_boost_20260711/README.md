# Exp32 Small Area Score Boost

## Purpose

在不重新训练、不破坏 Exp30 整体优势的前提下，验证“小目标分数增强”是否能提高弱小目标 TopK 命中率。

Exp30 是当前 overall 主结果：

- full mAP@0.5 = `0.1521`
- full grounding R@1 = `0.3890`

但弱小类仍有短板，尤其 `people / awning tricycle / motor`。

## Method

基于 Exp30 最终预测文件做后处理：

```text
final_score = model_score + alpha[class] * small_area_prior
```

其中：

```text
small_area_prior = clamp((area_thr[class] - box_area_ratio) / area_thr[class], 0, 1)
```

只对小目标类生效：

```text
pedestrian, people, bicycle, tricycle, awning tricycle, motor
```

强类别如 `car / van / truck / bus` 不改分数。

## Safety Constraint

在 val 上搜索 boost profile 时，必须满足：

```text
val mAP >= Exp30 val mAP - 0.005
val prediction-based R@1 >= Exp30 val prediction-based R@1 - 0.005
```

满足约束后，最大化：

```text
0.6 * small_target_R@1 + 0.4 * small_target_R@5
```

## Notes

本实验的 grounding 指标是基于最终预测文件的 prediction-based TopK，用于比较后处理前后的排序变化；它和 Exp30 脚本中“模型对全候选直接打分”的 grounding 指标口径不同。

## Run

```bash
tmux new -s exp32_small_area_score_boost
python experiment/exp32_small_area_score_boost_20260711/scripts/run_exp32_small_area_score_boost_20260711.py
```
