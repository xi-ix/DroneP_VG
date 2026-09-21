# Exp33 Small Candidate Recall Union

## Purpose

Exp32 证明最终预测阶段的小面积 score boost 能守住整体分数，但无法恢复 `motor / awning tricycle` 等极弱小目标，说明问题更可能发生在候选保留阶段。

本实验前移到候选召回阶段：在 Exp30 最终预测基础上，从 Exp18 与 GroundingDINO baseline 的候选中低阈值捞回弱小类小框候选。

## Method

基础预测：

```text
Exp30 final predictions
```

补充候选源：

```text
Exp18 predictions
GroundingDINO baseline predictions
```

只恢复以下弱小类：

```text
people
tricycle
awning tricycle
motor
```

候选过滤：

```text
class in weak classes
source_score >= profile.score_thr
box_area_ratio <= area_threshold[class]
topK per class per image
```

补充候选重打分：

```text
new_score = base_score + 0.05 * source_score + area_bonus * small_area_prior + class_bonus
```

随后与 Exp30 最终预测合并，并做 class-aware NMS。

## Search

在 val 上搜索多个安全/召回 profile，并设置约束：

```text
val mAP >= Exp30 val prediction mAP - 0.006
val prediction-based R@1 >= Exp30 val prediction-based R@1 - 0.006
```

在满足整体约束下，最大化：

```text
0.45 * small_R@1 + 0.35 * small_R@5 + 0.20 * weak_R@5
```

## Run

```bash
tmux new -s exp33_small_candidate_recall_union
python experiment/exp33_small_candidate_recall_union_20260711/scripts/run_exp33_small_candidate_recall_union_20260711.py
```
