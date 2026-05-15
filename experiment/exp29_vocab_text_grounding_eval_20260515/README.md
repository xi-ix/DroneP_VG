# Exp29 Vocab Text Grounding Evaluation

## Purpose

在 Exp28 真 query 输入原型基础上做两件事：

1. 用可学习词表 embedding 替代 hash text feature。
2. 增加 grounding 专用评估：`query -> Top1/Top5/Top10 box Recall`。

## Data

- 数据根目录：`dataset/VisDroneSplit1000Guarded`
- split：复用 `experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv`
- train/val/test：700 / 150 / 150

## Inputs

- Exp18 candidates：`experiment/exp18_exp14_method_large_data_20260423/log/predictions/full`
- GroundingDINO candidates：`experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions`
- Teacher：`experiment/exp23_metric_driven_fusion_20260512/log/predictions/full`

## Method

模型输入：

```text
A/B image region numeric feature + query semantic numeric priors + query token ids
```

文本分支：

```text
query tokens -> Embedding(dim=32) -> mean pooling -> MLP(dim=32)
```

融合分支：

```text
numeric feature + text feature -> MLP scorer -> query-box matching score
```

训练目标：

```text
loss = BCE_gt + 0.7 * BCE_teacher
```

## Grounding Evaluation

对每张图中存在的每个类别构造 canonical query，例如：

```text
car
pedestrian
truck
motor
```

模型对该图所有候选框打分，按分数排序，计算：

```text
Recall@1 / Recall@5 / Recall@10
```

只要 TopK 内任一候选框与该 query 对应类别的 GT IoU >= 0.5，就记为命中。

## Result

Best checkpoint:

- epoch: 12
- val objective: 0.2966
- val Acc@0.5: 0.4813
- val continuous mAP@0.5: 0.1118

Best val postprocess:

- threshold: 0.03
- NMS: 0.60
- val objective: 0.2968

Detection-style full metrics:

| metric | value |
| --- | ---: |
| Acc@0.5 | 0.4677 |
| Acc@0.75 | 0.3123 |
| continuous mAP@0.5 | 0.1025 |
| VOC2007 11-point mAP@0.5 | 0.1242 |
| prediction boxes | 123831 |

Grounding full metrics:

| metric | value |
| --- | ---: |
| query count | 5422 |
| Recall@1 | 0.2979 |
| Recall@5 | 0.4825 |
| Recall@10 | 0.5791 |
| mean first hit rank | 20.32 |

Strong classes:

- car: R@1=0.8482, R@5=0.9820, R@10=0.9904
- pedestrian: R@1=0.4969, R@5=0.7093, R@10=0.7640

Weak classes:

- awning tricycle: R@1=0.0217, R@10=0.1902
- motor: R@1=0.0267, R@10=0.3601
- people: R@1=0.0414, R@10=0.3638

## Notes

Initial grounding evaluation incorrectly reported `query_count=0` because the GT reader reused the 6-column prediction parser and skipped 5-column GT files. This was fixed in the Exp29 script and recomputed with:

```bash
python experiment/exp29_vocab_text_grounding_eval_20260515/scripts/recompute_exp29_grounding_topk_20260515.py
```

## Conclusion

- The learnable vocab text encoder improves continuous detection-style mAP over Exp28 (`0.1025` vs `0.1004`).
- Grounding TopK evaluation is now available and gives a more relevant signal for the final goal.
- The model is strong for common/visually clear classes such as `car` and `pedestrian`, but weak for visually ambiguous or sparse classes such as `awning tricycle`, `motor`, and `people`.
- Next improvement should focus on real referring-language data from RefDrone/AerialVG or a stronger pretrained text/vision-language encoder.
