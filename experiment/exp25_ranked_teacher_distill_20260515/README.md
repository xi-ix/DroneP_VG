# Exp25 Ranked Teacher Distillation

Purpose:

- Continue from Exp24 teacher distillation.
- Add explicit same-image positive/negative pair ranking loss to improve score ordering for mAP.

Data:

- `dataset/VisDroneSplit1000Guarded`

Inputs:

- Exp18 candidates: `experiment/exp18_exp14_method_large_data_20260423/log/predictions/full`
- GroundingDINO candidates: `experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions`
- Exp23 teacher: `experiment/exp23_metric_driven_fusion_20260512/log/predictions/full`

Training objective:

```text
loss = BCE_gt + 0.7 * BCE_teacher + 0.5 * ranking_loss
```

Ranking:

```text
positive_logit > negative_logit + 0.2
```

Run:

```bash
python experiment/exp25_ranked_teacher_distill_20260515/scripts/run_exp25_ranked_teacher_distill_20260515.py
```

## Result

Best checkpoint:

- epoch: 9
- val objective: 0.2917
- val Acc@0.5: 0.4809
- val continuous mAP@0.5: 0.1025

Full metrics:

| metric | value |
| --- | ---: |
| Acc@0.5 | 0.4661 |
| Acc@0.75 | 0.3110 |
| continuous mAP@0.5 | 0.0945 |
| VOC2007 11-point mAP@0.5 | 0.1288 |
| prediction boxes | 123013 |

Conclusion:

- Ranking loss made the pairwise margin easier to satisfy during training.
- VOC2007 11-point mAP is slightly above restored Exp24 (`0.1288` vs `0.1274`).
- Continuous AP and Acc both dropped relative to restored Exp24, so this is not a robust mainline improvement.
- Keep as an ablation result; do not replace Exp24 as the main student baseline.
