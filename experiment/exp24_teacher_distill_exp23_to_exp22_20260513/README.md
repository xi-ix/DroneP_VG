# Exp24 Teacher Distill Exp23 To Exp22

This experiment restores the Exp24 student distillation pipeline.

Goal:

- Use Exp23 fusion predictions as teacher.
- Train an Exp22-style external scorer over candidate boxes from Exp18 and GroundingDINO baseline.
- Select checkpoint by real val objective: `0.5 * Acc@0.5 + 0.5 * mAP@0.5`.

Data:

- `dataset/VisDroneSplit1000Guarded`

Inputs:

- Exp18 candidates: `experiment/exp18_exp14_method_large_data_20260423/log/predictions/full`
- GroundingDINO candidates: `experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions`
- Exp23 teacher: `experiment/exp23_metric_driven_fusion_20260512/log/predictions/full`

Run:

```bash
python experiment/exp24_teacher_distill_exp23_to_exp22_20260513/scripts/run_exp24_teacher_distill_exp23_to_exp22_20260513.py
```

## mAP calibration

Historical Exp23/Exp24 records use VOC 2007 11-point AP for `mAP@0.5`.
The restored scripts originally reported continuous AP as `map_05`, which is lower.

Calibration:

```bash
python experiment/exp24_teacher_distill_exp23_to_exp22_20260513/scripts/calibrate_exp24_map_20260515.py
```

Restored full-set results:

| case | continuous AP mAP@0.5 | VOC2007 11-point mAP@0.5 |
| --- | ---: | ---: |
| Exp23 teacher | 0.1021 | 0.1295 |
| Exp24 student | 0.1016 | 0.1274 |
