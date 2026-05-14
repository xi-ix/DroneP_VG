# Exp23 Metric Driven Fusion

This experiment rebuilds the metric-driven fusion upper bound on `dataset/VisDroneSplit1000Guarded`.

It fuses:

- Exp18 predictions: `experiment/exp18_exp14_method_large_data_20260423/log/predictions/full`
- GroundingDINO baseline predictions: `experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions`

Default restored config:

- score scale: `1.0` for both sources
- threshold: `0.01`
- class-aware NMS: `0.45`
- topK: all

Run:

```bash
python experiment/exp23_metric_driven_fusion_20260512/scripts/run_exp23_metric_driven_fusion_20260512.py
```

