# inspection_demo Final Evaluation Summary

## Config
- gt_dir: /home/wangzhe/DroneP_VG/scripts/data/test/annotations
- pred_dir: /home/wangzhe/DroneP_VG/scripts/data/pred
- eval_mode: class-aware
- gt_format: visdrone

## Metrics
- Label files considered: 2
- GT boxes: 73
- Prediction boxes: 138
- Acc@0.5: 0.6027 (44/73)
- Acc@0.75: 0.5068 (37/73)
- mAP@0.5: 0.2453

## Output Structure
- summary.json: machine-readable final result
- evaluation_summary_final_class_aware.md: human-readable final report
- metrics_table.csv: one-row key metric table for paper/report tables
- per_class_metrics.csv: per-class AP/recall/precision
- per_image_metrics.csv: per-image match statistics
