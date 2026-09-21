# Demo evaluation

Run the lightweight inspection demo from the repository root:

```bash
python scripts/demo_evaluate.py
```

The demo uses the bundled samples in `scripts/data`, evaluates prediction boxes
against ground-truth annotations, prints precision/recall/F1, and writes
visualized results to `scripts/demo_output`.

Useful options:

```bash
python scripts/demo_evaluate.py --conf-threshold 0.30 --iou-threshold 0.50
python scripts/demo_evaluate.py --json
```

# Final result evaluation

Build the final experiment-result structure:

```bash
python scripts/evaluate_final_results.py
```

The default command evaluates the bundled inspection samples and writes:

```text
scripts/final_results/
├── summary.json
├── evaluation_summary_final_class_aware.md
├── metrics_table.csv
├── per_class_metrics.csv
└── per_image_metrics.csv
```

For a formal experiment, point the script to the experiment log directories:

```bash
python scripts/evaluate_final_results.py \
  --experiment-name exp30_real_query_focal_rerank_20260711 \
  --gt-dir experiment/exp30_real_query_focal_rerank_20260711/log/gt_xyxy/full \
  --pred-dir experiment/exp30_real_query_focal_rerank_20260711/log/predictions/full \
  --gt-format xyxy \
  --output-dir experiment/exp30_real_query_focal_rerank_20260711/log/final_results
```
