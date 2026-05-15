# Exp27 Language-Aware Context-Refine Dual Region Distillation

## Purpose

在 Exp26 的 A/B 双区域图像处理基础上，加入轻量语言因素，验证类别文本提示和语义先验能否进一步提升小目标候选筛选。

## Data

- 数据根目录：`dataset/VisDroneSplit1000Guarded`
- split：复用 `experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv`
- train/val/test：700 / 150 / 150

## Inputs

- Exp18 candidates：`experiment/exp18_exp14_method_large_data_20260423/log/predictions/full`
- GroundingDINO candidates：`experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions`
- Teacher：`experiment/exp23_metric_driven_fusion_20260512/log/predictions/full`

## Method

沿用 Exp26 的双区域设计：

```text
A/context = candidate box expanded by 2.0x
B/refine = candidate box shrunk to 0.7x around center
```

新增语言/语义特征：

```text
class prompt stable hash embedding, dim=12
small-target prior
person / vehicle group prior
class-aware contrast interaction features
```

训练目标沿用 Exp24/Exp26：

```text
loss = BCE_gt + 0.7 * BCE_teacher
```

训练后在 val split 上搜索后处理：

```text
threshold in [0.03, 0.04, 0.05, 0.06, 0.08]
NMS in [0.45, 0.50, 0.55, 0.60]
objective = 0.5 * Acc@0.5 + 0.5 * continuous mAP@0.5
```

## Run

推荐在 tmux 中运行：

```bash
tmux new -s exp27_lang_dual
python experiment/exp27_lang_context_refine_dual_region_20260515/scripts/run_exp27_lang_context_refine_dual_region_20260515.py 2>&1 | tee experiment/exp27_lang_context_refine_dual_region_20260515/log/tmux_stdout_stderr.log
```

## Outputs

- `log/run_log.txt`
- `log/tmux_stdout_stderr.log`
- `log/exp27_lang_context_refine_dual_region_20260515_summary.json`
- `log/exp27_lang_context_refine_dual_region_20260515_best.pt`
- `log/postprocess_search_exp27_lang_dual_region.csv`
- `log/predictions/`
- `log/gt_xyxy/`
- `log/evaluation_summary_exp27_lang_dual_region_*_class_aware.md`

## Result

Best checkpoint:

- epoch: 11
- val objective: 0.2958
- val Acc@0.5: 0.4813
- val continuous mAP@0.5: 0.1102

Best val postprocess:

- threshold: 0.03
- NMS: 0.60
- val objective: 0.2960
- val Acc@0.5: 0.4818
- val continuous mAP@0.5: 0.1102

Full metrics:

| metric | value |
| --- | ---: |
| Acc@0.5 | 0.4677 |
| Acc@0.75 | 0.3121 |
| continuous mAP@0.5 | 0.1001 |
| VOC2007 11-point mAP@0.5 | 0.1341 |
| prediction boxes | 123825 |

## Conclusion

- 语言/语义先验让 val objective 和 full Acc@0.5 小幅提升。
- VOC2007 11-point mAP@0.5 从 Exp26 的 `0.1314` 提升到 `0.1341`，说明按历史实验表口径有收益。
- continuous AP 从 Exp26 的 `0.1015` 下降到 `0.1001`，说明当前轻量语言 hash 特征没有改善完整 precision-recall 排序。
- 结论是可以保留为 Exp26 的语言增强版，但还不足以替代主线；下一步更值得做冻结视觉语言 backbone 特征，而不是继续扩大手工 hash 特征。
