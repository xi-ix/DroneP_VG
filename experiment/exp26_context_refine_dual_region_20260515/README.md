# Exp26 Context-Refine Dual Region Distillation

## Purpose

验证一个两步图像区域处理想法：

1. 先从候选框外扩得到上下文区域 A，保留小目标周围背景信息。
2. 再从候选框中心收缩得到细化区域 B，突出目标局部信息。
3. 将原候选框几何特征、A 区域统计、B 区域统计和 B-A 差异同时输入 scorer 整合模块。

本实验不覆盖 Exp24，而是在 Exp24 teacher distillation 基线上新增图像双区域特征，作为结构消融。

## Data

- 数据根目录：`dataset/VisDroneSplit1000Guarded`
- split：复用 `experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv`
- train/val/test：700 / 150 / 150

## Inputs

- Exp18 candidates：`experiment/exp18_exp14_method_large_data_20260423/log/predictions/full`
- GroundingDINO candidates：`experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions`
- Teacher：`experiment/exp23_metric_driven_fusion_20260512/log/predictions/full`

## Method

保持 Exp24 的训练目标和后处理：

```text
loss = BCE_gt + 0.7 * BCE_teacher
threshold = 0.05
NMS = 0.50
epochs = 24
```

新增双区域特征：

```text
A/context = candidate box expanded by 2.0x
B/refine = candidate box shrunk to 0.7x around center
feature = box_geometry + A_rgb_stats + B_rgb_stats + (B - A)_stats + region_area_ratios
```

当前实现使用 PIL 裁剪区域并提取 RGB mean/std、brightness、contrast；这是轻量版本，不引入额外 CNN 或大模型图像特征。

## Run

```bash
python experiment/exp26_context_refine_dual_region_20260515/scripts/run_exp26_context_refine_dual_region_20260515.py
```

## Outputs

- `log/run_log.txt`
- `log/exp26_context_refine_dual_region_20260515_summary.json`
- `log/exp26_context_refine_dual_region_20260515_best.pt`
- `log/predictions/`
- `log/gt_xyxy/`
- `log/evaluation_summary_exp26_dual_region_*_class_aware.md`

## Result

Best checkpoint:

- epoch: 6
- val objective: 0.2951
- val Acc@0.5: 0.4816
- val continuous mAP@0.5: 0.1085

Full metrics:

| metric | value |
| --- | ---: |
| Acc@0.5 | 0.4670 |
| Acc@0.75 | 0.3124 |
| continuous mAP@0.5 | 0.1015 |
| VOC2007 11-point mAP@0.5 | 0.1314 |
| prediction boxes | 122999 |

## Conclusion

- A/B 双区域输入是合理方向：小目标确实需要上下文 A 和局部 B 同时参与候选打分。
- 当前轻量 RGB 统计版本没有显著提升 continuous AP，但 VOC2007 11-point mAP 从恢复版 Exp24 的 `0.1274` 小幅提升到 `0.1314`。
- 该实验可作为后续“真实图像特征版”的起点；若继续优化，应优先把 A/B 区域统计换成冻结视觉 backbone 特征，而不是继续手工堆 RGB 统计。
