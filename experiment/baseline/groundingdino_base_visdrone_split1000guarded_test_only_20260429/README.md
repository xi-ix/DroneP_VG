# GroundingDINO baseline on VisDroneSplit1000Guarded test only (2026-04-29)

## 目的
只在 [dataset/VisDroneSplit1000Guarded](../../../dataset/VisDroneSplit1000Guarded) 的 test split 上重跑 GroundingDINO 基线，便于和 1000 张整体结果对比。

## 数据范围
- 数据集根目录：/home/wangzhe/DroneP_VG/dataset/VisDroneSplit1000Guarded
- 划分：test only
- 评估模式：class-aware
- 阈值：box_threshold=0.20，text_threshold=0.20，nms_threshold=0.40

## 脚本
- scripts/run_groundingdino_test_only_20260429.py

## 输出
- log/predictions
- log/gt_xyxy
- log/run_log.txt
- log/evaluation_summary_groundingdino_test_only_20260429_class_aware.md
- log/groundingdino_test_only_20260429_summary.json

## 说明
- 这个版本只读取 test 目录，不会把 train/val 一起算进去。
- 可直接与 [baseline/groundingdino_base_visdrone_split1000guarded_20260429](../baseline/groundingdino_base_visdrone_split1000guarded_20260429) 的全量结果对比差距。
