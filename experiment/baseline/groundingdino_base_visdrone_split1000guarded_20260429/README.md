# GroundingDINO baseline on VisDroneSplit1000Guarded (2026-04-29)

## 目的
在新的扩容数据集 [dataset/VisDroneSplit1000Guarded](../../../dataset/VisDroneSplit1000Guarded) 上重跑 GroundingDINO 基线，作为新数据口径下的参考结果。

## 数据范围
- 数据集根目录：/home/wangzhe/DroneP_VG/dataset/VisDroneSplit1000Guarded
- 划分：train/val/test = 700/150/150
- 评估模式：class-aware
- 阈值：box_threshold=0.20，text_threshold=0.20，nms_threshold=0.40

## 脚本
- scripts/run_groundingdino_base_visdrone_split1000guarded_20260429.py

## 输出
- log/predictions
- log/gt_xyxy
- log/run_log.txt
- log/evaluation_summary_groundingdino_base_visdrone_split1000guarded_20260429_class_aware.md
- log/groundingdino_base_visdrone_split1000guarded_20260429_summary.json

## 说明
- 这个版本不再依赖根目录的单文件 manifest，而是直接读取 train/val/test 三个 split 目录。
- 结果可与历史 baseline 做横向对比，但要注意数据集规模已从 100 图扩展到 1000 图。
