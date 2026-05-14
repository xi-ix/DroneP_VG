# GroundingDINO baseline rerun (restored)

## Purpose
Restore the baseline experiment directory and rerun baseline metrics on visdrone_test_100.

## Script
- scripts/run_groundingdino_base_refdrone100_recovered_20260421.py

## Outputs
- log/predictions
- log/gt_xyxy
- log/run_log.txt
- log/evaluation_summary_groundingdino_base_refdrone100_recovered_20260421_class_aware.md
- log/groundingdino_base_refdrone100_recovered_20260421_summary.json
# GroundingDINO 基线（恢复的历史 100 图数据集）

## 目的
在恢复出的历史 100 图数据集上重新运行 GroundingDINO 基础 checkpoint，验证其数值是否与历史 Exp-01 基线保持一致。

## 范围
- 实验根目录：/home/wangzhe/DroneP_VG/experiment/baseline/groundingdino_base_refdrone100_recovered_20260421
- 恢复后的数据图像：/home/wangzhe/DroneP_VG/experiment/baseline/refdrone100_hist_recovered_20260421/images
- 恢复后的数据标注：/home/wangzhe/DroneP_VG/experiment/baseline/refdrone100_hist_recovered_20260421/annotations
- 模型 checkpoint：/home/wangzhe/GroundingDINO/weights/groundingdino_swint_ogc.pth
- 模型配置：/home/wangzhe/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py

## 评估口径
- box_threshold：0.20
- text_threshold：0.20
- nms_threshold：0.40
- 评估模式：class-aware
- 输出指标：Acc@0.5、Acc@0.75、mAP@0.5

## 运行方式
使用 qwen_vl Python 环境。

示例：
python scripts/run_groundingdino_base_refdrone100_recovered_20260421.py

## 输出
- 预测结果：log/predictions
- 转换后的 GT（XYXY）：log/gt_xyxy
- 运行日志：log/run_log.txt
- 指标汇总：log/evaluation_summary_groundingdino_base_refdrone100_recovered_20260421_class_aware.md
- 历史对齐报告：log/HISTORICAL_ALIGNMENT_RESULT_20260421.md

## 结果摘要
来自 log/evaluation_summary_groundingdino_base_refdrone100_recovered_20260421_class_aware.md：
- 统计到的标注文件：100
- GT boxes：5077
- Prediction boxes：3651
- Acc@0.5：0.3449（1751/5077）
- Acc@0.75：0.2630（1335/5077）
- mAP@0.5：0.1454

## 历史对齐
来自 log/HISTORICAL_ALIGNMENT_RESULT_20260421.md：
- 历史参考：GT 5077，Pred 3648，Acc@0.5 0.3443，Acc@0.75 0.2624，mAP@0.5 0.1451
- 差值（当前 - 历史）：Pred +3，Acc@0.5 +0.0006，Acc@0.75 +0.0006，mAP@0.5 +0.0003

结论：该基线的复现质量与历史结果几乎一致，可作为后续 Exp-13/Exp-14 重建的固定参考。
