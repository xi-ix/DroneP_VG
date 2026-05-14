# Exp17 更大数据集守卫（2026-04-22）

## 目标
- 按之前实验一致的切分标准构建一个更大的数据集。
- 冻结原始 100 子集中的 val/test 样本，保证与历史最优结果可直接对比。
- 为后续 attention 实验提供更稳定的数据基础。

## 结构
- scripts/build_large_dataset_guard_20260422.py：构建数据集并写入守卫信息。
- log/：运行日志、切分计划、统计信息和回归守卫文件。
- dataset/VisDroneSplit1000Guarded/：生成的实文件数据集副本，后续训练统一使用这里的数据。

## 数据来源与规则
- 小样本来源：/home/wangzhe/DroneP_VG/dataset/visdrone_test_100
- 完整来源：/home/wangzhe/DroneP_VG/dataset/RefDrone/VisDrone2019-DET-{train,val,test}
- 目标规模：1000 张图像
- 划分比例：train/val/test = 70/15/15
- 随机种子：42

## 回归守卫策略
- 使用 Exp14 现有 split_plan 作为锁定子集来源：
  /home/wangzhe/DroneP_VG/experiment/exp14_full_rebuild_20260421/log/split_plan.json
- 在扩容后的划分中保留原有 val/test 样本。
- 从完整来源中补充新增样本，直到达到目标数量。

## 运行
python3 /home/wangzhe/DroneP_VG/experiment/exp17_large_dataset_guard_20260422/scripts/build_large_dataset_guard_20260422.py

## 输出
- log/split_plan.json
- log/split_manifest.tsv
- log/dataset_stats.json
- log/regression_guard.json
- log/run_log.txt

## 后续建议
- 后续 attention 实验请同时报告两套指标：
  1) 扩容后全量划分上的指标
  2) 冻结回归子集上的指标（不得低于之前最优基线）
