# Exp18 大数据集上的 Exp14 方法（2026-04-23）

## 目标
- 在扩容后的守卫数据集上重新运行 Exp14 的原始训练方法。
- 保持与 Exp14 完全一致的模型、训练和搜索流程。
- 观察扩大数据后，原方法性能是否发生变化。

## 方法
- 脚本由 Exp14 复制而来，只修改为指向大数据集切分。
- 数据切分直接读取自：
  /home/wangzhe/DroneP_VG/dataset/VisDroneSplit1000Guarded
- 训练流程保持不变：
  - 监督训练（60 轮）
  - 强化训练（40 轮）
  - 在验证集上进行 threshold/NMS 网格搜索
  - 按类别进行评估

## 运行
python3 /home/wangzhe/DroneP_VG/experiment/exp18_exp14_method_large_data_20260423/scripts/run_exp18_exp14_method_large_data_20260423.py

## 输出
- log/exp18_exp14_method_large_data_20260423_summary.json
- log/evaluation_summary_exp18_exp14_method_large_data_trainval_class_aware.md
- log/evaluation_summary_exp18_exp14_method_large_data_test_class_aware.md
- log/evaluation_summary_exp18_exp14_method_large_data_full_class_aware.md
- log/run_log.txt
