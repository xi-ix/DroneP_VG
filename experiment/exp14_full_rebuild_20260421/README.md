# Exp14 全量重构（2026-04-21）

## 一句话结论
当前最好的可用权重是 [log/exp14_full_rebuild_20260421_best.pt](log/exp14_full_rebuild_20260421_best.pt)，配合最优后处理 threshold=0.08、nms=0.66，full Acc@0.5=0.4658。

## 目的
完全重构 Exp14 脚本链路：
- 以 Exp13 全量预测为主候选来源；
- 接入外接 SL/RL 双阶段训练；
- 在 100 图上输出 class-aware official 指标。

## 输入
- 数据：/home/wangzhe/DroneP_VG/visdrone_test_100
- Exp13 全量预测：/home/wangzhe/DroneP_VG/experiment/exp13_full_rebuild_20260421/log/predictions
- 辅助滑窗候选（如存在）：/home/wangzhe/DroneP_VG/experiment/lab9_reproduce_20260421/log/predictions/{train,val,test}_sw

## 训练口径
- split: 70/15/15（seed=42）
- SL: epochs=60, hidden_dim=64, lr=0.002
- RL: epochs=40, hidden_dim=32, lr=0.001, entropy_coef=0.001
- 搜索：threshold in [0.12,0.48] step=0.04, nms in {0.40,0.50,0.60,0.70}

## 运行
python scripts/run_exp14_full_rebuild_20260421.py

## 输出
- log/run_log.txt
- log/split_plan.json
- log/split_manifest.tsv
- log/exp14_full_rebuild_20260421_summary.json
- log/exp14_full_rebuild_20260421_best.pt
- log/evaluation_summary_exp14_full_rebuild_trainval_class_aware.md
- log/evaluation_summary_exp14_full_rebuild_test_class_aware.md
- log/evaluation_summary_exp14_full_rebuild_full_class_aware.md

## 当前最优结果
- 训练权重: [log/exp14_full_rebuild_20260421_best.pt](log/exp14_full_rebuild_20260421_best.pt)
- 最优后处理: threshold=0.08, nms=0.66
- 对应 full 指标: Acc@0.5=0.4658, Acc@0.75=0.3081, mAP@0.5=0.1430

## 结果对照
| 版本 | 权重/来源 | 后处理 | full Acc@0.5 | full Acc@0.75 | full mAP@0.5 | 预测框数 |
|---|---|---:|---:|---:|---:|---:|
| baseline | [GroundingDINO/weights/groundingdino_swint_ogc.pth](GroundingDINO/weights/groundingdino_swint_ogc.pth) | 0.20 / 0.40 | 0.3083 | 0.2407 | 0.1038 | 3130 |
| Exp14 full rebuild | [log/exp14_full_rebuild_20260421_best.pt](log/exp14_full_rebuild_20260421_best.pt) | 0.12 / 0.40 | 0.3074 | 0.2400 | 0.1057 | 3061 |
| Exp14 local refine best | [log/exp14_full_rebuild_20260421_best.pt](log/exp14_full_rebuild_20260421_best.pt) | 0.08 / 0.66 | 0.4658 | 0.3081 | 0.1430 | 8553 |

## 说明
- baseline 是 GroundingDINO 原始权重在 visdrone_test_100 上的重跑结果。
- Exp14 full rebuild 的 checkpoint 是外接 SL/RL 训练得到的权重，后处理可再单独搜索。
- local refine best 只是同一 checkpoint 的最优后处理组合，不是新的训练权重。
