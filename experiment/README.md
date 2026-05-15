# 实验总记录

这个文件用于持续记录每次实验的目的、数据、产物和当前结论。新的实验请继续追加到末尾，尽量保持每个实验一段，方便回溯。

## 统一约定
- 数据目录统一放在 [dataset](../dataset) 下。
- 每个实验单独放在 [experiment](.) 的独立子目录中，目录下保留 `scripts/`、`log/` 和 `README.md`。
- 训练时优先读取实验目录自己的 `README.md` 和 `log/` 产物，避免和别的实验混用。

## 当前阶段指标目标
- `mAP@0.5`：先回到 `0.15`，再向 `0.18~0.20` 争取；`0.22+` 视为很强。
- `Acc@0.5`：先稳定到 `0.45~0.50`，再争取 `0.50~0.55`；`0.55+` 视为很强。
- `Acc@0.75`：先稳定到 `0.30` 左右，再争取 `0.32~0.35`；`0.35+` 视为很强。
- 当前优先级：先把 `mAP@0.5` 拉回 `exp14` 水平并继续提升，再看更深的打分器或 set-level reranker。

## 小数据集（100图）对比表（class-aware）
| 实验 | 目录 | best threshold / nms | full Acc@0.5 | full Acc@0.75 | full mAP@0.5 | full 预测框数 | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| Exp14 全量重构 | [exp14_full_rebuild_20260421](exp14_full_rebuild_20260421) | 0.12 / 0.40 | 0.3081 | 0.2405 | 0.1069 | 3122 | 当前可复现小数据主基线 |
| Exp16 注意力 + 大模型 + MLP | [exp16_attn_bigmlp_20260422](exp16_attn_bigmlp_20260422) | 0.08 / 0.66 | 0.3083 | 0.2407 | 0.1011 | 3130 | 2026-05-09 架构更新后重跑，mAP 较旧版回升 |
| Exp21 Exp14+双输入预处理 | [exp21_preprocess_exp14_20260509](exp21_preprocess_exp14_20260509) | 0.12 / 0.40 | 0.3069 | 0.2400 | 0.1049 | 3081 | 独立新脚本，图像/语言预处理后 mAP 高于 Exp16，低于 Exp14 |
| Exp20 奖励重构 | [exp20_small_reward_20260507](exp20_small_reward_20260507) | 0.12 / 0.40 | 0.3081 | 0.2405 | 0.1062 | 3111 | 命中率导向 reward，mAP 小幅下降 |
| Exp22 冻结DINO双输入整合打分 | [exp22_frozen_dino_dual_preprocess_20260512](exp22_frozen_dino_dual_preprocess_20260512) | 0.08 / 0.40 | 0.3083 | 0.2407 | 0.0852 | 3130 | 不改 GroundingDINO，外接图像/语言预处理与整合打分；Acc 持平但 mAP 明显下降 |

## Exp21 Exp14 + 双输入预处理（1000 图，2026-05-09）
- 目录： [exp21_preprocess_exp14_large_20260509](exp21_preprocess_exp14_large_20260509)
- 作用：在 Exp18 的 1000 图扩容数据口径上，加入图像输入与语言输入预处理模块，观察对稳健性和 mAP 的影响。
- 关键约束：使用独立新脚本，不引用旧实验脚本。
- 主要改动：
	- 图像分支预处理：`Linear + LayerNorm + ReLU`
	- 语言分支预处理：`prompt embedding -> Linear + LayerNorm + ReLU`
	- 空文本分支：可学习 `null_prompt`
- 主要产物：`log/exp21_preprocess_exp14_large_20260509_summary.json`、三份评估报告、`log/run_log.txt`。
- 结果摘要：best threshold=0.02、best nms=0.40；full Acc@0.5=0.4394、Acc@0.75=0.2930、mAP@0.5=0.0934；full prediction boxes=113836。
- 对比结论：相比 Exp18，Acc 基本持平，mAP 略降；说明双预处理模块在当前后处理网格下没有带来收益。

## Exp13 全量重构（2026-04-21）
- 目录： [exp13_full_rebuild_20260421](exp13_full_rebuild_20260421)
- 作用：Exp14 的前置重构实验，主要产出全量候选预测和重构后的摘要文件。
- 主要产物：`log/exp13_full_rebuild_20260421_summary.json`、`log/predictions/`
- 记录结论：Exp13 的高保真 checkpoint 只在 20 图子集上验证过，直接迁移到完整校准流程后不稳定。
- 备注：后续 Exp14/Exp15/Exp16 的分析都以 Exp13 结果为前置基础。

## Exp14 全量重构（2026-04-21）
- 目录： [exp14_full_rebuild_20260421](exp14_full_rebuild_20260421)
- 作用：基于 Exp13 候选框做外接校准器，包含监督学习和强化学习两阶段训练。
- 数据口径：100 图，70/15/15 切分。
- 主要产物：`log/exp14_full_rebuild_20260421_best.pt`、`log/exp14_full_rebuild_20260421_summary.json`、三个 class-aware 评估报告。
- 当前结论：按当前可复现实验产物（`log/exp14_full_rebuild_20260421_summary.json` 与 `log/evaluation_summary_exp14_full_rebuild_full_class_aware.md`），full Acc@0.5=0.3081、Acc@0.75=0.2405、mAP@0.5=0.1069。
- 备注：历史文档中曾记录 `0.4658`（threshold=0.08、nms=0.66），但 `verify_best_result.json` 与相关复核结果无法复现该值，后续统一以可复现产物为准。
- 风险记录：历史上多个重跑分支会因为数据根目录或恢复口径不一致而退化，需要固定数据源和锁定子集。

## Exp15 语言注意力层（2026-04-22）
- 目录： [exp15_lang_attention_20260422](exp15_lang_attention_20260422)
- 作用：在 Exp14 外接校准器前加入语言条件注意力，观察文本门控对候选框打分的影响。
- 主要改动：prompt 经 token hash、embedding、MLP 后生成 gate，再对候选特征做门控缩放。
- 主要产物：`log/exp15_lang_attention_20260422_best.pt`、`log/exp15_lang_attention_20260422_summary.json`、三份评估报告。
- 当前结论：作为 Exp14 的结构变体保留，后续若继续做语言引导，优先从这一条线迭代。

## Exp16 注意力 + 大模型 + MLP（2026-04-22）
- 目录： [exp16_attn_bigmlp_20260422](exp16_attn_bigmlp_20260422)
- 作用：在与 Exp14 相同后处理口径下，测试更深更宽的 MLP backbone 和独立 scoring head。
- 主要改动：保留语言门控注意力，替换成更大的主干和单独打分头。
- 主要产物：`log/exp16_attn_bigmlp_20260422_best.pt`、`log/exp16_attn_bigmlp_20260422_summary.json`、三份评估报告。
- 当前结论：2026-05-09 在架构更新（图像/语言双输入预处理模块）后重跑，full Acc@0.5=0.3083、Acc@0.75=0.2407、mAP@0.5=0.1011；相比旧版 Exp16（mAP@0.5=0.0969）有回升，但仍低于 Exp14（0.1069）。

## Exp17 更大数据集守卫（2026-04-22）
- 目录： [exp17_large_dataset_guard_20260422](exp17_large_dataset_guard_20260422)
- 作用：把有用数据重新整理进 [dataset/VisDroneSplit1000Guarded](../dataset/VisDroneSplit1000Guarded) 作为后续训练统一数据源。
- 规则：冻结原始 100 图中的 val/test 子集，训练集继续扩容，目标总量 1000 张。
- 主要产物：`log/split_plan.json`、`log/split_manifest.tsv`、`log/dataset_stats.json`、`log/regression_guard.json`。
- 当前结论：这套数据已经重建为实文件副本，后续训练统一使用这里的数据，不再依赖软链接树。

## Exp18 大数据集上的 Exp14 方法（2026-04-23）
- 目录： [exp18_exp14_method_large_data_20260423](exp18_exp14_method_large_data_20260423)
- 作用：把 Exp14 的原始训练流程直接迁移到 Exp17 的扩容数据集上，观察扩大数据后的变化。
- 数据口径：直接读取 [dataset/VisDroneSplit1000Guarded](../dataset/VisDroneSplit1000Guarded)。
- 训练流程：监督训练 60 轮，强化训练 40 轮，再做 threshold/NMS 搜索和 class-aware 评估。
- 主要产物：`log/exp18_exp14_method_large_data_20260423_summary.json` 和三份评估报告。
- 当前结论：这是后续所有“大数据版”训练的主基线，和 Exp14 保持同方法对照。
- 更新（2026-05-06）：
	- 清理了失败运行遗留产物（旧 `summary`、旧评估 markdown、`run_log.txt`、`threshold_search_tmp/` 与旧 `predictions/{full,test,trainval,val}`）。
	- 为避免继续改动 Exp13 目录，将候选预测复制到本实验目录 `log/predictions/source_exp13/`，并把脚本默认候选源改为该路径。
	- 候选覆盖率确认：train/val/test = 700/150/150 全覆盖（共 1000 个预测 txt）。
	- 快速复跑（`QUICK_VAL_SEARCH=1`）结果：best threshold=0.02、nms=0.40；full Acc@0.5=0.4395、mAP@0.5=0.0962；test Acc@0.5=0.4568、mAP@0.5=0.0991。
	- 对比修复前（覆盖不足版本）full Acc@0.5≈0.0250、mAP@0.5≈0.0078，确认主要退化原因是候选覆盖缺失而非训练流程本身。

## Exp19 召回增强版（2026-05-06）
- 目录： [exp19_recall_union_20260506](exp19_recall_union_20260506)
- 作用：在与 Exp18 相同的数据集上，进一步提高候选召回，比较“多阈值并集 + 更高 top-K + 统一 NMS”是否能提升效果。
- 候选策略：保留 Exp13 本地预测为基础，同时加入 GroundingDINO 多阈值并集候选（0.02 / 0.04 / 0.06 / 0.08 / 0.10），每阈值保留较高 top-K 后统一 NMS。
- 主要产物：`log/exp19_recall_union_20260506_summary.json`、三份评估报告、`log/predictions/source_gdino_union/`。
- 快速验证结果：best threshold=0.02、best nms=0.70；full Acc@0.5=0.4635、mAP@0.5=0.0813；test Acc@0.5=0.4897、mAP@0.5=0.0868。
- 对比 Exp18：Acc@0.5 有提升，但 mAP@0.5 下降，说明召回增强确实抬高了可覆盖目标数，但同时引入了更多冗余候选与误报，需要继续做更强的去重/筛选。

## Exp20 小数据集奖励重构（2026-05-07）
- 目录： [exp20_small_reward_20260507](exp20_small_reward_20260507)
- 作用：在 Exp14 小数据集流程上仅修改 RL reward，目标是提高命中率并减少无用生成。
- Reward：`0.55 * F1 + 0.20 * recall + 0.15 * mean IoU - 0.10 * FP rate`。
- 主要产物：`log/exp20_small_reward_20260507_summary.json`、三份评估报告、`log/run_log.txt`。
- 结果摘要：best threshold=0.12、best nms=0.40；full Acc@0.5=0.3081、Acc@0.75=0.2405、mAP@0.5=0.1062；full prediction boxes=3111。
- 对比 Exp14：Acc 指标基本持平，mAP@0.5 从 0.1069 小幅下降到 0.1062，说明当前 reward 方向仍需继续调参。
- 当日复跑补充（2026-05-07）：在完整数据上执行 `RL_EPOCHS=20` 复跑，RL 日志显示 reward 与梯度均为非零（训练链路正常），阈值搜索最优仍为 `threshold=0.12`、`nms=0.40`，最终 `full_metrics` 与上表一致。

## Exp21 Exp14 + 双输入预处理（2026-05-09）
- 目录： [exp21_preprocess_exp14_20260509](exp21_preprocess_exp14_20260509)
- 作用：在 Exp14 主流程基础上增加图像输入与语言输入预处理模块，验证预处理对稳健性的影响。
- 关键约束：使用独立新脚本，不引用旧实验脚本。
- 主要改动：
	- 图像分支预处理：`Linear + LayerNorm + ReLU`
	- 语言分支预处理：`prompt embedding -> Linear + LayerNorm + ReLU`
	- 空文本分支：可学习 `null_prompt`
- 主要产物：`log/exp21_preprocess_exp14_20260509_summary.json`、三份评估报告、`log/run_log.txt`。
- 结果摘要：best threshold=0.12、best nms=0.40；full Acc@0.5=0.3069、Acc@0.75=0.2400、mAP@0.5=0.1049；full prediction boxes=3081。
- 对比结论：mAP@0.5 介于 Exp14 与 Exp16 之间（高于 Exp16，低于 Exp14）。

## Exp22 冻结DINO双输入整合打分（2026-05-12）
- 目录： [exp22_frozen_dino_dual_preprocess_20260512](exp22_frozen_dino_dual_preprocess_20260512)
- 作用：在不修改 GroundingDINO 模型本体的前提下，加入图像输入预处理、语言输入预处理、整合层和候选打分模块。
- 关键约束：实验脚本独立；直接读取 [dataset/visdrone_test_100](../dataset/visdrone_test_100)；不复制原始数据；GroundingDINO 代码、权重和模型结构均不修改。
- 工程折中：由于暂时不动 DINO，本实验复用 Exp13 冻结候选预测作为 DINO 输出代理，把图像/语言预处理与整合层放在外接 scoring head 之前。该折中已写入 `log/run_log.txt` 和实验 README。
- 主要改动：
	- 候选框几何/置信度特征、图像统计特征、语言 prompt embedding 三路输入。
	- 三路输入分别经过预处理层，再进入 `integration_layer`。
	- `score_head` 输出候选保留概率。
	- RL reward 同时包含 Acc proxy、mAP proxy、F1、mean IoU，并惩罚 FP rate。
- 主要产物：`log/exp22_frozen_dino_dual_preprocess_20260512_summary.json`、三份评估报告、`log/run_log.txt`、`log/exp22_frozen_dino_dual_preprocess_20260512_best.pt`。
- 结果摘要：best threshold=0.08、best nms=0.40；full Acc@0.5=0.3083、Acc@0.75=0.2407、mAP@0.5=0.0852；full prediction boxes=3130。
- 对比结论：Acc 指标基本持平，但 mAP@0.5 显著低于 Exp14/Exp21，说明当前外接整合打分模块损害了排序质量或置信度校准，不建议作为主线结果。

## Exp22 冻结DINO双输入整合打分（1000 图，2026-05-12）
- 目录： [exp22_frozen_dino_dual_preprocess_large_20260512](exp22_frozen_dino_dual_preprocess_large_20260512)
- 作用：将 Exp22 的冻结 DINO 双输入预处理与整合打分模块迁移到 [dataset/VisDroneSplit1000Guarded](../dataset/VisDroneSplit1000Guarded)。
- 数据口径：直接读取 guarded dataset 固定 split，train/val/test = 700/150/150。
- 候选来源：复用 Exp18 的冻结候选预测 `exp18_exp14_method_large_data_20260423/log/predictions/source_exp13`，不修改 GroundingDINO 本体。
- 主要产物：`log/exp22_frozen_dino_dual_preprocess_large_20260512_summary.json`、三份评估报告、`log/run_log.txt`、`log/exp22_frozen_dino_dual_preprocess_large_20260512_best.pt`。
- 结果摘要：best threshold=0.08、best nms=0.40；full Acc@0.5=0.3627、Acc@0.75=0.2618、mAP@0.5=0.0794；full prediction boxes=69627。
- 对比结论：相比 Exp18 large（full Acc@0.5=0.4395、mAP@0.5=0.0962）和 Exp21 large（full Acc@0.5=0.4394、mAP@0.5=0.0934）均明显退化，作为负结果保留，不建议作为主线。

## Exp23 真实指标驱动融合上限（1000 图，2026-05-12）
- 目录： [exp23_metric_driven_fusion_20260512](exp23_metric_driven_fusion_20260512)
- 作用：不训练新网络，直接用 val split 的真实 `Acc@0.5` 与 `mAP@0.5` 搜索已有预测源的融合、阈值和 NMS。该实验是预测融合/后处理上限实验，不是单模型结果。
- 数据口径：直接读取 [dataset/VisDroneSplit1000Guarded](../dataset/VisDroneSplit1000Guarded)，train/val/test = 700/150/150。
- 候选来源：Exp18 `predictions/full`、Exp19 `predictions/full`、GroundingDINO baseline large `predictions`。
- 最优配置：`exp18 + gdino`，score scale 均为 1.0，threshold=0.01，NMS=0.45，topK=all。
- 主要产物：`log/exp23_metric_driven_fusion_20260512_summary.json`、三份评估报告、`log/exp23_metric_driven_fusion_20260512_search_results.csv`、`log/run_log.txt`。
- 结果摘要：融合后 full Acc@0.5=0.4650、Acc@0.75=0.3102、mAP@0.5=0.1295；test Acc@0.5=0.4802、Acc@0.75=0.3237、mAP@0.5=0.1336；full prediction boxes=122566。
- 对比结论：Exp23 证明 Exp18 与 GroundingDINO baseline 预测存在互补性，融合后指标高于单独来源。但它不代表某个模型自身能力提升，应作为后处理融合上限/诊断结果，与 Exp18/Exp21 这类单流程实验分开解读。

## Exp24 Exp23 teacher 蒸馏到 Exp22 外接 scorer（1000 图，2026-05-13）
- 目录： [exp24_teacher_distill_exp23_to_exp22_20260513](exp24_teacher_distill_exp23_to_exp22_20260513)
- 作用：把 Exp23 的融合/后处理结果作为 teacher，训练一个 Exp22 风格的外接候选/图像/语言预处理头和 score head，目标是把 Exp23 的互补候选筛选能力转成 student scorer。
- 数据口径：直接读取 [dataset/VisDroneSplit1000Guarded](../dataset/VisDroneSplit1000Guarded)，train/val/test = 700/150/150。
- 候选来源：Exp18 `predictions/full` + GroundingDINO baseline large `predictions`。
- Teacher 来源：Exp23 `predictions/full`。注意 Exp23 是融合上限，不是单模型结果。
- 训练方式：GT hard-label BCE + Exp23 teacher soft-label BCE + pair ranking loss；每轮用真实 val `0.5 * Acc@0.5 + 0.5 * mAP@0.5` 选择 checkpoint。
- 最优配置：epoch 22，threshold=0.05，NMS=0.50，val objective=0.3009。
- 主要产物：`log/exp24_teacher_distill_exp23_to_exp22_20260513_summary.json`、三份评估报告、`log/run_log.txt`、`log/exp24_teacher_distill_exp23_to_exp22_20260513_best.pt`。
- 结果摘要：full Acc@0.5=0.4667、Acc@0.75=0.3124、mAP@0.5=0.1234；test Acc@0.5=0.4823、Acc@0.75=0.3264、mAP@0.5=0.1256；full prediction boxes=123000。
- 对比结论：相比 Exp22 large（full Acc@0.5=0.3627、mAP@0.5=0.0794）明显修复；相比 Exp18 large（full Acc@0.5=0.4395、mAP@0.5=0.0962）也有提升。相比 Exp23 融合上限（full Acc@0.5=0.4650、mAP@0.5=0.1295），Acc@0.5 略高但 mAP@0.5 仍低，说明 student 学到了候选保留能力，但排序/置信度校准仍弱于直接融合。

## Exp25 Ranking Loss 蒸馏消融（1000 图，2026-05-15）
- 目录： [exp25_ranked_teacher_distill_20260515](exp25_ranked_teacher_distill_20260515)
- 作用：在 Exp24 teacher distillation 基础上加入同图正负候选 pair ranking loss，验证显式排序约束是否改善 mAP。
- 数据口径：直接读取 [dataset/VisDroneSplit1000Guarded](../dataset/VisDroneSplit1000Guarded)，train/val/test = 700/150/150。
- 候选来源：Exp18 `predictions/full` + GroundingDINO baseline large `predictions`。
- Teacher 来源：Exp23 `predictions/full`。
- 训练目标：`BCE_gt + 0.7 * BCE_teacher + 0.5 * ranking_loss`，ranking margin=0.2。
- 最优配置：epoch 9，threshold=0.05，NMS=0.50，val objective=0.2917。
- 结果摘要：full Acc@0.5=0.4661、Acc@0.75=0.3110、continuous mAP@0.5=0.0945、VOC2007 11-point mAP@0.5=0.1288；full prediction boxes=123013。
- 对比结论：VOC2007 11-point mAP 相比恢复版 Exp24 略升（0.1288 vs 0.1274），但 continuous AP 和 Acc 均下降；说明当前 ranking loss 权重/采样方式没有带来稳健主线收益，作为消融结果保留，不替代 Exp24。

## Exp26 A/B 双区域上下文细化蒸馏（1000 图，2026-05-15）
- 目录： [exp26_context_refine_dual_region_20260515](exp26_context_refine_dual_region_20260515)
- 作用：实现“两步图像区域处理”想法：先由候选框外扩得到上下文区域 A，再由候选框中心收缩得到细化区域 B，把原几何特征、A 区域统计、B 区域统计和 B-A 差异一起输入 scorer 整合模块。
- 数据口径：直接读取 [dataset/VisDroneSplit1000Guarded](../dataset/VisDroneSplit1000Guarded)，train/val/test = 700/150/150。
- 候选来源：Exp18 `predictions/full` + GroundingDINO baseline large `predictions`。
- Teacher 来源：Exp23 `predictions/full`。
- 训练目标：沿用 Exp24 的 `BCE_gt + 0.7 * BCE_teacher`；threshold=0.05，NMS=0.50。
- 双区域特征：A/context 为候选框 2.0x 外扩区域，B/refine 为候选框中心 0.7x 收缩区域；当前实现提取 RGB mean/std、brightness、contrast 和区域面积比例。
- 最优配置：epoch 6，val objective=0.2951，val Acc@0.5=0.4816，val continuous mAP@0.5=0.1085。
- 结果摘要：full Acc@0.5=0.4670、Acc@0.75=0.3124、continuous mAP@0.5=0.1015、VOC2007 11-point mAP@0.5=0.1314；full prediction boxes=122999。
- 对比结论：该想法方向合理，VOC2007 11-point mAP 相比恢复版 Exp24 小幅提升（0.1314 vs 0.1274），但 continuous AP 基本持平；当前轻量 RGB 统计版收益有限，后续更值得尝试冻结视觉 backbone 对 A/B 区域提特征。

## Exp27 语言增强 A/B 双区域上下文细化蒸馏（1000 图，2026-05-15）
- 目录： [exp27_lang_context_refine_dual_region_20260515](exp27_lang_context_refine_dual_region_20260515)
- 作用：在 Exp26 的 A/B 双区域图像统计基础上加入轻量语言因素，验证类别文本提示、语义分组和小目标先验能否进一步提升候选筛选。
- 数据口径：直接读取 [dataset/VisDroneSplit1000Guarded](../dataset/VisDroneSplit1000Guarded)，train/val/test = 700/150/150。
- 候选来源：Exp18 `predictions/full` + GroundingDINO baseline large `predictions`。
- Teacher 来源：Exp23 `predictions/full`。
- 新增特征：类别 prompt 稳定 hash embedding（12 维）、small-target prior、person/vehicle group prior、class-aware contrast interaction。
- 训练目标：沿用 `BCE_gt + 0.7 * BCE_teacher`；训练后在 val split 上搜索 threshold `[0.03,0.04,0.05,0.06,0.08]` 和 NMS `[0.45,0.50,0.55,0.60]`。
- 最优配置：epoch 11；val 后处理 threshold=0.03、NMS=0.60，val objective=0.2960。
- 结果摘要：full Acc@0.5=0.4677、Acc@0.75=0.3121、continuous mAP@0.5=0.1001、VOC2007 11-point mAP@0.5=0.1341；full prediction boxes=123825。
- 对比结论：相比 Exp26，full Acc@0.5 和 VOC2007 11-point mAP 小幅上升，但 continuous AP 下降；说明轻量语言/语义先验对历史 mAP 口径有帮助，但没有解决完整 precision-recall 排序问题。保留为语言增强消融，不建议替代主线。

## Exp28 真 query 输入 Grounding Scorer（1000 图，2026-05-15）
- 目录： [exp28_text_query_grounding_scorer_20260515](exp28_text_query_grounding_scorer_20260515)
- 作用：把模型形式从固定类别先验改成真正 `image + text query + candidate boxes -> query-box score`，用 VisDrone 类别自动生成伪语言 query，跑通第一版语言定位链路。
- 数据口径：直接读取 [dataset/VisDroneSplit1000Guarded](../dataset/VisDroneSplit1000Guarded)，train/val/test = 700/150/150。
- 文本表示：动态 text hash feature + query semantic priors。
- 结果摘要：full Acc@0.5=0.4676、Acc@0.75=0.3121、continuous mAP@0.5=0.1004、VOC2007 11-point mAP@0.5=0.1295；full prediction boxes=123823。
- 对比结论：Exp28 证明了真 query 输入链路可运行，但 hash text 表达能力有限，指标没有超过 Exp26/Exp27；作为 grounding 原型保留。

## Exp29 词表文本编码 + Grounding TopK 评估（1000 图，2026-05-15）
- 目录： [exp29_vocab_text_grounding_eval_20260515](exp29_vocab_text_grounding_eval_20260515)
- 作用：在 Exp28 基础上，用可学习词表 embedding 替代 hash text，并新增 grounding 专用评估 `query -> Top1/Top5/Top10 box Recall`。
- 数据口径：直接读取 [dataset/VisDroneSplit1000Guarded](../dataset/VisDroneSplit1000Guarded)，train/val/test = 700/150/150。
- 文本分支：query tokens -> Embedding(dim=32) -> mean pooling -> MLP(dim=32)，再与 A/B 图像区域特征和几何特征融合。
- 最优配置：epoch 12；val 后处理 threshold=0.03、NMS=0.60，val objective=0.2968。
- Detection-style 结果摘要：full Acc@0.5=0.4677、Acc@0.75=0.3123、continuous mAP@0.5=0.1025、VOC2007 11-point mAP@0.5=0.1242；full prediction boxes=123831。
- Grounding 结果摘要：full query count=5422、Recall@1=0.2979、Recall@5=0.4825、Recall@10=0.5791、mean first hit rank=20.32。
- 类别观察：car 表现最强（R@1=0.8482、R@10=0.9904），pedestrian 次之（R@1=0.4969、R@10=0.7640）；awning tricycle、motor、people 较弱。
- 修正记录：初次 TopK 评估因 GT 读取器复用 6 列 prediction parser，跳过了 5 列 GT，导致 `query_count=0`；已修复 GT reader 并用 checkpoint 重算 TopK。
- 对比结论：相比 Exp28，learnable vocab text encoder 提高了 continuous mAP（0.1025 vs 0.1004），并提供了更贴近最终目标的 grounding 指标；下一步应接入 RefDrone/AerialVG 真实语言标注或更强的预训练文本/视觉语言编码器。

## GroundingDINO baseline
- 目录： [baseline/groundingdino_base_refdrone100_recovered_20260421](baseline/groundingdino_base_refdrone100_recovered_20260421)
- 作用：恢复历史 100 图数据集后，重跑 GroundingDINO 基线，作为所有实验的参考点。
- 评估口径：box_threshold=0.20、text_threshold=0.20、nms_threshold=0.40、class-aware。
- 结果摘要：Acc@0.5=0.3449、Acc@0.75=0.2630、mAP@0.5=0.1454。
- 结论：这是历史对齐最稳定的基线，可作为后续 Exp13/14/18 的对照。

## GroundingDINO baseline on VisDroneSplit1000Guarded（2026-04-29）
- 目录： [baseline/groundingdino_base_visdrone_split1000guarded_20260429](baseline/groundingdino_base_visdrone_split1000guarded_20260429)
- 作用：在 Exp17 重建后的扩容数据集上重新测试 GroundingDINO 基线，作为新数据口径的参考点。
- 数据口径：/home/wangzhe/DroneP_VG/dataset/VisDroneSplit1000Guarded，train/val/test = 700/150/150。
- 评估口径：box_threshold=0.20、text_threshold=0.20、nms_threshold=0.40、class-aware。
- 结果摘要：Acc@0.5=0.3391（18014/53121）、Acc@0.75=0.2600（13814/53121）、mAP@0.5=0.1099。
- 主要产物：`log/evaluation_summary_groundingdino_base_visdrone_split1000guarded_20260429_class_aware.md`、`log/groundingdino_base_visdrone_split1000guarded_20260429_summary.json`。
- 结论：这份结果是扩容数据集上的新 baseline，可直接作为后续 Exp18 和其他大数据实验的对照。

## GroundingDINO baseline test only on VisDroneSplit1000Guarded（2026-04-29）
- 目录： [baseline/groundingdino_base_visdrone_split1000guarded_test_only_20260429](baseline/groundingdino_base_visdrone_split1000guarded_test_only_20260429)
- 作用：只在扩容数据集的 test split 上重跑 GroundingDINO 基线，用来和全量 1000 张结果比较差距。
- 数据口径：/home/wangzhe/DroneP_VG/dataset/VisDroneSplit1000Guarded，仅 test split。
- 评估口径：box_threshold=0.20、text_threshold=0.20、nms_threshold=0.40、class-aware。
- 结果摘要：Acc@0.5=0.3607（2876/7973）、Acc@0.75=0.2779（2216/7973）、mAP@0.5=0.1255。
- 主要产物：`log/evaluation_summary_groundingdino_test_only_20260429_class_aware.md`、`log/groundingdino_test_only_20260429_summary.json`。
- 对比结论：test-only 的 Acc@0.5 和 mAP@0.5 都高于 1000 张整体结果，说明 val/train 部分拉低了全量平均表现。


## requirements
为了以后实验代码的可读性，每一次实验建立一个单独的文件夹在experiment/路径下，结构为：
.
├── scripts（保存实验脚本，写脚本的时候尽可能得将参数写道文件中，我只需要python3 文件名.py  就可以运行）
├── README.md(写清楚实验目的和内容)
└── log（保存实验输出数据）

不要生成在fail和baseline文件夹中，也不要在我没有要求的时候删除文件

更完整的工作区级实验规范见：[WORKSPACE_EXPERIMENT_RULES.md](../WORKSPACE_EXPERIMENT_RULES.md)。
