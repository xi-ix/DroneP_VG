# Exp43: Exp38 元数据特征消融

## 目的

验证 Exp38 的排序收益来自 rich metadata，而不是候选集合变化、模型参数量变化或测试集调参。

## 对照设计

六组实验使用完全相同的候选框、训练/验证/测试划分、监督标签、样本权重、MLP 结构和参数量。所有配置保留同一套 68 维输入槽位，未启用字段置零：

1. `Raw score`：仅 `final_score`。
2. `+ geometry`：加入面积、宽高、宽高比、中心和 rank。
3. `+ query`：加入 `query_match_score`。
4. `+ source`：加入 `source_prompt/source_type/source_class_id`。
5. `+ small prior`：加入 `small_area_bonus`、类别/尺寸先验。
6. `Full Exp38`：加入其余 `gdino_score`、Exp36 score 与分数交互项，使用完整 rich metadata。

每个配置默认运行 3 个随机种子。融合系数 alpha 只在 val holdout 上选择，test 仅评测一次。各组最终预测应始终保持 52,872 个候选框。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 python experiment/exp43_exp38_metadata_ablation_20260911/scripts/run_exp43_exp38_metadata_ablation_20260911.py
```

快速检查可缩短训练：

```bash
CUDA_VISIBLE_DEVICES=0 python experiment/exp43_exp38_metadata_ablation_20260911/scripts/run_exp43_exp38_metadata_ablation_20260911.py --seeds 42 --epochs 2
```

## 输出

- `log/exp43_exp38_metadata_ablation_20260911_summary.json`：逐 seed 原始指标及均值/标准差。
- `Exp38元数据特征消融结果.md`：可直接用于论文或答辩整理的结果表。
- `log/predictions/<config>/seed_<seed>/test`：固定候选集合上的排序结果。
