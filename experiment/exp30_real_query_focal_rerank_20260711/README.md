# Exp30 Real Query Focal Rerank

## Purpose

在 Exp29 的 `image + text query + candidate boxes -> query-box score` 链路上做一次更贴近最终 grounding 目标的改进实验。

核心目标：

1. 保持 Exp18 + GroundingDINO baseline 的高召回候选源。
2. 从 RefDrone / AerialVG 真实语言标注中抽取短语，增强 query 模板和词表。
3. 扩展 A/B 区域图像特征为多尺度上下文与细化区域统计。
4. 在 BCE + teacher distillation 之外加入 focal loss 和 batch ranking loss，改善难例与排序。

## Data

- 主训练/评估数据：`dataset/VisDroneSplit1000Guarded`
- split：复用 `experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv`
- 真实语言增强来源：
  - `dataset/RefDrone/RefDrone_train_mdetr.json`
  - `dataset/RefDrone/RefDrone_val_mdetr.json`
  - `dataset/RefDrone/RefDrone_test_mdetr.json`
  - `dataset/AerialVG/annotation/vg_*_odvg.jsonl`

## Inputs

- Exp18 candidates：`experiment/exp18_exp14_method_large_data_20260423/log/predictions/full`
- GroundingDINO candidates：`experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions`
- Teacher：`experiment/exp23_metric_driven_fusion_20260512/log/predictions/full`

## Method

基于 Exp29 的 vocab text scorer，主要改动如下：

- Query 增强：每类保留原有模板，并从 RefDrone/AerialVG 真实 caption 或 phrase 中按关键词抽取最多 3 条额外表达式。
- 文本分支：`query tokens -> Embedding(dim=32) -> mean pooling -> MLP(dim=32)`。
- 视觉数值分支：box geometry + A/B 区域 RGB/gray 统计 + 外层 context / 内层 refine 统计 + 多尺度 delta。
- 损失函数：

```text
loss = BCE_gt + 0.7 * BCE_teacher + 0.35 * focal_BCE + 0.15 * batch_ranking_loss
```

- 后处理搜索：
  - threshold：`[0.02, 0.03, 0.04, 0.05, 0.06, 0.08]`
  - NMS：`[0.45, 0.50, 0.55, 0.60, 0.65, 0.70]`
  - objective：`0.5 * Acc@0.5 + 0.5 * mAP@0.5`

## Run

```bash
tmux new -s exp30_real_query_focal_rerank
python experiment/exp30_real_query_focal_rerank_20260711/scripts/run_exp30_real_query_focal_rerank_20260711.py
```

脚本会写入：

- `log/run_log.txt`
- `log/exp30_real_query_focal_rerank_20260711_summary.json`
- `log/evaluation_summary_exp30_real_query_focal_rerank_*_class_aware.md`
- `log/grounding_eval_full_topk.json`
- `log/grounding_eval_full_topk.md`

## Expected Comparison

主对比对象是 Exp29：

- Detection-style full：`Acc@0.5=0.4677`，`Acc@0.75=0.3123`，`continuous mAP@0.5=0.1025`
- Grounding full：`R@1=0.2979`，`R@5=0.4825`，`R@10=0.5791`

本实验优先观察 grounding TopK 和 mAP 排序是否改善；如果 Acc 小幅波动但 TopK/R@1 或 mAP 提升，说明 query-aware reranking 有继续推进价值。
