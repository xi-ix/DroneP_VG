# Exp28 Text Query Grounding Scorer

## Purpose

首次将模型形式改为真正的 `image + text query + candidate boxes -> query-box matching score` 架构。之前实验（Exp25-27）的语言信号均以固定伪先验方式注入，Exp28 是第一次让文本 query 作为动态输入参与打分。

## Data

- 数据根目录：`dataset/VisDroneSplit1000Guarded`
- split：复用 `experiment/exp17_large_dataset_guard_20260422/log/split_manifest.tsv`
- train/val/test：700 / 150 / 150

## Inputs

- Exp18 candidates：`experiment/exp18_exp14_method_large_data_20260423/log/predictions/full`
- GroundingDINO candidates：`experiment/baseline/groundingdino_base_visdrone_split1000guarded_20260429/log/predictions`
- Teacher：`experiment/exp23_metric_driven_fusion_20260512/log/predictions/full`

## Method

### 模型架构：DistillScorer

```
input_feature -> Linear(input_dim, 96) -> LayerNorm -> ReLU
             -> Linear(96, 96) -> LayerNorm -> ReLU
             -> Linear(96, 48) -> ReLU
             -> Linear(48, 1)
```

输入特征拼接结构：

```text
[box geometry (11)] + [A context RGB stats (8)] + [B refined RGB stats (8)] + [B-A delta (8)] + [area ratios (2)]
+ [query semantic priors (7)] + [hashed text embedding (24)]
```

### 文本表示

1. **Hashed text embedding（24 维）**：将 query 文本分词后，对每个 token 计算 bucket hash（`sum(ord(ch)) % 24`）和 sign hash（`sum(ord(ch) * (idx+1)) % 2 -> ±1`），归一化后得到 24 维向量。
2. **Query semantic priors（7 维）**：
   - `is_small_target`：query 中是否含小目标关键词
   - `is_vehicle`：query 中是否含车辆关键词
   - `is_person`：query 中是否含行人关键词
   - `compactness_prior`：小目标先验 × 面积约束
   - `contrast_gain × is_small_target`：细化区域对比度增益与小目标的交互
   - `context_contrast × is_vehicle`：上下文对比度与车辆的交互
   - `refine_contrast × is_person`：细化对比度与行人的交互

### 推理流程

对每张图，用 10 个类别的 canonical query（如 `pedestrian`, `car`, `motor` 等）分别对所有候选框打分，按类别过滤后做 class-aware NMS，最后合并所有类别的预测。

### 训练目标

```text
loss = BCE_with_logits(gt_label) + 0.7 * BCE_with_logits(teacher_soft_label)
```

- GT hard label：候选框与同类别 GT IoU >= 0.5 为 1，否则为 0
- Teacher soft label：候选框与 Exp23 teacher 预测 IoU >= 0.7 时给高置信度，IoU >= 0.5 时按比例缩放，< 0.5 时为 0
- 仅当 query class == 候选框 class 时才计算 label，跨类 query-box pair 的 label 为 0

### Query Templates

每个类别 3 个模板（共 10 类 × 3 = 30 个 query），训练时全部展开：

| class_id | canonical query | template 2 | template 3 |
| --- | --- | --- | --- |
| 1 | pedestrian | person | small person in drone image |
| 2 | people | standing person | group of people |
| 3 | bicycle | small bicycle | bicycle rider |
| 4 | car | small car | vehicle on road |
| 5 | van | medium vehicle | white van |
| 6 | truck | large vehicle | cargo truck |
| 7 | tricycle | three wheel vehicle | small tricycle |
| 8 | awning tricycle | covered tricycle | covered three wheel vehicle |
| 9 | bus | large passenger vehicle | bus on road |
| 10 | motor | motorcycle | small motor vehicle |

推理时仅使用 canonical query（每类 1 个）。

### 后处理搜索

在 val 集上搜索最优 threshold/NMS 组合：
- threshold grid：[0.03, 0.04, 0.05, 0.06, 0.08]
- NMS grid：[0.45, 0.50, 0.55, 0.60]
- objective：`0.5 × Acc@0.5 + 0.5 × mAP@0.5`

## Result

Best checkpoint:

- epoch: 22
- val objective: 0.2956
- val Acc@0.5: 0.4811
- val continuous mAP@0.5: 0.1100

Best val postprocess:

- threshold: 0.03
- NMS: 0.60
- val objective: 0.2958

Detection-style full metrics:

| metric | value |
| --- | ---: |
| Acc@0.5 | 0.4676 |
| Acc@0.75 | 0.3121 |
| continuous mAP@0.5 | 0.1004 |
| VOC2007 11-point mAP@0.5 | 0.1295 |
| prediction boxes | 123823 |

## Training Curve Observations

- 训练损失从 0.158 快速下降到 ~0.1025，在第 5 epoch 后趋于平稳
- val Acc@0.5 在所有 epoch 几乎不变（~0.481），说明分类能力主要来自视觉特征
- val mAP@0.5 在 0.098~0.110 之间波动，best epoch 22 达到 0.110
- 整体训练信号偏弱，hash text 表达能力有限

## Conclusion

- **语言输入链路跑通**：这是第一个将 text query 作为模型动态输入的实验，推理时可根据不同 query 对同一候选框给出不同分数。
- **Hash text 表达能力弱**：24 维 hash embedding 区分度有限，不同类别的 query hash 可能碰撞；指标未超过 Exp26/Exp27 的固定伪语言先验方案。
- **下一步**：需要更强的文本编码方式（如可学习词表 embedding 或预训练语言模型），而非手工 hash。Exp29 在此基础上进行了改进。
