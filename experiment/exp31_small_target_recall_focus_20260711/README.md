# Exp31 Small Target Recall Focus

## Purpose

Exp30 显著提升了整体排序质量，但弱小目标类别仍然很差，尤其：

- `people`
- `tricycle`
- `awning tricycle`
- `motor`

本实验基于 Exp30，目标不再单纯追求 full mAP，而是优先提高弱小类的 grounding 命中率，尤其 `Recall@1` 和 `Recall@5`。

## Method

相对 Exp30 的改动：

1. 弱小类训练样本加权：
   - small target class 正样本权重更高。
   - weak small class 正样本权重最高。
   - weak small class 负样本也轻微加权，压制误排。

2. 弱小类 ranking loss：
   - batch ranking 优先从 `people / tricycle / awning tricycle / motor` 中采样正负对。
   - 目标是把弱类真框排到更靠前的位置。

3. 小目标验证目标：
   - checkpoint 选择不再只看 detection objective。
   - 新 objective 混入 weak small class `Recall@1`。

4. 弱类 canonical query 改写：
   - `people -> group of people`
   - `tricycle -> small tricycle`
   - `awning tricycle -> covered tricycle`
   - `motor -> small motorcycle`

## Expected Result

主比较对象是 Exp30：

- full mAP@0.5: `0.1521`
- full R@1: `0.3890`
- weak classes:
  - people R@1: `0.1190`
  - tricycle R@1: `0.2577`
  - awning tricycle R@1: `0.0326`
  - motor R@1: `0.0189`

Exp31 允许 full mAP 小幅回落，但希望 weak small class 的 R@1/R@5 明显提升。

## Run

```bash
tmux new -s exp31_small_target_recall_focus
CUDA_VISIBLE_DEVICES=2 python experiment/exp31_small_target_recall_focus_20260711/scripts/run_exp31_small_target_recall_focus_20260711.py
```
