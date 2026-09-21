# Exp37 Online Fusion Calibrator

## Purpose

Exp36 证明了小目标 hard-negative reranker 与多级别名特征可以提升排序：

```text
Exp35 mAP@0.5: 0.1464
Exp36 mAP@0.5: 0.1546
```

Exp37 进一步做统一在线融合器，将多个在线可获得的信号统一训练成最终 score：

```text
Exp35 online score
Exp36 alias calibrator score
box geometry and scale prior
multi-level alias hierarchy
class confusion group features
```

## Method

训练阶段：

```text
Exp35 online val predictions
  -> Exp36 calibrator score
  -> fusion features
  -> val train / val holdout split
  -> hard-negative weighted fusion model
  -> alpha/threshold search on val holdout
```

测试阶段：

```text
Exp35 online test predictions
  -> Exp36 calibrator score
  -> Exp37 fusion score
  -> final test predictions
```

Test labels are used only for final evaluation.

## Run

```bash
tmux new -s exp37_online_fusion
CUDA_VISIBLE_DEVICES=2 python experiment/exp37_online_fusion_calibrator_20260712/scripts/run_exp37_online_fusion_calibrator_20260712.py
```

## Result

Full test:

```text
Exp35:
  prediction boxes: 52872
  Acc@0.5:  0.5374
  Acc@0.75: 0.3463
  mAP@0.5:  0.1464

Exp36:
  prediction boxes: 52872
  Acc@0.5:  0.5374
  Acc@0.75: 0.3472
  mAP@0.5:  0.1546

Exp37:
  prediction boxes: 52872
  Acc@0.5:  0.5374
  Acc@0.75: 0.3472
  mAP@0.5:  0.1546
```

The learned fusion model did not outperform the Exp36 alias calibrator. The validation search selected `alpha=1.00`, which means the best online fusion setting falls back to the Exp36 calibrated score.

## Interpretation

This experiment is still useful: it shows that the currently saved prediction text files do not preserve enough source-level information for a stronger fusion module. To make fusion genuinely stronger, the online inference stage should export richer candidate metadata, especially:

```text
GroundingDINO raw score
prompt/source name
base prompt vs alias prompt
Exp30 scorer probability
Exp36 calibrator probability
geometry and hierarchy features
```
