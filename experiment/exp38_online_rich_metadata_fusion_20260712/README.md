# Exp38 Online Rich Metadata Fusion

## Purpose

Exp37 说明旧的预测 txt 只有 `class_id x1 y1 x2 y2 final_score`，缺少来源与中间分数，融合器无法超过 Exp36。

Exp38 在在线推理阶段补充 richer metadata：

```text
GroundingDINO raw score
source prompt / source type
source class id
query match score
small-area prior bonus
final score
area ratio
rank
```

然后基于这些在线可获得信息重新训练融合器。

## Outputs

For each split:

```text
log/predictions/<split>/<stem>.txt
log/metadata/<split>/<stem>.jsonl
```

The txt keeps compatibility with existing evaluation scripts. The JSONL is used by the rich fusion model.

## Run

Generate metadata:

```bash
tmux new -s exp38_metadata
CUDA_VISIBLE_DEVICES=2 python experiment/exp38_online_rich_metadata_fusion_20260712/scripts/run_exp38_online_metadata_inference_20260712.py --dataset-split val
CUDA_VISIBLE_DEVICES=2 python experiment/exp38_online_rich_metadata_fusion_20260712/scripts/run_exp38_online_metadata_inference_20260712.py --dataset-split test
```

Train fusion:

```bash
python experiment/exp38_online_rich_metadata_fusion_20260712/scripts/run_exp38_rich_metadata_fusion_20260712.py
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

Exp38:
  prediction boxes: 52872
  Acc@0.5:  0.5374
  Acc@0.75: 0.3491
  mAP@0.5:  0.1690
```

The prediction count is unchanged. The gain comes from better confidence ordering using richer online metadata.

## Interpretation

Exp37 could not beat Exp36 because the saved txt predictions did not preserve source-level information. Exp38 fixes that by writing JSONL metadata during online inference. Once `gdino_score`, `source_prompt`, `source_type`, `source_class_id`, `query_match_score`, and `small_area_bonus` are available, the fusion model improves mAP substantially.
