# Exp39 Online Crop Refine

## Purpose

Exp39 targets `Acc@0.5`, not only mAP ranking. The hypothesis is that UAV small objects are often too small in the full image, so a second online localization pass on enlarged local crops may recover missed or poorly localized small-object candidates.

## Method

The pipeline inherits Exp38 online candidate generation and metadata writing, then adds a lightweight crop-refine branch:

```text
full-image GroundingDINO base/alias candidates
-> select top-k small-object seeds per class
-> crop around each seed with enlarged context
-> run GroundingDINO again on the crop using the seed prompt
-> map refined crop boxes back to the original image
-> merge original + crop_refine candidates
-> query-aware scoring, NMS, TopK
```

Current crop-refine config:

```text
classes: pedestrian, people, bicycle, tricycle, awning tricycle, motor
topk_per_class: 2
crop_scale: 2.4
min_crop_size: 96
box_threshold: 0.03
text_threshold: 0.15
```

## Run

Small validation probe:

```bash
tmux new -s exp39_val50
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/mplconfig CUDA_VISIBLE_DEVICES=2 \
python experiment/exp39_online_crop_refine_20260713/scripts/run_exp39_online_crop_refine_20260713.py \
  --dataset-split val --limit 50
```

## Validation Probe Result

Same first 50 validation images:

```text
Exp38 online baseline:
  prediction boxes: 19997
  Acc@0.5:  0.5131
  Acc@0.75: 0.3488
  mAP@0.5:  0.1335

Exp39 crop refine:
  prediction boxes: 23334
  Acc@0.5:  0.5176
  Acc@0.75: 0.3464
  mAP@0.5:  0.1294
```

Crop-refine source statistics on val50:

```text
base boxes:        8380
alias boxes:       10824
crop_refine boxes: 4130
```

## Interpretation

Crop refine brings a small `Acc@0.5` gain on val50: `+0.0045`, or 15 additional TP@0.5 over 3329 GT boxes. However, it also increases prediction boxes by 3337 and reduces mAP. This means the current strategy is directionally useful but too conservative and noisy for the target of reaching a usable `Acc@0.5` level.

The main limitation is that crop refine only improves regions around already high-scoring candidates. If the full-image detector does not produce a reasonable seed near a missed small object, this branch cannot recover it.

## Next Recommendation

Do not spend a full test run on the current Exp39 configuration yet. The next stronger direction should be online tiled small-object recall or an Acc@0.5-oriented final selector:

```text
Exp40 option A: 2x2 overlap tiled inference for small-object prompts
Exp40 option B: group-wise final selector trained directly for TP@0.5
```
