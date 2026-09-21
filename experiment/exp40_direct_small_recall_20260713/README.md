# Exp40 Direct Small-object Recall

## Purpose

Exp40 directly tests whether `Acc@0.5` is limited by insufficient small-object candidate recall. Unlike Exp39 crop refine, this experiment does not run a second local crop pass. It expands the online candidate pool itself:

```text
lower base GroundingDINO thresholds
lower small-object alias thresholds
add more small-object alias prompts
increase final TopK for small classes
```

The goal is to recover more GT boxes at IoU >= 0.5.

## Main Changes

```text
base_box_threshold: 0.20 -> 0.16
base_text_threshold: 0.20 -> 0.18
small_alias_box_threshold: 0.03 -> 0.015
small_alias_text_threshold: 0.15 -> 0.10
```

Added alias prompts for:

```text
pedestrian / person / walking person
people / group of people / crowd
bicycle / bike / cyclist
tricycle / small tricycle / three wheel vehicle
covered tricycle / awning tricycle / canopy tricycle
motorcycle / motorbike / scooter / motor
```

For speed, all alias prompts are sent to GroundingDINO in one batched prompt call instead of one call per prompt.

## Results

### Validation probe, first 50 val images

```text
Exp38 baseline:
  boxes:    19997
  Acc@0.5:  0.5131
  Acc@0.75: 0.3488
  mAP@0.5:  0.1335

Exp39 crop refine:
  boxes:    23334
  Acc@0.5:  0.5176
  Acc@0.75: 0.3464
  mAP@0.5:  0.1294

Exp40 direct recall:
  boxes:    26825
  Acc@0.5:  0.5311
  Acc@0.75: 0.3497
  mAP@0.5:  0.1363
```

### Full test

```text
Exp38 baseline:
  boxes:    52872
  Acc@0.5:  0.5374
  Acc@0.75: 0.3463
  mAP@0.5:  0.1464

Exp40 direct recall:
  boxes:    73992
  Acc@0.5:  0.5477
  Acc@0.75: 0.3403
  mAP@0.5:  0.1334
```

Exp40 adds 82 TP@0.5 on the full test set:

```text
TP@0.5: 4285 -> 4367
```

## Per-class Test Recall@0.5

```text
class 1 pedestrian:       0.4985 -> 0.4728
class 2 people:           0.4968 -> 0.4581
class 3 bicycle:          0.4211 -> 0.4413
class 4 car:              0.6597 -> 0.6959
class 5 van:              0.1098 -> 0.0963
class 6 truck:            0.2835 -> 0.2629
class 7 tricycle:         0.7209 -> 0.8372
class 8 awning tricycle:  0.4630 -> 0.6019
class 9 bus:              0.1462 -> 0.1154
class 10 motor:           0.5681 -> 0.5913
```

## Interpretation

Direct small-object recall is more effective for `Acc@0.5` than Exp39 crop refine. It improves full-test `Acc@0.5` by `+0.0103`, and the strongest gains are on small vehicle classes:

```text
tricycle:        +0.1163 Recall@0.5
awning tricycle: +0.1389 Recall@0.5
motor:           +0.0232 Recall@0.5
bicycle:         +0.0202 Recall@0.5
```

However, the current broad alias expansion hurts person classes and increases prediction boxes substantially. The next step should not simply add more prompts. It should use class-specific recall control:

```text
keep strong recall expansion for tricycle / awning tricycle / motor / bicycle
restore stricter thresholds or prompt sets for pedestrian / people
add a recall-aware filtering stage to reduce noisy alias boxes
```
