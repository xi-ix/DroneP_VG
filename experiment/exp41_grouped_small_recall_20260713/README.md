# Exp41 Grouped Small-object Recall

## Purpose

Exp40 showed that direct small-object recall improves `Acc@0.5`, but broad alias expansion adds many noisy boxes and hurts `Acc@0.75` / mAP. Exp41 tests a grouped recall strategy:

```text
keep strong recall for small vehicle classes
curate person / people aliases
keep batched alias prompting to preserve GroundingDINO candidate behavior
```

An early split-threshold version was rejected because it reduced alias candidates too aggressively and dropped val50 `Acc@0.5` to `0.5044`.

The final Exp41 version uses:

```text
person aliases:
  pedestrian, person, people, group of people

small-vehicle aliases:
  bicycle, bike, cyclist
  tricycle, small tricycle, three wheel vehicle
  covered tricycle, awning tricycle, canopy tricycle
  motorcycle, motorbike, scooter, motor
  bicycle, tricycle for motor recovery

alias inference:
  one batched GroundingDINO call
  box_threshold = 0.015
  text_threshold = 0.10
```

## Results

### Validation probe, first 50 val images

```text
Exp38 baseline:
  boxes:    19997
  Acc@0.5:  0.5131
  Acc@0.75: 0.3488
  mAP@0.5:  0.1335

Exp40 direct recall:
  boxes:    26825
  Acc@0.5:  0.5311
  Acc@0.75: 0.3497
  mAP@0.5:  0.1363

Exp41 grouped recall:
  boxes:    26773
  Acc@0.5:  0.5335
  Acc@0.75: 0.3488
  mAP@0.5:  0.1380
```

### Full test

```text
Exp38 baseline:
  boxes:    52872
  Acc@0.5:  0.5374
  Acc@0.75: 0.3463
  mAP@0.5:  0.1464
  TP@0.5:   4285

Exp40 direct recall:
  boxes:    73992
  Acc@0.5:  0.5477
  Acc@0.75: 0.3403
  mAP@0.5:  0.1334
  TP@0.5:   4367

Exp41 grouped recall:
  boxes:    73400
  Acc@0.5:  0.5477
  Acc@0.75: 0.3399
  mAP@0.5:  0.1350
  TP@0.5:   4367
```

## Per-class Test Recall@0.5

```text
class 1 pedestrian:       Exp40 0.4728 -> Exp41 0.4744
class 2 people:           Exp40 0.4581 -> Exp41 0.4548
class 3 bicycle:          Exp40 0.4413 -> Exp41 0.4453
class 4 car:              Exp40 0.6959 -> Exp41 0.6959
class 5 van:              Exp40 0.0963 -> Exp41 0.0963
class 6 truck:            Exp40 0.2629 -> Exp41 0.2629
class 7 tricycle:         Exp40 0.8372 -> Exp41 0.8256
class 8 awning tricycle:  Exp40 0.6019 -> Exp41 0.6481
class 9 bus:              Exp40 0.1154 -> Exp41 0.1154
class 10 motor:           Exp40 0.5913 -> Exp41 0.5820
```

## Interpretation

Exp41 slightly improves val50 over Exp40 and gives a small mAP recovery on full test, but it does not improve full-test `Acc@0.5` beyond Exp40. The best current full-test `Acc@0.5` remains:

```text
0.5477
```

This confirms that direct recall expansion is useful, but prompt grouping alone is not enough to reach `0.60+`. The remaining gap likely requires either:

```text
1. class-specific score calibration / filtering after recall expansion
2. tiled inference for true missed small objects
3. training an Acc@0.5-oriented candidate selector on the expanded Exp40/Exp41 candidate pool
```

Recommended next step: use Exp40/Exp41 expanded candidates as the recall pool, then train a selector or calibrator that suppresses noisy alias boxes while preserving the additional TP@0.5.
