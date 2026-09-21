# Exp44 Reward Hyperparameter Validation Protocol

## Objective

Validate whether the current closed-loop reward is a defensible local trade-off,
or select a better reward without using the test split for model selection.

Current reward:

```text
0.40 * delta_recall_05
+ 0.22 * delta_recall_075
+ 0.45 * delta_weak_class_recall_05
+ 0.15 * delta_precision_05
- 0.010 * relative_added_boxes
- 0.012 * expansion_action_count
- 0.012 * stop_indicator
```

The positive weights always sum to `1.22` during sensitivity comparisons.

## Data protocol

- Hyperparameter development uses only the 150-image validation split.
- Validation images are assigned to five deterministic, group-aware folds. The
  first token of the image stem is the sequence group, so frames from one
  sequence cannot appear in both sides of a fold.
- Broad screening uses five folds and seed 42 with 180 epochs.
- Finalists use five folds and seeds 42, 43, and 44 with 300 epochs.
- The test split is evaluated only after one challenger has been selected.
- Current and selected rewards are retrained on all validation images for each
  final seed. No choice is changed after inspecting test results.

## Search space

1. Component ablations: equal weights, recall-only, removal of every positive
   term, removal of every cost term, and removal of all costs.
2. Small-target definition: current weak-class prior, pixel area at most
   `32 x 32`, and a 50/50 hybrid.
3. One-factor curves: each positive term uses multipliers
   `0, 0.5, 0.75, 1, 1.25, 1.5, 2`, with renormalization.
4. Cost curves use seven levels including zero, the current value, and values
   above and below it.
5. Pairwise response surfaces cover recall/small, small/precision, and
   action-cost/STOP-cost interactions.
6. Forty deterministic Dirichlet samples cover the four-weight simplex.

## Selection rule (pre-registered)

1. A configuration is feasible when mean expansion actions and mean prediction
   boxes are no more than 105% of Current.
2. Among feasible configurations, maximize held-out validation Acc@0.5.
3. Differences below 0.002 are treated as tied; break ties using area-small
   recall, then mAP@0.5, then fewer actions.
4. The selected configuration must be rechecked with five folds and three
   seeds. If it does not outperform Current robustly, Current is retained.

## Reported metrics

- Acc@0.5 and Acc@0.75
- mAP@0.5 for finalists and final test models
- area-small (`area <= 32 x 32`) recall
- weak-class recall for classes `{1, 2, 3, 7, 8, 10}`
- prediction count, expansion actions per image, and STOP ratio
- paired image bootstrap confidence intervals on the final test comparison

## Interpretation boundary

The experiment can establish empirical robustness and local superiority over a
registered search space. It does not claim a mathematically global optimum.

## Reproduction

Run from the repository root. Cache construction is CPU-only; training stages
should use a CUDA device.

```bash
python experiment/exp44_reward_hyperparameter_validation_20260916/scripts/run_exp44_reward_search.py cache --rebuild-cache
CUDA_VISIBLE_DEVICES=0 python experiment/exp44_reward_hyperparameter_validation_20260916/scripts/run_exp44_reward_search.py screen --device cuda:0
CUDA_VISIBLE_DEVICES=0 python experiment/exp44_reward_hyperparameter_validation_20260916/scripts/run_exp44_reward_search.py confirm --device cuda:0
CUDA_VISIBLE_DEVICES=0 python experiment/exp44_reward_hyperparameter_validation_20260916/scripts/run_exp44_reward_search.py test --device cuda:0
python experiment/exp44_reward_hyperparameter_validation_20260916/scripts/run_exp44_reward_search.py report
```

The completed run used three parallel confirmation shards. This changes only
wall-clock time; the folds, seeds, epochs, and selection rule are identical to
the single-process `confirm` command.
