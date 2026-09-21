# Exp43 RL Action-policy Ablation

## Purpose

Test whether the closed-loop improvement comes from the learned action policy rather than merely executing more candidate-generation actions.

## Compared methods

- **No expansion**: use only first-round query-aware candidates.
- **All expansion**: execute all 11 non-STOP actions in the fixed action space.
- **Random action**: sample the same number of non-STOP actions per image as Fixed-action RL; report 5-seed mean and standard deviation.
- **Prompt-rank**: select unique candidate sources in Qwen output order under the same per-image action budget as Fixed-action RL.
- **Fixed-action RL**: replay the saved decisions from the fixed 12-action policy.
- **Qwen dynamic RL**: replay the saved Qwen dynamic action-pool decisions.

All methods use the same `VisDroneSplit1000Guarded/test` split, first-round candidates, candidate deduplication, Exp38 rescoring, and class-aware evaluator.

## Metrics

- `Acc@0.5`, `Acc@0.75`, `mAP@0.5`
- prediction boxes, average non-STOP actions per image, STOP ratio
- small candidate oracle recall at IoU 0.5 before Exp38 rescoring
- small-object final recall at IoU 0.5 after Exp38 rescoring
- weak-class candidate/final recall for classes `{pedestrian, people, bicycle, tricycle, awning tricycle, motor}`

Small objects follow the COCO pixel-area definition: GT box area `<= 32 x 32` pixels.

## Run

```bash
python experiment/exp43_rl_action_policy_ablation_20260911/scripts/run_exp43_rl_action_policy_ablation_20260911.py
```

The run uses cached candidates and saved RL traces. It does not execute GroundingDINO or Qwen inference.

## Outputs

- `log/exp43_rl_action_policy_ablation_20260911_summary.json`
- `log/evaluation_summary_test_class_aware.md`
- `log/metrics.csv`
- `log/ablation_trace.json`
- `log/predictions/`
- `log/figures/main_metrics_and_candidates.png`
- `log/figures/round_actions_and_candidates.png`
- `log/figures/rl_action_class_distribution.png`

The measured results and conclusion are written into the generated Markdown report after running the experiment.
