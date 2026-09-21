# 38 + RL_loop 2-round Summary

- method: `2-round Exp38 rescoring over RL inference-loop candidates`
- checkpoint: `/home/wangzhe/DroneP_VG/38+RL_loop/outputs/rl_policy_exp38_reward.pt`
- device: `cuda:0`
- action_counts: `{'STOP': 42, 'ADD_GDINO_BASE': 0, 'ADD_PERSON': 87, 'ADD_PEOPLE': 13, 'ADD_GROUP_OF_PEOPLE': 0, 'ADD_TRICYCLE': 0, 'ADD_COVERED_TRICYCLE': 7, 'ADD_AWNING_TRICYCLE': 0, 'ADD_MOTORCYCLE': 1, 'ADD_MOTORBIKE': 0, 'ADD_SCOOTER': 0, 'ADD_BICYCLE': 0}`
- loop_rounds: `2`
- closed_loop_inference_sec: `11.2665`
- timing_scope: cached candidates + policy + merge/dedup + Exp38 rescoring; excludes GroundingDINO
- exp38_match_ratio: `0.5759`

| Method | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 |
| --- | ---: | ---: | ---: | ---: |
| query_base_exp38_scored | 22298 | 0.4651 | 0.3131 | 0.1343 |
| rl_plain | 8329 | 0.4168 | 0.3024 | 0.1265 |
| 38+RL_loop | 37487 | 0.5139 | 0.3253 | 0.1386 |
| exp38_full_reference | 52872 | 0.5374 | 0.3463 | 0.1464 |
