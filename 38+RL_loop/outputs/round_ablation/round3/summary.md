# 38 + RL_loop 3-round Summary

- method: `3-round Exp38 rescoring over RL inference-loop candidates`
- checkpoint: `/home/wangzhe/DroneP_VG/38+RL_loop/outputs/rl_policy_exp38_reward.pt`
- device: `cuda:0`
- action_counts: `{'STOP': 97, 'ADD_GDINO_BASE': 0, 'ADD_PERSON': 112, 'ADD_PEOPLE': 14, 'ADD_GROUP_OF_PEOPLE': 1, 'ADD_TRICYCLE': 0, 'ADD_COVERED_TRICYCLE': 7, 'ADD_AWNING_TRICYCLE': 0, 'ADD_MOTORCYCLE': 2, 'ADD_MOTORBIKE': 49, 'ADD_SCOOTER': 0, 'ADD_BICYCLE': 18}`
- loop_rounds: `3`
- closed_loop_inference_sec: `13.9735`
- timing_scope: cached candidates + policy + merge/dedup + Exp38 rescoring; excludes GroundingDINO
- exp38_match_ratio: `0.6113`

| Method | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 |
| --- | ---: | ---: | ---: | ---: |
| query_base_exp38_scored | 22298 | 0.4651 | 0.3131 | 0.1343 |
| rl_plain | 8329 | 0.4168 | 0.3024 | 0.1265 |
| 38+RL_loop | 40904 | 0.5293 | 0.3310 | 0.1397 |
| exp38_full_reference | 52872 | 0.5374 | 0.3463 | 0.1464 |
