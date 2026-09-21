# Closed-loop RL Summary

- checkpoint: `RL_loop/outputs/closed_loop_rl/rl_policy_best.pt`

## Train
- split: `val`
- image_count: `150`
- best: `{'epoch': 274.0, 'val_mean_reward': 0.014094577170908451, 'loss': -0.01404557190835476}`
- oracle_action_counts: `{'STOP': 135, 'USE_QUERY_ONLY': 4, 'ADD_QUERY_ALIAS': 7, 'ADD_QUERY_ALIAS_FALLBACK': 4}`

## Test
- action_counts: `{'STOP': 142, 'USE_QUERY_ONLY': 0, 'ADD_QUERY_ALIAS': 6, 'ADD_QUERY_ALIAS_FALLBACK': 2}`

| Method | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 |
| --- | ---: | ---: | ---: | ---: |
| closed_loop_rl | 8329 | 0.4168 | 0.3024 | 0.1265 |
