# LLM Dynamic Prompt RL Summary

- LLM: `Qwen2.5-VL-3B-Instruct`
- policy: `state encoder + prompt action encoder + scorer`
- loop_rounds: `3`
- action_counts: `{'person': 48, 'bicycle': 7, 'people': 28, 'STOP': 157, 'gdino_base': 57, 'tricycle': 1, 'motorcycle': 2}`

| Method | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 |
| --- | ---: | ---: | ---: | ---: |
| query_base_exp38_scored | 22298 | 0.4651 | 0.3131 | 0.1343 |
| llm_dynamic_38+RL_loop | 35869 | 0.5004 | 0.3248 | 0.1424 |
