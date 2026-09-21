# Qwen dynamic RL performance-cost comparison

| Method | Boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 | Small Recall | Actions/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original query | 22298 | 0.4651 | 0.3131 | 0.1343 | 0.3291 | 0.000 |
| All Qwen mapped actions | 50895 | 0.5180 | 0.3336 | 0.1465 | 0.3929 | 2.460 |
| Qwen dynamic RL | 35869 | 0.5004 | 0.3248 | 0.1424 | 0.3732 | 0.953 |

- RL retains 66.8% of the full Qwen Acc@0.5 gain over query-only.
- RL reduces actions by 61.2% and final boxes by 29.5% versus full Qwen expansion.
