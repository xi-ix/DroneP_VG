# Latest Test Summary

| Method | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 | TP@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 5801 | 0.3607 | 0.2779 | 0.1255 | 2876 |
| My model | 52872 | 0.5374 | 0.3491 | 0.1690 | 4285 |

- alpha: `0.15`
- alias checkpoint: `experiment/exp36_small_target_alias_rerank_20260711/log/exp36_small_target_alias_rerank_20260711_best.pt`
- fusion checkpoint: `experiment/exp38_online_rich_metadata_fusion_20260712/log/exp38_rich_metadata_fusion_20260712_best.pt`
- model predictions: `outputs/latest_test/predictions`
- random samples: `outputs/latest_test/samples`
