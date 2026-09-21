# Query-aware Prompt Ablation

- split: `val`
- image_count: `150`

| Variant | Candidates | Acc@0.5 | Acc@0.75 | mAP@0.5 | Cand Recall@0.5 | Small Cand Recall@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed10 | 7476 | 0.3719 | 0.2776 | 0.1065 | 0.3744 | 0.1579 |
| query_only | 9967 | 0.3691 | 0.2543 | 0.0891 | 0.3717 | 0.1556 |
| query_alias | 23633 | 0.4178 | 0.2580 | 0.0863 | 0.4202 | 0.3043 |
| query_alias_fallback | 26091 | 0.4504 | 0.3072 | 0.1062 | 0.4527 | 0.3187 |
