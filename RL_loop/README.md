# RL_loop 实验目录

本目录用于放置“强化学习 + 闭环推理”方向的新实验，先从最小可验证的 query-aware 候选扩充做起。

## 当前实验：Query-aware Prompt Ablation

实验目标：

```text
验证输入语言 query 是否应该参与候选集扩充；
比较固定 10 类 prompt 与 query-aware prompt 扩展的候选召回和最终检测指标。
```

对照组：

```text
fixed10
    只使用 VisDrone 固定 10 类 prompt。

query_only
    只使用用户原始 query。

query_alias
    使用用户 query + 词表解析得到的同义词/易混类别 prompt。

query_alias_fallback
    使用用户 query + alias prompt + 固定 10 类 fallback prompt。
```

这里的 query 解析是手写词表规则，不引入额外语言模型：

```text
query -> 小写化/去标点/分词 -> 类别关键词匹配 -> target class -> prompt set
```

## 运行方式

先跑小样本：

```bash
python RL_loop/query_prompt_ablation.py --split test --limit 5
```

跑完整测试集：

```bash
python RL_loop/query_prompt_ablation.py --split test --limit 0
```

只检查每类 query 会展开成哪些 prompt，不运行 GroundingDINO：

```bash
python RL_loop/query_prompt_ablation.py --dry-run
```

## 输出

```text
RL_loop/outputs/query_prompt_ablation/
    predictions/
        fixed10/
        query_only/
        query_alias/
        query_alias_fallback/
    summary.json
    summary.md
```

`summary.md` 会给出每个变体的：

```text
candidate_count
Acc@0.5
Acc@0.75
mAP@0.5
candidate_recall@0.5
small_candidate_recall@0.5
```

## 后续闭环/RL 接入

如果 query-aware prompt 扩展相比 fixed10 有收益，下一步在本目录继续加入：

```text
1. feedback state extractor
2. rule-based closed-loop controller
3. RL policy controller
4. closed-loop inference evaluator
```

推荐顺序：

```text
query-aware prompt ablation
-> rule-based closed-loop inference
-> RL policy learning for action selection
```
