# Exp15 语言注意力层（基于 Exp14）

## 目的
在 Exp14 外接校准器基础上加入语言注意力层（language-conditioned attention），观察指标变化。

## 目录结构
- scripts: 实验脚本
- log: 运行日志与评估产物
- README.md: 实验说明

## 核心改动
- 在 `HybridExternalCalibrator` 前增加语言门控：
  - 输入文本由 `LANGUAGE_PROMPT` 指定。
  - 将 prompt 经过 token hash + embedding + MLP 得到门控向量。
  - 对候选特征做 `x * (1 + scale * gate)` 后再送入原 MLP。

## 运行
python3 scripts/run_exp15_lang_attention_20260422.py

## 主要输出
- log/run_log.txt
- log/exp15_lang_attention_20260422_best.pt
- log/exp15_lang_attention_20260422_summary.json
- log/evaluation_summary_exp15_lang_attention_trainval_class_aware.md
- log/evaluation_summary_exp15_lang_attention_test_class_aware.md
- log/evaluation_summary_exp15_lang_attention_full_class_aware.md
