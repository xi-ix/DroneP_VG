# Exp16 注意力 + 大模型 + MLP 打分层（基于 Exp14）

## 目的
在与 Exp14 最优后处理完全对齐的前提下，尝试把结构改成“注意力 + 大模型 + MLP 打分层”，观察是否比 Exp14 的外接校准器更稳。

## 目录结构
- scripts: 实验脚本
- log: 运行日志与评估产物
- README.md: 实验说明

## 核心改动
- 在图像输入和语言输入分支分别增加预处理模块：
  - 图像分支：`image_preprocess = Linear + LayerNorm + ReLU`，在进入注意力融合前先做一次特征规整。
  - 语言分支：`text_preprocess = Linear + LayerNorm + ReLU`，在 prompt embedding 后、生成 gate 前进行规整。
  - 空文本分支：新增可学习 `null_prompt`，替代原先固定全 1 gate 路径。
- 在候选特征进入主干之前先做语言门控注意力：
  - 输入文本由 `LANGUAGE_PROMPT` 指定。
  - 将 prompt 经过 token hash + embedding + MLP 得到 gate。
  - 对候选特征做 `x * (1 + scale * gate)`。
- 主体改为更深更宽的 MLP backbone。
- backbone 后再接一个独立的 MLP scoring head 输出最终分数。
- 推理后处理默认固定为 Exp14 local refine 最优配置：`threshold=0.08, nms=0.66`，方便公平对比。

## 运行
python3 scripts/run_exp16_attn_bigmlp_20260422.py

## 主要输出
- log/run_log.txt
- log/exp16_attn_bigmlp_20260422_best.pt
- log/exp16_attn_bigmlp_20260422_summary.json
- log/evaluation_summary_exp16_attn_bigmlp_trainval_class_aware.md
- log/evaluation_summary_exp16_attn_bigmlp_test_class_aware.md
- log/evaluation_summary_exp16_attn_bigmlp_full_class_aware.md
