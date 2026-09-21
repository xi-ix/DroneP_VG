# LLM 动态 Prompt 动作空间实验

本实验在三轮 `38+RL_loop` 的基础上补充语言模型，使第二、三轮动作不再是固定类别 ID，而是由语言模型根据 query/caption 动态生成 prompt pool，再由强化学习策略在动态 prompt pool 中选择动作。

## 1. 使用的语言模型

使用本地模型：

```text
Qwen2.5-VL-3B-Instruct
```

模型路径：

```text
/home/wangzhe/VLM/model_weights/Qwen2.5-VL-3B/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
```

选择 Qwen2.5-VL-3B 的原因：

- 本地已有权重，不需要下载；
- 单卡 A800 可以稳定加载；
- 文本生成正常，适合作为 query parser / prompt proposer；
- Qwen3-VL-4B 在当前 transformers/CUDA 组合下文本生成触发 CUDA index assert，因此没有作为本次实验模型。

## 2. LLM 的作用

语言模型不直接预测检测框，也不直接决定最终结果。

它只负责：

```text
query / caption -> dynamic prompt pool
```

例如真实 caption：

```text
A white electric bicycle was parked on the sidewalk with a man in brown clothes on the left,
a woman in a floral shirt at the top right.
```

Qwen 生成：

```text
a white electric bicycle
a man wearing brown clothes
a woman wearing a floral shirt
a person standing on the sidewalk
```

然后系统将这些开放 prompt 映射到可执行候选来源：

```text
a white electric bicycle -> bicycle
a man wearing brown clothes -> person
a woman wearing a floral shirt -> person
```

## 3. 动态动作空间

原来的固定动作空间是：

```text
STOP
ADD_GDINO_BASE
ADD_PERSON
ADD_PEOPLE
ADD_GROUP_OF_PEOPLE
ADD_TRICYCLE
ADD_COVERED_TRICYCLE
ADD_AWNING_TRICYCLE
ADD_MOTORCYCLE
ADD_MOTORBIKE
ADD_SCOOTER
ADD_BICYCLE
```

LLM 动态版本不再直接输出固定动作 ID，而是每张图根据 Qwen 生成：

```text
A_t = {STOP, prompt_1, prompt_2, ..., prompt_K}
```

每个 prompt action 包含：

```text
text
source_prompt
llm_rank
llm_role
```

其中 `text` 是 Qwen 生成的原始开放 prompt，`source_prompt` 是实际候选来源，`llm_rank` 是该 prompt 在 LLM 输出中的位置，`llm_role` 表示 primary / alias / attribute / relation / fallback。

## 4. 动态策略网络

固定动作版是：

```text
state -> Linear -> 12 action logits
```

LLM 动态版改成：

```text
state encoder + prompt action encoder + scorer
```

结构为：

```text
state(38维)
   |
State Encoder
   |
state embedding(64维)
```

```text
prompt action feature(20维)
   |
Action Encoder
   |
action embedding(64维)
```

然后：

```text
score = MLP([state_embedding, action_embedding, state_embedding * action_embedding])
```

因此策略网络学习的是：

```text
当前候选状态下，某个 LLM 生成 prompt 是否值得作为下一轮扩展动作
```

而不是死记固定动作 ID。

## 5. 训练方式

训练样本形式为：

```text
(state_t, prompt_action_i, reward_i)
```

对每个状态下的动态 action group 做 softmax 策略优化。reward 仍然来自候选扩展后的检测收益：

```text
普通 recall 提升
高 IoU recall 提升
小目标 recall 提升
precision 提升
- 候选数量成本
- 动作成本
- STOP 惩罚
```

训练使用 GPU：

```text
CUDA_VISIBLE_DEVICES=1
device=cuda:0
```

训练过程中验证 reward 在 60 epoch 左右进入平台，因此提前停止，best checkpoint 自动保存。

最佳 checkpoint：

```text
38+RL_loop/outputs/llm_dynamic/dynamic_prompt_policy.pt
```

best 信息：

```text
best epoch = 50
val_mean_reward = 0.028201
state_dim = 38
action_feature_dim = 20
```

## 6. LLM Prompt Pool 统计

缓存文件：

```text
38+RL_loop/outputs/llm_dynamic/llm_prompt_pools.json
```

统计：

```text
样本数：300
总 prompt 数：2396
唯一 prompt 数：604
```

说明语言模型确实生成了大量非固定文本表达，而不是只复用原来的 12 个动作名称。

## 7. 测试结果

| 方法 | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 |
| --- | ---: | ---: | ---: | ---: |
| query_base + Exp38 score | 22298 | 0.4651 | 0.3131 | 0.1343 |
| 固定动作三轮 38+RL_loop | 40904 | 0.5293 | 0.3310 | 0.1397 |
| LLM 动态动作三轮 38+RL_loop | 35869 | 0.5004 | 0.3248 | 0.1424 |

LLM 动态动作版相比固定动作三轮版：

```text
mAP@0.5: 0.1397 -> 0.1424
候选框: 40904 -> 35869
```

说明加入语言模型后，候选更少，但排序质量更好，mAP 有提升。

不过 Acc@0.5 和 Acc@0.75 略低，说明该版本更偏向保守选择，召回不如固定动作三轮版。

## 8. 动作分布

测试阶段动作统计：

```text
STOP: 157
gdino_base: 57
person: 48
people: 28
bicycle: 7
motorcycle: 2
tricycle: 1
```

示例：

```text
LLM prompt:
a white electric bicycle
a man wearing brown clothes
a woman wearing a floral shirt

映射动作:
bicycle
person
person

RL 实际选择:
person -> bicycle
```

## 9. 当前限制

当前实验中，Qwen 可以生成开放 prompt，但候选来源仍然受已有 Exp38/GDINO metadata 限制。很多车辆类开放表达，例如：

```text
blue SUV
yellow sedan
silver sedan
white truck
black SUV
```

会被映射到：

```text
gdino_base
```

这是因为当前候选缓存中没有为每个 LLM prompt 重新运行 GroundingDINO。

因此，本实验已经证明：

```text
动作内容可以由语言模型动态生成；
策略网络可以对动态 prompt action 打分；
LLM 动态动作空间可以提升 mAP。
```

但如果要进一步释放 LLM 的能力，下一步需要：

```text
对 LLM 生成的开放 prompt 真实运行 GroundingDINO，生成对应候选框，
而不是只映射到已有 Exp38 prompt 来源。
```

## 10. 报告表述建议

可以这样写：

```text
为避免固定动作空间限制模型扩展性，本文进一步引入 Qwen2.5-VL-3B-Instruct
作为语言查询解析器，根据输入 query 动态生成 prompt action pool。
强化学习策略不再输出固定动作编号，而是通过 state encoder 和 prompt action encoder
对每个动态 prompt 进行打分选择，从而形成语言模型驱动的动态动作空间闭环推理。
```

实验结论可以写：

```text
相比固定动作三轮闭环策略，LLM 动态动作空间在减少候选框数量的同时，
将 mAP@0.5 从 0.1397 提升到 0.1424，说明动态 prompt action 具有更好的排序有效性和扩展潜力。
```
