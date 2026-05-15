# 工作区实验规范

本文件是 `DroneP_VG` 工作区的实验执行规范。后续新增、复现、整理实验时优先遵守这里的规则。

## 目录约定

所有正式实验必须放在 `experiment/` 下的独立子目录中，不要直接散落在仓库根目录。

标准结构：

```text
experiment/<exp_name>/
├── README.md
├── scripts/
└── log/
```

要求：

- `README.md`：写清楚实验目的、数据来源、方法改动、运行方式、主要产物和结论。
- `scripts/`：保存实验脚本，参数尽量写在脚本顶部常量区，使运行方式保持简单。
- `log/`：保存实验输出，包括日志、预测结果、评估报告、summary json、checkpoint 等。

运行方式应尽量保持为：

```bash
python experiment/<exp_name>/scripts/<run_script>.py
```

开启新的正式实验，尤其是训练或长时间推理实验时，优先在 `tmux` 会话中运行，避免 SSH 断开导致任务中断。标准做法是把终端输出同时维护到实验 `log/` 目录下的单一运行日志文件，例如：

```bash
tmux new -s <exp_name>
python experiment/<exp_name>/scripts/<run_script>.py 2>&1 | tee experiment/<exp_name>/log/run_log.txt
```

如果实验脚本内部已经写入 `log/run_log.txt`，也应保证 tmux 中的 stdout/stderr 不丢失；必要时可使用 `tee -a` 追加到同一个日志文件。

## 命名规范

实验目录推荐格式：

```text
expNN_short_description_YYYYMMDD
```

示例：

```text
exp18_exp14_method_large_data_20260423
exp24_teacher_distill_exp23_to_exp22_20260513
```

命名要求：

- `expNN` 必须递增，避免复用已有编号。
- 简短描述使用小写英文和下划线。
- 日期使用 `YYYYMMDD`。
- 不要把新实验生成在 `fail/` 或 `baseline/` 目录中，除非用户明确要求。

## 产物规范

每个实验至少应尽量产出：

```text
log/run_log.txt
log/<exp_name>_summary.json
log/evaluation_summary_<scope>_class_aware.md
```

如果有模型训练，还应保存：

```text
log/<exp_name>_best.pt
```

如果有预测结果，推荐结构：

```text
log/predictions/
log/gt_xyxy/
```

大数据实验如果区分 split，推荐：

```text
log/predictions/train/
log/predictions/val/
log/predictions/test/
log/predictions/full/

log/gt_xyxy/train/
log/gt_xyxy/val/
log/gt_xyxy/test/
log/gt_xyxy/full/
```

实验日志和输出产物保存在 `log/` 中，但默认不上传 Git。仓库 `.gitignore` 必须忽略：

```gitignore
experiment/**/log/
```

## 数据规范

统一数据目录在：

```text
dataset/
```

所有实验的数据都必须从 `dataset/` 中读取。实验脚本不得把 `experiment/`、仓库根目录或临时目录中的数据副本当作主要数据源。

已有重要数据源：

```text
dataset/visdrone_test_100
dataset/VisDroneSplit1000Guarded
dataset/RefDrone
dataset/AerialVG
```

规则：

- 实验脚本应明确写出使用的数据根目录。
- 不要在没有说明的情况下混用不同数据口径。
- 大数据实验优先使用 `dataset/VisDroneSplit1000Guarded`。
- 小数据复现实验如果依赖历史路径 `visdrone_test_100`，只能用兼容软链接指向 `dataset/visdrone_test_100`；真实数据来源仍然必须是 `dataset/`。

## 评估规范

检测实验默认报告：

```text
Acc@0.5
Acc@0.75
mAP@0.5
GT boxes
Prediction boxes
```

默认评估口径：

- class-aware matching
- IoU threshold for mAP: `0.5`
- 每张图片一个 `.txt` 文件
- GT 和 prediction 文件名 stem 对齐

常见格式：

```text
# GT
class_id x1 y1 x2 y2

# prediction
class_id x1 y1 x2 y2 score
```

如果实验使用无类别 GT 或特殊格式，必须在实验 README 中写清楚转换方式。

## 复现实验流程

复现实验时按以下顺序进行：

1. 检查数据目录是否存在。
2. 检查实验脚本是否存在。
3. 检查依赖实验的产物是否存在，例如候选预测、checkpoint、split plan。
4. 先复现数据构建实验，再复现训练/推理实验。
5. 跑完后核对 `summary.json` 和 markdown 评估报告。
6. 在实验 README 或 `experiment/README.md` 中记录最终指标。

推荐依赖顺序：

```text
Exp17 dataset guard
baseline
Exp13 candidates
Exp14
Exp15 / Exp16
Exp18 large
Exp19+
```

## 禁止事项

- 不要在用户没有明确要求时删除实验目录、日志、checkpoint 或预测结果。
- 不要使用 `git reset --hard`、`git checkout --` 等破坏性命令清理实验。
- 不要把新实验放进 `fail/` 或 `baseline/`，除非用户明确要求。
- 不要覆盖用户正在编辑的脚本，除非当前任务明确要求修改该文件。
- 不要把临时 demo 输出、缓存、`__pycache__` 当作正式实验产物提交。

## Git 约定

实验产物通常较大，是否提交由用户决定。

建议提交：

```text
experiment/<exp_name>/README.md
experiment/<exp_name>/scripts/*.py
WORKSPACE_EXPERIMENT_RULES.md
```

谨慎提交：

```text
experiment/<exp_name>/log/
dataset/
*.pt
*.pth
```

默认规则：实验 `log/` 目录不提交到 Git。如果某些日志文件已经被 Git 跟踪，需要只从 Git 索引移除，不删除本地文件：

```bash
git rm -r --cached experiment/**/log
```

如果必须保证实验可完整复现，应至少保留脚本、README、summary 和关键配置；大文件产物可以单独备份。
