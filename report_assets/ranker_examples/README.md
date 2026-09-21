# 最终排序器真实图片案例

## 推荐主图

答辩优先使用 `clear_query_ranker_example.png`。该图来自 RefDrone test，原始提示词为 `The pedestrians on the right bottom corner of the image.`，正确框在无 Exp36 source 排序器中由第 6 名升至第 1 名。对应审计数据见 `clear_query_case.json`，生成命令为 `python report_assets/figures/generate_clear_query_ranker_example.py`。

## 补充案例

这些图使用 VisDroneSplit1000Guarded 测试集真实图片与标注生成。重排序前采用 metadata 中的 `final_score`，重排序后采用 Exp38 `fusion_predictions/test` 中的融合分数；候选框集合和坐标完全不变。

基础 GroundingDINO 使用联合提示词：`pedestrian, people, bicycle, car, van, truck, tricycle, awning tricycle, bus, motor`。小目标召回还会使用 `person, people, group of people, tricycle, covered tricycle, awning tricycle, motorcycle, motorbike, scooter, bicycle` 等别名提示。每个局部框下方显示该候选的实际来源提示；`[基础]` 表示来自联合提示词，`[别名]` 表示来自单独的别名提示。

判定口径：候选框与同类别 GT 的最大 IoU，`IoU >= 0.5` 记为高质量框。图中绿色或红色为预测框，黄黑双线且带 `GT` 标签的框为同类别真实标注。

| 案例 | 图像 | 变化 | Top-5 高质量框 | Top-5 平均 IoU |
|---|---|---|---:|---:|
| 1 | `0000087_01580_d_0000005` | 误检 Top-1 被抑制，高质量行人框整体前移 | 3 → 5 | 0.535 → 0.824 |
| 2 | `0000293_03601_d_0000940` | 错误行人框退出榜首，汽车真框占据 Top-5 | 4 → 5 | 0.742 → 0.926 |
| 3 | `9999972_00000_d_0000104` | 靠后的高质量面包车框进入 Top-5 | 3 → 4 | 0.442 → 0.664 |
| 4 | `9999955_00000_d_0000034` | 低质量人群框被替换，行人框排序更集中 | 4 → 5 | 0.690 → 0.815 |

## 输出文件

- `ranker_examples_overview.png`：四个案例汇总长图。
- `case_01.png` 至 `case_04.png`：适合论文或答辩单独使用的高清图。
- `case_details.json`：每个 Top-5 框的原排名、类别、查询词、来源提示词、分数、IoU 和坐标。

重新生成：

```bash
python report_assets/figures/generate_ranker_examples.py
```
