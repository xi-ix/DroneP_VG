from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).resolve().parent

W, H = 1600, 820


def text(x, y, content, cls="text", anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(content)}</text>'


def rect(x, y, w, h, cls="box", rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'


def line(x1, y1, x2, y2, cls="arrow"):
    return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}"/>'


def polyline(points, cls="arrow"):
    d = " ".join(("M" if i == 0 else "L") + f"{x},{y}" for i, (x, y) in enumerate(points))
    return f'<path d="{d}" class="{cls}"/>'


def box(x, y, w, h, title, lines=(), cls="box", accent=None):
    parts = [rect(x, y, w, h, cls)]
    if accent:
        parts.append(f'<rect x="{x}" y="{y}" width="8" height="{h}" rx="4" class="{accent}"/>')
    parts.append(text(x + 28, y + 38, title, "label"))
    for i, item in enumerate(lines):
        parts.append(text(x + 28, y + 72 + i * 28, item, "small"))
    return "\n".join(parts)


def panel(x, y, w, h, idx, title, cls="panel"):
    return "\n".join([
        rect(x, y, w, h, cls),
        f'<circle cx="{x + 31}" cy="{y + 32}" r="17" class="badge"/>',
        text(x + 31, y + 39, str(idx), "badgeText", "middle"),
        text(x + 58, y + 40, title, "panelTitle"),
    ])


def svg(title, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="10" markerHeight="10" orient="auto-start-reverse">
      <path d="M2,2 L10,6 L2,10 Z" fill="#1f2937"/>
    </marker>
    <marker id="arrowSoft" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
      <path d="M2,2 L10,6 L2,10 Z" fill="#64748b"/>
    </marker>
    <style>
      svg {{ background: #fbfcff; }}
      .title {{ font: 700 36px "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; fill: #111827; }}
      .panelTitle {{ font: 700 27px "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; fill: #111827; }}
      .label {{ font: 700 26px "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; fill: #111827; }}
      .text {{ font: 22px "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; fill: #111827; }}
      .small {{ font: 20px "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; fill: #475569; }}
      .note {{ font: 700 22px "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; fill: #b91c1c; }}
      .badgeText {{ font: 700 21px "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; fill: #111827; }}
      .panel {{ fill: #f8fafc; stroke: #64748b; stroke-width: 1.8; }}
      .panelBlue {{ fill: #eef5ff; stroke: #64748b; stroke-width: 1.8; }}
      .panelGreen {{ fill: #effaf3; stroke: #64748b; stroke-width: 1.8; }}
      .panelWarm {{ fill: #fff7ed; stroke: #64748b; stroke-width: 1.8; }}
      .box {{ fill: #ffffff; stroke: #334155; stroke-width: 2; }}
      .boxBlue {{ fill: #eef5ff; stroke: #2563eb; stroke-width: 2; }}
      .boxGreen {{ fill: #ecfdf5; stroke: #059669; stroke-width: 2; }}
      .boxWarm {{ fill: #fff7ed; stroke: #f59e0b; stroke-width: 2; }}
      .boxRed {{ fill: #fff1f2; stroke: #ef4444; stroke-width: 2; }}
      .boxPurple {{ fill: #f1efff; stroke: #6d5bd0; stroke-width: 2; }}
      .accentBlue {{ fill: #3b82f6; stroke: none; }}
      .accentGreen {{ fill: #10b981; stroke: none; }}
      .accentWarm {{ fill: #f59e0b; stroke: none; }}
      .accentRed {{ fill: #ef4444; stroke: none; }}
      .badge {{ fill: #ffffff; stroke: #334155; stroke-width: 1.5; }}
      .arrow {{ fill: none; stroke: #1f2937; stroke-width: 3; marker-end: url(#arrow); }}
      .arrowSoft {{ fill: none; stroke: #64748b; stroke-width: 2.4; stroke-dasharray: 8 8; marker-end: url(#arrowSoft); }}
      .thin {{ fill: none; stroke: #94a3b8; stroke-width: 1.6; }}
      .divider {{ stroke: #111827; stroke-width: 2; }}
    </style>
  </defs>
  <rect x="0" y="0" width="{W}" height="{H}" fill="#fbfcff"/>
  {text(W / 2, 42, title, "title", "middle")}
  <line x1="44" y1="62" x2="{W - 44}" y2="62" class="divider"/>
  {body}
</svg>
'''


def figure1():
    body = []
    body.append(panel(46, 92, 248, 560, 1, "输入", "panel"))
    body.append(box(78, 180, 184, 124, "航拍图像", ("UAV image", "复杂背景"), "boxBlue"))
    body.append(box(78, 412, 184, 124, "文本问题", ("目标类别 / 语义约束", "query text"), "box"))

    body.append(panel(332, 92, 332, 560, 2, "候选生成", "panelBlue"))
    body.append(box(378, 170, 240, 112, "视觉-语言骨干", ("图像特征 + 文本特征",), "boxGreen", "accentGreen"))
    body.append(box(378, 364, 240, 132, "开放词表候选框", ("GroundingDINO", "box / score / prompt"), "boxPurple", "accentBlue"))

    body.append(panel(704, 92, 332, 560, 3, "小目标增强", "panelWarm"))
    body.append(box(750, 162, 240, 112, "提示词集合", ("基础类别 + 别名类别",), "boxWarm", "accentWarm"))
    body.append(box(750, 322, 240, 112, "别名恢复", ("person / pedestrian", "van / vehicle 等"), "boxRed", "accentRed"))
    body.append(box(750, 482, 240, 96, "候选压缩排序", ("保留可靠小目标候选",), "box"))

    body.append(panel(1076, 92, 226, 560, 4, "融合评分", "panelBlue"))
    body.append(box(1112, 172, 154, 112, "元信息", ("score", "area / rank", "source"), "box"))
    body.append(box(1112, 382, 154, 112, "融合器", ("calibrated", "final score"), "boxPurple"))

    body.append(panel(1336, 92, 218, 560, 5, "输出", "panelGreen"))
    body.append(box(1372, 205, 146, 112, "类别感知NMS", ("去除重复框",), "box"))
    body.append(box(1372, 430, 146, 112, "定位结果", ("类别/框/分数",), "boxGreen"))

    body.extend([
        line(262, 242, 378, 226),
        line(262, 474, 378, 226),
        line(498, 282, 498, 364),
        line(618, 430, 750, 218),
        line(618, 430, 750, 378),
        line(870, 274, 870, 322),
        line(870, 434, 870, 482),
        line(990, 530, 1112, 438),
        line(1266, 438, 1372, 261),
        line(1445, 317, 1445, 430),
        polyline([(1258, 382), (1294, 348), (1294, 220), (1372, 261)], "arrowSoft"),
        polyline([(1445, 542), (1036, 640), (870, 578)], "arrowSoft"),
        text(1000, 724, "小目标命中奖励反馈到排序与提示优先级", "note", "middle"),
    ])
    return svg("强化学习驱动的航拍图像视觉定位整体框架", "\n".join(body))


def figure2():
    body = []
    body.append(panel(56, 102, 330, 520, 1, "文本提示构建", "panel"))
    body.append(box(94, 182, 252, 96, "原始问题", ("需要定位的目标语义",), "box"))
    body.append(box(94, 342, 252, 128, "多级别名词表", ("基础类别", "小目标别名", "弱势类别映射"), "boxWarm", "accentWarm"))

    body.append(panel(430, 102, 342, 520, 2, "候选池补充", "panelBlue"))
    body.append(box(476, 176, 250, 104, "基础候选源", ("常规类别提示生成",), "boxBlue", "accentBlue"))
    body.append(box(476, 340, 250, 104, "小目标候选源", ("别名提示扩大召回",), "boxRed", "accentRed"))
    body.append(box(476, 500, 250, 72, "候选合并", ("去重并保留来源",), "box"))

    body.append(panel(816, 102, 342, 520, 3, "小目标排序", "panelWarm"))
    body.append(box(862, 168, 250, 104, "面积与尺度先验", ("小框不直接丢弃",), "boxGreen", "accentGreen"))
    body.append(box(862, 318, 250, 104, "语义一致性", ("查询匹配 / 别名匹配",), "boxPurple"))
    body.append(box(862, 486, 250, 72, "候选压缩", ("按可靠性重排",), "box"))

    body.append(panel(1202, 102, 342, 520, 4, "结果恢复", "panelGreen"))
    body.append(box(1248, 190, 250, 104, "弱小类恢复", ("person / pedestrian", "vehicle / van 等"), "boxGreen", "accentGreen"))
    body.append(box(1248, 388, 250, 104, "最终候选框", ("提升小目标召回", "控制总预测框数量"), "box"))

    body.extend([
        line(346, 230, 476, 228),
        line(346, 406, 476, 392),
        line(601, 280, 601, 340),
        line(601, 444, 601, 500),
        line(726, 536, 862, 220),
        line(726, 536, 862, 370),
        line(987, 272, 987, 318),
        line(987, 422, 987, 486),
        line(1112, 522, 1248, 242),
        line(1112, 522, 1248, 440),
        polyline([(1498, 242), (1530, 242), (1530, 610), (601, 610), (601, 574)], "arrowSoft"),
        text(1050, 724, "核心思想：从候选阶段补回可能漏检的小目标，而不是只在后处理加分", "note", "middle"),
    ])
    return svg("小目标候选召回与多级别名恢复机制", "\n".join(body))


def figure3():
    body = []
    body.append(panel(52, 104, 350, 520, 1, "候选记录", "panel"))
    body.append(box(92, 190, 270, 92, "候选框", ("类别 / 坐标 / 原始分",), "boxBlue", "accentBlue"))
    body.append(box(92, 338, 270, 92, "提示来源", ("基础提示 / 别名提示",), "boxWarm", "accentWarm"))
    body.append(box(92, 486, 270, 92, "查询上下文", ("query text / class prior",), "boxGreen", "accentGreen"))

    body.append(panel(452, 104, 350, 520, 2, "富元信息特征", "panelBlue"))
    body.append(box(492, 168, 270, 84, "分数特征", ("gdino / query / alias",), "boxGreen"))
    body.append(box(492, 286, 270, 84, "来源特征", ("来源类型 / 提示编号",), "boxWarm"))
    body.append(box(492, 404, 270, 84, "几何特征", ("area ratio / rank / size",), "boxBlue"))
    body.append(box(492, 512, 270, 84, "小目标先验", ("小面积优先",), "boxRed"))

    body.append(panel(852, 104, 350, 520, 3, "强化学习融合决策", "panelWarm"))
    body.append(box(898, 178, 258, 210, "融合评分器", ("特征归一化", "小目标重加权", "困难负样本约束", "输出最终置信度"), "boxPurple", "accentBlue"))
    body.append(box(898, 474, 258, 90, "奖励信号", ("IoU 命中 / 小目标召回",), "box", "accentRed"))

    body.append(panel(1252, 104, 298, 520, 4, "输出与闭环", "panelGreen"))
    body.append(box(1292, 178, 218, 94, "融合置信度", ("final score",), "boxGreen", "accentGreen"))
    body.append(box(1292, 348, 218, 94, "类别感知NMS", ("去重 / TopK",), "box"))
    body.append(box(1292, 510, 218, 80, "最终定位框", ("类别 / 框 / 分数",), "boxGreen"))

    body.extend([
        line(362, 236, 492, 210),
        line(362, 384, 492, 328),
        line(362, 532, 492, 446),
        line(762, 210, 898, 228),
        line(762, 328, 898, 280),
        line(762, 446, 898, 332),
        line(762, 557, 898, 356),
        line(1156, 284, 1292, 225),
        line(1401, 272, 1401, 348),
        line(1401, 442, 1401, 510),
        polyline([(1401, 590), (1220, 668), (1028, 564)], "arrowSoft"),
        line(1027, 474, 1027, 388),
        text(1130, 724, "闭环作用：根据定位收益调节候选排序，而非改变基础检测器结构", "note", "middle"),
    ])
    return svg("富元信息融合与强化学习闭环优化", "\n".join(body))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "fig1_overall_framework.svg": figure1(),
        "fig2_small_object_recall.svg": figure2(),
        "fig3_metadata_rl_fusion.svg": figure3(),
    }
    for name, content in figures.items():
        (OUT / name).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
