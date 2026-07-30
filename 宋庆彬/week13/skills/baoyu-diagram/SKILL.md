---
name: baoyu-diagram
description: 创建自包含的暗色主题 SVG 图表，支持架构图、流程图、时序图和结构图；适合系统组件、流程、交互顺序及实体关系的可视化。
version: 1.0.0
triggers:
  - 画图
  - 架构图
  - 流程图
  - 时序图
  - diagram
allowed_tools:
  - read_skill_resource
  - write_workspace_file
---

# Baoyu Diagram

## 目标

根据用户描述生成一个自包含 SVG 文件，保存在：

```text
diagram/<topic-slug>.svg
```

## 执行流程

1. 判断最合适的图表类型：
   - 系统组件和依赖：架构图。
   - 步骤、条件分支：流程图。
   - 多角色按时间交互：时序图。
   - 类、实体、组织关系：结构图。
2. 只读取匹配类型的一份参考：
   - `references/architecture.md`
   - `references/flowchart.md`
   - `references/sequence.md`
   - `references/structural.md`
3. 根据参考规划组件、连接关系和布局。
4. 使用 `write_workspace_file` 写出完整 SVG。
5. 最终回答给出文件路径和所选图表类型。

## 通用视觉约束

- 根元素必须包含 `xmlns="http://www.w3.org/2000/svg"` 和 `viewBox`。
- SVG 内嵌全部样式，不依赖本地图片或 CSS。
- 背景使用 `#0f172a`，主要文字使用 `#f8fafc`。
- 节点之间至少保留 28px 间距。
- 先绘制连接线，再绘制节点，避免连线覆盖文字。
- 中文字体使用：
  `Inter, "PingFang SC", "Microsoft YaHei", sans-serif`。
- 生成前检查文字是否超出节点、连接线是否穿过节点。

## 边界

- 不要一次读取全部参考文件。
- 不要生成用户没有要求的多个版本。
- 不要声称生成了 PNG；本 Skill 只输出 SVG。

