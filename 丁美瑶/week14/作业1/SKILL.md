---
name: plot-telemetry-column
description: >-
  解析目录中 Tab 分隔的遥测 txt 文件，按选定列拉曲线，可选解析工程值或 16 进制原码。
  触发词：遥测 txt、遥测数据、拉曲线、筛选列、原码/解析数据、data_raw。
---

# 遥测列曲线绘制

解析 `--data-dir` 下所有 `*.txt` 遥测文件，按选定列画曲线。每个采样在文件中占两行
（共享时间戳）：**解析行**（工程浮点数 / 中文状态字 / `NaN`）与**原码行**（16 进制）。
脚本按"该行所有非空 cell 是否都像 hex token"自动判别两层，并跳过重复表头行。

## 依赖

`pip install matplotlib`（仅画图需要；`--list-columns` 不需要）。脚本用 Agg 后端写 PNG，
Windows 自动检测中文字体。

## 用法（在项目根目录运行）

```bash
# 1. 列出列名（含 0 基索引，col 0 = 时间戳）
python .cursor/skills/plot-telemetry-column/scripts/plot_column.py \
  --data-dir data_raw --list-columns

# 2. 画某列曲线：--mode parsed（工程值）或 hex（原码）；--column 接列名或索引，可重复叠加
python .cursor/skills/plot-telemetry-column/scripts/plot_column.py \
  --data-dir data_raw --column 63 --mode parsed --output plots/out.png

# 3. 可选：--files 指定子集；--x-mode time 用时间戳作 x 轴
```

输出 PNG 到 `--output`（先建目录，如 `mkdir plots`）。列名含特殊字符时建议用索引。
完整说明、全部示例与工作原理见 [reference.md](reference.md)。
