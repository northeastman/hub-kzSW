# plot-telemetry-column 参考文档

本文件为 SKILL.md 的按需加载详情，仅在需要完整示例或工作原理时读取，不随 skill 触发自动加载（节省 token）。

## 工作原理

- 读取 `--data-dir` 下所有 `*.txt`（按文件名排序），除非用 `--files` 指定子集。
- 对每个文件：解析表头，然后按 hex-token 检测把数据行分成 `parsed` 和 `hex` 两池。
- 把所选模式（parsed/hex）的行跨文件拼接；当文件数大于 1 时，在文件边界处
  画橙色虚线竖线。
- `--mode parsed`：
  - 数值 cell → float。
  - 非数值 cell（如 `加电`/`断电`/`健康`）→ 按首次出现顺序编码为整数，
    画阶梯曲线；映射关系会打印并在图上标注。
  - `NaN` / 空 → 曲线缺口（gap）。
  - 仅含 `NaN` 的数值列不会被误判为 categorical，`NaN` 直接作为缺口。
- `--mode hex`：每个 cell 通过 `int(v, 16)` 转整数；非 hex cell 变为缺口。
- 输出：PNG 写入 `--output`（如需请先创建父目录）。

## 注意事项

- 若未传 `--column`，脚本会打印列名列表后退出，不画图——可作为发现步骤。
- 运行前请先创建输出目录，例如 `mkdir plots`。
- 列名含前导空格或特殊字符（如 ` +Xxyw...`）时按字面匹配，建议改用
  0 基索引更稳妥。
- 文件较大时脚本按行流式读取；内存随行数增长，与磁盘文件大小无关。

## 完整用法示例（核心 SKILL.md 仅保留 2 个，其余在此）

在**项目根目录**（即包含 `data_raw/` 的目录）下运行。

1. **列出所有可用列**（推荐先做这一步，无需 matplotlib）：

```bash
python .cursor/skills/plot-telemetry-column/scripts/plot_column.py \
  --data-dir data_raw --list-columns
```

2. **按索引画某列的解析值曲线**（col 63 = `+Xxyw/-Xxyw/+Yxyw1温度第一次采集`，
   有真实温度数据）：

```bash
python .cursor/skills/plot-telemetry-column/scripts/plot_column.py \
  --data-dir data_raw \
  --column 63 \
  --mode parsed \
  --output plots/temp1.png
```

3. **画同一列的 16 进制原码曲线**：

```bash
python .cursor/skills/plot-telemetry-column/scripts/plot_column.py \
  --data-dir data_raw \
  --column 63 \
  --mode hex \
  --output plots/temp1_hex.png
```

4. **按列名画**（col 0 = 时间戳；col 23 = 某电压列）：

```bash
python .cursor/skills/plot-telemetry-column/scripts/plot_column.py \
  --data-dir data_raw \
  --column "对面试验面电池A1/A3/A5电压1" \
  --mode parsed --output plots/voltage.png
```

5. **叠加多列**（重复 `--column`）：

```bash
python .cursor/skills/plot-telemetry-column/scripts/plot_column.py \
  --data-dir data_raw \
  --column 63 --column 64 \
  --mode parsed --output plots/temp12.png
```

6. **画状态列**（categorical，例如 col 1 `试验数据采集单元主机状态`，
   取值如 `加电`/`断电`）。脚本自动把状态编码为整数并画阶梯曲线，
   映射关系会打印并在图上标注：

```bash
python .cursor/skills/plot-telemetry-column/scripts/plot_column.py \
  --data-dir data_raw --column 1 --mode parsed --output plots/host_state.png
```

7. **只处理指定文件**（相对 `--data-dir` 的文件名）：

```bash
python .cursor/skills/plot-telemetry-column/scripts/plot_column.py \
  --data-dir data_raw \
  --files 20230719182031遥测数据.txt \
  --column 63 --mode parsed --output plots/a.png
```

8. **用时间戳作 x 轴**（替代默认的采样序号）：

```bash
python .cursor/skills/plot-telemetry-column/scripts/plot_column.py \
  --data-dir data_raw --column 63 --mode parsed \
  --x-mode time --output plots/time.png
```
