---
name: random-math
description: 生成十个随机数，进行随机加减乘除运算，并生成美观的 HTML 页面展示
---

# 随机数运算展示

生成十个随机整数，将它们用随机的加减乘除运算符串联计算，最终生成一个美观的 HTML 页面展示完整表达式和结果。

## 执行流程

### 步骤 1：生成随机数

用 Python 生成 10 个范围在 1~100 的随机整数：

```bash
python -c "import random; nums = [random.randint(1, 100) for _ in range(10)]; print(nums)"
```

### 步骤 2：串联随机运算

将这 10 个数字依次用随机运算符（+、-、*、/）串联，得到一个最终计算结果。运算遵循标准数学优先级（先乘除，后加减）。

```bash
python -c "
import random
nums = [/* 步骤1生成的数字列表 */]
ops = ['+', '-', '*', '/']
result = nums[0]
expr = str(nums[0])

for i in range(1, len(nums)):
    op = random.choice(ops)
    b = nums[i]
    expr += ' ' + op + ' ' + str(b)
    if op == '+':
        result += b
    elif op == '-':
        result -= b
    elif op == '*':
        result *= b
    else:
        result /= b

print('Expression:', expr)
print('Result:', result)
"
```

### 步骤 3：生成 HTML 页面

根据步骤 1 的数字和步骤 2 的表达式与结果，生成一个深色主题的美观 HTML 页面，要求：

- **背景**：深色（`#0f172a`），带彩色渐变光晕
- **数字展示**：10 个彩色标签排成一行，每个数字用不同颜色
- **表达式展示**：在代码风格的卡片中展示完整运算式，数字和运算符用不同颜色区分
- **结果展示**：大号渐变色字体醒目显示最终结果
- **优先级提示**：底部标注"先乘除，后加减"

将 HTML 文件写入工作目录下的 `random_calc.html`，并用浏览器打开：

```bash
start "" "random_calc.html"
```

## 页面样式参考

页面的 CSS 变量和配色方案：

```css
:root {
    --bg: #0f172a;
    --card-bg: #1e293b;
    --border: #334155;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --accent: #38bdf8;      /* 数字颜色 */
    --orange: #fb923c;       /* 运算符颜色 */
    --green: #4ade80;        /* 结果颜色 */
}
```

数字标签使用 10 种不同的彩色背景：
```
#ef4444, #f97316, #eab308, #22c55e, #14b8a6,
#3b82f6, #6366f1, #8b5cf6, #a855f7, #ec4899
```

## 交互要求

1. 每次执行都重新生成随机数和随机运算符，保证结果不同
2. 最终在浏览器中展示 HTML 页面
3. 向用户汇报生成的文件路径和主要结果
