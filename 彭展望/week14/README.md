# 第十四周作业：编写一个 Skill 并优化它

> **作业要求**：下载或让大模型编写一个 skill，然后尝试让模型优化它，可以从执行效率
> 或 token 消耗等角度进行要求，对比下优化前后的效果。

## 一、我做了什么

1. **写了一个 skill：`log-stats`（日志统计分析）** —— 输入一个日志文件，输出各级别
   数量、Top N 错误、每小时请求量的文本报告。
2. **让模型从两个角度优化它**：
   - **执行效率**：优化分析脚本（`analyze.py`）的算法与 I/O。
   - **token 消耗**：优化 `SKILL.md` 本身的上下文占用。
3. **用真实数据对比优化前后**（200 万行 / 105 MB 日志，见 `benchmark/results.md`）。

## 二、目录结构

```
作业-skill优化/
├── README.md                        # 本文件：作业说明 + 优化对比结论
├── skill-v1-未优化/log-stats/
│   ├── SKILL.md                     # 冗长版（所有细节内联，1860 tokens）
│   └── scripts/analyze.py           # 低效版（全量读内存 + 3 趟遍历 + 循环内编译正则）
├── skill-v2-优化后/log-stats/
│   ├── SKILL.md                     # 精简版（渐进式披露，362 tokens）
│   ├── reference.md                 # 按需加载的格式细节
│   └── scripts/analyze.py           # 高效版（流式单趟 + Counter + 预编译正则）
└── benchmark/
    ├── gen_log.py                   # 生成可复现的测试日志
    ├── benchmark.py                 # 执行效率对比（含正确性校验）
    ├── count_tokens.py              # SKILL.md token 消耗对比
    └── results.md                   # 实测结果
```

## 三、优化点与对比结果

### 角度 1：执行效率（优化脚本 `analyze.py`）

| 优化项 | v1 未优化 | v2 优化后 |
|---|---|---|
| 文件读取 | `readlines()` 全量读入内存 | `for line in f` 流式逐行 |
| 遍历次数 | 3 趟（级别 / 错误 / 每小时各一趟） | 1 趟全部搞定 |
| 正则编译 | 循环内每行都重新编译 | 循环外预编译一次复用 |
| Top N 取法 | 全量 `sorted()` 再切片 | `Counter.most_common(n)`（堆，O(n·log k)） |
| 计数结构 | 手写 dict + `if k in d` | `collections.Counter` |

**实测（105 MB 日志）**：

| 指标 | v1 | v2 | 提升 |
|---|---|---|---|
| 耗时 | 3.047 s | 1.316 s | **2.32x** |
| 峰值内存 | 249.4 MB | 11.4 MB | **21.89x** |
| 输出一致性 | — | — | 逐字节一致 ✅ |

内存差距尤为关键：v1 内存随文件线性增长，几百 MB 日志就会吃掉几百 MB 内存，GB 级
日志可能直接 OOM；v2 内存恒定，多大的文件都能处理。

### 角度 2：Token 消耗（优化 `SKILL.md`）

核心手法是 **渐进式披露（progressive disclosure）**：SKILL.md 只保留"何时触发 +
怎么调用"的最小必要信息，把日志格式说明、报告样例、边界情况等**冗长且不常用**的细节
下沉到 `reference.md`，仅在模型确实要核对格式时才加载。

| 文件 | v1 | v2 |
|---|---|---|
| SKILL.md（每次触发都加载） | 1860 tokens | **362 tokens** |
| reference.md（按需才加载） | —（内联在上面） | 447 tokens |

- **常态 token 降幅 80.5%**（每次触发省下约 1498 tokens）。
- 即便偶尔需要加载 reference.md，总量 809 tokens 仍比 v1 少 **56.5%**。
- skill 越多、触发越频繁，这种"瘦身 + 按需加载"省下的上下文越可观。

## 四、结论

同一个 skill，仅通过让模型从"执行效率"和"token 消耗"两个角度优化：

- 脚本：**快 2.3 倍、省内存 22 倍**，且结果完全不变；
- SKILL.md：常态上下文 **省 80% token**。

这也印证了 skill 设计的两条基本原则：**脚本负责把重活做高效，SKILL.md 负责把上下文
占用压到最小（该说的说清楚，不常用的按需加载）**。

## 五、如何复现

```bash
cd benchmark
python gen_log.py test.log 2000000
python benchmark.py test.log 3
python count_tokens.py
```
