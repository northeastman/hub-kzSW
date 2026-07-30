# 第十三周作业 — 可实现「渐进式加载执行 Skills」的 Harness

> 作业要求：写一套可以实现渐进式加载执行 skills 的 harness。
> 完整代码见同目录 [`skill_harness.py`](./skill_harness.py)，运行记录见 [`run_output.log`](./run_output.log)。

---

## 一、什么是「渐进式加载执行 skills 的 harness」

课件《skills》Part 3 讲了一个核心矛盾：**能力越强，上下文越重，推理越低效**。
20 个 skill 若把每个的完整 `SKILL.md` 一次性塞进 system prompt，光工具/技能定义
就占掉 60%+ 的 context，模型还没开始想问题，注意力就被稀释了。

**渐进式披露（Progressive Disclosure）** 就是解法——用到哪一层才加载哪一层。
Harness 就是把这套「按需加载 + 在 skill 约束下执行」机制固化下来的运行时。本作业
实现的 harness 严格对应课件的三层模型：

| 层 | 名称 | 加载什么 | 量级 |
|----|------|----------|------|
| **L1** | 常驻层 Always Loaded | 每个 skill 只暴露 **1 行摘要**（`name` + `description`），拼成索引常驻 | 课堂 2 个 skill 实测 **277 tokens** |
| **L2** | 触发层 On Demand | 用户消息命中某 skill 后，**才加载它完整的 SKILL.md 正文** | flash-card 正文 **713 tokens** |
| **L3** | 执行层 In Context | 执行时**再按需** `read` references/、跑 scripts/，产物写回工作目录 | 用几个读几个 |

一句话：**L1 只给"目录"，L2 才翻"正文"，L3 才查"附录"、动"工具"。**

## 二、Harness 干了哪几件事（对应源码模块）

```
用户一句话
   │
   ▼
[L1] SkillRegistry.discover()   扫 skills/*/SKILL.md，只解析 YAML frontmatter
   │                            拿 name+description，正文一个字都不读 → 建索引
   ▼
[L2] SkillRegistry.match()      只把 L1 索引喂给调度器，选出唯一最合适的 skill
   │                            （有 API Key → LLM 判定；无 → 关键词打分降级）
   │  Skill.load_body()         这时才真正读该 skill 的完整正文进 context
   ▼
[L3] SkillHarness._exec_llm()   在 <SKILL> 正文约束下跑 ReAct 工具循环：
        Toolbox.read_reference    ← 按需读 references/xxx.md（每次都记 token 账）
        Toolbox.run_script        ← 执行 skill 的脚本
        Toolbox.write_file        ← 产物写回工作目录
   ▼
[账单] 渐进式实际占用 vs 全量加载，算出省了多少 token
```

关键设计点：

1. **frontmatter 与 body 物理分离加载**。`discover()` 里 `_parse_frontmatter()` 解析后
   **丢弃 body**，`Skill._body` 保持 `None`；只有 `load_body()` 被调用（即 L2 命中）时
   才真正 `read_text` 正文并缓存。这是"没命中的 skill 一分 token 都不占正文"的保证。

2. **匹配阶段只能看见 L1 索引**。`match()` 拿到的上下文就是那几行摘要，看不到任何
   skill 正文——这正是渐进式披露省 token 的物理原因，而不是"假装省"。

3. **references 是 L3 按需加载**。baoyu-diagram 有 4 个 references（architecture/
   flowchart/sequence/structural），画架构图时**只**会 `read_reference(architecture.md)`，
   其余 3 个从不进 context。`Toolbox.loaded_ref_tokens` 把这笔实际加载量记下来。

4. **Harness 给 skill 提供"运行环境"**。课堂 flash-card 的 SKILL.md 里写的是
   `.cursor/skills/flash-card/...` 这种部署专用路径，skill 换个地方就找不到脚本。
   harness 在执行层 system prompt 里注入该 skill 的**真实 `baseDir` 绝对路径 +
   脚本/参考文件清单 + 工作目录**，让 skill 的相对/占位路径能正确解析——这本就是
   harness 该兜住的隐性知识（对应课件 Harness Engineering 的"隐性知识暴露给 Agent"）。

5. **元技能思想 / 与 week12 的关系**。week12 让 agent 学会"循环调用工具"；本作业在其
   之上加了一层"**先决定加载哪个 skill、再在 skill 指令约束下调用工具**"的调度器
   （课件 Part 4 的"元技能"形态）。执行层复用了 week12 的手写 ReAct 解析
   （Thought / Action / Action Input / Final Answer），LLM 客户端也复用课堂的
   DashScope/DeepSeek OpenAI 兼容配置。

## 三、token 账单（实测，指向课堂 skills 目录）

以 flash-card 那次执行为例（`run_output.log` 有完整记录）：

```
全量加载(2 个 skill 正文 + 所有 references 全塞进去): 6960 tokens
渐进式实际占用:                                     990 tokens
   = L1 索引 277 + L2 flash-card 正文 713 + L3 按需 reference 0
节省: 5970 tokens (86%)
```

只有 2 个 skill 就省 86%；skill 越多，"全量加载"分母越大，渐进式披露的优势
越夸张——这与课件"20 个 skill 场景省 60~90%"的量化完全一致。

## 四、怎么运行

```bash
cd week13-作业答案-zw

# ① 只看 harness 发现了哪些 skill、L1 索引多大（只解析 frontmatter，不读正文）
python skill_harness.py --list

# ② 给一句话，走完 L1→L2→L3 并打印 token 账单
python skill_harness.py --query "给我做一张 resilient 的单词闪卡"

# ③ 一次跑通内置多条示例（含"无 skill 命中"的 NONE 分支）——用于生成 run_output.log
python skill_harness.py --demo

# 默认指向课堂素材 ../week13 skills和harness/skills（含 flash-card、baoyu-diagram）
# 也可 --skills-dir 指向任意 skills 目录，harness 是通用的
```

**LLM 说明**：设了 `DASHSCOPE_API_KEY`（或 `DEEPSEEK_API_KEY`）时，L2 匹配和 L3 工具
循环交给真实 LLM；没设 Key 会自动降级到"确定性规划器（离线演示）"——L1 匹配用
关键词打分，L3 用内置迷你词库 + 真实脚本产出 HTML，**离线也能跑通并复现产物**。
两种模式下渐进式披露的加载/计量机制完全一致。`run_output.log` 是有 Key 的真实
LLM 跑法。

## 五、验证结果

- `--list`：正确发现 2 个 skill，L1 索引 277 tokens，全量基线 6960 tokens。
- `--query resilient 闪卡`：LLM 从索引选中 flash-card → 加载正文 → 写 `resilient.json`
  → 跑 `make_flashcard.py` → 产出真实 `resilient.html`（约 3.9 KB，可浏览器打开）。
- `--demo`：闪卡、架构图、以及"帮我算天气"（**无 skill 命中，正确返回 NONE**）三条
  分支均符合预期，token 账单逐条打印。其中架构图那条最能体现 L3 按需加载：
  baoyu-diagram 有 4 个 references，画架构图时**只加载了 `architecture.md`（706 tokens）**，
  另外 3 个从不进 context，最重的 skill 也省了 43%。（该 skill 最后一步把 SVG 转 @2x PNG
  依赖 `bun`+`sharp` 原生模块，本机环境未装成而到达步数上限，属课堂脚本的环境依赖问题，
  非 harness 逻辑问题——**真实 SVG 产物 `diagram/microservices-architecture.svg` 已正常产出**。）
