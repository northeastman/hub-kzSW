---
name: flash-card
description: 为一个英语单词生成静态 HTML 学习闪卡（音标/词性/释义/近义词/3条中英例句）。当用户说"给我做张 crazy 的闪卡""做一个 resilient 的单词卡"等时使用。
run: scripts/make_flashcard.py
---

# Flash Card 单词闪卡生成

为英语单词生成一张静态 HTML 学习卡片。版面顺序：单词+音标 → 释义 → 近义词 → 3 条中英对照例句。

## 触发场景

- "给我做张 crazy 词的闪卡"
- "给我做 crazy 的 flash card"
- "做一个 resilient 的单词卡"
- "帮我生成 meticulous 的闪卡"

## 执行流程

1. **识别单词**：从用户话语中提取目标英语单词。

2. **自己编写学习数据**：为该单词写出下列字段，组成一个 JSON 对象：
   - `word`：单词（小写）
   - `phonetic`：音标（如 `/rɪˈzɪliənt/`）
   - `pos`：词性（如 `adj.`）
   - `definition`：中文释义
   - `examples`：**恰好 3 条**，每条含 `en`（英文例句）和 `zh`（中文翻译）
   - `synonyms`：近义词列表（4-6 个为宜）

   例句要地道、长度适中、体现典型用法；近义词贴近该词核心含义。

3. **调用脚本生成 HTML**：用 `run_skill_script` 执行本能力，把上面的 JSON 对象**序列化成一个 JSON 字符串**作为第一个参数传入。脚本会：
   - 生成 `<word>.html`（默认输出到服务当前工作目录）
   - 自动把这份数据归档到 skill 的 `data/<word>.json` 以便复用
   - 在标准输出打印生成的 HTML 文件路径

   例如 args 传：`["{\"word\":\"resilient\",\"phonetic\":\"...\",...}"]`

4. **告知用户**：把脚本回报的 HTML 路径自然地告诉用户，说明可在浏览器打开查看，不要暴露脚本细节。

## 数据 JSON 示例

```json
{
  "word": "resilient",
  "phonetic": "/rɪˈzɪliənt/",
  "pos": "adj.",
  "definition": "能迅速从困难、挫折中恢复过来的；有韧性的，适应力强的",
  "examples": [
    {"en": "She is a resilient child who bounces back quickly from setbacks.", "zh": "她是个有韧性的孩子，遇到挫折能很快恢复过来。"},
    {"en": "The economy proved remarkably resilient during the crisis.", "zh": "在危机期间，经济表现出了惊人的韧性。"},
    {"en": "A resilient mindset helps you cope with life's challenges.", "zh": "一种有韧性的心态能帮你应对生活中的挑战。"}
  ],
  "synonyms": ["tough", "flexible", "strong", "hardy", "buoyant", "springy"]
}
```

## 注意事项

- 例句固定 3 条，脚本会自动截断或补占位，但生成数据时应直接给齐 3 条。
- 数据以内联 JSON 字符串传入脚本，脚本会自动归档到 `data/` 目录。
