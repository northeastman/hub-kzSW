---
name: flash-card
description: 为一个英语单词生成静态 HTML 学习闪卡，包含音标、词性、中文释义、近义词和三条中英对照例句。
version: 1.0.0
triggers:
  - 英语单词闪卡
  - flash card
  - 单词卡
allowed_tools:
  - write_workspace_file
  - run_skill_script
---

# Flash Card 单词闪卡

## 目标

根据用户给出的一个英语单词，在当前工作目录生成 `<word>.html`。

## 执行流程

1. 从用户请求中提取一个英语单词，并转换为小写。
2. 准备以下 JSON 数据：
   - `word`：单词。
   - `phonetic`：音标。
   - `pos`：词性。
   - `definition`：简洁、准确的中文释义。
   - `examples`：恰好三条，每条包含 `en` 和 `zh`。
   - `synonyms`：四到六个近义词。
3. 调用 `write_workspace_file`，写入：
   `skills/flash-card/data/<word>.json`
4. 调用 `run_skill_script`：
   - `skill_name`: `flash-card`
   - `script`: `make_flashcard.py`
   - `args`: `["skills/flash-card/data/<word>.json"]`
5. 检查工具的 `exit_code`。只有 `exit_code=0` 才能告诉用户生成成功。
6. 最终回答给出生成文件的相对路径。

## 数据格式

```json
{
  "word": "resilient",
  "phonetic": "/rɪˈzɪliənt/",
  "pos": "adj.",
  "definition": "有韧性的；能迅速从困难中恢复的",
  "examples": [
    {
      "en": "She remained resilient after the setback.",
      "zh": "遭遇挫折后，她依然很坚韧。"
    },
    {
      "en": "The local economy is surprisingly resilient.",
      "zh": "当地经济有着惊人的韧性。"
    },
    {
      "en": "Resilient teams learn quickly from mistakes.",
      "zh": "有韧性的团队能迅速从错误中学习。"
    }
  ],
  "synonyms": ["tough", "strong", "hardy", "flexible"]
}
```

## 质量要求

- 例句必须自然，并体现该单词的典型用法。
- 中文翻译需要与英文逐句对应。
- 不得捏造已经生成的文件；必须以脚本实际返回结果为准。
- 不需要读取或修改本 Skill 之外的文件。

