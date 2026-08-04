"""根据结构化 JSON 生成一个无外部依赖的英语学习闪卡。"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"字段 {key!r} 必须是非空字符串")
    return value.strip()


def _validate(data: dict[str, Any]) -> dict[str, Any]:
    word = _required_text(data, "word")
    if not re.fullmatch(r"[A-Za-z][A-Za-z-]*", word):
        raise ValueError("word 只能包含英文字母和连字符")

    examples = data.get("examples")
    if not isinstance(examples, list) or len(examples) != 3:
        raise ValueError("examples 必须恰好包含三条")
    clean_examples = []
    for index, item in enumerate(examples, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"examples[{index}] 必须是对象")
        clean_examples.append(
            {
                "en": _required_text(item, "en"),
                "zh": _required_text(item, "zh"),
            }
        )

    synonyms = data.get("synonyms")
    if not isinstance(synonyms, list) or not 4 <= len(synonyms) <= 6:
        raise ValueError("synonyms 必须包含四到六个单词")
    clean_synonyms = []
    for item in synonyms:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("synonyms 中的每一项都必须是非空字符串")
        clean_synonyms.append(item.strip())

    return {
        "word": word.lower(),
        "phonetic": _required_text(data, "phonetic"),
        "pos": _required_text(data, "pos"),
        "definition": _required_text(data, "definition"),
        "examples": clean_examples,
        "synonyms": clean_synonyms,
    }


def _render(data: dict[str, Any]) -> str:
    word = html.escape(data["word"])
    phonetic = html.escape(data["phonetic"])
    pos = html.escape(data["pos"])
    definition = html.escape(data["definition"])
    synonym_html = "".join(
        f"<span class=\"chip\">{html.escape(item)}</span>"
        for item in data["synonyms"]
    )
    example_html = "".join(
        (
            "<li>"
            f"<p class=\"en\">{html.escape(item['en'])}</p>"
            f"<p class=\"zh\">{html.escape(item['zh'])}</p>"
            "</li>"
        )
        for item in data["examples"]
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{word} · Flash Card</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, "PingFang SC", "Microsoft YaHei", sans-serif;
      background: #07111f;
      color: #e5eef8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      min-height: 100vh;
      margin: 0;
      display: grid;
      place-items: center;
      padding: 32px;
      background:
        radial-gradient(circle at 15% 15%, #12375b 0, transparent 34%),
        radial-gradient(circle at 85% 80%, #143f36 0, transparent 32%),
        #07111f;
    }}
    main {{
      width: min(760px, 100%);
      border: 1px solid #2b5877;
      border-radius: 24px;
      padding: clamp(28px, 6vw, 58px);
      background: rgba(8, 24, 39, .92);
      box-shadow: 0 30px 80px rgba(0, 0, 0, .4);
    }}
    h1 {{
      margin: 0;
      font: 700 clamp(52px, 11vw, 92px)/1 Georgia, serif;
      letter-spacing: -.04em;
      color: #74e6cf;
    }}
    .sound {{ margin: 12px 0 36px; color: #a9bfd2; font-size: 20px; }}
    .definition {{
      margin: 0 0 28px;
      padding-left: 18px;
      border-left: 3px solid #ffbd70;
      font-size: 20px;
      line-height: 1.7;
    }}
    .pos {{ color: #ffbd70; font-weight: 700; margin-right: 8px; }}
    h2 {{
      margin: 30px 0 14px;
      color: #8ecaff;
      font-size: 15px;
      letter-spacing: .14em;
      text-transform: uppercase;
    }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 9px; }}
    .chip {{
      padding: 7px 12px;
      border: 1px solid #31546e;
      border-radius: 999px;
      background: #10283b;
      color: #cbe7f8;
    }}
    ol {{ margin: 0; padding-left: 24px; }}
    li {{ margin: 0 0 18px; padding-left: 8px; }}
    p {{ margin: 0; }}
    .en {{ font-size: 17px; line-height: 1.55; color: #f3f7fb; }}
    .zh {{ margin-top: 5px; line-height: 1.55; color: #91a9bc; }}
  </style>
</head>
<body>
  <main>
    <h1>{word}</h1>
    <p class="sound">{phonetic}</p>
    <p class="definition"><span class="pos">{pos}</span>{definition}</p>
    <h2>Synonyms</h2>
    <div class="chips">{synonym_html}</div>
    <h2>Examples</h2>
    <ol>{example_html}</ol>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    input_path = Path(args.json_path)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("JSON 根节点必须是对象")
    data = _validate(raw)

    output_path = Path(args.output) if args.output else Path(f"{data['word']}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render(data), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

