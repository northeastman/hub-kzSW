#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the publish pack (markdown) for the daily knowledge card.

Usage:
    python make_publish_pack.py --content card.json --copy copy.json \
        --template publish_pack_template.md --output pack.md

copy.json schema:
{
  "title": str,      # <= 20 chars, follows content_style.md
  "body": str,       # 200-400 chars
  "tags": [str, ...] # 3-5 hashtags without '#'
  "image": str       # PNG filename to attach
  "sources": [{"title": str, "url": str}, ...]  # 1-3 verified references
}
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the XHS publish pack markdown")
    ap.add_argument("--content", required=True)
    ap.add_argument("--copy", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    content = json.loads(Path(args.content).read_text(encoding="utf-8"))
    copy = json.loads(Path(args.copy).read_text(encoding="utf-8"))
    template = Path(args.template).read_text(encoding="utf-8")

    tags_line = " ".join(f"#{t}" for t in copy["tags"])
    sources_lines = "\n".join(
        f"- [{s['title']}]({s['url']})" for s in copy.get("sources", [])
    )
    pack = (
        template
        .replace("{{TITLE}}", copy["title"].strip())
        .replace("{{BODY}}", copy["body"].strip())
        .replace("{{TAGS}}", tags_line)
        .replace("{{SOURCES}}", sources_lines)
        .replace("{{DATE}}", content["date"])
        .replace("{{DAY}}", f"{int(content['day']):02d}")
        .replace("{{IMAGE}}", copy["image"])
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pack, encoding="utf-8")
    print(f"pack={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
