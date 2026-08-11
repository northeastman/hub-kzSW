#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a knowledge card PNG (1242x1656, 3:4) from content JSON + HTML template.

Usage:
    python render_card.py --content card.json --template card_template.html --output card.png

Content JSON schema:
{
  "title": str, "category": str, "lead": str, "quote": str,
  "points": [{"title": str, "desc": str}, ...],   # 3-5 items
  "date": "YYYY-MM-DD", "day": int, "card_no": int, "total": int
}

Uses headless Chrome/Edge; override the binary with RENDER_CHROME env var.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

WIDTH = 1242
HEIGHT = 1656


def find_chrome() -> str:
    override = os.environ.get("RENDER_CHROME", "").strip()
    if override and Path(override).exists():
        return override
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    which = shutil.which("chrome") or shutil.which("msedge")
    if which:
        return which
    print("ERROR: Chrome/Edge not found; set RENDER_CHROME", file=sys.stderr)
    sys.exit(1)


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_points_html(points: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(points, start=1):
        blocks.append(
            '      <div class="point">\n'
            f'        <div class="num">{i}</div>\n'
            '        <div class="point-body">\n'
            f'          <div class="point-title">{esc(p["title"])}</div>\n'
            f'          <div class="point-desc">{esc(p["desc"])}</div>\n'
            '        </div>\n'
            '      </div>'
        )
    return "\n".join(blocks)


def fill_template(template: str, content: dict) -> str:
    total = int(content.get("total", 100))
    day = int(content.get("day", 1))
    card_no = int(content.get("card_no", day))
    module_no = content.get("module_no")
    module_progress = (
        f"模块 {module_no}/{content['module_total']} · "
        if module_no is not None else ""
    )
    return (
        template
        .replace("{{TOPIC_TAG}}", esc(content["category"]))
        .replace("{{TITLE}}", esc(content["title"]))
        .replace("{{LEAD}}", esc(content["lead"]))
        .replace("{{POINTS}}", build_points_html(content["points"]))
        .replace("{{QUOTE}}", esc(content["quote"]))
        .replace("{{DAY}}", f"DAY {day:02d}")
        .replace("{{DATE}}", esc(content["date"]))
        .replace("{{MODULE_PROGRESS}}", module_progress)
        .replace("{{PROGRESS}}", f"第 {card_no}/{total} 张")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Render knowledge card PNG")
    ap.add_argument("--content", required=True, help="content JSON path")
    ap.add_argument("--template", required=True, help="HTML template path")
    ap.add_argument("--output", required=True, help="output PNG path")
    args = ap.parse_args()

    content = json.loads(Path(args.content).read_text(encoding="utf-8"))
    template = Path(args.template).read_text(encoding="utf-8")
    html = fill_template(template, content)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_html = out.with_suffix(".html")
    tmp_html.write_text(html, encoding="utf-8")

    chrome = find_chrome()
    profile = out.parent / "_render_tmp" / "profile"
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-software-rasterizer",
        "--hide-scrollbars",
        f"--user-data-dir={profile}",
        f"--window-size={WIDTH},{HEIGHT}",
        "--virtual-time-budget=4000",
        f"--screenshot={out}",
        tmp_html.as_uri(),
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    shutil.rmtree(profile.parent, ignore_errors=True)
    if not out.exists() or out.stat().st_size == 0:
        print(f"ERROR: render failed, no output at {out}", file=sys.stderr)
        return 1
    print(f"rendered={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
