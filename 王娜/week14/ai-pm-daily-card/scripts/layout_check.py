#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check a rendered card HTML for overflow / text clipping.

Usage:
    python layout_check.py --html card.html [--strict]

Injects a small JS probe, dumps the DOM via headless Chrome, and reports
any element that overflows the .card bounds or clips its own text.
Exits non-zero when issues are found (always if --strict, otherwise only
for real overflow beyond the card boundary).
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROBE_JS = """
<script>
window.addEventListener('load', function () {
  var out = [];
  var card = document.querySelector('.card');
  var c = card.getBoundingClientRect();
  var bad = [];
  document.querySelectorAll('.card *').forEach(function (el) {
    var r = el.getBoundingClientRect();
    if (r.bottom > c.bottom + 1 || r.right > c.right + 1 || r.left < c.left - 1 || r.top < c.top - 1) {
      bad.push('out:' + (el.className || el.tagName) + ' b=' + Math.round(r.bottom));
    }
    if (el.scrollHeight > el.clientHeight + 1 && el.clientHeight > 0) {
      bad.push('clip:' + (el.className || el.tagName));
    }
  });
  out.push(bad.length ? bad.slice(0, 20).join(' | ') : 'OK');
  var pre = document.createElement('pre');
  pre.id = 'checkout';
  pre.textContent = out.join('\\n');
  document.body.appendChild(pre);
});
</script>
"""


def chrome_bin() -> str:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    which = shutil.which("chrome") or shutil.which("msedge")
    if which:
        return which
    print("ERROR: Chrome/Edge not found", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Layout check for rendered card HTML")
    ap.add_argument("--html", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="fail on any clip issue too")
    args = ap.parse_args()

    html_path = Path(args.html)
    html = html_path.read_text(encoding="utf-8")
    if "</body>" not in html:
        print("ERROR: no </body> in html", file=sys.stderr)
        return 1
    check_html = html.replace("</body>", PROBE_JS + "\n</body>")

    tmpdir = html_path.parent / "_layout_tmp"
    tmpdir.mkdir(exist_ok=True)
    profile = tmpdir / "profile"
    profile.mkdir(exist_ok=True)
    check_path = tmpdir / "check.html"
    check_path.write_text(check_html, encoding="utf-8")

    cmd = [
        chrome_bin(), "--headless=new", "--disable-gpu", "--no-sandbox",
        "--disable-dev-shm-usage", f"--user-data-dir={profile}",
        "--virtual-time-budget=4000", "--dump-dom", check_path.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    dom = proc.stdout
    m = re.search(r'<pre id="checkout">(.*?)</pre>', dom, re.S)
    if not m:
        print("CHECK-FAILED: probe did not run")
        if proc.stderr:
            print(proc.stderr[-500:])
        return 1

    result = m.group(1).strip().split("\n")
    lines = [ln for ln in result if ln.strip()]
    issues = lines[-1] if lines else "OK"
    print("layout=" + issues)
    if issues != "OK":
        if args.strict:
            return 1
        if "out:" in issues:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
