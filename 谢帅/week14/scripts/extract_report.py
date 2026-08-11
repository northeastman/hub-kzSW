#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract structured data from a monthly log report (.xls/.xlsx).

Handles two recurring pain points on this setup:
  1. Chinese characters in Windows paths get mangled through bash/argv, so we
     resolve the target file by listing the directory rather than trusting an
     interpolated path. Pass the *folder* and (optionally) a filename filter.
  2. Old-format .xls (Composite Document) needs the xlrd engine, not openpyxl.

Usage:
    python extract_report.py "<folder>" [--contains "工作总结"] [--sheet 0]

Prints a JSON blob + a human-readable dump of every non-empty cell so the
caller can eyeball the raw daily reports and synthesize the summary from them.
Never invents data — it only surfaces what is in the sheet.
"""
import sys, os, json, argparse

# Windows consoles default to gbk and turn the CJK dump into mojibake; force
# UTF-8 so the printed cells stay readable (the whole point of the dump).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def find_file(folder, contains):
    names = os.listdir(folder)  # avoids passing a CJK path back through argv
    cands = [n for n in names if n.lower().endswith((".xls", ".xlsx"))]
    if contains:
        pref = [n for n in cands if contains in n]
        if pref:
            cands = pref
    if not cands:
        raise SystemExit(f"No .xls/.xlsx found in {folder!r}. Saw: {names}")
    # newest by mtime, so re-runs pick up the latest report
    cands.sort(key=lambda n: os.path.getmtime(os.path.join(folder, n)))
    return os.path.join(folder, cands[-1])


def load(path, sheet):
    import pandas as pd
    engine = "xlrd" if path.lower().endswith(".xls") else "openpyxl"
    xls = pd.ExcelFile(path, engine=engine)
    name = xls.sheet_names[sheet] if isinstance(sheet, int) else sheet
    df = xls.parse(name, header=None)
    return name, df


def dump(df):
    """Every non-empty cell as (row, col, value) — the raw material to read."""
    out = []
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            v = df.iat[r, c]
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() != "nan":
                out.append({"row": r, "col": c, "value": s})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--contains", default="")
    ap.add_argument("--sheet", default="0")
    a = ap.parse_args()
    sheet = int(a.sheet) if a.sheet.isdigit() else a.sheet

    path = find_file(a.folder, a.contains)
    name, df = load(path, sheet)
    cells = dump(df)

    meta = {"file": os.path.basename(path), "sheet": name,
            "shape": list(df.shape), "cell_count": len(cells)}
    print("=== META ===")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\n=== CELLS (row,col: value) ===")
    for c in cells:
        print(f"[{c['row']:>3},{c['col']:>2}] {c['value']}")


if __name__ == "__main__":
    main()
