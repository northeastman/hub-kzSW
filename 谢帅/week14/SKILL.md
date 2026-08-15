---
name: work-summary-dashboard
description: >-
  Generate a polished single-file HTML "work summary" (工作总结) dashboard from a
  monthly log/日志报告 spreadsheet (.xls/.xlsx). Use this whenever the user wants
  to turn a log report, 日报/工作日志, or monthly activity export into a work
  summary, 月度工作总结, performance review page, or data 看板/dashboard — even if
  they only say "根据这个报告生成工作总结" or "把日志做成一个HTML页面". It reads the
  report, synthesizes the month's work into themed streams, renders KPIs plus a
  work-distribution donut, keeps AI提效分享 / 额外贡献 / 技术体系梳理沉淀 as separate
  modules, and ends with a 自我评价. Trigger it for review/总结/看板 requests over
  timesheet or log-report spreadsheets.
---

# 工作总结数据看板生成

Turn a monthly log report into a self-contained HTML dashboard the user can open,
print, or hand to a manager. The output is one `.html` file with inline CSS + a
tiny scroll-reveal script — no build step, no external assets.

## Workflow

### 1. Locate & read the report
The source is a spreadsheet in a folder the user names (often a Chinese path).
Do NOT paste the CJK path into a bash heredoc — it gets mangled. Instead run the
bundled extractor, which resolves the file by listing the folder:

```bash
python scripts/extract_report.py "<folder>" --contains "工作总结"
```

It prints META (file/sheet/shape) then every non-empty cell as `[row,col] value`.
Old `.xls` needs `xlrd`, new `.xlsx` needs `openpyxl` — the script picks the
right engine. If a dependency is missing, `pip install pandas xlrd openpyxl`.

Read the dumped cells carefully. Daily reports are usually in one long column;
the header rows hold name, department, month, total hours, completion %, and
deviation %. This raw dump is your only source of truth.

### 2. Synthesize the content (this is the real work)
The report is raw daily entries; the dashboard is a synthesis. From the cells:
- **Headline KPIs** (4): pull real numbers — total hours, task completion %,
  workload deviation %, count of daily reports. Use what's there; don't invent.
- **Work streams** (2–4): cluster the daily entries into the month's main
  threads (e.g. a platform build, a data-matching task, production ops). Give
  each a short title + 3–4 bullet outcomes. This is where you add value by
  organizing scattered entries into a coherent story.
- **AI 提效分享 / 额外贡献 / 技术体系梳理沉淀**: scan for these specifically —
  they map to review criteria and the user wants them as **separate modules**,
  not folded into the streams. Keep each independent. Drop a module only if the
  report genuinely has nothing for it.
- **自我评价**: 3–5 honest, specific points grounded in the above (交付把控 /
  技术攻坚 / 提效创新 / 责任担当 style), plus a 改进方向 list. Keep it credible,
  not inflated.

### 3. Build the HTML
Copy `assets/template.html` to the output location, then replace its example
content section by section with the synthesized data. The template is a filled
reference (annotated at the top) — follow its structure and class names rather
than reinventing markup. Read `references/design-system.md` before editing so
you preserve the look: one accent color per series, `.num` on every meaningful
number, donut slice %s matching the legend, the three modules kept separate.

Save alongside the source with a clear name, e.g.
`工作总结-<姓名>-<YYYY>年<MM>月.html`.

### 4. Validate structure before finishing
Multi-section edits easily drop a closing tag. After building, confirm the tags
balance — an unclosed `</style>` or mismatched `<div>` silently breaks the page:

```bash
python - <<'PY'
import re,sys
h=open(r"<output.html>",encoding="utf-8").read()
for t in ("style","script","head","body"):
    assert h.count(f"<{t}")==h.count(f"</{t}>"), f"{t} unbalanced"
o,c=len(re.findall(r"<div\b",h)),h.count("</div>")
assert o==c, f"div unbalanced {o} vs {c}"
print("OK: tags balanced")
PY
```

## Guardrails
- **Data honesty.** Render only what the report supports. If a number is an
  estimate (overlapping parallel workdays inflating a distribution), flag it in
  the panel hint rather than presenting it as exact. Never fabricate a KPI to
  fill a slot — remove the slot.
- **Let the user pick style only if they push back.** This dashboard style is
  the settled default. If the user calls the result ugly or wants a different
  feel, don't keep guessing — offer 2–3 concrete directions and let them choose.
- **Reply in the user's language** (Chinese for these reports).
