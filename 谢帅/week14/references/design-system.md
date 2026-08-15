# 数据看板设计系统 (data-dashboard style)

The look the user settled on after rejecting gradient-tech and logbook styles.
It reads as a clean BI dashboard: white cards on a soft grey field, one accent
color per data series, monospace numerals. Keep these invariants when adapting
`assets/template.html` — they are what make it feel designed rather than busy.

## Color tokens (CSS custom properties, already in template)
- Neutrals: `--bg:#eef1f6` field, `--card:#fff`, `--ink:#161a22` text,
  `--sub:#5b6472` secondary, `--faint:#98a1b0` tertiary, `--line:#eaedf2`.
- Accents, each with a soft `-s` tint for icon chips:
  indigo `#4f46e5`, teal `#0d9488`, amber `#e08a00`, rose `#e11d48`, sky `#0284c7`.
- Assign ONE accent per data series and reuse it everywhere that series appears
  (KPI stripe, donut slice, legend dot, stream card top-border). Consistency of
  color→meaning is the whole point; do not randomize.

## Typography
- Body: CJK sans stack (`--sans`), 14px, line-height 1.7.
- Every number that carries meaning gets `class="num"` → monospace + tabular
  figures. This is the single biggest "it looks like a dashboard" signal.

## Layout rhythm (top → bottom)
1. **topbar** — dark gradient banner: eyebrow + h1 title + crumbs (name · dept),
   right side 2–3 glass `.tbstat` chips for headline numbers.
2. **.kpis** — 4 KPI cards, colored left stripe + icon chip. Headline metrics.
3. **.cols** — two panels: a donut (work distribution) + progress bars (goal
   attainment). Donut is `conic-gradient`; slice %s MUST match the legend.
4. **§01 概述** — 2–4 `.scard` stream cards, one accent border each.
5. **Modules** — AI / EXT / DOC as separate `.section`+`.mod` blocks, each with
   a colored `.tag`. These stay independent even if short; that separation was
   an explicit user requirement.
6. **.eval 自我评价** — dark panel, 3–5 points in a 2-col grid + 改进方向 list.
7. **footer** — mono line citing the source file.

## Motion & robustness
- `.reveal` + IntersectionObserver fade-in; respects `prefers-reduced-motion`.
- Responsive `@media(max-width:820px)` collapses grids to 1 col.
- `@media print` drops shadows, keeps eval panel colors — it prints cleanly.

## Data honesty
- Render only what the report contains. If a distribution is an estimate (e.g.
  overlapping parallel workdays), say so in the panel `.hint`, don't present it
  as exact. Never invent KPIs to fill a slot — remove the slot instead.
