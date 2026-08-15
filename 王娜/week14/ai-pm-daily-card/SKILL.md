---
name: ai-pm-daily-card
description: 生成 AI 产品经理每日知识卡片并输出小红书发布包（3:4 卡片 PNG、标题、正文、话题标签、参考来源、打卡记录）。当用户说「今天的知识卡片」「每日知识卡」「AI产品经理知识卡」「生成知识卡片」、或要求制作 AI 产品经理/大模型主题的小红书知识卡内容、或提到 ai-pm-daily-card 时使用。
---

# AI 产品经理每日知识卡

## 概览

每天为 AI 产品经理生成一张 3:4 知识卡片（1242×1656 PNG）和配套小红书发布包。工作流：选题 → 写内容 → 写文案（含验证过的参考来源）→ 渲染 → 布局自检 → 组装发布包 → 用户发布后打卡。

学习路径：题库按 9 大模块由浅入深排列（大模型基础 → RAG → Agent → 提示词工程 → AI 产品设计 → 评测与数据 → 商业化与增长 → 合规与伦理 → 行业案例与趋势），默认按题库顺序每天连载一张，知识前后承接、不跳题。卡片底部显示「模块 X/9 · 第 X/100 张」让连载进度一目了然。

## 文件位置

- Skill 资源：本目录（scripts / assets / references）
- 数据目录：`C:\Users\17717\Documents\Codex\xhs-knowledge-card-data\`
  - `tracker.csv`：打卡记录（date,topic_id,category,title,day,status,note_url）
  - `cards\`：每日内容 JSON、渲染用 HTML、卡片 PNG
  - `packs\`：发布包 Markdown
- 运行时把 tracker 和产出写到数据目录，不要写进 skill 目录。

## 工作流（按顺序执行）

### 1. 选题

```powershell
python scripts/pick_topic.py --bank references/topic_bank.json --tracker <数据目录>\tracker.csv
```

输出今日题目 JSON（id/category/title/date/day/status）。脚本默认按题库顺序选择下一个未用题目（即沿学习路径连载），同一天重复运行幂等；如需随机轮转模式可加 `--mode rotating`。

### 2. 写卡片内容

按选题撰写 `cards\YYYY-MM-DD_content.json`：

```json
{
  "title": "题目",
  "category": "分类",
  "lead": "一句话导语",
  "points": [{"title": "小标题", "desc": "说明"}, ...],
  "quote": "金句",
  "date": "YYYY-MM-DD",
  "day": 2,
  "module_no": 1,
  "module_total": 9,
  "card_no": 2,
  "total": 100
}
```

要求：内容真实准确、不编造；要点 3-5 个且能讲清楚；专业术语给一句话解释。module_no/module_total 从当前题目所属模块计算（1-9），文件用 UTF-8 写入。

### 3. 写小红书文案

遵循 `references/content_style.md` 撰写 `cards\YYYY-MM-DD_copy.json`：

```json
{
  "title": "标题（≤20字）",
  "body": "正文（四段式，200-400字）",
  "tags": ["AI产品经理", "大模型", ...],
  "image": "YYYY-MM-DD_card.png",
  "sources": [{"title": "来源标题", "url": "https://..."}]
}
```

要点：
- 标题、正文、标签规则见 content_style.md，必须遵守。
- sources 必须 1-3 个；**本轮用搜索或打开网页验证每个链接可访问**，禁止编造链接；失效就替换或删除，宁缺毋滥。
- image 用渲染出的卡片 PNG 文件名。

### 4. 渲染卡片

```powershell
python scripts/render_card.py --content cards\YYYY-MM-DD_content.json --template assets/card_template.html --output cards\YYYY-MM-DD_card.png
```

### 5. 布局自检

```powershell
python scripts/layout_check.py --html cards\YYYY-MM-DD_card.html --strict
```

必须输出 `layout=OK`。出现 out:/clip: 问题时缩短导语或要点说明后重新渲染，直到通过。

### 6. 组装发布包

```powershell
python scripts/make_publish_pack.py --content cards\YYYY-MM-DD_content.json --copy cards\YYYY-MM-DD_copy.json --template assets/publish_pack_template.md --output packs\发布包_YYYY-MM-DD.md
```

### 7. 交付

把卡片 PNG 和发布包（标题/正文/标签/参考来源）展示给用户，说明半自动发布：用户在小红书 App 上传图片、粘贴文案、发布。

### 8. 打卡

用户发布后把笔记链接发回来时，更新 `tracker.csv` 对应行：`status=published`、`note_url=笔记链接`。

## 环境注意

- Windows PowerShell 运行 python 前设置 `$env:PYTHONIOENCODING='utf-8'`；含中文参数的 Python 脚本请写成 .py 文件再运行，避免管道编码损坏中文。
- 中文文件一律显式 UTF-8 写入（`[System.IO.File]::WriteAllText(path, content, [System.Text.UTF8Encoding]::new($false))` 或 Python `open(..., encoding="utf-8")`），不要用 `Set-Content` 默认编码。
- 渲染用本机 Chrome/Edge 无头截图；脚本自动找常见安装路径，可用环境变量 `RENDER_CHROME` 覆盖。
- 题库当前 100 题，顺序即学习路径；新增题目按路径插入对应模块并重排 id，保持 JSON 合法。