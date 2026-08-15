# -*- coding: utf-8 -*-
"""
AI 每日新闻自动推送 - 调度入口脚本（RSS + DeepSeek 版）

工作流程：
    1. 从国内可直连的 RSS 源（36氪/IT之家/少数派）抓取新闻
    2. 用 AI 关键词过滤 + 近 48 小时时效筛选，得到 AI 新闻素材
    3. 读取近 7 天已生成文章的标题，用于跨天去重
    4. 把素材 + 历史标题 + SKILL.md 指令一起发给 DeepSeek
    5. 模型筛选、撰稿，保存为 output/yyyy-mm-dd.md

分工说明：
    - 联网抓取由本脚本完成（RSS，国内可直连，无需代理）
    - 大模型只负责基于素材筛选、撰写文章

使用前准备：
    1. pip install openai feedparser python-dotenv requests
    2. 复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY
    3. 配置 Windows 任务计划程序每天定时运行 run_daily.bat
"""

import os
import re
import socket
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import requests
from dotenv import load_dotenv
from openai import OpenAI

# feedparser 超时控制（parse 不支持 timeout 参数，需全局设置）
socket.setdefaulttimeout(20)

# ======================== 路径定义 ========================
SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_PATH = SCRIPT_DIR / ".trae" / "skills" / "ai-news-daily" / "SKILL.md"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOG_PATH = SCRIPT_DIR / "run_daily.log"
ENV_PATH = SCRIPT_DIR / ".env"

# ======================== 读取配置 ========================
load_dotenv(ENV_PATH)
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com").strip()
MODEL = os.getenv("MODEL", "deepseek-chat").strip()

# RSS 源（可在 .env 用 RSS_FEEDS="url1|url2" 覆盖）
# (url, weight, source_name) —— 权重越高代表来源越权威，打分时加权
# 核心3源（权重3）：AI/技术专业媒体，质量最高，不拉低均分
# 补充2源（权重1）：量大但需筛选，仅作补充
# 已剔除：博客园(个人博客非新闻)、钛媒体(偏商业)、36氪(偏创投) —— 拉低均分
DEFAULT_RSS = [
    # 官方一手源（权重4）：最准确，英文内容，更新慢但保真
    ("https://openai.com/blog/rss.xml", 4, "OpenAI官方"),
    ("https://blog.google/technology/ai/rss/", 4, "Google官方"),
    ("https://blogs.nvidia.com/feed/", 4, "NVIDIA官方"),
    # 官方仓库发布（权重4）：DeepSeek 无官网 RSS，用 GitHub releases 替代
    ("https://github.com/deepseek-ai/DeepSeek-V3/releases.atom", 4, "DeepSeek官方"),
    # 核心中文源（权重3）：AI/技术专业媒体
    ("https://www.qbitai.com/feed", 3, "量子位"),
    ("https://www.leiphone.com/feed", 3, "雷峰网"),
    ("https://www.infoq.cn/feed", 3, "InfoQ"),
    # 产业向媒体（权重2）：补充视角
    ("https://www.geekpark.net/rss", 2, "极客公园"),
    # 补充源（权重1）
    ("https://juejin.cn/rss", 1, "掘金"),
    ("https://www.ithome.com/rss/", 1, "IT之家"),
]
# 环境变量覆盖：用户自定义源无法指定权重，默认权重 1
_rss_env = os.getenv("RSS_FEEDS", "").strip()
RSS_FEEDS = (
    [(u.strip(), 1, "自定义") for u in _rss_env.split("|") if u.strip()]
    or DEFAULT_RSS
)

# 近多少天的标题用于去重
DEDUP_DAYS = int(os.getenv("DEDUP_DAYS", "7"))

# 抓取多少小时内的新闻
FRESH_HOURS = int(os.getenv("FRESH_HOURS", "48"))

# AI 关键词（用于从综合 RSS 中过滤出 AI 新闻）
AI_KEYWORDS = re.compile(
    r"AI|人工智能|大模型|GPT|LLM|深度学习|机器学习|智能体|Agent|"
    r"Claude|Gemini|通义|文心|千问|DeepSeek|Sora|多模态|AIGC|生成式|"
    r"Copilot|Cursor|Claude Code|Codeium|代码生成|编程助手|IDE|"
    r"开源模型|开源权重|HuggingFace|GPU|算力|芯片|"
    r"具身智能|机器人|自动驾驶|RAG|微调|推理|训练",
    re.IGNORECASE,
)

# 重磅词（标题含这些词说明是"事件性新闻"，加分）
HEAVY_WORDS = re.compile(
    r"发布|开源|突破|登顶|首发|更新|降价|涨价|参数|榜单|超越|碾压|"
    r"里程碑|上线|公测|集成|支持|推出|开源权重|开源模型|开源框架|"
    r"崩了|宕机|泄露|离职|收购|融资",
    re.IGNORECASE,
)
# 噪音词（标题含这些词说明是"非新闻内容"，强力降权排除）
NOISE_WORDS = re.compile(
    r"体验|测评|评测|源码阅读|源码解析|笔记|教程|手把手|入门|攻略|"
    r"盘点|汇总|解读|分析|展望|预测|周报|月报|weekly|Weekly|"
    r"如何|为什么|聊聊|谈谈|拆解|指南|面试|简历|招聘",
    re.IGNORECASE,
)

# 官方源域名（用于判断"官宣"类型）
OFFICIAL_DOMAINS = (
    "openai.com", "blog.google", "blogs.nvidia.com",
    "anthropic.com", "ai.meta.com", "blogs.microsoft.com",
    "huggingface.co",
    "github.com/deepseek-ai",  # DeepSeek 官方仓库 releases
)
# 传闻词（标题含这些词 → 标注"传闻"）
RUMOR_WORDS = re.compile(
    r"据传|或将|疑似|传闻|可能|或成|曝光|泄密|泄露|"
    r"leak|rumor|may|possibly|reportedly|said to|purported",
    re.IGNORECASE,
)
# 分析词（标题含这些词 → 标注"分析"）
ANALYSIS_WORDS = re.compile(
    r"分析|解读|深度|视角|复盘|趋势|展望|思考|拆解|"
    r"analysis|deep dive|perspective|explainer|takeaway",
    re.IGNORECASE,
)


def log(msg: str) -> None:
    """写日志并打印"""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_skill_instruction() -> str:
    """读取 SKILL.md 全文作为系统指令"""
    if not SKILL_PATH.exists():
        raise FileNotFoundError(f"找不到 SKILL.md：{SKILL_PATH}")
    return SKILL_PATH.read_text(encoding="utf-8")


def _entry_time(entry) -> datetime | None:
    """从 feedparser entry 解析发布时间，失败返回 None"""
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6])
            except Exception:
                pass
    return None


def _infer_info_type(title: str, url: str) -> str:
    """判断信息类型：官宣/传闻/分析/报道"""
    for domain in OFFICIAL_DOMAINS:
        if domain in url:
            return "官宣"
    if RUMOR_WORDS.search(title):
        return "传闻"
    if ANALYSIS_WORDS.search(title):
        return "分析"
    return "报道"


def _score_item(title: str, weight: int, t, now: datetime) -> float:
    """素材打分：来源权重 + 重磅词 + 时效 - 噪音词"""
    score = weight * 2.0
    if HEAVY_WORDS.search(title):
        score += 3.0
    if NOISE_WORDS.search(title):
        score -= 6.0  # 非新闻强力降权
    if t:
        hours_ago = (now - t).total_seconds() / 3600
        if hours_ago < 24:
            score += 3.0
        elif hours_ago < 48:
            score += 1.0
        else:
            score -= 2.0
    return score


def search_news() -> list:
    """从 RSS 源抓取 AI 新闻，按关键词和时效过滤，打分粗排后返回 top 30"""
    items = []
    seen_urls = set()
    now = datetime.now()
    cutoff = now - timedelta(hours=FRESH_HOURS)
    for url, weight, source_name in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                log(f"RSS 源无内容：{source_name}（{url}，bozo={feed.bozo}）")
                continue
            matched = 0
            for e in feed.entries:
                title = (e.get("title") or "").strip()
                link = (e.get("link") or "").strip()
                summary = (e.get("summary") or "").strip()
                if not title or not link or link in seen_urls:
                    continue
                # AI 关键词过滤
                if not AI_KEYWORDS.search(title + " " + summary):
                    continue
                # 时效过滤（解析不到时间的保留，交给模型判断）
                t = _entry_time(e)
                if t and t < cutoff:
                    continue
                seen_urls.add(link)
                # 清理 summary 里的 HTML 标签
                summary_clean = re.sub(r"<[^>]+>", "", summary)[:120]
                # 打分
                score = _score_item(title, weight, t, now)
                # 时间格式化
                time_str = t.strftime("%Y-%m-%d %H:%M") if t else "未知"
                # 信息类型标注
                info_type = _infer_info_type(title, link)
                items.append({
                    "title": title,
                    "url": link,
                    "snippet": summary_clean,
                    "time": t,
                    "time_str": time_str,
                    "source": source_name,
                    "score": score,
                    "info_type": info_type,
                })
                matched += 1
            log(f"RSS {source_name} → 命中 AI 新闻 {matched} 条")
        except Exception as e:
            log(f"RSS 源抓取失败 {source_name}（{url}）：{e}")
    # 按分数降序，取 top 20 给模型（省 token + 提质）
    items.sort(key=lambda x: x["score"], reverse=True)
    top = items[:20]
    log(f"素材打分粗排：{len(items)} 条 → 取 top {len(top)} 条喂给模型")
    return top


def fetch_github_trending() -> list:
    """抓取 GitHub Trending，返回 AI 相关热门开源项目（补充素材）"""
    items = []
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(
            "https://github.com/trending?since=daily",
            timeout=15, headers=headers,
        )
        if r.status_code != 200:
            log(f"GitHub Trending 抓取失败：HTTP {r.status_code}")
            return items
        # 解析仓库名和描述
        repos = re.findall(
            r'<h2 class="h3 lh-condensed">.*?<a href="(/[^"]+)"', r.text, re.S
        )
        descs = re.findall(
            r'<p class="col-9 color-fg-muted my-1 pr-4">\s*(.*?)\s*</p>',
            r.text, re.S,
        )
        matched = 0
        for i, repo_path in enumerate(repos):
            repo_path = repo_path.strip()
            if "/stargazers" in repo_path:
                repo_path = repo_path.replace("/stargazers", "")
            desc = re.sub(r"<[^>]+>", "", descs[i]).strip() if i < len(descs) else ""
            # AI 关键词过滤
            if not AI_KEYWORDS.search(repo_path + " " + desc):
                continue
            items.append({
                "title": f"🔥 {repo_path[1:]}",
                "url": f"https://github.com{repo_path}",
                "snippet": desc[:120],
                "time": None,
                "time_str": "今日",
                "source": "GitHub Trending",
                "score": 2.0,
                "info_type": "项目",
            })
            matched += 1
            if matched >= 5:
                break
        log(f"GitHub Trending → 命中 AI 相关项目 {matched} 个")
    except Exception as e:
        log(f"GitHub Trending 抓取失败：{e}")
    return items


def get_recent_titles(days: int = 7) -> list:
    """读取近 days 天 output 里的文章，提取 ## 标题用于跨天去重"""
    titles = []
    for i in range(1, days + 1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        p = OUTPUT_DIR / f"{d}.md"
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("## "):
                    titles.append(line[3:].strip())
        except Exception:
            pass
    return titles[-15:]


def build_user_prompt(items: list, recent_titles: list) -> str:
    """拼接 user message：素材(含时间和来源) + 近7天已推送标题"""
    today = datetime.now().strftime("%Y-%m-%d")
    material = "\n\n".join(
        f"[{i + 1}] {it['title']}\n类型: {it['info_type']} | 来源: {it['source']} | 发布: {it['time_str']} | {it['url']}\n摘要: {it['snippet']}"
        for i, it in enumerate(items)
    )
    dup_block = ""
    if recent_titles:
        dup_block = "\n\n## 近 7 天已推送标题（必须避开，不要重复推送）：\n" + "\n".join(
            f"- {t}" for t in recent_titles
        )
    return (
        f"今天是 {today}。以下是从 RSS 源采集到的 AI 新闻素材（已按重要性打分排序）：\n\n"
        f"{material}\n{dup_block}\n\n"
        f"请严格按照 SKILL.md 的步骤：从上述素材中筛选 6-10 条近 24 小时的新闻"
        f"（重磅不足时可少写几条，宁缺毋滥，同源事件合并），"
        f"避开已推送标题，撰写一篇中文公众号文章。"
    )


def call_llm_with_retry(system_prompt: str, user_prompt: str, retries: int = 3) -> str:
    """调用 DeepSeek API，失败自动重试"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=2500,
                timeout=180,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            wait = 5 * attempt
            log(f"API 第 {attempt}/{retries} 次调用失败：{e}，{wait}秒后重试...")
            time.sleep(wait)
    raise RuntimeError(f"API 调用 {retries} 次均失败：{last_err}")


def save_article(content: str) -> Path:
    """把生成的文章保存为 output/yyyy-mm-dd.md"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y-%m-%d')}.md"
    out_path = OUTPUT_DIR / filename
    out_path.write_text(content, encoding="utf-8")
    return out_path


def main() -> int:
    log("==== AI 每日新闻任务开始 ====")
    try:
        if not API_KEY:
            log("!!!! 未配置 DEEPSEEK_API_KEY，请检查 .env 文件")
            return 1

        skill = load_skill_instruction()
        log(f"已加载 SKILL.md（{len(skill)} 字符）")

        items = search_news()
        gh_items = fetch_github_trending()
        if gh_items:
            items.extend(gh_items)
        log(f"共搜索到素材 {len(items)} 条（RSS {len(items)-len(gh_items)} + GitHub {len(gh_items)}）")
        if not items:
            log("!!!! 未抓到任何素材，跳过本次生成（不调用 API，省 token）")
            return 1
        if len(items) < 5:
            log(f"!! 素材仅 {len(items)} 条（不足5条），仍继续，模型会按实际情况处理")

        recent = get_recent_titles(DEDUP_DAYS)
        log(f"近 {DEDUP_DAYS} 天已推送 {len(recent)} 条标题，将用于去重")

        user_prompt = build_user_prompt(items, recent)
        article = call_llm_with_retry(skill, user_prompt)
        log(f"大模型返回内容长度：{len(article)} 字符")

        out_path = save_article(article)
        log(f"文章已保存：{out_path}")
        log("==== 任务完成 ====")
        return 0
    except Exception as e:
        log(f"!!!! 任务失败：{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
