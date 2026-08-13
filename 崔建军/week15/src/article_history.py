"""文章历史追踪：记录已写过的人物，支撑「推荐下一篇」的系列感

- history.json：累积记录已写人物列表（figure / title / timestamp / 摘要）
- outputs/<时间戳>_<人物>.md：每篇文章单独存一份
- get_last_figure() / get_written_figures()：供主 agent 推荐时读取，避免重复、
  并让推荐能与上一篇形成反差/关联，增强公众号系列感
"""
import json, time, re, logging
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
HISTORY_FILE = OUTPUT_DIR / "history.json"


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def get_written_figures() -> list[str]:
    """已写过的人物名列表（按写入顺序）。"""
    return [h["figure"] for h in load_history() if h.get("figure")]


def get_last_figure() -> str | None:
    """最近一次写的人物名，用于「根据上一篇推荐下一篇」。"""
    hist = load_history()
    return hist[-1]["figure"] if hist else None


def _safe_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "_-") or "figure"


def save_article(figure: str, article: str, title: str = "") -> dict:
    """落盘：写入 history.json + 单篇 md 文件。返回本次记录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 若 history.json 存在但损坏（解析失败），先备份再重置，避免被覆盖而彻底丢失
    hist = []
    if HISTORY_FILE.exists():
        raw = ""
        try:
            raw = HISTORY_FILE.read_text(encoding="utf-8")
            hist = json.loads(raw)
        except Exception as e:
            logger.warning(f"history.json 解析失败，备份后重置: {e}")
            if raw.strip():
                bak = HISTORY_FILE.with_suffix(".json.bak")
                try:
                    HISTORY_FILE.rename(bak)
                except Exception:
                    pass
            hist = []
    entry = {
        "figure": figure,
        "title": title or figure,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "excerpt": (article or "")[:120].replace("\n", " "),
    }
    fname = OUTPUT_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{_safe_filename(figure)}.md"
    entry["file"] = fname.name  # 写入 history.json 前带上文件名，供前端点击查看
    hist.append(entry)
    try:
        HISTORY_FILE.write_text(json.dumps(hist, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    except Exception as e:
        logger.warning(f"写入 history.json 失败: {e}")

    try:
        fname.write_text(f"# {title or figure}\n\n{article}", encoding="utf-8")
    except Exception as e:
        logger.warning(f"写入单篇文章失败: {e}")
    return entry


if __name__ == "__main__":
    print("已写人物:", get_written_figures())
    print("上一篇:", get_last_figure())
