"""扫描 data/raw_pdf 生成 data/manifest.json（替代原版 download_reports.py）。"""
import re, json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_pdf"

# 无法提取文本的文件（CID 字体乱码、无文本层且无 OCR）——排除出语料库。
# GB+811-2022.pdf 为 CID 编码乱码、无 ToUnicode 映射、页面无图片，任何文本提取器均读不出。
EXCLUDE_FILES = {"GB+811-2022.pdf"}

GB_PAT = re.compile(r"GB\s*[+]?\s*T?\s*(\d+)\s*[-—.]\s*(\d+)")

def _clean_title(filename: str) -> str:
    name = Path(filename).stem
    name = name.replace("+", " ").strip()
    return name

def infer_meta(filename: str) -> dict:
    stem = Path(filename).stem
    m = GB_PAT.search(filename)
    if m:
        doc_id = f"GB{m.group(1)}-{m.group(2)}"
        doc_type = "标准"
    elif "法" in stem:
        doc_id = f"LAW-{stem}"
        doc_type = "法律"
    else:
        doc_id = stem
        doc_type = "标准"
    return {"doc_id": doc_id, "doc_title": _clean_title(filename), "doc_type": doc_type}

def main():
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error(f"未找到 PDF：{RAW_DIR}")
        return
    manifest = []
    for p in pdfs:
        if p.name in EXCLUDE_FILES:
            logger.warning(f"  跳过（无法提取文本，已排除）：{p.name}")
            continue
        meta = infer_meta(p.name)
        meta["filename"] = p.name
        manifest.append(meta)
        logger.info(f"  {p.name} → doc_id={meta['doc_id']} type={meta['doc_type']}")
    out = RAW_DIR.parent / "manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"已写入 {out}（共 {len(manifest)} 项）")

if __name__ == "__main__":
    main()
