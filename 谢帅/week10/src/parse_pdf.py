"""
PDF 解析脚本：将原始 标准/法律 PDF 转换为结构化文本

教学重点（企业级 RAG 的真实挑战）：
  1. 数字 PDF vs 扫描件：处理方式完全不同
  2. 表格提取：标准里有参数/限值表格，直接按文字流提取会乱序
  3. 页眉/页脚噪声：每页都有标准号、页码，必须去除
  4. 章节/条款识别：利用字体大小/加粗 + 编号规律猜测标题层级
     （第X章/第X条、条款号 4.1.2、附录A 等）
  5. 输出格式：保留元信息（页码、章节路径），供后续溯源用

依赖安装：
  pip install pdfplumber pymupdf pytesseract pillow
  # tesseract-ocr 需要单独安装并配置 PATH
  # Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pdfplumber          # 擅长表格提取
import fitz                # PyMuPDF，擅长文字+图片提取

# OCR 依赖可选（需要同时安装 pytesseract 包 + tesseract-ocr 二进制）
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw_pdf"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# 如果 tesseract 不在 PATH，手动指定（Windows 常见路径）
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    """
    一个解析块 = 年报里的一段连续内容（文字段落 or 表格）

    保留 page_num 和 section_path 非常重要——
    RAG 答案引用时能告诉用户"来自第38页，财务报告/资产负债表"
    """
    block_type:   str            # "text" | "table" | "title"
    content:      str            # 文字内容（表格转为 markdown）
    page_num:     int
    section_path: list[str]      # ["第三章 管理层讨论", "一、经营情况概述"]
    is_ocr:       bool = False   # 是否经过 OCR，质量可能有误
    raw_table:    Optional[list] = field(default=None, repr=False)  # 原始表格数据


# ── 工具函数 ──────────────────────────────────────────────────────────────────

# 标准/法律里常见的章节/条款标题模式（粗略匹配，不求完美）
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百零]+[章节]"),     # 第一章、第三节
    re.compile(r"^第[一二三四五六七八九十百零\d]+条"),        # 第X条（法律）
    re.compile(r"^\d+(\.\d+)+\s*\S"),                        # 4.1 / 4.1.2 条款号
    re.compile(r"^\d+\s+\S"),                                # 顶层编号 "5 要求"
    re.compile(r"^附录[A-Z]"),                               # 附录A
    re.compile(r"^[A-Z]\.\d+"),                              # A.1
    re.compile(r"^[一二三四五六七八九十]、"),                 # 一、二、
]

NOISE_PATTERNS = [
    re.compile(r"^GB\s*[/T]*\s*\d+\s*[-—.]\s*\d+\s*$"),      # 页脚标准号 GB 17761—2024
    re.compile(r"^\d+\s*$"),                                 # 独立页码
    re.compile(r"^[－—\-]\s*\d+\s*[－—\-]$"),                 # — 38 — / －3－（含全角破折号）
    re.compile(r"^ICS\b"),                                   # 封面代号 ICS
    re.compile(r"^CCS\b"),                                   # 封面代号 CCS
]


def _cjk_ratio(text: str) -> float:
    """中日韩汉字占比。用于判断文本是否为可读中文。"""
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    printable = sum(1 for ch in text if not ch.isspace())
    return cjk / max(printable, 1)


def is_garbled(text: str) -> bool:
    """
    判断文本是否为乱码（CID 字体无 ToUnicode 映射时，提取出的是无意义 ASCII）。
    对中文文档：正文较长却几乎不含汉字 → 判为乱码。
    """
    stripped = text.strip()
    if len(stripped) < 30:
        return False
    return _cjk_ratio(stripped) < 0.05


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 2:
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_title_line(
    line: str,
    fontsize: Optional[float] = None,
    is_bold: bool = False,
    body_size: Optional[float] = None,
) -> bool:
    """
    判断一行是否是标题。
    有字体信息时用「相对正文字号明显更大」判断（body_size 为该文档正文主字号），
    没有 body_size 时回退到绝对阈值；同时结合编号规律（第X章/第X条/4.1 等）。
    """
    text = line.strip()
    if fontsize:
        if body_size:
            if fontsize >= body_size + 1.5:      # 明显大于正文
                return True
        elif fontsize >= 14:                     # 无正文基准时的回退阈值
            return True
    if is_bold and len(text) < 50:
        return True
    return any(p.match(text) for p in CHAPTER_PATTERNS)


def table_to_markdown(table: list[list]) -> str:
    """把 pdfplumber 提取的表格转成 markdown 格式，方便 LLM 理解。"""
    if not table:
        return ""

    # 清洗单元格：None 变空字符串，去掉换行
    rows = []
    for row in table:
        cleaned = [str(cell or "").replace("\n", " ").strip() for cell in row]
        rows.append(cleaned)

    if not rows:
        return ""

    # 构建 markdown 表格
    header = rows[0]
    lines  = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        # 对齐列数（有些 PDF 表格行列不整齐）
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")

    return "\n".join(lines)


def detect_if_scanned(page: fitz.Page, text: str) -> bool:
    """
    启发式判断：文字极少但图片多 → 很可能是扫描页。
    年报中扫描件多见于附件（审计报告原件）。
    """
    if len(text.strip()) > 50:
        return False
    image_list = page.get_images(full=True)
    return len(image_list) > 0


def ocr_page(page: fitz.Page, dpi: int = 200) -> str:
    """对扫描页做 OCR（中文）。需要 pytesseract + tesseract-ocr 二进制。"""
    if not OCR_AVAILABLE:
        return "[扫描页，OCR 不可用（未安装 pytesseract/tesseract），内容跳过]"
    try:
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        clip = page.rect
        pix  = page.get_pixmap(matrix=mat, clip=clip)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text
    except Exception as e:
        logger.warning(f"  OCR 失败，跳过此页: {e}")
        return "[扫描页，OCR 失败，内容跳过]"


# ── 主解析逻辑 ────────────────────────────────────────────────────────────────

class DocParser:
    """
    标准/法律 PDF 解析器。

    策略：
      - 用 pdfplumber 提取表格（它的表格算法更准）
      - 用 PyMuPDF (fitz) 提取带字体信息的文字（用于判断标题）
      - 对扫描页降级为 OCR
    """

    def __init__(self, pdf_path: Path, meta: dict = None):
        self.pdf_path = pdf_path
        self.meta     = meta or {}
        self.blocks: list[ParsedBlock] = []
        self._section_stack: list[str] = []

    def _update_section(self, title: str):
        """维护章节栈：根据编号层级推断层次（章 > 节/条 > 数字条款）。"""
        t = title.strip()
        if re.match(r"^第[一二三四五六七八九十百零]+章", t):
            self._section_stack = [t]                                  # 顶级章
        elif re.match(r"^第[一二三四五六七八九十百零\d]+[节条]", t):
            self._section_stack = self._section_stack[:1] + [t]        # 二级 节/条
        elif re.match(r"^\d+(\.\d+)+", t):
            depth = t.split()[0].count(".")                            # 4.1→1, 4.1.2→2
            self._section_stack = self._section_stack[:depth] + [t]
        elif re.match(r"^\d+\s+\S", t) or re.match(r"^附录[A-Z]", t):
            self._section_stack = [t]                                  # 顶层编号章/附录
        else:
            self._section_stack = self._section_stack[:3] + [t]

    def _detect_body_size(self, fitz_doc) -> Optional[float]:
        """扫描全文 span 字号，取出现次数最多的作为「正文主字号」。"""
        from collections import Counter
        counter: Counter = Counter()
        for page_num in range(len(fitz_doc)):
            for block in fitz_doc[page_num].get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        if txt:
                            counter[round(span.get("size", 0), 1)] += len(txt)
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def parse(self) -> list[ParsedBlock]:
        logger.info(f"开始解析: {self.pdf_path.name}")

        # 同时打开两个解析器
        plumber_doc = pdfplumber.open(self.pdf_path)
        fitz_doc    = fitz.open(str(self.pdf_path))

        # 预扫描正文主字号，供相对阈值的标题判断使用
        self.body_size = self._detect_body_size(fitz_doc)
        logger.info(f"  正文主字号 ≈ {self.body_size}")

        for page_num in range(len(fitz_doc)):
            fitz_page   = fitz_doc[page_num]
            plumb_page  = plumber_doc.pages[page_num]

            # ── 1. 先用 PyMuPDF 获取带字体信息的文字 ──
            raw_text = fitz_page.get_text("text")
            # 扫描页 或 CID 字体乱码页 → 走 OCR 降级
            if detect_if_scanned(fitz_page, raw_text) or is_garbled(raw_text):
                logger.debug(f"  第{page_num+1}页：扫描/乱码，启动 OCR")
                ocr_text = ocr_page(fitz_page)
                self.blocks.append(ParsedBlock(
                    block_type="text",
                    content=ocr_text,
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                    is_ocr=True,
                ))
                continue

            # ── 2. 提取表格（用 pdfplumber）──
            table_bboxes = []
            for table in plumb_page.extract_tables():
                if table:
                    md = table_to_markdown(table)
                    if md:
                        self.blocks.append(ParsedBlock(
                            block_type="table",
                            content=md,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                            raw_table=table,
                        ))
            # 记录表格所在区域（后续跳过这些区域的文字提取）
            for table_obj in plumb_page.find_tables():
                table_bboxes.append(table_obj.bbox)

            # ── 3. 提取文字（跳过表格区域，逐行处理）──
            page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            current_para_lines = []

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:   # 0=文字，1=图片
                    continue

                for line in block.get("lines", []):
                    line_text = "".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()

                    if not line_text or is_noise_line(line_text):
                        continue

                    # 判断是否标题
                    spans    = line.get("spans", [])
                    fontsize = spans[0].get("size", 0) if spans else 0
                    is_bold  = any("Bold" in span.get("font", "") for span in spans)

                    if is_title_line(line_text, fontsize, is_bold, self.body_size):
                        # 先把积累的段落存起来
                        if current_para_lines:
                            self.blocks.append(ParsedBlock(
                                block_type="text",
                                content="\n".join(current_para_lines),
                                page_num=page_num + 1,
                                section_path=list(self._section_stack),
                            ))
                            current_para_lines = []

                        self._update_section(line_text)
                        self.blocks.append(ParsedBlock(
                            block_type="title",
                            content=line_text,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                        ))
                    else:
                        current_para_lines.append(line_text)

            # 最后一段
            if current_para_lines:
                self.blocks.append(ParsedBlock(
                    block_type="text",
                    content="\n".join(current_para_lines),
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                ))

        plumber_doc.close()
        fitz_doc.close()

        logger.info(f"  解析完成: {len(self.blocks)} 个块")
        return self.blocks

    def save(self):
        """将解析结果保存为 JSON，保留所有元信息。"""
        stem     = self.pdf_path.stem
        out_path = PARSED_DIR / f"{stem}.json"

        output = {
            "meta":   self.meta,
            "source": str(self.pdf_path),
            "blocks": [asdict(b) for b in self.blocks],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存 → {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    manifest_path = RAW_DIR.parent / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        # 没有 manifest 就直接扫目录
        manifest = [
            {"filename": p.name, "doc_id": "", "doc_title": "", "doc_type": ""}
            for p in RAW_DIR.glob("*.pdf")
        ]

    if not manifest:
        logger.error("没有找到任何 PDF，请先运行 build_manifest.py")
        return

    for item in manifest:
        pdf_path = RAW_DIR / item["filename"]
        if not pdf_path.exists():
            logger.warning(f"文件不存在，跳过: {pdf_path}")
            continue

        parser = DocParser(pdf_path, meta=item)
        parser.parse()
        parser.save()

    logger.info(f"\n全部解析完成，结果在 {PARSED_DIR}")


if __name__ == "__main__":
    main()
