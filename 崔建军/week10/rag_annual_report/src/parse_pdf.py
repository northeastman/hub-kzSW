"""
PDF 解析脚本：将原始深度学习课程笔记 PDF 转换为结构化文本

教学重点（企业级 RAG 的真实挑战）：
  1. 数字 PDF vs 扫描件：处理方式完全不同
  2. 表格提取：笔记中的公式和表格，直接按文字流提取会乱序
  3. 页眉/页脚噪声：每页都有页码，必须去除
  4. 章节识别：利用字体大小/加粗猜测标题层级
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
    一个解析块 = 深度学习课程笔记里的一段连续内容（文字段落 or 表格）

    保留 page_num 和 section_path 非常重要——
    RAG 答案引用时能告诉用户"来自第38页，某章节"
    """
    block_type:   str            # "text" | "table" | "title"
    content:      str            # 文字内容（表格转为 markdown）
    page_num:     int
    section_path: list[str]      # ["第一章 神经网络基础", "1.1 线性回归"]
    is_ocr:       bool = False   # 是否经过 OCR，质量可能有误
    raw_table:    Optional[list] = field(default=None, repr=False)  # 原始表格数据


# ── 工具函数 ──────────────────────────────────────────────────────────────────

CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百]+[章节篇]\s"),  # 第一章、第三节、第一篇（后面必须有空格，排除"第一节xxx"）
    re.compile(r"^第[一二三四五六七八九十百]+门课\s"),       # 第一门课（后面必须有空格）
    re.compile(r"^[一二三四五六七八九十]、\s"),             # 一、二、（后面必须有空格）
    re.compile(r"^[1-9]\d*\.\s+(?![\d.])"),                # 1. 2.（整数部分不为0，排除小数如0.5）
    re.compile(r"^[1-9]\d*\.\d+(?!\d)\s+"),                # 1.1 2.3（整数部分不为0，排除小数如0.05）
    re.compile(r"^[1-9]\d*\.\d+\.\d+(?!\d)\s+"),           # 1.1.1 2.3.4（整数部分不为0）
    re.compile(r"^摘要\s*$"),                               # 摘要（单行标题）
    re.compile(r"^目录\s*$"),                               # 目录（单行标题）
    re.compile(r"^参考文献\s*$"),                           # 参考文献（单行标题）
    re.compile(r"^DeepLearning\.ai\s*$"),                  # DeepLearning.ai（单行标题）
]

NOISE_PATTERNS = [
    re.compile(r"^\d+\s*$"),                # 独立页码
    re.compile(r"^—\s*\d+\s*—$"),          # — 38 —
    re.compile(r"^主编：\S+\s*$"),          # 主编：XXX
    re.compile(r"^[\w.]+@[\w.]+\s*$"),      # 邮箱地址
    re.compile(r"^知乎\(\S+\)\s*$"),        # 知乎(XXX)
]


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 2:
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_toc_line(line: str) -> bool:
    """判断是否是目录行（含页码的引导点）"""
    line = line.strip()
    if len(line) < 20:
        return False
    # 如果行匹配"第X门课"模式且包含引导点和页码，则是目录行（过滤目录页的课程标题）
    if re.match(r'^第[一二三四五六七八九十百]+门课\s', line):
        # 如果包含引导点和页码，则是目录条目，不是真正的章节标题
        if re.search(r'\.{3,}', line) and re.search(r'\d{1,3}\s*$', line):
            return True
        return False
    # 目录行必须包含引导点或特定格式的页码
    if re.search(r'\.{5,}', line) and re.search(r'\d{2,3}\s*$', line):
        return True
    if re.search(r'\.\s*\d{2,3}\s*$', line):
        return True
    # 以章节编号开头且长度大于50字符，必须同时包含引导点才判定为目录行
    if re.match(r'^\d+\.\d+\s+', line) and len(line) > 50:
        if re.search(r'\.{3,}', line):
            return True
    return False


def is_title_line(line: str, fontsize: Optional[float] = None, is_bold: bool = False, 
                  dominant_fontsize: float = 0) -> bool:
    """
    判断一行是否是标题。
    
    改进策略：
    1. 过滤目录行（含大量引导点和页码）
    2. 使用相对字号：比主导字号大2pt以上才算标题
    3. 加粗文字必须匹配标题模式才判定为标题
    4. 长度限制：标题一般不会太长（>150字符）也不会太短（<2字符）
       - "第X门课"类型的标题可能较长（包含中英文标题）
    5. 章节编号模式（如1.1, 1.2, 2.1等）的行，直接判定为标题
       这是为了确保所有章节标题都能被识别，防止section_path泄漏到后续页面
       即使字号和正文相同或更小，只要匹配章节编号模式就判定为标题
    """
    line = line.strip()
    
    if len(line) < 2 or len(line) > 150:
        return False
    
    if is_toc_line(line):
        return False
    
    # 章节编号模式（如1.1, 1.2, 2.1等）的行，直接判定为标题
    # 这是最重要的规则，确保所有章节标题都能被识别
    if any(p.match(line) for p in CHAPTER_PATTERNS):
        # 章节标题长度通常在5-100字符之间
        if 5 <= len(line) <= 100:
            return True
    
    # 其他情况必须有字号优势才判定为标题
    has_font_advantage = fontsize and dominant_fontsize > 0 and fontsize >= dominant_fontsize + 2
    if has_font_advantage:
        return True
    
    return False


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
    课程笔记中扫描件多见于手写笔记或截图。
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

class DeepLearningNoteParser:
    """
    深度学习课程笔记 PDF 解析器。

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
        """维护章节栈：根据缩进/编号层级推断层次。"""
        if re.match(r"^第[一二三四五六七八九十]+章", title):
            self._section_stack = [title]           # 顶级章
        elif re.match(r"^第[一二三四五六七八九十]+门课", title):
            self._section_stack = [title]           # 顶级课（和章同级）
        elif re.match(r"^第[一二三四五六七八九十]+节", title):
            self._section_stack = self._section_stack[:1] + [title]  # 二级节
        elif re.match(r"^[一二三四五六七八九十]、", title):
            self._section_stack = self._section_stack[:2] + [title]  # 三级
        elif re.match(r"^\d+\.\s+", title):
            self._section_stack = self._section_stack[:1] + [title]  # 1. 2.（二级）
        elif re.match(r"^\d+\.\d+\s+", title):
            self._section_stack = self._section_stack[:2] + [title]  # 1.1 2.3（三级）
        elif re.match(r"^\d+\.\d+\.\d+\s+", title):
            self._section_stack = self._section_stack[:3] + [title]  # 1.1.1（四级）
        else:
            self._section_stack = self._section_stack[:3] + [title]

    def parse(self) -> list[ParsedBlock]:
        logger.info(f"开始解析: {self.pdf_path.name}")

        # 同时打开两个解析器
        plumber_doc = pdfplumber.open(self.pdf_path)
        fitz_doc    = fitz.open(str(self.pdf_path))

        for page_num in range(len(fitz_doc)):
            fitz_page   = fitz_doc[page_num]
            plumb_page  = plumber_doc.pages[page_num]
            
            # 检测当前页是否是目录页（包含大量引导点）
            page_text = fitz_page.get_text()
            is_toc_page = page_text.count('...') > 10 or page_text.count('....') > 5

            # ── 1. 先用 PyMuPDF 获取带字体信息的文字 ──
            raw_text = fitz_page.get_text("text")
            is_scanned = detect_if_scanned(fitz_page, raw_text)

            if is_scanned:
                logger.debug(f"  第{page_num+1}页：检测到扫描件，启动 OCR")
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

            # 获取页面高度，用于判断页眉区域（顶部 15%）
            page_height = fitz_page.rect.height
            header_threshold = page_height * 0.15

            # 第一次遍历：统计每页的字号分布，找出主导字号（众数）
            font_sizes = []
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0)
                        if size > 0:
                            font_sizes.append(size)
            
            # 计算主导字号（出现次数最多的字号）
            dominant_fontsize = 0
            if font_sizes:
                from collections import Counter
                size_counter = Counter(font_sizes)
                dominant_fontsize = size_counter.most_common(1)[0][0]

            # 收集所有文字行（含排版信息），用于跨行标题合并
            text_lines = []
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:   # 0=文字，1=图片
                    continue

                for line in block.get("lines", []):
                    line_text = "".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()

                    if not line_text or is_noise_line(line_text):
                        continue

                    line_y0 = line.get("bbox", [0, 0, 0, 0])[1]
                    if line_y0 < header_threshold:
                        if re.match(r'^\d+\.\d+\s+', line_text):
                            pass
                        else:
                            continue

                    spans    = line.get("spans", [])
                    fontsize = spans[0].get("size", 0) if spans else 0
                    is_bold  = any("Bold" in span.get("font", "") for span in spans)
                    text_lines.append({
                        "text": line_text,
                        "fontsize": fontsize,
                        "is_bold": is_bold,
                        "y0": line_y0,
                    })

            # 第二次遍历：处理跨行标题合并并提取文字
            i = 0
            while i < len(text_lines):
                line = text_lines[i]
                line_text = line["text"]
                fontsize = line["fontsize"]
                is_bold = line["is_bold"]

                # 判断是否标题
                if is_title_line(line_text, fontsize, is_bold, dominant_fontsize):
                    # 先把积累的段落存起来
                    if current_para_lines:
                        self.blocks.append(ParsedBlock(
                            block_type="text",
                            content="\n".join(current_para_lines),
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                        ))
                        current_para_lines = []

                    # 目录页不识别标题
                    if is_toc_page:
                        current_para_lines.append(line_text)
                        i += 1
                        continue
                    
                    # 清理标题中的引导点和页码（目录页常见排版）
                    cleaned_title = re.sub(r'\.{3,}', '', line_text)
                    cleaned_title = re.sub(r'\s*\d{1,3}\s*$', '', cleaned_title).strip()
                    
                    # 尝试合并跨行标题（只合并紧跟在标题后面的一行）
                    full_title = cleaned_title
                    # 检查下一行是否是标题的续行
                    if i + 1 < len(text_lines):
                        next_line = text_lines[i + 1]
                        next_text = next_line["text"]
                        # 标题续行的特征：字号相同、距离近、包含引导点和页码（目录页常见排版）
                        y_diff = abs(next_line["y0"] - line["y0"])
                        has_dots = bool(re.search(r'\.{3,}', next_text))
                        has_page_num = bool(re.search(r'\d{2,3}\s*$', next_text))
                        # 如果下一行字号相同、距离近、且包含引导点+页码，则认为是标题续行
                        if next_line["fontsize"] == fontsize and y_diff < fontsize * 2 and has_dots and has_page_num:
                            # 清理续行中的引导点和页码
                            cleaned_next = re.sub(r'\.{3,}', '', next_text)
                            cleaned_next = re.sub(r'\s*\d{2,3}\s*$', '', cleaned_next).strip()
                            if cleaned_next:
                                # 如果当前标题末尾是字母且续行开头也是字母，添加空格
                                if full_title and full_title[-1].isalpha() and cleaned_next[0].isalpha():
                                    full_title += ' ' + cleaned_next
                                else:
                                    full_title += cleaned_next
                            i += 1

                    self._update_section(full_title)
                    self.blocks.append(ParsedBlock(
                        block_type="title",
                        content=full_title,
                        page_num=page_num + 1,
                        section_path=list(self._section_stack),
                    ))
                else:
                    current_para_lines.append(line_text)
                i += 1

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

        # 按页码排序blocks，确保顺序正确
        self.blocks.sort(key=lambda b: b.page_num)

        # 后处理：自动检测课程标题并插入到section_path
        self._post_process_course_titles()

        logger.info(f"  解析完成: {len(self.blocks)} 个块")
        return self.blocks

    def _post_process_course_titles(self):
        """
        后处理：自动检测课程标题并修复section_path的层级结构。
        由于PDF正文中的课程标题可能没有被正确识别，需要根据内容中的关键词进行推断。
        
        策略：
        1. 收集所有"1.1"开头的标题作为课程边界（每门课从1.1开始）
        2. 按页码排序后依次分配给五门课程
        3. 修复section_path为标准层级：[摘要, 课程标题, DeepLearning.ai, 章节标题]
        """
        courses = [
            '第一门课 神经网络与深度学习(Neural Networks and Deep Learning)',
            '第二门课 改善深层神经网络：超参数调试、正则化以及优化(Improving Deep Neural Networks:Hyperparameter tuning, Regularization and Optimization)',
            '第三门课 结构化机器学习项目(Structuring Machine Learning Projects)',
            '第四门课 卷积神经网络(Convolutional Neural Networks)',
            '第五门课 序列模型(Sequence Models)',
        ]
        
        # 收集所有"1.1"开头的标题作为课程边界
        one_one_titles = []
        chapter_titles = {}
        
        for idx, block in enumerate(self.blocks):
            if block.block_type == 'title':
                content = block.content
                # 检测章节编号（如1.1, 2.3, 4.1等）
                match = re.match(r'^([1-9]\d*)\.(\d+)\s+', content)
                if match:
                    chapter_titles[idx] = content
                    # 记录所有"1.1"开头的标题（课程开始标志）
                    if int(match.group(1)) == 1 and int(match.group(2)) == 1:
                        one_one_titles.append((block.page_num, idx, content))
        
        # 按页码排序
        one_one_titles.sort()
        
        # 根据"1.1"标题分配课程
        course_boundaries = []
        for i, course in enumerate(courses):
            if i < len(one_one_titles):
                course_boundaries.append((one_one_titles[i][1], course))
            else:
                # 如果没有足够的"1.1"标题，使用最后一个边界
                if course_boundaries:
                    course_boundaries.append((course_boundaries[-1][0] + 1, course))
                else:
                    course_boundaries.append((0, course))
        
        # 第二遍：根据块索引范围为每个块修复section_path
        updated_blocks = []
        
        for idx, block in enumerate(self.blocks):
            # 找到当前块对应的课程
            current_course = None
            for i, (boundary_idx, course_title) in enumerate(course_boundaries):
                if idx >= boundary_idx:
                    current_course = course_title
                else:
                    break
            
            # 如果没有找到课程，使用最后一个课程
            if current_course is None and course_boundaries:
                current_course = course_boundaries[-1][1]
            
            # 修复section_path为标准层级
            if current_course:
                new_path = ['摘要', current_course, 'DeepLearning.ai']
                
                # 找到当前块所属的章节标题（向前查找最近的标题）
                chapter_title = None
                for prev_idx in range(idx, -1, -1):
                    if prev_idx in chapter_titles:
                        chapter_title = chapter_titles[prev_idx]
                        break
                
                if chapter_title:
                    new_path.append(chapter_title)
                
                block.section_path = new_path
            
            updated_blocks.append(block)
        
        self.blocks = updated_blocks

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
            {"filename": p.name, "source_file": p.stem}
            for p in RAW_DIR.glob("*.pdf")
        ]

    if not manifest:
        logger.error("没有找到任何 PDF，请将深度学习课程笔记 PDF 放入 data/raw_pdf 目录")
        return

    for item in manifest:
        pdf_path = RAW_DIR / item["filename"]
        if not pdf_path.exists():
            logger.warning(f"文件不存在，跳过: {pdf_path}")
            continue

        parser = DeepLearningNoteParser(pdf_path, meta=item)
        parser.parse()
        parser.save()

    logger.info(f"\n全部解析完成，结果在 {PARSED_DIR}")


if __name__ == "__main__":
    main()
