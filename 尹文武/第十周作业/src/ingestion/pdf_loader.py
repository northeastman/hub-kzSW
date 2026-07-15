from pathlib import Path
from typing import List, Dict, Any

import pdfplumber

from config.settings import MIN_TEXT_LENGTH, OCR_OUTPUT_DIR
from src.ingestion.text_cleaner import clean_text
from src.ingestion.ocr_engine import BaiduUnlimitedOCREngine


class PDFLoader:
    def __init__(self):
        self.ocr_engine = BaiduUnlimitedOCREngine()

    def load_pdf_by_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_index, page in enumerate(pdf.pages):
                raw_text = page.extract_text() or ""
                cleaned = clean_text(raw_text)

                pages.append(
                    {
                        "text": cleaned,
                        "metadata": {
                            "source": str(pdf_path),
                            "file_name": pdf_path.name,
                            "page": page_index + 1,
                            "parser": "pdfplumber",
                        },
                    }
                )

        total_text_length = sum(len(p["text"]) for p in pages)

        # 如果整份 PDF 基本没解析出文字，走 OCR
        if total_text_length < MIN_TEXT_LENGTH:
            return self._load_by_ocr(pdf_path)

        return pages

    def _load_by_ocr(self, pdf_path: Path) -> List[Dict[str, Any]]:
        output_txt_path = OCR_OUTPUT_DIR / f"{pdf_path.stem}.txt"

        raw_text = self.ocr_engine.run_ocr(pdf_path, output_txt_path)
        cleaned = clean_text(raw_text)

        return [
            {
                "text": cleaned,
                "metadata": {
                    "source": str(pdf_path),
                    "file_name": pdf_path.name,
                    "page": None,
                    "parser": "unlimited-ocr",
                    "ocr_output": str(output_txt_path),
                },
            }
        ]

    def load_folder(self, folder: Path) -> List[Dict[str, Any]]:
        folder = Path(folder)
        all_docs = []

        for pdf_path in folder.glob("*.pdf"):
            docs = self.load_pdf_by_pages(pdf_path)
            all_docs.extend(docs)

        return all_docs
