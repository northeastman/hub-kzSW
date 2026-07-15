import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config.settings import RAW_PDF_DIR, OCR_OUTPUT_DIR
from src.ingestion.pdf_loader import PDFLoader


def main():
    loader = PDFLoader()
    docs = loader.load_folder(RAW_PDF_DIR)

    output_path = OCR_OUTPUT_DIR / "parsed_docs.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"[OK] Parsed docs: {len(docs)}")
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
