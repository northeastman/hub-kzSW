from __future__ import annotations

import sys
from pathlib import Path

# Compatibility adapter: the old project already contains OCR implementations.
# Public ingestion code imports the neutral names below.
LEGACY_PROJECT_ROOT = Path("/Users/songqingbin/PycharmProjects/rag_learn/integrated_qa_system")
if str(LEGACY_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_PROJECT_ROOT))

from rag_qa.edu_document_loaders import OCRDOCLoader, OCRIMGLoader, OCRPDFLoader, OCRPPTLoader


WordOcrLoader = OCRDOCLoader
ImageOcrLoader = OCRIMGLoader
PdfOcrLoader = OCRPDFLoader
PresentationOcrLoader = OCRPPTLoader
