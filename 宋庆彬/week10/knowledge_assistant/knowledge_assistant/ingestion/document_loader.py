from __future__ import annotations

from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

from knowledge_assistant.ingestion.ocr_loaders import (
    ImageOcrLoader,
    PdfOcrLoader,
    PresentationOcrLoader,
    WordOcrLoader,
)


DOCUMENT_LOADERS = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PdfOcrLoader,
    ".docx": WordOcrLoader,
    ".ppt": PresentationOcrLoader,
    ".pptx": PresentationOcrLoader,
    ".jpg": ImageOcrLoader,
    ".png": ImageOcrLoader,
}


class DocumentLoader:
    def load_directory(self, directory_path: str | Path):
        directory = Path(directory_path)
        documents = []
        source = directory.name.replace("_data", "")
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            loader_class = DOCUMENT_LOADERS.get(file_path.suffix.lower())
            if loader_class is None:
                continue
            loader = loader_class(str(file_path), encoding="utf-8") if file_path.suffix.lower() == ".txt" else loader_class(str(file_path))
            for document in loader.load():
                document.metadata["source"] = source
                document.metadata["file_path"] = str(file_path)
                document.metadata["timestamp"] = datetime.now().isoformat()
                documents.append(document)
        return documents
