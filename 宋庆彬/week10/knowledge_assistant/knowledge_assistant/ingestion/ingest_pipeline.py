from __future__ import annotations

from logging import Logger
from pathlib import Path

from knowledge_assistant.core.settings import RetrievalSettings
from knowledge_assistant.ingestion.document_loader import DocumentLoader
from knowledge_assistant.ingestion.text_splitter import ParentChildTextSplitter
from knowledge_assistant.rag_retrieval.vector_index import MilvusVectorIndex


class IngestPipeline:
    def __init__(self, vector_index: MilvusVectorIndex, retrieval_settings: RetrievalSettings, logger: Logger):
        self.vector_index = vector_index
        self.logger = logger
        self.loader = DocumentLoader()
        self.splitter = ParentChildTextSplitter(
            parent_chunk_size=retrieval_settings.parent_chunk_size,
            child_chunk_size=retrieval_settings.child_chunk_size,
            chunk_overlap=retrieval_settings.chunk_overlap,
        )

    def ingest_directory(self, directory_path: str | Path) -> int:
        documents = self.loader.load_directory(directory_path)
        chunks = self.splitter.split(documents)
        self.vector_index.add_documents(chunks)
        self.logger.info(f"Ingested {len(chunks)} chunks from {directory_path}")
        return len(chunks)

