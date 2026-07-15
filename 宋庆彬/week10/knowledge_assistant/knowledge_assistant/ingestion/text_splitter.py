from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import MarkdownTextSplitter

from knowledge_assistant.ingestion.chinese_recursive_text_splitter import ChineseRecursiveTextSplitter


class ParentChildTextSplitter:
    def __init__(self, parent_chunk_size: int, child_chunk_size: int, chunk_overlap: int):
        self.parent_splitter = ChineseRecursiveTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.child_splitter = ChineseRecursiveTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.markdown_parent_splitter = MarkdownTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.markdown_child_splitter = MarkdownTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents):
        child_chunks = []
        for document_index, document in enumerate(documents):
            suffix = Path(document.metadata.get("file_path", "")).suffix.lower()
            parent_splitter = self.markdown_parent_splitter if suffix == ".md" else self.parent_splitter
            child_splitter = self.markdown_child_splitter if suffix == ".md" else self.child_splitter
            for parent_index, parent_doc in enumerate(parent_splitter.split_documents([document])):
                parent_id = f"doc_{document_index}_parent_{parent_index}"
                parent_doc.metadata["parent_id"] = parent_id
                parent_doc.metadata["parent_content"] = parent_doc.page_content
                for child_index, child_doc in enumerate(child_splitter.split_documents([parent_doc])):
                    child_doc.metadata["parent_id"] = parent_id
                    child_doc.metadata["parent_content"] = parent_doc.page_content
                    child_doc.metadata["id"] = f"{parent_id}_child_{child_index}"
                    child_chunks.append(child_doc)
        return child_chunks
