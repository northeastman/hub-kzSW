from __future__ import annotations

import hashlib
from logging import Logger

import torch
from langchain_core.documents import Document
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker
from sentence_transformers import CrossEncoder

from knowledge_assistant.core.settings import MilvusSettings, ModelSettings, RetrievalSettings


class MilvusVectorIndex:
    def __init__(
        self,
        milvus_settings: MilvusSettings,
        model_settings: ModelSettings,
        retrieval_settings: RetrievalSettings,
        logger: Logger,
    ):
        self.settings = milvus_settings
        self.retrieval_settings = retrieval_settings
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=str(model_settings.embedding_model_path),
            use_fp16=(self.device == "cuda"),
            device=self.device,
        )
        self.reranker = CrossEncoder(str(model_settings.reranker_model_path), device=self.device)
        self.dense_dim = self.embedding_function.dim["dense"]
        self._ensure_database_exists()
        self.client = MilvusClient(uri=f"http://{self.settings.host}:{self.settings.port}", db_name=self.settings.database)
        self._create_or_load_collection()

    def _ensure_database_exists(self) -> None:
        admin_client = MilvusClient(uri=f"http://{self.settings.host}:{self.settings.port}")
        try:
            if self.settings.database not in admin_client.list_databases():
                admin_client.create_database(db_name=self.settings.database)
        finally:
            if hasattr(admin_client, "close"):
                admin_client.close()

    def _create_or_load_collection(self) -> None:
        if not self.client.has_collection(self.settings.collection):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )
            self.client.create_collection(
                collection_name=self.settings.collection,
                schema=schema,
                index_params=index_params,
            )
            self.logger.info(f"Created Milvus collection: {self.settings.collection}")
        else:
            self.logger.info(f"Loaded Milvus collection: {self.settings.collection}")
        self.client.load_collection(self.settings.collection)

    def add_documents(self, documents: list[Document]) -> None:
        texts = [document.page_content for document in documents]
        if not texts:
            return

        embeddings = self.embedding_function(texts)
        data = []
        for index, document in enumerate(documents):
            data.append(
                {
                    "id": hashlib.md5(document.page_content.encode("utf-8")).hexdigest(),
                    "text": document.page_content,
                    "dense_vector": embeddings["dense"][index],
                    "sparse_vector": self._sparse_row_to_dict(embeddings["sparse"], index),
                    "parent_id": document.metadata["parent_id"],
                    "parent_content": document.metadata["parent_content"],
                    "source": document.metadata.get("source", "unknown"),
                    "timestamp": document.metadata.get("timestamp", "unknown"),
                }
            )
        self.client.upsert(collection_name=self.settings.collection, data=data)
        self.logger.info(f"Upserted {len(data)} document chunks")

    def hybrid_search(self, query: str, top_k: int | None = None, source_filter: str | None = None) -> list[Document]:
        limit = top_k or self.retrieval_settings.top_k
        query_embeddings = self.embedding_function([query])
        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        dense_request = AnnSearchRequest(
            data=[query_embeddings["dense"][0]],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit,
            expr=filter_expr,
        )
        sparse_request = AnnSearchRequest(
            data=[self._sparse_row_to_dict(query_embeddings["sparse"], 0)],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=limit,
            expr=filter_expr,
        )
        results = self.client.hybrid_search(
            collection_name=self.settings.collection,
            reqs=[dense_request, sparse_request],
            ranker=WeightedRanker(self.retrieval_settings.dense_weight, self.retrieval_settings.sparse_weight),
            limit=limit,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"],
        )[0]
        parent_docs = self._unique_parent_docs([self._doc_from_hit(hit["entity"]) for hit in results])
        if len(parent_docs) < 2:
            return parent_docs[: self.retrieval_settings.final_context_count]

        scores = self.reranker.predict([[query, doc.page_content] for doc in parent_docs])
        ranked_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        return ranked_docs[: self.retrieval_settings.final_context_count]

    @staticmethod
    def _sparse_row_to_dict(sparse_rows, index: int) -> dict[int, float]:
        try:
            row = sparse_rows[index]
            indices = row.col if hasattr(row, "col") else row.indices
            values = row.data
        except Exception:
            row = sparse_rows.getrow(index)
            indices = row.indices
            values = row.data
        return {int(idx): float(value) for idx, value in zip(indices, values)}

    @staticmethod
    def _unique_parent_docs(chunks: list[Document]) -> list[Document]:
        seen: set[str] = set()
        docs: list[Document] = []
        for chunk in chunks:
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            if parent_content and parent_content not in seen:
                docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                seen.add(parent_content)
        return docs

    @staticmethod
    def _doc_from_hit(hit) -> Document:
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp"),
            },
        )

