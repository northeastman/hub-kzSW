from __future__ import annotations

import argparse

from knowledge_assistant.core.logging import setup_logger
from knowledge_assistant.core.settings import load_settings


def import_faq(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    logger = setup_logger(settings.app.log_file)
    from knowledge_assistant.faq_retrieval.mysql_repository import MysqlFaqRepository

    repository = MysqlFaqRepository(settings.mysql, logger)
    repository.ensure_table()
    count = repository.import_csv(args.csv)
    repository.close()
    print(f"已导入 FAQ 数据 {count} 条。")


def ingest_documents(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    logger = setup_logger(settings.app.log_file)
    from knowledge_assistant.ingestion.ingest_pipeline import IngestPipeline
    from knowledge_assistant.rag_retrieval.vector_index import MilvusVectorIndex

    vector_index = MilvusVectorIndex(settings.milvus, settings.models, settings.retrieval, logger)
    pipeline = IngestPipeline(vector_index, settings.retrieval, logger)
    count = pipeline.ingest_directory(args.directory)
    print(f"已写入向量库 {count} 个文档块。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Assistant management commands")
    parser.add_argument("--config", default=None, help="Path to knowledge_assistant.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    faq_parser = subparsers.add_parser("import-faq", help="Import FAQ CSV into MySQL")
    faq_parser.add_argument("csv", help="CSV path, columns: 学科名称, 问题, 答案")
    faq_parser.set_defaults(func=import_faq)

    ingest_parser = subparsers.add_parser("ingest-docs", help="Ingest documents into Milvus")
    ingest_parser.add_argument("directory", help="Directory containing source documents")
    ingest_parser.set_defaults(func=ingest_documents)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
