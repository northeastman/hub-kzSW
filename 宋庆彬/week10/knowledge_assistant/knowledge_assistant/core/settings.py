from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PROJECT_ROOT = Path("/Users/songqingbin/PycharmProjects/rag_learn/integrated_qa_system")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _env(name: str, default: Any) -> Any:
    value = os.getenv(name)
    return default if value in (None, "") else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, default))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, default))


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in value.split(",") if item.strip()]


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class MysqlSettings:
    host: str
    user: str
    password: str
    database: str
    port: int = 3306


@dataclass(frozen=True)
class RedisSettings:
    host: str
    port: int
    password: str | None
    db: int


@dataclass(frozen=True)
class MilvusSettings:
    host: str
    port: str
    database: str
    collection: str


@dataclass(frozen=True)
class LlmSettings:
    model: str
    api_key: str
    base_url: str
    timeout_seconds: int


@dataclass(frozen=True)
class RetrievalSettings:
    parent_chunk_size: int
    child_chunk_size: int
    chunk_overlap: int
    top_k: int
    final_context_count: int
    faq_threshold: float
    dense_weight: float
    sparse_weight: float


@dataclass(frozen=True)
class ModelSettings:
    embedding_model_path: Path
    reranker_model_path: Path
    classifier_base_model_path: Path
    classifier_model_path: Path


@dataclass(frozen=True)
class AppSettings:
    valid_sources: list[str]
    support_phone: str
    log_file: Path
    max_history_turns: int
    max_prompt_chars: int


@dataclass(frozen=True)
class Settings:
    mysql: MysqlSettings
    redis: RedisSettings
    milvus: MilvusSettings
    llm: LlmSettings
    retrieval: RetrievalSettings
    models: ModelSettings
    app: AppSettings


def load_settings(config_path: str | Path | None = None) -> Settings:
    config_file = Path(config_path) if config_path else PROJECT_ROOT / "knowledge_assistant.toml"
    config: dict[str, Any] = {}
    if config_file.exists():
        with config_file.open("rb") as fp:
            config = tomllib.load(fp)

    mysql = _section(config, "mysql")
    redis = _section(config, "redis")
    milvus = _section(config, "milvus")
    llm = _section(config, "llm")
    retrieval = _section(config, "retrieval")
    models = _section(config, "models")
    app = _section(config, "app")

    return Settings(
        mysql=MysqlSettings(
            host=str(_env("KA_MYSQL_HOST", mysql.get("host", "localhost"))),
            port=_env_int("KA_MYSQL_PORT", int(mysql.get("port", 3306))),
            user=str(_env("KA_MYSQL_USER", mysql.get("user", "root"))),
            password=str(_env("KA_MYSQL_PASSWORD", mysql.get("password", "123456"))),
            database=str(_env("KA_MYSQL_DATABASE", mysql.get("database", "subjects_kg"))),
        ),
        redis=RedisSettings(
            host=str(_env("KA_REDIS_HOST", redis.get("host", "localhost"))),
            port=_env_int("KA_REDIS_PORT", int(redis.get("port", 6379))),
            password=_env("KA_REDIS_PASSWORD", redis.get("password", None)),
            db=_env_int("KA_REDIS_DB", int(redis.get("db", 0))),
        ),
        milvus=MilvusSettings(
            host=str(_env("KA_MILVUS_HOST", milvus.get("host", "127.0.0.1"))),
            port=str(_env("KA_MILVUS_PORT", milvus.get("port", "19530"))),
            database=str(_env("KA_MILVUS_DATABASE", milvus.get("database", "knowledge_base"))),
            collection=str(_env("KA_MILVUS_COLLECTION", milvus.get("collection", "knowledge_chunks"))),
        ),
        llm=LlmSettings(
            model=str(_env("KA_LLM_MODEL", llm.get("model", "qwen-plus"))),
            api_key=str(_env("KA_LLM_API_KEY", llm.get("api_key", ""))),
            base_url=str(
                _env(
                    "KA_LLM_BASE_URL",
                    llm.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                )
            ),
            timeout_seconds=_env_int("KA_LLM_TIMEOUT_SECONDS", int(llm.get("timeout_seconds", 30))),
        ),
        retrieval=RetrievalSettings(
            parent_chunk_size=_env_int("KA_PARENT_CHUNK_SIZE", int(retrieval.get("parent_chunk_size", 512))),
            child_chunk_size=_env_int("KA_CHILD_CHUNK_SIZE", int(retrieval.get("child_chunk_size", 128))),
            chunk_overlap=_env_int("KA_CHUNK_OVERLAP", int(retrieval.get("chunk_overlap", 50))),
            top_k=_env_int("KA_RETRIEVAL_TOP_K", int(retrieval.get("top_k", 5))),
            final_context_count=_env_int(
                "KA_FINAL_CONTEXT_COUNT", int(retrieval.get("final_context_count", 2))
            ),
            faq_threshold=_env_float("KA_FAQ_THRESHOLD", float(retrieval.get("faq_threshold", 0.85))),
            dense_weight=_env_float("KA_DENSE_WEIGHT", float(retrieval.get("dense_weight", 1.0))),
            sparse_weight=_env_float("KA_SPARSE_WEIGHT", float(retrieval.get("sparse_weight", 0.7))),
        ),
        models=ModelSettings(
            embedding_model_path=_path(
                str(
                    _env(
                        "KA_EMBEDDING_MODEL_PATH",
                        models.get("embedding_model_path", str(LEGACY_PROJECT_ROOT / "rag_qa/models/bge-m3")),
                    )
                )
            ),
            reranker_model_path=_path(
                str(
                    _env(
                        "KA_RERANKER_MODEL_PATH",
                        models.get(
                            "reranker_model_path",
                            str(LEGACY_PROJECT_ROOT / "rag_qa/models/bge-reranker-large"),
                        ),
                    )
                )
            ),
            classifier_base_model_path=_path(
                str(
                    _env(
                        "KA_CLASSIFIER_BASE_MODEL_PATH",
                        models.get(
                            "classifier_base_model_path",
                            str(LEGACY_PROJECT_ROOT / "rag_qa/models/bert-base-chinese"),
                        ),
                    )
                )
            ),
            classifier_model_path=_path(
                str(
                    _env(
                        "KA_CLASSIFIER_MODEL_PATH",
                        models.get(
                            "classifier_model_path",
                            str(LEGACY_PROJECT_ROOT / "rag_qa/models/bert_query_classifier"),
                        ),
                    )
                )
            ),
        ),
        app=AppSettings(
            valid_sources=_env_list(
                "KA_VALID_SOURCES", [str(item) for item in app.get("valid_sources", ["ai", "java", "ops", "bigdata"])]
            ),
            support_phone=str(_env("KA_SUPPORT_PHONE", app.get("support_phone", "12345678"))),
            log_file=_path(str(_env("KA_LOG_FILE", app.get("log_file", "logs/knowledge_assistant.log")))),
            max_history_turns=_env_int("KA_MAX_HISTORY_TURNS", int(app.get("max_history_turns", 5))),
            max_prompt_chars=_env_int("KA_MAX_PROMPT_CHARS", int(app.get("max_prompt_chars", 4096))),
        ),
    )
