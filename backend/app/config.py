"""
NyayaMitra — Centralized Application Configuration.

Reads all settings from environment variables / .env file.
Every service imports from here — single source of truth.

Usage:
    from app.config import settings
    print(settings.POSTGRES_HOST)
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: two levels up from this file (backend/app/config.py → nyayamitra/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ─── Application ─────────────────────────────────────────────────────────
    APP_NAME: str = "nyayamitra"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_VERSION: str = "0.1.0"
    APP_SECRET_KEY: str = "change-this-to-a-random-secret-key-in-production"

    # ─── Backend (FastAPI) ───────────────────────────────────────────────────
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8080
    BACKEND_WORKERS: int = 1
    BACKEND_LOG_LEVEL: str = "INFO"

    # ─── PostgreSQL ──────────────────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "nyayamitra"
    POSTGRES_USER: str = "nyayamitra"
    POSTGRES_PASSWORD: str = "change-this-password"

    @property
    def database_url(self) -> str:
        """Async PostgreSQL connection string for SQLAlchemy."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def database_url_sync(self) -> str:
        """Sync PostgreSQL connection string (for Alembic migrations)."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ─── Redis ───────────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    @property
    def redis_url(self) -> str:
        """Redis connection string."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ─── Qdrant (Vector Database) ────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str = ""

    # ─── Elasticsearch ───────────────────────────────────────────────────────
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_USER: str = "elastic"
    ELASTICSEARCH_PASSWORD: str = "change-this-password"

    @property
    def elasticsearch_url(self) -> str:
        """Elasticsearch connection URL."""
        return f"http://{self.ELASTICSEARCH_HOST}:{self.ELASTICSEARCH_PORT}"

    # ─── Neo4j (Knowledge Graph) ─────────────────────────────────────────────
    NEO4J_HOST: str = "localhost"
    NEO4J_BOLT_PORT: int = 7687
    NEO4J_HTTP_PORT: int = 7474
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "change-this-password"

    @property
    def neo4j_uri(self) -> str:
        """Neo4j Bolt connection URI."""
        return f"bolt://{self.NEO4J_HOST}:{self.NEO4J_BOLT_PORT}"

    # ─── LLM Inference (vLLM) ────────────────────────────────────────────────
    VLLM_HOST: str = "localhost"
    VLLM_PORT: int = 8000
    VLLM_MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"
    VLLM_MAX_MODEL_LEN: int = 8192
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.90

    @property
    def vllm_api_url(self) -> str:
        """vLLM OpenAI-compatible API URL."""
        return f"http://{self.VLLM_HOST}:{self.VLLM_PORT}/v1"

    # ─── Embedding Model ─────────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32

    # ─── Re-ranker ───────────────────────────────────────────────────────────
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    RERANKER_TOP_K: int = 30

    # ─── Retrieval Configuration ─────────────────────────────────────────────
    RETRIEVAL_TOP_K_DENSE: int = 20
    RETRIEVAL_TOP_K_SPARSE: int = 20
    RETRIEVAL_RRF_K: int = 60
    RETRIEVAL_FINAL_TOP_K: int = 10

    # ─── Query Router ────────────────────────────────────────────────────────
    ROUTER_MODEL_PATH: str = "models/router/query_router"
    ROUTER_CONFIDENCE_THRESHOLD: float = 0.7

    # ─── Citation Verifier ───────────────────────────────────────────────────
    CITATION_VERIFICATION_ENABLED: bool = True
    CITATION_FAILURE_THRESHOLD: float = 0.3

    # ─── LLM Provider ───────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    # Generic OpenAI-compatible endpoint (Groq, Together, etc.)
    LLM_API_URL: str = ""
    LLM_API_KEY: str = ""
    LLM_MODEL_NAME: str = ""

    # ─── Data Ingestion ──────────────────────────────────────────────────────
    INDIAN_KANOON_API_URL: str = "https://api.indiankanoon.org"
    INDIAN_KANOON_API_TOKEN: str = ""
    SCRAPER_RATE_LIMIT_REQUESTS: int = 10
    SCRAPER_RATE_LIMIT_PERIOD: int = 1
    SCRAPER_RETRY_MAX: int = 3
    SCRAPER_RETRY_BACKOFF: float = 2.0

    # ─── Security ────────────────────────────────────────────────────────────
    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60
    ENCRYPTION_KEY: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache ensures the .env file is read only once.
    Call get_settings.cache_clear() to reload in tests.
    """
    return Settings()


# Convenience alias — import this directly
settings = get_settings()