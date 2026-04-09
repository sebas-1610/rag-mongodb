"""
config/settings.py
==================
S — Una sola responsabilidad: leer y exponer la configuración.
D — Todo el proyecto importa Settings, nunca os.environ directamente.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # MongoDB
    mongodb_uri: str
    mongodb_db: str = "rag_ia"

    # Groq
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking
    chunk_size: int = 256
    chunk_overlap: int = 32
    sentence_max: int = 5
    sentence_overlap: int = 1
    semantic_threshold: float = 0.75

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """
    Singleton cacheado — se crea una vez y se reutiliza en toda la app.
    Uso: from config.settings import get_settings; cfg = get_settings()
    """
    return Settings()
