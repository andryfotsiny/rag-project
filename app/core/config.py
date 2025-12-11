
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    app_name: str = "RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data/raw"))
    index_dir: Path = Field(default_factory=lambda: Path("data/processed"))
    log_dir: Path = Field(default_factory=lambda: Path("logs"))

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    normalize_embeddings: bool = True

    chunk_size: int = 300
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 500

    faiss_index_type: str = "IndexFlatIP"
    faiss_index_path: str = "data/processed/index.faiss"
    faiss_metadata_path: str = "data/processed/metadata.pkl"

    #  RAG
    default_top_k: int = 5
    min_similarity_score: float = 0.65
    max_context_length: int = 2000

    # PERFORMANCE
    enable_cache: bool = True
    cache_ttl: int = 3600
    max_workers: int = 4

    @field_validator("data_dir", "index_dir", "log_dir")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Crée les répertoires s'ils n'existent pas."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("min_similarity_score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Valide que le score est entre 0 et 1."""
        if not 0 <= v <= 1:
            raise ValueError("min_similarity_score doit être entre 0 et 1")
        return v

    @property
    def faiss_index_full_path(self) -> Path:
        """Path complet vers l'index FAISS."""
        return self.base_dir / self.faiss_index_path

    @property
    def faiss_metadata_full_path(self) -> Path:
        """Path complet vers les métadonnées."""
        return self.base_dir / self.faiss_metadata_path


@lru_cache()
def get_settings() -> Settings:
    """Singleton pour les settings."""
    return Settings()


settings = get_settings()