from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 120
    llm_model: str = "qwen2.5:3b"
    embedding_model: str = "embeddinggemma"
    chroma_dir: Path = Path("chroma_db")
    max_upload_mb: int = 20

    @property
    def chroma_path(self) -> Path:
        return self.chroma_dir if self.chroma_dir.is_absolute() else BACKEND_DIR / self.chroma_dir

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
