"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings

APP_VERSION = "0.1.0"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # File handling
    upload_dir: Path = Path(__file__).parent.parent / "uploads"
    max_file_size_mb: int = 500

    # Analysis defaults
    default_analysis_depth: str = "standard"
    analysis_timeout_seconds: int = 120

    # Optional features
    enable_vocal_analysis: bool = True
    enable_watermark_detection: bool = True

    class Config:
        env_prefix = "AI_DETECTOR_"


settings = Settings()
