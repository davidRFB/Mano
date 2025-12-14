"""
API Configuration using Pydantic Settings.

Settings can be overridden via environment variables or .env file.
Example:
    MODEL_RUN_ID=abc123 uvicorn src.api.main:app
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Settings
    api_title: str = "LSC-Connect API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Model Settings
    model_run_id: str = "587ca0fd066a4a1fbf1a5a26971c3284"

    # MLflow Settings
    # Path relative to project root where mlruns directory lives
    mlflow_tracking_uri: str = "models/mlruns"
    
    # Image Settings (must match training)
    image_size: int = 224

    # Inference Settings
    confidence_threshold: float = 0.5  # Minimum confidence to return prediction

    # CORS Settings (for frontend access)
    cors_origins: list[str] = ["*"]  # In production, restrict this!

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # This allows MODEL_RUN_ID env var to map to model_run_id
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Using lru_cache ensures we only create one Settings instance,
    which only reads the .env file once at startup.
    """
    return Settings()