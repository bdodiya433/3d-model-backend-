from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings configuration using Pydantic Settings.
    Loads configurations from environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    APP_NAME: str = Field(default="Image-to-3D AI Platform Backend")
    DEBUG: bool = Field(default=True)
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8000)
    LOG_LEVEL: str = Field(default="INFO")
    API_VERSION: str = Field(default="v1")

    # File System Configurations
    MODEL_DIRECTORY: str = Field(default="weights")
    UPLOAD_DIRECTORY: str = Field(default="uploads")
    OUTPUT_DIRECTORY: str = Field(default="outputs")
    TEMP_DIRECTORY: str = Field(default="temp")
    LOGS_DIRECTORY: str = Field(default="logs")

    # Model Activations
    ENABLE_HUNYUAN: bool = Field(default=True)
    ENABLE_TRIPOSR: bool = Field(default=True)
    ENABLE_INSTANTMESH: bool = Field(default=True)

    @property
    def model_dir(self) -> Path:
        return Path(self.MODEL_DIRECTORY)

    @property
    def upload_dir(self) -> Path:
        return Path(self.UPLOAD_DIRECTORY)

    @property
    def output_dir(self) -> Path:
        return Path(self.OUTPUT_DIRECTORY)

    @property
    def temp_dir(self) -> Path:
        return Path(self.TEMP_DIRECTORY)

    @property
    def logs_dir(self) -> Path:
        return Path(self.LOGS_DIRECTORY)


@lru_cache
def get_settings() -> Settings:
    """
    Dependency injection helper to retrieve settings.
    Uses LRU cache to prevent re-parsing .env/environment files.
    """
    return Settings()
