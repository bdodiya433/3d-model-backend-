from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings configuration class utilizing Pydantic Settings.
    Loads configurations from .env file or system environment variables.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    DEBUG: bool = Field(default=True)
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8000)
    APP_NAME: str = Field(default="Image-to-3D AI Backend")
    API_VERSION: str = Field(default="v1")
    LOG_LEVEL: str = Field(default="INFO")

    # File System Configurations
    MAX_UPLOAD_SIZE: int = Field(default=104857600)  # 100 MB in bytes
    UPLOAD_FOLDER: str = Field(default="uploads")
    OUTPUT_FOLDER: str = Field(default="outputs")
    TEMP_FOLDER: str = Field(default="temp")
    WEIGHTS_FOLDER: str = Field(default="weights")
    LOGS_FOLDER: str = Field(default="logs")

    @property
    def upload_dir(self) -> Path:
        return Path(self.UPLOAD_FOLDER)

    @property
    def output_dir(self) -> Path:
        return Path(self.OUTPUT_FOLDER)

    @property
    def temp_dir(self) -> Path:
        return Path(self.TEMP_FOLDER)

    @property
    def weights_dir(self) -> Path:
        return Path(self.WEIGHTS_FOLDER)

    @property
    def logs_dir(self) -> Path:
        return Path(self.LOGS_FOLDER)


@lru_cache
def get_settings() -> Settings:
    """
    Dependency injection helper to retrieve settings.
    Uses LRU cache to prevent re-parsing of .env/environment files.
    """
    return Settings()
