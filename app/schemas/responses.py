from typing import Any, Dict, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    """
    Standard envelope pattern for all API responses.
    """
    success: bool = Field(default=True, description="Indicates if the API request was successful")
    data: Optional[T] = Field(default=None, description="The payload of the response")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information in case of failure")


class HealthStatus(BaseModel):
    """
    Details about the system's operational health.
    """
    status: str = Field(default="healthy", description="Application status (healthy, degraded, unhealthy)")
    uptime_seconds: float = Field(..., description="Total system uptime in seconds")
    timestamp: str = Field(..., description="Current ISO 8601 timestamp")
    directories: Dict[str, bool] = Field(..., description="Status of required workspace directories")


class VersionInfo(BaseModel):
    """
    Metadata representation of the application's software version.
    """
    app_name: str = Field(..., description="Name of the application")
    api_version: str = Field(..., description="Current API major version (e.g. v1)")
    debug_mode: bool = Field(..., description="Whether the application is running in debug mode")


class RootInfo(BaseModel):
    """
    Information returned on root query.
    """
    message: str = Field(..., description="Welcome message")
    docs_url: str = Field(..., description="URL pointing to Swagger Interactive API documentation")
    redoc_url: str = Field(..., description="URL pointing to ReDoc API documentation")


class ModelStatusItem(BaseModel):
    """
    Operational details for a single model provider.
    """
    enabled: bool = Field(..., description="Indicates if the model is enabled in configuration")
    loaded: bool = Field(..., description="Indicates if the model is loaded in memory")
    load_time_seconds: Optional[float] = Field(default=None, description="Time taken to load weights (seconds)")
    health: bool = Field(..., description="Diagnostic status of the loaded model")


class ModelsStatusResponse(BaseModel):
    """
    Consolidated operational status report of the AI subsystem.
    """
    models: Dict[str, ModelStatusItem] = Field(..., description="Detailed status of each provider")
    memory_info: Dict[str, Any] = Field(..., description="System RAM and virtual memory metrics")
    device: str = Field(..., description="Current hardware accelerator (mps, cuda, cpu)")


class ModelsListResponse(BaseModel):
    """
    Overview list of enabled and currently resident models.
    """
    loaded_models: list[str] = Field(..., description="List of models currently resident in memory")
    enabled_models: list[str] = Field(..., description="List of models enabled in settings")

