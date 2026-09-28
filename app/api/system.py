import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Request
from app.config.settings import Settings, get_settings
from app.schemas.responses import BaseResponse, HealthStatus, VersionInfo, RootInfo

router = APIRouter(tags=["System"])


@router.get("/", response_model=BaseResponse[RootInfo])
async def get_root(settings: Settings = Depends(get_settings)) -> BaseResponse[RootInfo]:
    """
    Root endpoint serving application welcome metadata and interactive API links.
    """
    data = RootInfo(
        message=f"Welcome to the {settings.APP_NAME} API.",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    return BaseResponse(data=data)


@router.get("/health", response_model=BaseResponse[HealthStatus])
async def get_health(request: Request, settings: Settings = Depends(get_settings)) -> BaseResponse[HealthStatus]:
    """
    Health check endpoint monitoring server availability, uptime, and write permissions 
    on storage directories.
    """
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time

    # Verify key directories are present and writable
    target_dirs = {
        "uploads": settings.upload_dir,
        "outputs": settings.output_dir,
        "temp": settings.temp_dir,
        "weights": settings.model_dir,
        "logs": settings.logs_dir
    }

    dir_status = {}
    is_healthy = True

    for name, path in target_dirs.items():
        try:
            # Check exist and try to resolve directory
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            dir_status[name] = path.exists() and os_is_writable(path)
            if not dir_status[name]:
                is_healthy = False
        except Exception:
            dir_status[name] = False
            is_healthy = False

    status_str = "healthy" if is_healthy else "degraded"

    data = HealthStatus(
        status=status_str,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        directories=dir_status
    )
    return BaseResponse(data=data)


@router.get("/version", response_model=BaseResponse[VersionInfo])
async def get_version(settings: Settings = Depends(get_settings)) -> BaseResponse[VersionInfo]:
    """
    Version check endpoint displaying current build configurations and metadata.
    """
    data = VersionInfo(
        app_name=settings.APP_NAME,
        api_version=settings.API_VERSION,
        debug_mode=settings.DEBUG
    )
    return BaseResponse(data=data)


def os_is_writable(path) -> bool:
    """
    Checks if a directory path is writable.
    """
    import os
    return os.access(path, os.W_OK)
