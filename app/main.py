import time
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi

from starlette.exceptions import HTTPException as StarletteHTTPException
from app.config.settings import get_settings, Settings
from app.core.logging import setup_logging
from app.middleware.request_context import RequestContextMiddleware
from app.api.exception_handlers import (
    app_exception_handler,
    validation_exception_handler,
    global_exception_handler,
    http_exception_handler,
)
from app.core.exceptions import AppException
from app.api.router import api_router
from app.services.jobs import JobManager
from app.ai.manager.model_manager import ModelManager


def customize_openapi(app: FastAPI, settings: Settings) -> None:
    """
    Applies custom descriptive headers and branding info to the OpenAPI schema.
    """
    def custom_openapi_schema():
        if app.openapi_schema:
            return app.openapi_schema
        
        schema = get_openapi(
            title=f"🚀 {settings.APP_NAME} Documentation",
            version=settings.API_VERSION,
            description=(
                "**Production-grade Python API Backend** designed for 3D AI Reconstruction.<br>"
                "Later features will include queue management, mesh export, and Blender workflows.<br>"
                "*This API includes tracing correlation headers and structured error envelopes.*"
            ),
            routes=app.routes,
        )
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi_schema


def create_app() -> FastAPI:
    """
    FastAPI Application Factory pattern.
    Bootstraps logging, middlewares, exception translation filters, routers, and lifecycles.
    """
    settings = get_settings()

    # Limit PyTorch CPU thread usage to 2 cores so Mac remains cool and UI does not lag
    try:
        import torch
        torch.set_num_threads(2)
    except Exception:
        pass

    # 1. Setup system-wide logging
    setup_logging(settings)
    logger = logging.getLogger("app.lifecycle")

    # 2. Instantiate FastAPI
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.API_VERSION,
        debug=settings.DEBUG,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
    )

    # Attach settings to state for middleware/dependency use
    app.state.settings = settings

    # 3. Startup & Shutdown Events
    @app.on_event("startup")
    async def startup_event() -> None:
        app.state.start_time = time.time()
        
        # Verify and create runtime directories
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        settings.logs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize and register JobManager
        app.state.job_manager = JobManager(settings)

        # Initialize ModelManager & trigger non-blocking model pre-warm task
        app.state.model_manager = ModelManager(settings)
        
        async def _warmup_model():
            try:
                logger.info("Pre-warming Hunyuan3D model into accelerator memory...")
                await asyncio.to_thread(app.state.model_manager.load, "hunyuan3d")
                logger.info("Hunyuan3D model successfully pre-warmed into memory.")
            except Exception as e:
                logger.warning(f"Startup model pre-warm note: {e}")

        asyncio.create_task(_warmup_model())

        logger.info(
            f"Successfully booted {settings.APP_NAME} in "
            f"{'DEBUG' if settings.DEBUG else 'PRODUCTION'} mode."
        )

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        logger.info(f"Starting shutdown sequence for {settings.APP_NAME}.")
        # Unload Hunyuan3D model weights to release accelerator memory
        if hasattr(app.state, "model_manager"):
            try:
                app.state.model_manager.unload("hunyuan3d")
            except Exception as e:
                logger.error(f"Error unloading Hunyuan3D model during shutdown: {e}")
        logger.info("Shutdown sequence completed.")

    # 4. Standard CORS Settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict this to domain lists in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 5. Custom Request Tracking Middleware
    app.add_middleware(RequestContextMiddleware)

    # 6. Global Exception Translation Filter Registrations
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    # 7. Router Mounting
    app.include_router(api_router)

    # 8. Swagger / Redoc customized specs
    customize_openapi(app, settings)

    return app


# Export application instance for WSGI/ASGI servers (e.g. Uvicorn)
app = create_app()
