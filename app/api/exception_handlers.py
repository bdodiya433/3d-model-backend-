import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.exceptions import AppException


logger = logging.getLogger("app.api.exception_handlers")


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """
    Intercepts and formats custom business exceptions (AppException).
    """
    if exc.status_code >= 500:
        logger.error(f"Internal Application Error: {exc.message} - Code: {exc.error_code} - Details: {exc.details}")
    else:
        logger.warning(f"Application Request Exception: {exc.message} - Code: {exc.error_code}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Intercepts FastAPI's default RequestValidationError, translating it into a uniform response payload.
    """
    details = []
    for error in exc.errors():
        loc = error.get("loc", [])
        field = ".".join(str(l) for l in loc[1:]) if len(loc) > 1 else str(loc[0]) if loc else "request"
        details.append({
            "field": field,
            "type": error.get("type"),
            "message": error.get("msg")
        })

    logger.warning(f"Request validation failure on path {request.url.path} - errors: {details}")

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Input validation failed",
                "details": details
            }
        }
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Fallback global handler intercepting all unhandled native exceptions.
    Prevents raw tracebacks from leaking to customers in production.
    """
    logger.exception(f"Unhandled Exception: {exc}")

    # Safely determine debug state from FastAPI app state configuration
    settings = getattr(request.app.state, "settings", None)
    is_debug = settings.DEBUG if settings else True

    message = str(exc) if is_debug else "An unexpected error occurred"
    details = exc.__class__.__name__ if is_debug else None

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": message,
                "details": details
            }
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Intercepts and formats standard Starlette HTTP exceptions (e.g. 404, 405).
    """
    error_code = "HTTP_ERROR"
    if exc.status_code == 404:
        error_code = "NOT_FOUND"
    elif exc.status_code == 405:
        error_code = "METHOD_NOT_ALLOWED"
    elif exc.status_code == 403:
        error_code = "FORBIDDEN"
    elif exc.status_code == 401:
        error_code = "UNAUTHORIZED"

    logger.warning(f"HTTP error {exc.status_code}: {exc.detail} on path {request.url.path}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": error_code,
                "message": exc.detail,
                "details": None
            }
        }
    )

