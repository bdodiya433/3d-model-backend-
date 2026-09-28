import time
import uuid
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from app.core.logging import request_id_ctx_var

logger = logging.getLogger("app.middleware.request_context")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that manages:
    1. Retrieval or generation of X-Request-ID.
    2. Setting the Request ID in the thread-safe context variable for loggers.
    3. Calculating request execution duration.
    4. Attaching tracing information back to the HTTP Response headers.
    5. Logging request telemetry.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Retrieve incoming request ID or generate a fresh UUID4
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Set request ID in the context variable for log formats
        token = request_id_ctx_var.set(request_id)
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration = time.perf_counter() - start_time

            # Inject tracing headers into the client response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{duration:.4f}s"

            # Log completion statistics
            logger.info(
                f"Request completed: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Duration: {duration:.4f}s"
            )
            return response
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} - "
                f"Duration: {duration:.4f}s - Exception: {exc.__class__.__name__}: {exc}"
            )
            raise exc
        finally:
            # Always reset the token to restore the context after the request lifecycle
            request_id_ctx_var.reset(token)
