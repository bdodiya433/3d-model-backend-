import contextvars
import logging
from logging.handlers import RotatingFileHandler
from app.config.settings import Settings

# Context variable to hold request_id for the lifetime of an async task/request
request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIDFilter(logging.Filter):
    """
    Logging Filter to inject the request_id context variable into log records.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx_var.get("-")
        return True


def setup_logging(settings: Settings) -> None:
    """
    Configures the root logger with console and rotating file handlers.
    Creates logs directory if it does not exist.
    """
    log_dir = settings.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    # Define log format containing timestamp, log level, request ID context, and logger location
    log_format = "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(log_format)

    # Filter to inject the request ID context variable
    req_filter = RequestIDFilter()

    # Root Logger setup
    root_logger = logging.getLogger()
    # Safely parse log level
    numeric_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicate log entries
    root_logger.handlers.clear()

    # Console Handler configuration
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(req_filter)
    root_logger.addHandler(console_handler)

    # Rotating File Handler configuration (10 MB files, keeping 5 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(req_filter)
    root_logger.addHandler(file_handler)

    # Adjust third-party logs level to prevent console noise in production
    if not settings.DEBUG:
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
