from typing import Any, Optional


class AppException(Exception):
    """
    Base Exception class for all application-related business errors.
    """
    def __init__(
        self,
        status_code: int = 500,
        message: str = "An unexpected error occurred",
        error_code: str = "INTERNAL_SERVER_ERROR",
        details: Optional[Any] = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_code = error_code
        self.details = details


class ValidationException(AppException):
    """
    Exception raised when inputs or request objects fail validation.
    """
    def __init__(self, message: str = "Validation failed", details: Optional[Any] = None) -> None:
        super().__init__(
            status_code=422,
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class NotFoundException(AppException):
    """
    Exception raised when a requested resource is not found.
    """
    def __init__(self, message: str = "Resource not found", details: Optional[Any] = None) -> None:
        super().__init__(
            status_code=404,
            message=message,
            error_code="NOT_FOUND",
            details=details
        )


class InternalServerException(AppException):
    """
    Exception raised for unhandled exceptions or critical system errors.
    """
    def __init__(self, message: str = "Internal server error", details: Optional[Any] = None) -> None:
        super().__init__(
            status_code=500,
            message=message,
            error_code="INTERNAL_SERVER_ERROR",
            details=details
        )
