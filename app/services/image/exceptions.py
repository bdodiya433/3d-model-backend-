from app.core.exceptions import AppException


class ImageProcessingError(AppException):
    """
    Raised when image loading or basic image manipulation fails.
    """
    def __init__(self, message: str = "Failed to process image", details: dict | None = None) -> None:
        super().__init__(
            status_code=400,
            message=message,
            error_code="IMAGE_PROCESSING_ERROR",
            details=details,
        )


class ObjectDetectionError(AppException):
    """
    Raised when no valid object can be isolated from the input image.
    """
    def __init__(self, message: str = "No valid object detected in image", details: dict | None = None) -> None:
        super().__init__(
            status_code=422,
            message=message,
            error_code="OBJECT_NOT_FOUND",
            details=details,
        )
