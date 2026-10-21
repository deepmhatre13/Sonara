from typing import Any

from fastapi import HTTPException, status


class SonaraException(HTTPException):
    """Base application exception for Sonara."""

    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "An internal error occurred",
        code: str = "INTERNAL_ERROR",
        extra: dict[str, Any] | None = None,
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code
        self.extra = extra or {}


class EntityNotFoundError(SonaraException):
    def __init__(self, entity_name: str, entity_id: str | int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{entity_name} with identifier '{entity_id}' was not found.",
            code="ENTITY_NOT_FOUND",
            extra={"entity_name": entity_name, "entity_id": str(entity_id)},
        )


class AuthorizationError(SonaraException):
    def __init__(self, detail: str = "Permission denied for this resource."):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            code="PERMISSION_DENIED",
        )


class AuthenticationRequiredError(SonaraException):
    def __init__(self, detail: str = "Authentication credentials were not provided or invalid."):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            code="UNAUTHENTICATED",
        )


class AudioValidationError(SonaraException):
    def __init__(self, detail: str, extra: dict[str, Any] | None = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            code="INVALID_AUDIO_FORMAT",
            extra=extra,
        )


class ProviderUnavailableError(SonaraException):
    def __init__(self, provider_name: str, detail: str = "External provider unavailable."):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{provider_name}: {detail}",
            code="PROVIDER_UNAVAILABLE",
            extra={"provider": provider_name},
        )

