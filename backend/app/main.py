import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import SonaraException
from app.core.logging import setup_logging
from app.schemas.health import ErrorDetail, ErrorResponse

logger = logging.getLogger("sonara.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle events."""
    setup_logging()
    logger.info("Initializing Sonara Backend Application...")
    yield
    logger.info("Shutting down Sonara Backend Application...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend for Sonara — Audio Intelligence & Knowledge Retrieval",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """
    Assigns or preserves a correlation ID for every request, attaches it to the
    response headers, and tracks request timing.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# Global Exception Handlers
@app.exception_handler(SonaraException)
async def sonara_exception_handler(request: Request, exc: SonaraException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    error_response = ErrorResponse(
        error=ErrorDetail(
            code=exc.code,
            message=str(exc.detail),
            details=exc.extra if exc.extra else None,
        ),
        request_id=request_id,
    )
    return JSONResponse(status_code=exc.status_code, content=error_response.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    error_response = ErrorResponse(
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            details={"errors": exc.errors()},
        ),
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        f"Unhandled exception during request {request_id}: {exc}",
        exc_info=True,
        extra={"correlation_id": request_id},
    )
    # Never expose internal stack traces or internal exception strings to client
    error_response = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected server error occurred. Please contact support.",
            details=None,
        ),
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )


# Mount versioned API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

