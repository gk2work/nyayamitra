"""
NyayaMitra — FastAPI Application Entry Point.

This is the main application file. It configures:
- CORS middleware for frontend access
- Request logging middleware
- Health check endpoints
- API router mounting

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

Or via Makefile:
    make backend
"""

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import health, query

# ─── Structured Logging Setup ────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if settings.APP_DEBUG else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.get_config()["wrapper_class"].level if hasattr(structlog.get_config().get("wrapper_class", object), "level") else 0
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# ─── Application Lifespan ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.

    Startup: Initialize database connections, load models, warm up caches.
    Shutdown: Close connections gracefully.
    """
    # ── Startup ──
    logger.info(
        "nyayamitra_starting",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        debug=settings.APP_DEBUG,
    )

    # TODO (Sprint 2): Initialize PostgreSQL connection pool
    # TODO (Sprint 3): Initialize Qdrant client
    # TODO (Sprint 3): Initialize Elasticsearch client
    # TODO (Sprint 5): Initialize Neo4j driver
    # TODO (Sprint 4): Initialize Redis client

    logger.info("nyayamitra_ready", port=settings.BACKEND_PORT)

    yield

    # ── Shutdown ──
    logger.info("nyayamitra_shutting_down")

    # TODO: Close all database connections gracefully


# ─── Create FastAPI Application ──────────────────────────────────────────────
app = FastAPI(
    title="NyayaMitra API",
    description="AI-Powered Legal Assistant for Indian Citizens",
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.APP_DEBUG else None,
    redoc_url="/redoc" if settings.APP_DEBUG else None,
    lifespan=lifespan,
)


# ─── CORS Middleware ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request Logging Middleware ──────────────────────────────────────────────
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every request with timing, method, path, and status code."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Bind request context for structured logging
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown",
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(
            "request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id,
            },
        )

    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )

    # Add request ID to response headers for debugging
    response.headers["X-Request-ID"] = request_id
    return response


# ─── Register Routers ────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(query.router, tags=["Query"])

# TODO (Sprint 6): Mount feedback router
# app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])


# ─── Root Endpoint ───────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — basic API info."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "AI-Powered Legal Assistant for Indian Citizens",
        "docs": "/docs" if settings.APP_DEBUG else "disabled",
        "disclaimer": (
            "This is legal information, not legal advice. "
            "For case-specific advice, consult a qualified advocate."
        ),
    }