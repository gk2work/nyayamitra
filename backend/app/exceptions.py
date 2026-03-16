"""
NyayaMitra — Custom Exception Hierarchy.

Central exception classes for the entire project. Every module imports
from here instead of using bare exceptions. This ensures consistent
error handling, structured logging, and clean error propagation.

Hierarchy:
    NyayaMitraError
    ├── ScraperError
    │   ├── RateLimitError
    │   ├── FetchError
    │   └── ParseError
    ├── ValidationError
    ├── DatabaseError
    ├── PipelineError
    ├── RetrievalError
    ├── LLMError
    └── CitationError

Usage:
    from app.exceptions import ScraperError, ParseError

    try:
        sections = parser.parse_html(html)
    except ParseError as e:
        logger.error("parse_failed", error=str(e), details=e.details)
"""

from __future__ import annotations

from typing import Any


class NyayaMitraError(Exception):
    """
    Base exception for all NyayaMitra errors.

    All custom exceptions inherit from this class, making it easy to
    catch any project-specific error with a single except clause.

    Args:
        message: Human-readable error description.
        details: Optional dict with structured context for logging.
    """

    def __init__(self, message: str = "", details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | {self.details}"
        return self.message


# ═══════════════════════════════════════════════════════════════════════════════
# Scraper Exceptions
# ═══════════════════════════════════════════════════════════════════════════════


class ScraperError(NyayaMitraError):
    """Base exception for all scraping failures."""

    pass


class RateLimitError(ScraperError):
    """Raised when rate limit is hit after all retries are exhausted."""

    def __init__(self, source: str, url: str = "", retry_after: float | None = None):
        details = {"source": source, "url": url}
        if retry_after is not None:
            details["retry_after_seconds"] = retry_after
        super().__init__(
            message=f"Rate limit exceeded for {source}",
            details=details,
        )


class FetchError(ScraperError):
    """Raised on HTTP fetch failures (timeout, connection, 4xx/5xx)."""

    def __init__(
        self,
        url: str,
        status_code: int | None = None,
        reason: str = "",
    ):
        details: dict[str, Any] = {"url": url}
        if status_code is not None:
            details["status_code"] = status_code
        if reason:
            details["reason"] = reason
        super().__init__(
            message=f"Failed to fetch {url}" + (f" (HTTP {status_code})" if status_code else ""),
            details=details,
        )


class ParseError(ScraperError):
    """Raised when HTML/PDF/text parsing fails to extract expected data."""

    def __init__(self, source: str, reason: str = "", document_hint: str = ""):
        details: dict[str, Any] = {"source": source}
        if reason:
            details["reason"] = reason
        if document_hint:
            details["document"] = document_hint[:200]
        super().__init__(
            message=f"Failed to parse content from {source}: {reason}",
            details=details,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Exceptions
# ═══════════════════════════════════════════════════════════════════════════════


class ValidationError(NyayaMitraError):
    """
    Raised when data fails validation checks.

    Carries a list of specific validation errors for reporting.
    """

    def __init__(
        self,
        message: str = "Data validation failed",
        errors: list[str] | None = None,
        record_type: str = "",
        record_id: str = "",
    ):
        details: dict[str, Any] = {}
        if errors:
            details["errors"] = errors
        if record_type:
            details["record_type"] = record_type
        if record_id:
            details["record_id"] = record_id
        self.errors = errors or []
        super().__init__(message=message, details=details)


# ═══════════════════════════════════════════════════════════════════════════════
# Database Exceptions
# ═══════════════════════════════════════════════════════════════════════════════


class DatabaseError(NyayaMitraError):
    """Raised on database operation failures (connection, query, constraint)."""

    def __init__(self, operation: str, reason: str = "", table: str = ""):
        details: dict[str, Any] = {"operation": operation}
        if table:
            details["table"] = table
        if reason:
            details["reason"] = reason
        super().__init__(
            message=f"Database error during {operation}: {reason}",
            details=details,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Exceptions
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineError(NyayaMitraError):
    """Raised when a pipeline stage fails."""

    def __init__(self, stage: str, reason: str = ""):
        details: dict[str, Any] = {"stage": stage}
        if reason:
            details["reason"] = reason
        super().__init__(
            message=f"Pipeline failed at stage '{stage}': {reason}",
            details=details,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval & LLM Exceptions (used by Sprint 3-4+ services)
# ═══════════════════════════════════════════════════════════════════════════════


class RetrievalError(NyayaMitraError):
    """Raised when the retrieval pipeline fails (Qdrant, ES, or fusion)."""

    def __init__(self, source: str, reason: str = ""):
        details: dict[str, Any] = {"source": source}
        if reason:
            details["reason"] = reason
        super().__init__(
            message=f"Retrieval failed from {source}: {reason}",
            details=details,
        )


class LLMError(NyayaMitraError):
    """Raised when LLM generation fails (vLLM, OpenAI, or fallback)."""

    def __init__(self, provider: str, reason: str = ""):
        details: dict[str, Any] = {"provider": provider}
        if reason:
            details["reason"] = reason
        super().__init__(
            message=f"LLM generation failed ({provider}): {reason}",
            details=details,
        )


class CitationError(NyayaMitraError):
    """Raised when citation verification detects invalid citations."""

    def __init__(
        self,
        invalid_count: int = 0,
        total_count: int = 0,
        reason: str = "",
    ):
        details: dict[str, Any] = {
            "invalid_count": invalid_count,
            "total_count": total_count,
        }
        if reason:
            details["reason"] = reason
        failure_rate = invalid_count / total_count if total_count > 0 else 0
        details["failure_rate"] = round(failure_rate, 3)
        super().__init__(
            message=(
                f"Citation verification failed: {invalid_count}/{total_count} "
                f"citations invalid ({failure_rate:.1%})"
            ),
            details=details,
        )