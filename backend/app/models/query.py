"""
NyayaMitra — Pydantic Schemas for Legal Queries and Responses.

These schemas define the data contract for the entire system.
Every component (router, retrieval, LLM, verifier) reads and writes
using these models, ensuring consistency across the pipeline.

Usage:
    from app.models.query import QueryRequest, QueryResponse
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════


class LegalDomain(str, Enum):
    """Legal domains supported by NyayaMitra."""

    CRIMINAL = "criminal"
    PROPERTY = "property"
    FAMILY = "family"
    LABOR = "labor"
    CONSUMER = "consumer"
    CONSTITUTIONAL = "constitutional"
    IP = "ip"
    GENERAL = "general"


class QueryType(str, Enum):
    """Type of legal query the user is asking."""

    RIGHTS = "rights"
    PROCEDURE = "procedure"
    CASE_OUTCOME = "case_outcome"
    GENERAL = "general"


class DetailLevel(str, Enum):
    """Level of detail requested in the response."""

    DETAILED = "detailed"
    SUMMARY = "summary"
    PROCEDURE_ONLY = "procedure_only"


class Language(str, Enum):
    """Supported languages (ISO 639-1 codes)."""

    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    TELUGU = "te"
    BENGALI = "bn"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"


class Confidence(str, Enum):
    """Confidence level of the response."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LegalStatus(str, Enum):
    """Current status of a legal provision."""

    ACTIVE = "active"
    REPEALED = "repealed"
    AMENDED = "amended"
    PARTIALLY_REPEALED = "partially_repealed"


# ═══════════════════════════════════════════════════════════════════════════════
# Request Schema
# ═══════════════════════════════════════════════════════════════════════════════


class QueryRequest(BaseModel):
    """
    Legal query request from the user.

    Example:
        {
            "query": "Can police arrest me without a warrant?",
            "language": "en",
            "jurisdiction": "Maharashtra",
            "domain_hint": "criminal",
            "detail_level": "detailed"
        }
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's legal question in natural language.",
        examples=["Can police arrest me without a warrant?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity. Omit for new session.",
    )
    language: Language = Field(
        default=Language.ENGLISH,
        description="Preferred response language (ISO 639-1 code).",
    )
    jurisdiction: str | None = Field(
        default=None,
        description="User's state/jurisdiction. If not provided, central law is used.",
        examples=["Maharashtra", "Karnataka", "Tamil Nadu"],
    )
    domain_hint: LegalDomain | None = Field(
        default=None,
        description="Optional hint for the legal domain. Auto-classified if not provided.",
    )
    detail_level: DetailLevel = Field(
        default=DetailLevel.DETAILED,
        description="Level of detail in the response.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Response Sub-schemas
# ═══════════════════════════════════════════════════════════════════════════════


class ApplicableLaw(BaseModel):
    """A specific legal provision cited in the response."""

    act: str = Field(
        ...,
        description="Name of the act.",
        examples=["Code of Criminal Procedure, 1973"],
    )
    section: str = Field(
        ...,
        description="Section number within the act.",
        examples=["41", "41A", "41(1)(b)"],
    )
    text: str = Field(
        ...,
        description="Relevant text or summary of the section.",
    )
    status: LegalStatus = Field(
        default=LegalStatus.ACTIVE,
        description="Current status of this provision.",
    )


class Precedent(BaseModel):
    """A judicial precedent (case law) cited in the response."""

    case: str = Field(
        ...,
        description="Case name.",
        examples=["D.K. Basu v. State of West Bengal"],
    )
    year: int = Field(
        ...,
        description="Year of the judgment.",
        examples=[1997],
    )
    court: str = Field(
        ...,
        description="Court that delivered the judgment.",
        examples=["Supreme Court", "Bombay High Court"],
    )
    citation: str = Field(
        ...,
        description="Official citation.",
        examples=["(1997) 1 SCC 416", "AIR 1997 SC 610"],
    )
    relevance: str = Field(
        ...,
        description="Why this case is relevant to the user's query.",
    )


class ProcedureStep(BaseModel):
    """A single step in a legal procedure."""

    step: int = Field(
        ...,
        description="Step number in the procedure.",
        examples=[1, 2, 3],
    )
    action: str = Field(
        ...,
        description="What the user should do in this step.",
        examples=["File a First Information Report (FIR) at the nearest police station"],
    )
    details: str = Field(
        ...,
        description="Detailed explanation of this step.",
    )
    forms: list[str] = Field(
        default_factory=list,
        description="Any forms required for this step.",
    )
    court: str | None = Field(
        default=None,
        description="Which court or authority handles this step.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Response Schema
# ═══════════════════════════════════════════════════════════════════════════════


class QueryResponse(BaseModel):
    """
    Structured legal response returned to the user.

    Every response includes applicable law, precedents, procedure,
    jurisdiction notes, confidence level, and a legal disclaimer.
    """

    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this response.",
    )
    answer: str = Field(
        ...,
        description="Plain language explanation answering the user's question.",
    )
    applicable_law: list[ApplicableLaw] = Field(
        default_factory=list,
        description="Specific legal provisions applicable to the query.",
    )
    precedents: list[Precedent] = Field(
        default_factory=list,
        description="Relevant judicial precedents with citations.",
    )
    procedure: list[ProcedureStep] = Field(
        default_factory=list,
        description="Step-by-step procedural guidance.",
    )
    jurisdiction_notes: str | None = Field(
        default=None,
        description="State-specific variations or notes.",
    )
    confidence: Confidence = Field(
        default=Confidence.MEDIUM,
        description="System confidence in the response accuracy.",
    )
    disclaimer: str = Field(
        default=(
            "This is legal information, not legal advice. "
            "For case-specific advice, consult a qualified advocate."
        ),
        description="Legal disclaimer (always present).",
    )
    sources_verified: bool = Field(
        default=False,
        description="Whether all citations passed verification.",
    )
    language: Language = Field(
        default=Language.ENGLISH,
        description="Language of the response.",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for conversation continuity.",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of response generation.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Router Classification Schema
# ═══════════════════════════════════════════════════════════════════════════════


class RouterClassification(BaseModel):
    """Output of the Query Router — classifies the user's query."""

    domain: LegalDomain = Field(
        ...,
        description="Classified legal domain.",
    )
    query_type: QueryType = Field(
        default=QueryType.GENERAL,
        description="Type of legal query.",
    )
    jurisdiction: str | None = Field(
        default=None,
        description="Detected or provided jurisdiction.",
    )
    language: Language = Field(
        default=Language.ENGLISH,
        description="Detected input language.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Classification confidence score (0.0 to 1.0).",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback Schema
# ═══════════════════════════════════════════════════════════════════════════════


class FeedbackRequest(BaseModel):
    """User feedback on a response."""

    response_id: str = Field(
        ...,
        description="ID of the response being rated.",
    )
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1 (poor) to 5 (excellent).",
    )
    feedback_type: str | None = Field(
        default=None,
        description="Category: 'wrong_citation', 'incomplete', 'wrong_jurisdiction', 'helpful', 'other'.",
    )
    comment: str | None = Field(
        default=None,
        max_length=2000,
        description="Free-text feedback or correction.",
    )
    corrected_answer: str | None = Field(
        default=None,
        description="User-provided correction (for training data pipeline).",
    )