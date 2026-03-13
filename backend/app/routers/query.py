"""
NyayaMitra — Query Router (API Endpoint).

This is the main endpoint users hit to ask legal questions.
It orchestrates the full pipeline:

    User Query → Router Classification → Hybrid Retrieval → LLM Generation
    → Citation Verification → Translation → Response

For Sprint 1-4, we build this incrementally:
    Sprint 1: Endpoint skeleton with mock response
    Sprint 3: Plug in hybrid retrieval
    Sprint 4: Plug in LLM generation
    Sprint 5: Plug in query router + knowledge graph
    Sprint 6: Plug in citation verification

Endpoints:
    POST /api/v1/query          Full legal query (JSON response)
    POST /api/v1/query/stream   Streaming legal query (Server-Sent Events)
"""

import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.query import (
    ApplicableLaw,
    Confidence,
    DetailLevel,
    Language,
    LegalDomain,
    LegalStatus,
    Precedent,
    ProcedureStep,
    QueryRequest,
    QueryResponse,
    RouterClassification,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1")


@router.post("/query", response_model=QueryResponse)
async def legal_query(request: QueryRequest) -> QueryResponse:
    """
    Process a legal query through the full NyayaMitra pipeline.

    Pipeline stages:
        1. Query Router — classify domain, jurisdiction, query type
        2. Hybrid Retrieval — fetch relevant acts, judgments, procedures
        3. Context Assembly — format retrieved chunks for LLM
        4. LLM Generation — generate structured legal response
        5. Citation Verification — verify every citation is real
        6. Translation — translate response if needed

    Returns a structured legal response with citations, precedents,
    and step-by-step procedure.
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        "query_received",
        query_length=len(request.query),
        language=request.language.value,
        jurisdiction=request.jurisdiction,
        domain_hint=request.domain_hint.value if request.domain_hint else None,
        session_id=session_id,
    )

    try:
        # ── Stage 1: Query Router ────────────────────────────────────────
        classification = await _classify_query(request)
        logger.info(
            "query_classified",
            domain=classification.domain.value,
            query_type=classification.query_type.value,
            confidence=classification.confidence,
        )

        # ── Stage 2: Hybrid Retrieval ────────────────────────────────────
        retrieved_context = await _retrieve_context(request, classification)
        logger.info(
            "context_retrieved",
            num_chunks=len(retrieved_context),
        )

        # ── Stage 3: LLM Generation ─────────────────────────────────────
        response = await _generate_response(request, classification, retrieved_context, session_id)
        logger.info(
            "response_generated",
            num_laws=len(response.applicable_law),
            num_precedents=len(response.precedents),
            num_steps=len(response.procedure),
            confidence=response.confidence.value,
        )

        # ── Stage 4: Citation Verification ───────────────────────────────
        if settings.CITATION_VERIFICATION_ENABLED:
            response = await _verify_citations(response)
            logger.info(
                "citations_verified",
                sources_verified=response.sources_verified,
            )

        # ── Stage 5: Translation ─────────────────────────────────────────
        if request.language != Language.ENGLISH:
            response = await _translate_response(response, request.language)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "query_completed",
            duration_ms=duration_ms,
            session_id=session_id,
        )

        return response

    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(
            "query_failed",
            error=str(e),
            duration_ms=duration_ms,
            session_id=session_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process legal query",
                "message": str(e) if settings.APP_DEBUG else "Internal server error",
            },
        )


@router.post("/query/stream")
async def legal_query_stream(request: QueryRequest):
    """
    Streaming legal query using Server-Sent Events (SSE).

    The frontend uses this for real-time response display.
    Response streams as JSON chunks separated by newlines.

    TODO (Sprint 4): Implement actual streaming from vLLM.
    Currently returns the full response as a single SSE event.
    """
    response = await legal_query(request)

    async def event_generator():
        """Generate SSE events from the response."""
        import json

        # Send the answer text in chunks (simulating streaming)
        words = response.answer.split()
        chunk_size = 5
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            event = json.dumps({"type": "text", "content": chunk})
            yield f"data: {event}\n\n"

        # Send structured data at the end
        law_data = json.dumps({"type": "applicable_law", "content": response.model_dump(include={"applicable_law"})}, default=str)
        yield f"data: {law_data}\n\n"

        prec_data = json.dumps({"type": "precedents", "content": response.model_dump(include={"precedents"})}, default=str)
        yield f"data: {prec_data}\n\n"

        proc_data = json.dumps({"type": "procedure", "content": response.model_dump(include={"procedure"})}, default=str)
        yield f"data: {proc_data}\n\n"

        done_data = json.dumps({"type": "done", "response_id": response.response_id})
        yield f"data: {done_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Stage Implementations
# ═══════════════════════════════════════════════════════════════════════════════
# Each stage is a separate async function. In Sprint 1, these return mock data.
# They will be replaced with real implementations in subsequent sprints.
# ═══════════════════════════════════════════════════════════════════════════════


async def _classify_query(request: QueryRequest) -> RouterClassification:
    """
    Stage 1: Classify the query into domain, type, jurisdiction.

    Sprint 1: Uses domain_hint if provided, otherwise defaults to GENERAL.
    Sprint 5: Will use fine-tuned DistilBERT classifier.
    """
    # TODO (Sprint 5): Replace with actual router model
    return RouterClassification(
        domain=request.domain_hint or LegalDomain.GENERAL,
        query_type="general",
        jurisdiction=request.jurisdiction,
        language=request.language,
        confidence=0.5 if request.domain_hint is None else 0.95,
    )


async def _retrieve_context(
    request: QueryRequest,
    classification: RouterClassification,
) -> list:
    """
    Stage 2: Retrieve relevant legal context from Qdrant.

    Sprint 3: Dense retrieval via Qdrant.
    Sprint 3+: Will add Elasticsearch BM25 + RRF fusion.
    Sprint 5: Will add Neo4j knowledge graph traversal.
    """
    from app.services.retrieval import get_retrieval_service

    service = await get_retrieval_service()

    results = await service.search(
        query=request.query,
        domain=classification.domain.value if classification.domain.value != "general" else None,
        jurisdiction=classification.jurisdiction,
    )

    return results


async def _generate_response(
    request: QueryRequest,
    classification: RouterClassification,
    context: list,
    session_id: str,
) -> QueryResponse:
    """
    Stage 3: Generate a structured legal response from retrieved context.

    Sprint 3: Builds response directly from retrieval results (no LLM yet).
    Sprint 4: Will use vLLM with Llama 3.1 8B + retrieved context.
    Sprint 9: Will use fine-tuned Llama 3.1 70B + domain LoRA adapters.
    """
    # Build applicable_law from retrieved sections
    applicable_laws = []
    for result in context:
        if result.source_type == "act" and result.section_number:
            applicable_laws.append(
                ApplicableLaw(
                    act=result.act_name or "Unknown Act",
                    section=result.section_number,
                    text=result.text[:500] if result.text else "",
                    status=LegalStatus(result.status) if result.status in ["active", "repealed", "amended"] else LegalStatus.ACTIVE,
                )
            )

    # Build precedents from retrieved judgments
    precedents = []
    seen_cases = set()
    for result in context:
        if result.source_type == "judgment" and result.case_name:
            if result.case_name in seen_cases:
                continue
            seen_cases.add(result.case_name)
            precedents.append(
                Precedent(
                    case=result.case_name,
                    year=result.year or 2024,
                    court=result.court or "Supreme Court",
                    citation=result.citation or "",
                    relevance=result.text[:300] if result.text else "",
                )
            )

    # Build answer from context
    if applicable_laws or precedents:
        answer_parts = [f"Based on the retrieved legal provisions and precedents relevant to your query:"]

        if applicable_laws:
            answer_parts.append("\n\nApplicable Legal Provisions:")
            for law in applicable_laws:
                answer_parts.append(f"\n- {law.act}, Section {law.section}: {law.text[:200]}")

        if precedents:
            answer_parts.append("\n\nRelevant Judicial Precedents:")
            for prec in precedents:
                answer_parts.append(f"\n- {prec.case} ({prec.year}), {prec.court}: {prec.relevance[:200]}")

        answer = "".join(answer_parts)
        confidence = Confidence.MEDIUM
    else:
        answer = (
            "I could not find specific legal provisions directly matching your query "
            "in the current database. Please try rephrasing your question or specifying "
            "the legal domain (criminal, property, family, etc.)."
        )
        confidence = Confidence.LOW

    return QueryResponse(
        answer=answer,
        applicable_law=applicable_laws,
        precedents=precedents,
        procedure=[],
        jurisdiction_notes=(
            f"Jurisdiction: {classification.jurisdiction or 'Central Law (state not specified)'}. "
            "For state-specific variations, please specify your state."
        ),
        confidence=confidence,
        sources_verified=False,
        language=request.language,
        session_id=session_id,
    )


async def _verify_citations(response: QueryResponse) -> QueryResponse:
    """
    Stage 4: Verify all citations in the response.

    Sprint 1: Passes through without verification.
    Sprint 6: Will check every section number and case name against the database.
    """
    # TODO (Sprint 6): Implement citation verification
    # - Check section numbers exist in acts DB
    # - Check case names exist in judgments DB
    # - Check if cited cases have been overruled
    # - Regenerate if failure rate > CITATION_FAILURE_THRESHOLD
    return response


async def _translate_response(
    response: QueryResponse,
    target_language: Language,
) -> QueryResponse:
    """
    Stage 5: Translate the response to the user's preferred language.

    Sprint 1: Returns response unchanged.
    Sprint 11: Will use IndicTrans2 for Hindi, Tamil, Telugu, Bengali, Marathi.
    """
    # TODO (Sprint 11): Implement translation via IndicTrans2
    response.language = target_language
    return response