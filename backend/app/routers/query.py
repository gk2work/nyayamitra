"""
NyayaMitra — Query Router (API Endpoint).

Main endpoint for legal questions. Orchestrates the full pipeline:

    User Query -> Router Classification -> Hybrid Retrieval -> Graph Enrichment
    -> Context Assembly -> LLM Generation -> Response Parsing -> Citation Verification -> Response

Endpoints:
    POST /api/v1/query          Full legal query (JSON response)
    POST /api/v1/query/stream   Streaming legal query (real SSE from LLM)

Sprint history:
    Sprint 1: Endpoint skeleton with mock response
    Sprint 3: Hybrid retrieval (Qdrant + ES + cross-encoder)
    Sprint 4: Real LLM generation + streaming + session history + response parsing
    Sprint 5: Query router (DistilBERT + keyword hybrid) + knowledge graph enrichment
    Sprint 6: Citation verification
"""

import json
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


# Map router domain strings to LegalDomain enum
_DOMAIN_MAP = {
    "criminal": LegalDomain.CRIMINAL,
    "property": LegalDomain.PROPERTY,
    "family": LegalDomain.FAMILY,
    "constitutional": LegalDomain.CONSTITUTIONAL,
    "labor": LegalDomain.LABOR,
    "consumer": LegalDomain.CONSUMER,
    "ip": LegalDomain.IP,
    "general": LegalDomain.GENERAL,
}


# =====================================================================
# POST /api/v1/query -- Full JSON Response
# =====================================================================


@router.post("/query", response_model=QueryResponse)
async def legal_query(request: QueryRequest) -> QueryResponse:
    """
    Process a legal query through the full NyayaMitra pipeline.

    Pipeline:
        1. Session -- load or create conversation session
        2. Query Router -- classify domain, jurisdiction, query type
        3. Hybrid Retrieval -- fetch relevant acts, judgments
        4. Graph Enrichment -- add interpreting judgments, principles
        5. LLM Generation -- generate structured legal response
        6. Response Parsing -- extract citations from LLM output
        7. Citation Verification -- verify every citation is real
        8. Translation -- translate response if needed
        9. Session -- save messages to history
    """
    start_time = time.time()

    # -- Stage 1: Session Management --
    session_id = request.session_id or str(uuid.uuid4())
    history = await _load_session_history(session_id)

    logger.info(
        "query_received",
        query_length=len(request.query),
        language=request.language.value,
        jurisdiction=request.jurisdiction,
        domain_hint=request.domain_hint.value if request.domain_hint else None,
        session_id=session_id,
        history_turns=len(history),
    )

    try:
        # -- Stage 2: Query Router --
        classification = await _classify_query(request)
        logger.info(
            "query_classified",
            domain=classification.domain.value,
            query_type=classification.query_type.value,
            confidence=classification.confidence,
        )

        # -- Stage 3: Hybrid Retrieval --
        retrieved_context = await _retrieve_context(request, classification)
        logger.info("context_retrieved", num_chunks=len(retrieved_context))

        # -- Stage 4: Graph Enrichment --
        retrieved_context = await _enrich_with_graph(retrieved_context)

        # -- Stage 5: LLM Generation + Response Parsing --
        response = await _generate_response(
            request, classification, retrieved_context, session_id, history
        )
        logger.info(
            "response_generated",
            num_laws=len(response.applicable_law),
            num_precedents=len(response.precedents),
            num_steps=len(response.procedure),
            confidence=response.confidence.value,
        )

        # -- Stage 6: Citation Verification --
        if settings.CITATION_VERIFICATION_ENABLED:
            response = await _verify_citations(response)

        # -- Stage 7: Translation --
        if request.language != Language.ENGLISH:
            response = await _translate_response(response, request.language)

        # -- Stage 8: Save to Session --
        await _save_to_session(session_id, request.query, response.answer)

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


# =====================================================================
# POST /api/v1/query/stream -- Real SSE Streaming
# =====================================================================


@router.post("/query/stream")
async def legal_query_stream(request: QueryRequest):
    """
    Streaming legal query using Server-Sent Events (SSE).

    Streams the LLM answer token-by-token, then sends structured
    data (applicable_law, precedents, procedure) as final events.

    SSE event types:
        text           -- answer text chunk (streamed)
        applicable_law -- structured law citations (after streaming)
        precedents     -- structured case citations (after streaming)
        procedure      -- structured procedure steps (after streaming)
        metadata       -- confidence, jurisdiction notes, session_id, router info
        done           -- signals end of stream
        error          -- error occurred during processing
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            # Load session history
            history = await _load_session_history(session_id)

            # Classify query
            classification = await _classify_query(request)

            # Retrieve context
            retrieved_context = await _retrieve_context(request, classification)

            # Graph enrichment
            retrieved_context = await _enrich_with_graph(retrieved_context)

            # Build structured data from retrieval results
            applicable_laws = _build_applicable_laws(retrieved_context)
            precedents_list = _build_precedents(retrieved_context)

            # Format context for LLM (retrieval + graph)
            formatted_context = await _format_full_context(retrieved_context)

            # Stream LLM response
            from app.services.llm_service import get_llm_service

            llm_service = await get_llm_service()

            full_answer = []
            async for chunk in llm_service.generate_stream(
                query=request.query,
                context=formatted_context,
                jurisdiction=classification.jurisdiction,
                history=history,
            ):
                full_answer.append(chunk)
                event = json.dumps({"type": "text", "content": chunk})
                yield f"data: {event}\n\n"

            # Parse the full answer for additional structured data
            answer_text = "".join(full_answer)
            from app.services.response_parser import (
                parse_llm_response,
                merge_parsed_with_retrieval,
            )

            parsed = parse_llm_response(answer_text)
            merged = merge_parsed_with_retrieval(
                parsed, applicable_laws, precedents_list
            )

            # Add any LLM-extracted citations not in retrieval
            for law_dict in merged["new_laws"]:
                applicable_laws.append(
                    ApplicableLaw(
                        act=law_dict["act"],
                        section=law_dict["section"],
                        text=law_dict.get("text", ""),
                    )
                )

            for prec_dict in merged["new_precedents"]:
                precedents_list.append(
                    Precedent(
                        case=prec_dict["case"],
                        year=prec_dict.get("year", 2024),
                        court=prec_dict.get("court", "Supreme Court"),
                        citation=prec_dict.get("citation", ""),
                        relevance=prec_dict.get("relevance", ""),
                    )
                )

            # Extract procedure steps from parsed response
            procedure_steps = []
            for step_dict in parsed.get("procedure", []):
                procedure_steps.append(
                    ProcedureStep(
                        step=step_dict["step"],
                        action=step_dict["action"],
                        details=step_dict.get("details", ""),
                    )
                )

            # Citation verification (before sending structured data)
            sources_verified = False
            verification_accuracy = None
            if settings.CITATION_VERIFICATION_ENABLED:
                try:
                    from app.services.citation_verifier import get_citation_verifier

                    verifier = await get_citation_verifier()
                    v_report = await verifier.verify_response(
                        applicable_law=applicable_laws,
                        precedents=precedents_list,
                    )

                    # Annotate overruled
                    if v_report.overruled_cases > 0:
                        precedents_list = verifier.annotate_overruled_precedents(
                            precedents_list, v_report
                        )

                    # Filter unverified
                    if v_report.failed_sections > 0 or v_report.failed_cases > 0:
                        applicable_laws, precedents_list = verifier.filter_unverified_citations(
                            applicable_laws, precedents_list, v_report
                        )

                    sources_verified = v_report.all_verified
                    verification_accuracy = round(v_report.accuracy, 3)
                except Exception as e:
                    logger.debug("stream_verification_skipped", error=str(e))

            # Send structured data events
            law_data = json.dumps(
                {
                    "type": "applicable_law",
                    "content": [l.model_dump() for l in applicable_laws],
                },
                default=str,
            )
            yield f"data: {law_data}\n\n"

            prec_data = json.dumps(
                {
                    "type": "precedents",
                    "content": [p.model_dump() for p in precedents_list],
                },
                default=str,
            )
            yield f"data: {prec_data}\n\n"

            if procedure_steps:
                proc_data = json.dumps(
                    {
                        "type": "procedure",
                        "content": [s.model_dump() for s in procedure_steps],
                    },
                    default=str,
                )
                yield f"data: {proc_data}\n\n"

            # Determine confidence
            llm_used = bool(llm_service.provider)
            if not applicable_laws and not precedents_list:
                confidence = Confidence.LOW
            elif llm_used:
                confidence = Confidence.HIGH
            else:
                confidence = Confidence.MEDIUM

            meta_data = json.dumps(
                {
                    "type": "metadata",
                    "content": {
                        "confidence": confidence.value,
                        "domain": classification.domain.value,
                        "query_type": classification.query_type.value,
                        "jurisdiction_notes": (
                            f"Jurisdiction: {classification.jurisdiction or 'Central Law'}. "
                            "For state-specific variations, specify your state."
                        ),
                        "session_id": session_id,
                        "router_confidence": classification.confidence,
                        "sources_verified": sources_verified,
                        "verification_accuracy": verification_accuracy,
                    },
                }
            )
            yield f"data: {meta_data}\n\n"

            # Save to session
            await _save_to_session(session_id, request.query, answer_text)

            # Done
            duration_ms = round((time.time() - start_time) * 1000, 2)
            done_data = json.dumps(
                {
                    "type": "done",
                    "response_id": str(uuid.uuid4()),
                    "duration_ms": duration_ms,
                }
            )
            yield f"data: {done_data}\n\n"

        except Exception as e:
            logger.error("stream_error", error=str(e))
            error_data = json.dumps(
                {"type": "error", "content": str(e) if settings.APP_DEBUG else "Internal error"}
            )
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =====================================================================
# Pipeline Stage Implementations
# =====================================================================


async def _load_session_history(session_id: str) -> list[dict]:
    """Load conversation history from Redis session."""
    try:
        from app.services.session import get_session_manager

        manager = await get_session_manager()
        return await manager.get_history(session_id)
    except Exception as e:
        logger.debug("session_load_skipped", error=str(e))
        return []


async def _save_to_session(session_id: str, query: str, answer: str) -> None:
    """Save the query and answer to the session history."""
    try:
        from app.services.session import get_session_manager

        manager = await get_session_manager()
        await manager.add_message(session_id, "user", query)
        await manager.add_message(session_id, "assistant", answer)
    except Exception as e:
        logger.debug("session_save_skipped", error=str(e))


async def _classify_query(request: QueryRequest) -> RouterClassification:
    """
    Stage 2: Classify the query into domain, type, jurisdiction.

    Uses the trained DistilBERT model with keyword fallback.
    Falls back to domain_hint if provided.
    """
    try:
        from app.services.query_router import get_query_router

        router_service = await get_query_router()
        result = router_service.classify(request.query)

        # Map string domain to LegalDomain enum
        domain_str = result.get("domain", "general")
        domain = _DOMAIN_MAP.get(domain_str, LegalDomain.GENERAL)

        # If user provided a domain_hint, prefer it over low-confidence router
        if request.domain_hint and result.get("confidence", 0) < 0.8:
            domain = request.domain_hint

        # Use router-detected jurisdiction, unless user explicitly provided one
        jurisdiction = request.jurisdiction or result.get("jurisdiction", "central")
        if jurisdiction == "central":
            jurisdiction = None  # None = no state filter in retrieval

        return RouterClassification(
            domain=domain,
            query_type=result.get("query_type", "rights"),
            jurisdiction=jurisdiction,
            language=request.language,
            confidence=result.get("confidence", 0.5),
        )

    except Exception as e:
        logger.warning("router_fallback", error=str(e))
        # Fallback: use domain_hint or default to GENERAL
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
    Stage 3: Retrieve relevant legal context via hybrid search.

    Uses Qdrant (dense) + Elasticsearch (BM25) + cross-encoder re-ranking.
    """
    from app.services.retrieval import get_retrieval_service

    service = await get_retrieval_service()

    domain = classification.domain.value
    if domain == "general":
        domain = None

    results = await service.search(
        query=request.query,
        domain=domain,
        jurisdiction=classification.jurisdiction,
    )

    return results


async def _enrich_with_graph(results: list) -> list:
    """
    Stage 4: Enrich retrieval results with knowledge graph context.

    Adds interpreting judgments, legal principles, and related cases
    to each retrieval result using the Neo4j knowledge graph.
    Non-fatal if Neo4j is unavailable.
    """
    try:
        from app.services.graph_service import get_graph_service

        graph_service = await get_graph_service()
        if graph_service.available:
            results = await graph_service.enrich_results(results)
    except Exception as e:
        logger.debug("graph_enrichment_skipped", error=str(e))

    return results


async def _format_full_context(results: list) -> str:
    """
    Format retrieval results + graph context for the LLM prompt.

    Combines the standard retrieval context with knowledge graph
    enrichment context (interpreting judgments, legal principles).
    """
    from app.services.retrieval import get_retrieval_service

    retrieval_service = await get_retrieval_service()
    formatted = retrieval_service.format_context_for_llm(results)

    # Append graph context if available
    try:
        from app.services.graph_service import get_graph_service

        graph_service = await get_graph_service()
        if graph_service.available:
            graph_context = graph_service.format_graph_context_for_llm(results)
            if graph_context:
                formatted += graph_context
    except Exception as e:
        logger.debug("graph_context_format_skipped", error=str(e))

    return formatted


def _build_applicable_laws(context: list) -> list[ApplicableLaw]:
    """Extract ApplicableLaw objects from retrieval results."""
    laws = []
    for result in context:
        if result.source_type == "act" and result.section_number:
            laws.append(
                ApplicableLaw(
                    act=result.act_name or "Unknown Act",
                    section=result.section_number,
                    text=result.text[:500] if result.text else "",
                    status=(
                        LegalStatus(result.status)
                        if result.status in ["active", "repealed", "amended"]
                        else LegalStatus.ACTIVE
                    ),
                )
            )
    return laws


def _build_precedents(context: list) -> list[Precedent]:
    """Extract Precedent objects from retrieval results."""
    precedents = []
    seen_cases: set[str] = set()
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
    return precedents


async def _generate_response(
    request: QueryRequest,
    classification: RouterClassification,
    context: list,
    session_id: str,
    history: list[dict] | None = None,
) -> QueryResponse:
    """
    Stage 5: Generate the full legal response using LLM.

    Steps:
        1. Build structured data from retrieval results
        2. Format context for the LLM prompt (retrieval + graph)
        3. Call LLM with conversation history
        4. Parse LLM output for additional structured citations
        5. Merge retrieval-based and LLM-extracted citations
        6. Build QueryResponse
    """
    from app.services.llm_service import get_llm_service
    from app.services.response_parser import parse_llm_response, merge_parsed_with_retrieval

    # Step 1: Build structured data from retrieval
    applicable_laws = _build_applicable_laws(context)
    precedents_list = _build_precedents(context)

    # Step 2: Format context for LLM (includes graph enrichment)
    formatted_context = await _format_full_context(context)

    # Step 3: Generate answer using LLM (with conversation history)
    llm_service = await get_llm_service()
    llm_result = await llm_service.generate_legal_response(
        query=request.query,
        context=formatted_context,
        jurisdiction=classification.jurisdiction,
        history=history,
    )

    answer = llm_result.get("answer", "")
    llm_used = llm_result.get("llm_used", False)

    # Step 4: Parse LLM output for additional citations
    parsed = parse_llm_response(answer)
    merged = merge_parsed_with_retrieval(parsed, applicable_laws, precedents_list)

    # Step 5: Add LLM-extracted citations not already in retrieval
    for law_dict in merged["new_laws"]:
        applicable_laws.append(
            ApplicableLaw(
                act=law_dict["act"],
                section=law_dict["section"],
                text=law_dict.get("text", ""),
            )
        )

    for prec_dict in merged["new_precedents"]:
        precedents_list.append(
            Precedent(
                case=prec_dict["case"],
                year=prec_dict.get("year", 2024),
                court=prec_dict.get("court", "Supreme Court"),
                citation=prec_dict.get("citation", ""),
                relevance=prec_dict.get("relevance", ""),
            )
        )

    # Extract procedure steps from parsed response
    procedure_steps = []
    for step_dict in parsed.get("procedure", []):
        procedure_steps.append(
            ProcedureStep(
                step=step_dict["step"],
                action=step_dict["action"],
                details=step_dict.get("details", ""),
            )
        )

    # Determine confidence
    if not applicable_laws and not precedents_list:
        confidence = Confidence.LOW
    elif llm_used:
        confidence = Confidence.HIGH
    else:
        confidence = Confidence.MEDIUM

    return QueryResponse(
        answer=answer,
        applicable_law=applicable_laws,
        precedents=precedents_list,
        procedure=procedure_steps,
        jurisdiction_notes=(
            f"Jurisdiction: {classification.jurisdiction or 'Central Law (state not specified)'}. "
            f"Domain: {classification.domain.value}. "
            "For state-specific variations, please specify your state."
        ),
        confidence=confidence,
        sources_verified=False,
        language=request.language,
        session_id=session_id,
    )


async def _verify_citations(response: QueryResponse) -> QueryResponse:
    """
    Stage 6: Verify all citations in the response.

    Checks every section and case citation against the database.
    If >30% fail, triggers regeneration with a stricter grounding prompt.
    Overruled judgments are annotated with warnings.
    Unverified citations are filtered out.
    """
    try:
        from app.services.citation_verifier import get_citation_verifier

        verifier = await get_citation_verifier()
        report = await verifier.verify_response(
            applicable_law=response.applicable_law,
            precedents=response.precedents,
        )

        # Annotate overruled precedents with warnings
        if report.overruled_cases > 0:
            response.precedents = verifier.annotate_overruled_precedents(
                response.precedents, report
            )

        # If too many citations failed, attempt regeneration
        if report.regeneration_triggered:
            logger.warning(
                "citation_regeneration_triggered",
                accuracy=round(report.accuracy, 3),
                failed_sections=report.failed_sections,
                failed_cases=report.failed_cases,
            )
            response = await _regenerate_with_verified_citations(
                response, report, verifier
            )
        else:
            # Filter out unverified citations (don't show fabricated ones)
            if report.failed_sections > 0 or report.failed_cases > 0:
                filtered_laws, filtered_precs = verifier.filter_unverified_citations(
                    response.applicable_law,
                    response.precedents,
                    report,
                )
                response.applicable_law = filtered_laws
                response.precedents = filtered_precs

            response.sources_verified = report.all_verified

        logger.info(
            "verification_complete",
            accuracy=round(report.accuracy, 3),
            verified=report.all_verified,
            regenerated=report.regeneration_triggered,
        )

    except Exception as e:
        logger.warning("verification_skipped", error=str(e))
        # Non-fatal — response passes through unverified

    return response


async def _regenerate_with_verified_citations(
    original_response: QueryResponse,
    report,
    verifier,
) -> QueryResponse:
    """
    Regenerate the LLM answer using only verified citations.

    Called when >30% of citations in the original response failed
    verification. Uses a strict grounding prompt that lists only
    the sections and cases confirmed to exist in the database.

    Max 1 retry — if regeneration also fails verification,
    we return the filtered original rather than looping.
    """
    try:
        from app.services.llm_service import get_llm_service

        # Build strict prompt with only verified citations
        strict_prompt = verifier.build_strict_grounding_prompt(
            query="",  # Query context is in the conversation history
            verified_sections=report.section_results,
            verified_cases=report.case_results,
        )

        llm_service = await get_llm_service()
        llm_result = await llm_service.generate_legal_response(
            query=strict_prompt,
            context="",  # Context is embedded in the strict prompt
        )

        regenerated_answer = llm_result.get("answer", "")
        if regenerated_answer and len(regenerated_answer) > 50:
            original_response.answer = regenerated_answer
            logger.info("citation_regeneration_success")

        # Filter to only verified citations regardless
        filtered_laws, filtered_precs = verifier.filter_unverified_citations(
            original_response.applicable_law,
            original_response.precedents,
            report,
        )
        original_response.applicable_law = filtered_laws
        original_response.precedents = filtered_precs
        original_response.sources_verified = (
            report.verified_citations == report.total_citations
        )

    except Exception as e:
        logger.warning("citation_regeneration_failed", error=str(e))
        # Fall back to filtered original
        filtered_laws, filtered_precs = verifier.filter_unverified_citations(
            original_response.applicable_law,
            original_response.precedents,
            report,
        )
        original_response.applicable_law = filtered_laws
        original_response.precedents = filtered_precs

    return original_response


async def _translate_response(
    response: QueryResponse,
    target_language: Language,
) -> QueryResponse:
    """
    Stage 7: Translate the response to the user's preferred language.

    Sprint 5: Returns response unchanged.
    Sprint 11: Will use IndicTrans2 for Hindi, Tamil, Telugu, Bengali, Marathi.
    """
    # TODO (Sprint 11): Implement translation via IndicTrans2
    response.language = target_language
    return response