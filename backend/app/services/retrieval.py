"""
NyayaMitra — Hybrid Retrieval Service.

Searches across Qdrant (semantic/dense) and Elasticsearch (BM25/sparse)
to find relevant legal provisions and judgments for a given query.
Results from both backends are fused using Reciprocal Rank Fusion (RRF)
and optionally re-ranked with a cross-encoder model.

Pipeline:
    Query → Embed → Qdrant Dense Search ─┐
    Query → ─── → ES BM25 Search ────────┼→ RRF Fusion → Re-rank → Top-K
                                          │
    Filters: domain, jurisdiction,        │
             act_name, court, year_range ─┘

Sprint 3: Dense (Qdrant) + Sparse (ES) + RRF fusion + re-ranking.
Sprint 5: Will add Neo4j knowledge graph traversal.

Usage:
    from app.services.retrieval import get_retrieval_service

    service = await get_retrieval_service()
    results = await service.search("Can police arrest me without a warrant?")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

from app.config import settings
from app.exceptions import RetrievalError

logger = structlog.get_logger()

# Collection/index names (shared across Qdrant and Elasticsearch)
COLLECTION_SECTIONS = "legal_sections"
COLLECTION_JUDGMENTS = "legal_judgments"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""

    text: str = ""
    score: float = 0.0
    chunk_id: str = ""
    chunk_type: str = ""
    source_type: str = ""

    # Section metadata
    act_name: str | None = None
    act_short_name: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    chapter: str | None = None

    # Judgment metadata
    case_name: str | None = None
    court: str | None = None
    year: int | None = None
    citation: str | None = None

    # Common
    domain: str | None = None
    jurisdiction: str | None = None
    status: str | None = None

    # Retrieval source tracking
    retrieval_source: str = ""  # "dense", "sparse", "both"


@dataclass
class SearchFilters:
    """Filters applied to both Qdrant and Elasticsearch searches."""

    domain: str | None = None
    jurisdiction: str | None = None
    act_name: str | None = None
    court: str | None = None
    year_from: int | None = None
    year_to: int | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reciprocal Rank Fusion
# ═══════════════════════════════════════════════════════════════════════════════


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[dict],
    k: int = 60,
) -> list[RetrievalResult]:
    """
    Fuse dense (Qdrant) and sparse (ES) results using RRF.

    RRF score = sum over all rankings of: 1 / (k + rank)

    This is rank-based, not score-based, so it handles the different
    score scales of cosine similarity (0-1) and BM25 (unbounded).

    Args:
        dense_results: RetrievalResult objects from Qdrant.
        sparse_results: Dicts from Elasticsearch with chunk_id + metadata.
        k: RRF constant (default 60, standard in literature).

    Returns:
        Fused list of RetrievalResult sorted by RRF score.
    """
    # Build a map of chunk_id → (rrf_score, RetrievalResult)
    fused: dict[str, dict] = {}

    # Score dense results by rank
    for rank, result in enumerate(dense_results, start=1):
        cid = result.chunk_id
        if cid not in fused:
            fused[cid] = {"result": result, "rrf_score": 0.0, "sources": set()}
        fused[cid]["rrf_score"] += 1.0 / (k + rank)
        fused[cid]["sources"].add("dense")

    # Score sparse results by rank
    for rank, hit in enumerate(sparse_results, start=1):
        cid = hit.get("chunk_id", "")
        if not cid:
            continue

        if cid not in fused:
            # Create a new RetrievalResult from ES hit
            fused[cid] = {
                "result": _es_hit_to_result(hit),
                "rrf_score": 0.0,
                "sources": set(),
            }
        fused[cid]["rrf_score"] += 1.0 / (k + rank)
        fused[cid]["sources"].add("sparse")

    # Build final list sorted by RRF score
    results = []
    for cid, entry in fused.items():
        result = entry["result"]
        result.score = entry["rrf_score"]
        sources = entry["sources"]
        if "dense" in sources and "sparse" in sources:
            result.retrieval_source = "both"
        elif "dense" in sources:
            result.retrieval_source = "dense"
        else:
            result.retrieval_source = "sparse"
        results.append(result)

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _es_hit_to_result(hit: dict) -> RetrievalResult:
    """Convert an Elasticsearch hit dict to a RetrievalResult."""
    return RetrievalResult(
        text=hit.get("text", ""),
        score=0.0,
        chunk_id=hit.get("chunk_id", ""),
        chunk_type=hit.get("chunk_type", ""),
        source_type=hit.get("source_type", ""),
        act_name=hit.get("act_name"),
        act_short_name=hit.get("act_short_name"),
        section_number=hit.get("section_number"),
        section_title=hit.get("section_title"),
        chapter=hit.get("chapter"),
        case_name=hit.get("case_name"),
        court=hit.get("court"),
        year=hit.get("year"),
        citation=hit.get("citation"),
        domain=hit.get("domain"),
        jurisdiction=hit.get("jurisdiction"),
        status=hit.get("status"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Encoder Re-ranker (imported from standalone module)
# ═══════════════════════════════════════════════════════════════════════════════

# LegalReranker lives in data/embeddings/reranker.py for clean testability.
# It exposes .load(), .rerank(query, results, top_k), and .score_pairs().
# Import is deferred to initialize() to avoid loading torch at module level.


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Retrieval Service
# ═══════════════════════════════════════════════════════════════════════════════


class RetrievalService:
    """
    Hybrid retrieval service for legal documents.

    Combines:
    - Qdrant dense/semantic search (BGE-large embeddings)
    - Elasticsearch BM25 sparse/keyword search
    - Reciprocal Rank Fusion to merge results
    - Cross-encoder re-ranking for final precision

    Gracefully degrades: if ES is unavailable, falls back to
    Qdrant-only. If re-ranker fails to load, skips re-ranking.
    """

    def __init__(self):
        self.embedder = None
        self.reranker = None  # LegalReranker from data.embeddings.reranker
        self.es_client = None
        self._initialized = False
        self._es_available = False
        self._reranker_available = False

    async def initialize(self) -> None:
        """
        Initialize all retrieval backends.

        Loads embedding model, connects to Elasticsearch, and
        optionally loads the cross-encoder re-ranker.
        """
        if self._initialized:
            return

        logger.info("retrieval_service_initializing")

        # Load embedding model for query encoding
        import sys
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))

        from data.embeddings.indexer import LegalEmbedder

        self.embedder = LegalEmbedder()
        self.embedder.load()

        # Connect to Elasticsearch (non-fatal if unavailable)
        try:
            from data.embeddings.es_indexer import ElasticsearchIndexer

            self.es_client = ElasticsearchIndexer()
            await self.es_client.connect()
            self._es_available = True
            logger.info("retrieval_es_connected")
        except Exception as e:
            logger.warning(
                "retrieval_es_unavailable",
                error=str(e),
                message="Elasticsearch unavailable. Falling back to Qdrant-only retrieval.",
            )
            self._es_available = False

        # Load cross-encoder re-ranker (non-fatal if unavailable)
        try:
            from data.embeddings.reranker import LegalReranker

            self.reranker = LegalReranker()
            self.reranker.load()
            self._reranker_available = True
            logger.info("retrieval_reranker_loaded")
        except Exception as e:
            logger.warning(
                "retrieval_reranker_unavailable",
                error=str(e),
                message="Cross-encoder re-ranker unavailable. Skipping re-ranking.",
            )
            self._reranker_available = False

        self._initialized = True
        logger.info(
            "retrieval_service_ready",
            es_available=self._es_available,
            reranker_available=self._reranker_available,
        )

    async def search(
        self,
        query: str,
        domain: str | None = None,
        jurisdiction: str | None = None,
        act_name: str | None = None,
        court: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        top_k: int | None = None,
        use_reranker: bool = True,
    ) -> list[RetrievalResult]:
        """
        Hybrid search across dense and sparse backends.

        Pipeline:
            1. Qdrant dense search (both collections)
            2. Elasticsearch BM25 search (both indices)
            3. RRF fusion of dense + sparse results
            4. Cross-encoder re-ranking (if available)
            5. Return top-K

        Args:
            query: The user's legal question.
            domain: Filter by legal domain (criminal, property, etc.).
            jurisdiction: Filter by jurisdiction (central, Maharashtra, etc.).
            act_name: Filter by act name (exact or partial).
            court: Filter by court name.
            year_from: Filter judgments from this year onwards.
            year_to: Filter judgments up to this year.
            top_k: Number of final results to return.
            use_reranker: Whether to apply cross-encoder re-ranking.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        if not self._initialized:
            await self.initialize()

        start = time.time()
        top_k = top_k or settings.RETRIEVAL_FINAL_TOP_K

        filters = SearchFilters(
            domain=domain,
            jurisdiction=jurisdiction,
            act_name=act_name,
            court=court,
            year_from=year_from,
            year_to=year_to,
        )

        # ── Step 1: Qdrant dense search ──────────────────────────────────
        query_vector = self.embedder.embed_query(query)
        qdrant_filter = self._build_qdrant_filter(filters)

        dense_sections = self._qdrant_search(
            COLLECTION_SECTIONS, query_vector,
            limit=settings.RETRIEVAL_TOP_K_DENSE,
            query_filter=qdrant_filter,
        )
        dense_judgments = self._qdrant_search(
            COLLECTION_JUDGMENTS, query_vector,
            limit=settings.RETRIEVAL_TOP_K_DENSE,
            query_filter=qdrant_filter,
        )
        dense_results = dense_sections + dense_judgments

        # ── Step 2: Elasticsearch BM25 search ────────────────────────────
        sparse_results: list[dict] = []
        if self._es_available:
            es_filters = self._build_es_filters(filters)
            try:
                sparse_sections, sparse_judgments = await asyncio.gather(
                    self.es_client.search(
                        COLLECTION_SECTIONS, query,
                        top_k=settings.RETRIEVAL_TOP_K_SPARSE,
                        filters=es_filters,
                    ),
                    self.es_client.search(
                        COLLECTION_JUDGMENTS, query,
                        top_k=settings.RETRIEVAL_TOP_K_SPARSE,
                        filters=es_filters,
                    ),
                )
                sparse_results = sparse_sections + sparse_judgments
            except Exception as e:
                logger.warning("es_search_failed_fallback", error=str(e))

        # ── Step 3: Build candidate set ─────────────────────────────────
        # When the cross-encoder is available, it is the quality arbiter.
        # Dense (Qdrant) provides the best candidate pool for semantic
        # re-ranking. ES (BM25) supplements by boosting candidates that
        # match both semantically AND lexically — these get a source tag
        # of "both" for transparency.
        #
        # When the reranker is NOT available, RRF fusion of dense+sparse
        # provides the ranking signal instead.

        if sparse_results:
            # Build a set of chunk_ids that ES also found (for tagging)
            es_chunk_ids = {h.get("chunk_id", "") for h in sparse_results if h.get("chunk_id")}

            # Tag dense results that also appeared in ES
            for r in dense_results:
                if r.chunk_id in es_chunk_ids:
                    r.retrieval_source = "both"
                else:
                    r.retrieval_source = "dense"

            if use_reranker and self._reranker_available:
                # Reranker path: use dense results as candidates, but
                # also inject ES-only results that dense missed (these
                # are keyword matches the embedding model couldn't find).
                # Cap ES-only injection to keep total candidates ~50
                # for cross-encoder latency budget (~400ms on Apple Silicon).
                dense_ids = {r.chunk_id for r in dense_results if r.chunk_id}
                es_only = [
                    _es_hit_to_result(h) for h in sparse_results
                    if h.get("chunk_id") and h["chunk_id"] not in dense_ids
                ][:5]  # top 5 ES-only results (already ranked by BM25)
                for r in es_only:
                    r.retrieval_source = "sparse"

                candidates = dense_results + es_only
            else:
                # No reranker: use RRF fusion as the ranking signal
                candidates = reciprocal_rank_fusion(
                    dense_results, sparse_results,
                    k=settings.RETRIEVAL_RRF_K,
                )
        else:
            # Qdrant-only fallback
            candidates = dense_results
            candidates.sort(key=lambda r: r.score, reverse=True)
            for r in candidates:
                r.retrieval_source = "dense"

        # ── Step 4: Cross-encoder re-ranking ─────────────────────────────
        if use_reranker and self._reranker_available and candidates:
            try:
                final_results = self.reranker.rerank(query, candidates, top_k=top_k)
            except Exception as e:
                logger.warning("reranker_failed_fallback", error=str(e))
                final_results = candidates[:top_k]
        else:
            final_results = candidates[:top_k]

        duration_ms = round((time.time() - start) * 1000, 2)

        logger.info(
            "retrieval_complete",
            query_length=len(query),
            domain=domain,
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            candidates_count=len(candidates),
            final_count=len(final_results),
            top_score=round(final_results[0].score, 4) if final_results else 0,
            es_used=self._es_available and bool(sparse_results),
            reranker_used=use_reranker and self._reranker_available,
            duration_ms=duration_ms,
        )

        return final_results

    # ─── Qdrant Search ───────────────────────────────────────────────────

    def _qdrant_search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 20,
        query_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        """
        Search a Qdrant collection using REST API.

        Uses direct HTTP to avoid qdrant-client version mismatch
        with the Qdrant server (client v1.17 vs server v1.8).
        """
        try:
            import httpx

            body: dict = {
                "vector": query_vector,
                "limit": limit,
                "with_payload": True,
            }

            if query_filter:
                body["filter"] = query_filter

            url = (
                f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
                f"/collections/{collection_name}/points/search"
            )

            with httpx.Client(timeout=10) as client:
                resp = client.post(url, json=body)
                resp.raise_for_status()
                data = resp.json()

            results = []
            for hit in data.get("result", []):
                payload = hit.get("payload", {})
                result = RetrievalResult(
                    text=payload.get("text", ""),
                    score=hit.get("score", 0.0),
                    chunk_id=payload.get("chunk_id", hit.get("id", "")),
                    chunk_type=payload.get("chunk_type", ""),
                    source_type=payload.get("source_type", ""),
                    act_name=payload.get("act_name"),
                    act_short_name=payload.get("act_short_name"),
                    section_number=payload.get("section_number"),
                    section_title=payload.get("section_title"),
                    chapter=payload.get("chapter"),
                    case_name=payload.get("case_name"),
                    court=payload.get("court"),
                    year=payload.get("year"),
                    citation=payload.get("citation"),
                    domain=payload.get("domain"),
                    jurisdiction=payload.get("jurisdiction"),
                    status=payload.get("status"),
                    retrieval_source="dense",
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(
                "qdrant_search_error",
                collection=collection_name,
                error=str(e),
            )
            return []

    # ─── Filter Builders ─────────────────────────────────────────────────

    def _build_qdrant_filter(self, filters: SearchFilters) -> dict | None:
        """Build a Qdrant filter dict from SearchFilters."""
        conditions = []

        if filters.domain and filters.domain != "general":
            conditions.append({
                "key": "domain",
                "match": {"value": filters.domain},
            })

        if filters.jurisdiction:
            conditions.append({
                "key": "jurisdiction",
                "match": {"value": filters.jurisdiction},
            })

        if filters.act_name:
            conditions.append({
                "key": "act_name",
                "match": {"value": filters.act_name},
            })

        if filters.court:
            conditions.append({
                "key": "court",
                "match": {"value": filters.court},
            })

        if conditions:
            return {"must": conditions}

        return None

    def _build_es_filters(self, filters: SearchFilters) -> dict | None:
        """Build an Elasticsearch filter dict from SearchFilters."""
        es_filters = {}

        if filters.domain and filters.domain != "general":
            es_filters["domain"] = filters.domain

        if filters.jurisdiction:
            es_filters["jurisdiction"] = filters.jurisdiction

        if filters.act_name:
            es_filters["act_name.keyword"] = filters.act_name

        if filters.court:
            es_filters["court.keyword"] = filters.court

        if filters.year_from or filters.year_to:
            year_range = {}
            if filters.year_from:
                year_range["gte"] = filters.year_from
            if filters.year_to:
                year_range["lte"] = filters.year_to
            es_filters["year"] = year_range

        return es_filters if es_filters else None

    # ─── LLM Context Formatting ──────────────────────────────────────────

    def format_context_for_llm(self, results: list[RetrievalResult]) -> str:
        """
        Format retrieval results into a context string for the LLM prompt.

        The LLM receives this as grounding context to generate its response.
        Format is designed to be clear and parseable by the model.
        """
        if not results:
            return "No relevant legal provisions or judgments found."

        context_parts = []

        # Group by type
        sections = [r for r in results if r.source_type == "act"]
        judgments = [r for r in results if r.source_type == "judgment"]

        if sections:
            context_parts.append("=== RELEVANT LEGAL PROVISIONS ===")
            for i, sec in enumerate(sections, 1):
                entry = f"\n[Section {i}]"
                entry += f"\nAct: {sec.act_name}"
                entry += f"\nSection: {sec.section_number}"
                if sec.section_title:
                    entry += f" - {sec.section_title}"
                if sec.chapter:
                    entry += f"\nChapter: {sec.chapter}"
                entry += f"\nStatus: {sec.status or 'active'}"
                entry += f"\nText: {sec.text}"
                entry += f"\nRelevance Score: {sec.score:.4f}"
                context_parts.append(entry)

        if judgments:
            context_parts.append("\n\n=== RELEVANT JUDICIAL PRECEDENTS ===")
            for i, j in enumerate(judgments, 1):
                entry = f"\n[Precedent {i}]"
                entry += f"\nCase: {j.case_name}"
                if j.court:
                    entry += f"\nCourt: {j.court}"
                if j.year:
                    entry += f"\nYear: {j.year}"
                if j.citation:
                    entry += f"\nCitation: {j.citation}"
                entry += f"\nType: {j.chunk_type}"
                entry += f"\nContent: {j.text}"
                entry += f"\nRelevance Score: {j.score:.4f}"
                context_parts.append(entry)

        return "\n".join(context_parts)

    # ─── Cleanup ─────────────────────────────────────────────────────────

    async def close(self) -> None:
        """
        Close all backend connections gracefully.

        Call this on application shutdown to avoid 'Unclosed connector'
        warnings from aiohttp (used by AsyncElasticsearch).
        """
        if self.es_client:
            try:
                await self.es_client.close()
                logger.info("retrieval_es_closed")
            except Exception as e:
                logger.warning("retrieval_es_close_error", error=str(e))

        self._initialized = False
        logger.info("retrieval_service_closed")


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_retrieval_service: RetrievalService | None = None


async def get_retrieval_service() -> RetrievalService:
    """Get or create the singleton retrieval service."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
        await _retrieval_service.initialize()
    return _retrieval_service


async def close_retrieval_service() -> None:
    """Close the singleton retrieval service. Call on app shutdown."""
    global _retrieval_service
    if _retrieval_service is not None:
        await _retrieval_service.close()
        _retrieval_service = None