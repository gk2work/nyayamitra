"""
NyayaMitra — Hybrid Retrieval Service.

Searches across Qdrant (semantic/dense) to find relevant legal
provisions and judgments for a given query.

Pipeline:
    Query → Embed → Qdrant Search (sections + judgments) → Merge & Rank → Return Top-K

Sprint 3: Dense retrieval via Qdrant only.
Sprint 3+: Add Elasticsearch BM25 (sparse) + RRF fusion.
Sprint 5: Add Neo4j knowledge graph traversal.

Usage:
    from app.services.retrieval import RetrievalService

    service = RetrievalService()
    await service.initialize()
    results = await service.search("Can police arrest me without a warrant?")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import settings

logger = structlog.get_logger()

# Collection names (must match indexer.py)
COLLECTION_SECTIONS = "legal_sections"
COLLECTION_JUDGMENTS = "legal_judgments"


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""

    text: str = ""
    score: float = 0.0
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


class RetrievalService:
    """
    Hybrid retrieval service for legal documents.

    Currently implements dense retrieval via Qdrant.
    Will be extended with Elasticsearch BM25 and Neo4j graph traversal.
    """

    def __init__(self):
        self.qdrant: QdrantClient | None = None
        self.embedder = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the retrieval service.

        Connects to Qdrant and loads the embedding model.
        Call this once at application startup.
        """
        if self._initialized:
            return

        logger.info("retrieval_service_initializing")

        # Connect to Qdrant
        self.qdrant = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
            check_compatibility=False,
        )

        # Load embedding model for query encoding
        # Import here to avoid loading torch at module level
        import sys
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))

        from data.embeddings.indexer import LegalEmbedder

        self.embedder = LegalEmbedder()
        self.embedder.load()

        self._initialized = True
        logger.info("retrieval_service_ready")

    async def search(
        self,
        query: str,
        domain: str | None = None,
        jurisdiction: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Search for relevant legal content across all collections.

        Args:
            query: The user's legal question
            domain: Optional domain filter (criminal, property, family, etc.)
            jurisdiction: Optional jurisdiction filter (central, Maharashtra, etc.)
            top_k: Number of results to return (default from config)

        Returns:
            Ranked list of RetrievalResult objects.
        """
        if not self._initialized:
            await self.initialize()

        start = time.time()
        top_k = top_k or settings.RETRIEVAL_FINAL_TOP_K

        # Embed the query
        query_vector = self.embedder.embed_query(query)

        # Build filters
        qdrant_filter = self._build_filter(domain, jurisdiction)

        # Search both collections in parallel
        section_results = self._search_collection(
            COLLECTION_SECTIONS,
            query_vector,
            limit=settings.RETRIEVAL_TOP_K_DENSE,
            query_filter=qdrant_filter,
        )

        judgment_results = self._search_collection(
            COLLECTION_JUDGMENTS,
            query_vector,
            limit=settings.RETRIEVAL_TOP_K_DENSE,
            query_filter=qdrant_filter,
        )

        # Merge and rank all results by score
        all_results = section_results + judgment_results
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Take top K
        final_results = all_results[:top_k]

        duration_ms = round((time.time() - start) * 1000, 2)

        logger.info(
            "retrieval_complete",
            query_length=len(query),
            domain=domain,
            sections_found=len(section_results),
            judgments_found=len(judgment_results),
            final_results=len(final_results),
            top_score=round(final_results[0].score, 4) if final_results else 0,
            duration_ms=duration_ms,
        )

        return final_results

    def _build_filter(
        self,
        domain: str | None,
        jurisdiction: str | None,
    ) -> Filter | None:
        """Build a Qdrant filter from domain and jurisdiction."""
        conditions = []

        if domain and domain != "general":
            conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )

        if jurisdiction:
            conditions.append(
                FieldCondition(key="jurisdiction", match=MatchValue(value=jurisdiction))
            )

        if conditions:
            return Filter(must=conditions)

        return None

    def _search_collection(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 20,
        query_filter: Filter | None = None,
    ) -> list[RetrievalResult]:
        """
        Search a single Qdrant collection using REST API.

        Uses direct HTTP to avoid qdrant-client version mismatch
        with the Qdrant server (client v1.17 vs server v1.8).
        """
        try:
            import httpx

            # Build the search request body
            body: dict = {
                "vector": query_vector,
                "limit": limit,
                "with_payload": True,
            }

            if query_filter:
                # Convert filter to dict format
                filter_dict = {"must": []}
                for condition in query_filter.must:
                    filter_dict["must"].append({
                        "key": condition.key,
                        "match": {"value": condition.match.value},
                    })
                body["filter"] = filter_dict

            url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/collections/{collection_name}/points/search"

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


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton instance for the application
# ═══════════════════════════════════════════════════════════════════════════════

_retrieval_service: RetrievalService | None = None


async def get_retrieval_service() -> RetrievalService:
    """Get or create the singleton retrieval service."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
        await _retrieval_service.initialize()
    return _retrieval_service