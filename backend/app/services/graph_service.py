"""
NyayaMitra — Knowledge Graph Service.

Async service layer wrapping the Neo4j knowledge graph. Provides
graph-based enrichment for retrieval results and direct graph queries
for the API.

Key capabilities:
    1. Enrich retrieval results — add interpreting judgments and legal
       principles to sections, add interpreted sections to judgments
    2. Graph traversal queries — exposed to the query pipeline
    3. Full-text graph search — supplementary to Qdrant/ES retrieval

The Neo4j Python driver is synchronous, so all graph operations run
in a thread executor to avoid blocking the async event loop.

Usage:
    from app.services.graph_service import get_graph_service

    service = await get_graph_service()
    enriched = await service.enrich_results(retrieval_results)
    judgments = await service.get_interpreting_judgments("IPC", "302")
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import structlog

from app.config import settings

logger = structlog.get_logger()

# Thread pool for offloading synchronous Neo4j calls
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="neo4j")


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Service
# ═══════════════════════════════════════════════════════════════════════════════


class GraphService:
    """
    Async wrapper around the Neo4j knowledge graph.

    Manages the Neo4j driver lifecycle and exposes graph queries
    as async methods by running them in a thread executor.
    """

    def __init__(self):
        self.driver = None
        self.query_executor = None
        self._initialized = False
        self._available = False

    async def initialize(self) -> None:
        """
        Connect to Neo4j and initialize the query executor.

        Non-fatal if Neo4j is unavailable — the service degrades
        gracefully and enrichment is skipped.
        """
        if self._initialized:
            return

        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )

            # Verify connectivity in a thread (synchronous call)
            await asyncio.get_event_loop().run_in_executor(
                _executor, self.driver.verify_connectivity
            )

            from data.knowledge_graph.graph_queries import GraphQueryExecutor

            self.query_executor = GraphQueryExecutor(self.driver)
            self._available = True
            self._initialized = True

            server_info = await asyncio.get_event_loop().run_in_executor(
                _executor, self.driver.get_server_info
            )
            logger.info(
                "graph_service_ready",
                uri=settings.neo4j_uri,
                server=str(server_info.agent),
            )

        except Exception as e:
            logger.warning(
                "graph_service_unavailable",
                error=str(e),
                message="Neo4j unavailable. Graph enrichment will be skipped.",
            )
            self._available = False
            self._initialized = True

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()
            self._available = False
            logger.info("graph_service_closed")

    @property
    def available(self) -> bool:
        """Whether the graph service is connected and usable."""
        return self._available

    # ─── Async Wrappers for Graph Queries ────────────────────────────────

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run a synchronous graph query in the thread executor."""
        if not self._available:
            return None
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                _executor, partial(func, *args, **kwargs)
            )
        except Exception as e:
            logger.warning("graph_query_error", func=func.__name__, error=str(e))
            return None

    async def get_interpreting_judgments(
        self, act_short_name: str, section_number: str, limit: int = 10
    ) -> list[dict]:
        """Find judgments that interpreted a section."""
        result = await self._run_in_thread(
            self.query_executor.get_interpreting_judgments,
            act_short_name, section_number, limit,
        )
        return result or []

    async def get_interpreted_sections(
        self, case_name: str, limit: int = 20
    ) -> list[dict]:
        """Find sections that a judgment interpreted."""
        result = await self._run_in_thread(
            self.query_executor.get_interpreted_sections,
            case_name, limit,
        )
        return result or []

    async def get_section_with_context(
        self, act_short_name: str, section_number: str
    ) -> dict | None:
        """Get a section with full graph context."""
        return await self._run_in_thread(
            self.query_executor.get_section_with_context,
            act_short_name, section_number,
        )

    async def get_related_judgments(
        self, case_name: str, limit: int = 10
    ) -> list[dict]:
        """Find judgments related to a given judgment."""
        result = await self._run_in_thread(
            self.query_executor.get_related_judgments,
            case_name, limit,
        )
        return result or []

    async def get_legal_principles(
        self, domain: str, limit: int = 20
    ) -> list[dict]:
        """Find legal principles in a domain."""
        result = await self._run_in_thread(
            self.query_executor.get_legal_principles,
            domain, limit,
        )
        return result or []

    async def get_section_principles(
        self, act_short_name: str, section_number: str, limit: int = 10
    ) -> list[dict]:
        """Find legal principles derived from a section."""
        result = await self._run_in_thread(
            self.query_executor.get_section_principles,
            act_short_name, section_number, limit,
        )
        return result or []

    async def get_act_structure(self, act_name: str) -> dict | None:
        """Get full act structure with interpretation counts."""
        return await self._run_in_thread(
            self.query_executor.get_act_structure,
            act_name,
        )

    async def find_connection(
        self,
        from_label: str, from_name: str,
        to_label: str, to_name: str,
        max_depth: int = 4,
    ) -> list[dict]:
        """Find shortest path between two nodes."""
        result = await self._run_in_thread(
            self.query_executor.find_connection,
            from_label, from_name, to_label, to_name, max_depth,
        )
        return result or []

    async def search_sections_fulltext(
        self, query: str, limit: int = 10
    ) -> list[dict]:
        """Full-text search across sections."""
        result = await self._run_in_thread(
            self.query_executor.search_sections_fulltext,
            query, limit,
        )
        return result or []

    async def search_judgments_fulltext(
        self, query: str, limit: int = 10
    ) -> list[dict]:
        """Full-text search across judgments."""
        result = await self._run_in_thread(
            self.query_executor.search_judgments_fulltext,
            query, limit,
        )
        return result or []

    async def get_domain_stats(self) -> list[dict]:
        """Get per-domain statistics."""
        result = await self._run_in_thread(
            self.query_executor.get_domain_stats,
        )
        return result or []

    async def get_most_interpreted_sections(self, limit: int = 20) -> list[dict]:
        """Find most-interpreted sections."""
        result = await self._run_in_thread(
            self.query_executor.get_most_interpreted_sections,
            limit,
        )
        return result or []

    # ─── Retrieval Enrichment ────────────────────────────────────────────

    async def enrich_results(self, results: list) -> list:
        """
        Enrich retrieval results with knowledge graph context.

        For each section result: adds interpreting judgments and
        legal principles as additional metadata.

        For each judgment result: adds the list of sections it
        interpreted and related judgments.

        This runs graph queries concurrently for all results using
        asyncio.gather for maximum throughput.

        Args:
            results: List of RetrievalResult objects from the
                     retrieval service.

        Returns:
            The same list with graph_context added to each result.
        """
        if not self._available or not results:
            return results

        async def enrich_single(result):
            try:
                if result.source_type == "act" and result.act_short_name and result.section_number:
                    # Enrich section with interpreting judgments + principles
                    judgments = await self.get_interpreting_judgments(
                        result.act_short_name, result.section_number, limit=3
                    )
                    principles = await self.get_section_principles(
                        result.act_short_name, result.section_number, limit=2
                    )
                    result.graph_context = {
                        "interpreting_judgments": [
                            {
                                "case_name": j.get("case_name", ""),
                                "year": j.get("year"),
                                "court": j.get("court", ""),
                                "citation": j.get("citation_scc", ""),
                                "is_overruled": j.get("is_overruled", False),
                            }
                            for j in judgments
                        ],
                        "legal_principles": [
                            {
                                "text": p.get("principle_text", "")[:200],
                                "source_case": p.get("source_case", ""),
                                "year": p.get("year"),
                            }
                            for p in principles
                        ],
                    }

                elif result.source_type == "judgment" and result.case_name:
                    # Enrich judgment with interpreted sections + related cases
                    sections = await self.get_interpreted_sections(
                        result.case_name, limit=5
                    )
                    related = await self.get_related_judgments(
                        result.case_name, limit=3
                    )
                    result.graph_context = {
                        "interpreted_sections": [
                            {
                                "section_number": s.get("section_number", ""),
                                "act_short_name": s.get("act_short_name", ""),
                                "act_name": s.get("act_name", ""),
                                "title": s.get("title", ""),
                            }
                            for s in sections
                        ],
                        "related_judgments": [
                            {
                                "case_name": r.get("case_name", ""),
                                "year": r.get("year"),
                                "rel_type": r.get("rel_type", ""),
                                "is_overruled": r.get("is_overruled", False),
                            }
                            for r in related
                        ],
                    }
                else:
                    result.graph_context = {}

            except Exception as e:
                logger.debug(
                    "enrichment_error",
                    source_type=result.source_type,
                    error=str(e),
                )
                result.graph_context = {}

        # Enrich all results concurrently
        await asyncio.gather(*[enrich_single(r) for r in results])

        enriched_count = sum(1 for r in results if getattr(r, "graph_context", None))
        logger.info(
            "results_enriched",
            total=len(results),
            enriched=enriched_count,
        )

        return results

    def format_graph_context_for_llm(self, results: list) -> str:
        """
        Format graph enrichment context for injection into the LLM prompt.

        Appends graph-derived information after the retrieval context
        to give the LLM additional relational knowledge.
        """
        context_parts = []

        for result in results:
            graph_ctx = getattr(result, "graph_context", None)
            if not graph_ctx:
                continue

            # Section enrichment
            interp_judgments = graph_ctx.get("interpreting_judgments", [])
            if interp_judgments and result.source_type == "act":
                header = (
                    f"\n[Graph Context for Section {result.section_number} "
                    f"of {result.act_short_name or result.act_name}]"
                )
                context_parts.append(header)
                context_parts.append("Interpreted by:")
                for j in interp_judgments:
                    overruled_tag = " [OVERRULED]" if j.get("is_overruled") else ""
                    context_parts.append(
                        f"  - {j['case_name']} ({j.get('year', '?')}) "
                        f"— {j.get('court', '')}{overruled_tag}"
                    )

                principles = graph_ctx.get("legal_principles", [])
                if principles:
                    context_parts.append("Established principles:")
                    for p in principles:
                        context_parts.append(
                            f"  - \"{p['text']}\" — {p.get('source_case', '')} ({p.get('year', '?')})"
                        )

            # Judgment enrichment
            interp_sections = graph_ctx.get("interpreted_sections", [])
            if interp_sections and result.source_type == "judgment":
                header = f"\n[Graph Context for {result.case_name}]"
                context_parts.append(header)
                context_parts.append("Sections interpreted:")
                for s in interp_sections:
                    context_parts.append(
                        f"  - Section {s['section_number']} of "
                        f"{s.get('act_short_name') or s.get('act_name', '?')}"
                        f"{' — ' + s['title'] if s.get('title') else ''}"
                    )

                related = graph_ctx.get("related_judgments", [])
                if related:
                    context_parts.append("Related judgments:")
                    for r in related:
                        overruled_tag = " [OVERRULED]" if r.get("is_overruled") else ""
                        context_parts.append(
                            f"  - {r['case_name']} ({r.get('year', '?')}) "
                            f"[{r.get('rel_type', 'related')}]{overruled_tag}"
                        )

        if context_parts:
            return "\n\n=== KNOWLEDGE GRAPH CONTEXT ===\n" + "\n".join(context_parts)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_graph_service: GraphService | None = None


async def get_graph_service() -> GraphService:
    """Get or create the singleton graph service."""
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
        await _graph_service.initialize()
    return _graph_service


async def close_graph_service() -> None:
    """Close the singleton graph service. Call on app shutdown."""
    global _graph_service
    if _graph_service is not None:
        await _graph_service.close()
        _graph_service = None