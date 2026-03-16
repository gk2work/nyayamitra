"""
NyayaMitra — Elasticsearch BM25 Indexer.

Creates and manages Elasticsearch indices for sparse (keyword/BM25)
retrieval of legal documents. Works alongside Qdrant (dense/semantic)
to enable hybrid search via Reciprocal Rank Fusion.

Index design:
    - legal_sections: sections from acts, optimized for section number
      and keyword matching (e.g., "Section 302 IPC murder")
    - legal_judgments: judgment chunks (headnote, ratio, facts),
      optimized for case name and legal principle matching

Each index mirrors the corresponding Qdrant collection with identical
chunk IDs, enabling score fusion across dense and sparse results.

Mapping features:
    - text field: analyzed with custom legal analyzer for BM25
    - keyword fields: domain, jurisdiction, status for exact filtering
    - section_number: keyword for exact "Section 302" matches
    - case_name: text + keyword for both fuzzy and exact matching
    - year: integer for range filtering

Usage:
    from data.embeddings.es_indexer import ElasticsearchIndexer

    indexer = ElasticsearchIndexer()
    await indexer.connect()
    indexer.create_indices()
    indexer.index_chunks("legal_sections", section_chunks)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings
from app.exceptions import RetrievalError

logger = structlog.get_logger()

# Index names — match the Qdrant collection names for consistency
INDEX_SECTIONS = "legal_sections"
INDEX_JUDGMENTS = "legal_judgments"

# ═══════════════════════════════════════════════════════════════════════════════
# Index Mappings
# ═══════════════════════════════════════════════════════════════════════════════

# Custom analyzer for Indian legal text:
# - Lowercase for case-insensitive matching
# - Standard tokenizer handles Hindi/Devanagari Unicode
# - Edge ngram for partial section number matching ("41" matches "41A")
# - Stop words kept (legal text needs "the", "of", "under")

INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "legal_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "legal_synonyms"],
                },
            },
            "filter": {
                "legal_synonyms": {
                    "type": "synonym",
                    "synonyms": [
                        # Act abbreviations
                        "ipc, indian penal code",
                        "crpc, code of criminal procedure",
                        "cpc, code of civil procedure",
                        "tpa, transfer of property act",
                        "hma, hindu marriage act",
                        "sma, special marriage act",
                        "cpa, consumer protection act",
                        "rti, right to information",
                        "dv act, domestic violence act",
                        "posh, sexual harassment at workplace",
                        "rera, real estate regulation",
                        "bns, bharatiya nyaya sanhita",
                        "bnss, bharatiya nagarik suraksha sanhita",
                        "it act, information technology act",
                        # Court abbreviations
                        "fir, first information report",
                        "sc, supreme court",
                        "hc, high court",
                        # Legal concepts ↔ constitutional provisions
                        "pil, public interest litigation, writ petition, article 32",
                        "fundamental rights, part iii, part 3",
                        # Digital/cyber law terms
                        "social media, internet, online, information technology, cyber",
                        "intermediary, platform, website, service provider",
                        "offensive, objectionable, defamatory, obscene",
                        # Marriage/family law terms
                        "inter religion, interfaith, different religion, special marriage",
                        "maintenance, alimony, support",
                        "divorce, dissolution of marriage",
                        # Criminal law terms
                        "bail, anticipatory bail, regular bail",
                        "arrest, custody, detention",
                        "cognizable, non bailable",
                    ],
                },
            },
        },
    },
}

SECTIONS_MAPPING = {
    "mappings": {
        "properties": {
            # ─── Core text (BM25 searchable) ─────────────────────────
            "text": {
                "type": "text",
                "analyzer": "legal_analyzer",
            },
            # ─── Chunk identification ────────────────────────────────
            "chunk_id": {"type": "keyword"},
            "chunk_type": {"type": "keyword"},
            "source_type": {"type": "keyword"},
            "source_id": {"type": "keyword"},
            # ─── Act/Section metadata ────────────────────────────────
            "act_name": {
                "type": "text",
                "analyzer": "legal_analyzer",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "act_short_name": {"type": "keyword"},
            "section_number": {"type": "keyword"},
            "section_title": {
                "type": "text",
                "analyzer": "legal_analyzer",
            },
            "chapter": {"type": "text"},
            # ─── Filtering fields ────────────────────────────────────
            "domain": {"type": "keyword"},
            "jurisdiction": {"type": "keyword"},
            "status": {"type": "keyword"},
        },
    },
}

JUDGMENTS_MAPPING = {
    "mappings": {
        "properties": {
            # ─── Core text (BM25 searchable) ─────────────────────────
            "text": {
                "type": "text",
                "analyzer": "legal_analyzer",
            },
            # ─── Chunk identification ────────────────────────────────
            "chunk_id": {"type": "keyword"},
            "chunk_type": {"type": "keyword"},
            "source_type": {"type": "keyword"},
            "source_id": {"type": "keyword"},
            # ─── Judgment metadata ───────────────────────────────────
            "case_name": {
                "type": "text",
                "analyzer": "legal_analyzer",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "court": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "year": {"type": "integer"},
            "citation": {"type": "keyword"},
            # ─── Filtering fields ────────────────────────────────────
            "domain": {"type": "keyword"},
            "jurisdiction": {"type": "keyword"},
            "status": {"type": "keyword"},
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Elasticsearch Indexer
# ═══════════════════════════════════════════════════════════════════════════════


class ElasticsearchIndexer:
    """
    Manages Elasticsearch indices and bulk-indexes legal chunks.

    Creates two indices (legal_sections, legal_judgments) with
    legal-optimized mappings and BM25 scoring. Each document uses
    the same chunk_id as the corresponding Qdrant point, enabling
    score fusion across dense and sparse results.
    """

    def __init__(self):
        self.client = None
        self._connected = False

    async def connect(self) -> None:
        """
        Connect to Elasticsearch.

        Uses async elasticsearch client for non-blocking operations.
        """
        from elasticsearch import AsyncElasticsearch

        es_url = settings.elasticsearch_url
        logger.info("es_connecting", url=es_url)

        self.client = AsyncElasticsearch(
            es_url,
            basic_auth=(
                (settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD)
                if settings.ELASTICSEARCH_PASSWORD
                else None
            ),
            request_timeout=30,
        )

        # Verify connection
        try:
            info = await self.client.info()
            version = info.get("version", {}).get("number", "unknown")
            cluster = info.get("cluster_name", "unknown")
            logger.info("es_connected", version=version, cluster=cluster)
            self._connected = True
        except Exception as e:
            logger.error("es_connection_failed", error=str(e))
            raise RetrievalError(source="elasticsearch", reason=f"Connection failed: {e}")

    async def close(self) -> None:
        """Close the Elasticsearch connection."""
        if self.client:
            await self.client.close()
            self._connected = False

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self._connected or not self.client:
            raise RetrievalError(
                source="elasticsearch",
                reason="Not connected. Call connect() first.",
            )

    async def create_indices(self, recreate: bool = False) -> None:
        """
        Create Elasticsearch indices for sections and judgments.

        Args:
            recreate: If True, delete and recreate existing indices.
                      Use when index settings/mappings have changed
                      (e.g. updated synonym list) or to clear duplicates.
        """
        self._ensure_connected()

        for index_name, mapping in [
            (INDEX_SECTIONS, SECTIONS_MAPPING),
            (INDEX_JUDGMENTS, JUDGMENTS_MAPPING),
        ]:
            try:
                exists = await self.client.indices.exists(index=index_name)
                if exists:
                    if recreate:
                        await self.client.indices.delete(index=index_name)
                        logger.info("es_index_deleted_for_recreate", index=index_name)
                    else:
                        logger.info("es_index_exists", index=index_name)
                        continue

                # Merge settings + mapping
                body = {**INDEX_SETTINGS, **mapping}
                await self.client.indices.create(index=index_name, body=body)
                logger.info("es_index_created", index=index_name)

            except Exception as e:
                logger.error("es_index_create_failed", index=index_name, error=str(e))
                raise RetrievalError(
                    source="elasticsearch",
                    reason=f"Failed to create index {index_name}: {e}",
                )

    async def delete_indices(self) -> None:
        """Delete all indices. USE WITH CAUTION — development/testing only."""
        self._ensure_connected()

        for index_name in [INDEX_SECTIONS, INDEX_JUDGMENTS]:
            try:
                exists = await self.client.indices.exists(index=index_name)
                if exists:
                    await self.client.indices.delete(index=index_name)
                    logger.warning("es_index_deleted", index=index_name)
            except Exception as e:
                logger.error("es_index_delete_failed", index=index_name, error=str(e))

    async def index_chunks(
        self,
        index_name: str,
        chunks: list,
    ) -> int:
        """
        Bulk-index chunks into an Elasticsearch index.

        Each chunk is indexed with its chunk_id as the document ID,
        matching the Qdrant point ID for cross-reference during
        hybrid retrieval.

        Args:
            index_name: Target index (INDEX_SECTIONS or INDEX_JUDGMENTS).
            chunks: List of LegalChunk objects from the chunker.

        Returns:
            Number of chunks successfully indexed.
        """
        self._ensure_connected()

        if not chunks:
            return 0

        from elasticsearch.helpers import async_bulk

        indexed = 0
        errors_count = 0

        # Build bulk actions
        actions = []
        for chunk in chunks:
            payload = chunk.to_payload()
            doc = {
                "_index": index_name,
                "_id": chunk.chunk_id,
                "text": chunk.text,
                **payload,
            }
            actions.append(doc)

        # Bulk index in batches
        batch_size = 100
        for i in range(0, len(actions), batch_size):
            batch = actions[i : i + batch_size]
            try:
                success, failed = await async_bulk(
                    self.client,
                    batch,
                    raise_on_error=False,
                    stats_only=True,
                )
                indexed += success
                errors_count += failed

                logger.info(
                    "es_batch_indexed",
                    index=index_name,
                    batch_success=success,
                    batch_failed=failed,
                    total_indexed=indexed,
                )

            except Exception as e:
                logger.error(
                    "es_bulk_error",
                    index=index_name,
                    batch_start=i,
                    error=str(e),
                )
                errors_count += len(batch)

        # Refresh index to make documents searchable immediately
        try:
            await self.client.indices.refresh(index=index_name)
        except Exception as e:
            logger.warning("es_refresh_failed", index=index_name, error=str(e))

        logger.info(
            "es_indexing_complete",
            index=index_name,
            total_indexed=indexed,
            total_errors=errors_count,
        )

        return indexed

    async def get_index_stats(self, index_name: str) -> dict:
        """Get document count and index size for an index."""
        self._ensure_connected()

        try:
            stats = await self.client.indices.stats(index=index_name)
            primaries = stats["indices"][index_name]["primaries"]
            return {
                "name": index_name,
                "doc_count": primaries["docs"]["count"],
                "size_bytes": primaries["store"]["size_in_bytes"],
                "size_mb": round(primaries["store"]["size_in_bytes"] / (1024 * 1024), 2),
            }
        except Exception as e:
            logger.error("es_stats_error", index=index_name, error=str(e))
            return {"name": index_name, "doc_count": 0, "error": str(e)}

    async def search(
        self,
        index_name: str,
        query: str,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Search an Elasticsearch index using BM25.

        Args:
            index_name: Index to search.
            query: Search query text.
            top_k: Number of results to return.
            filters: Optional dict of field -> value for exact match filtering.

        Returns:
            List of dicts with keys: chunk_id, text, score, and all metadata.
        """
        self._ensure_connected()

        # Build query body
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "text^1.0",
                        "act_name^2.0",
                        "section_title^2.0",
                        "case_name^2.0",
                        "section_number^3.0",
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                },
            },
        ]

        filter_clauses = []
        if filters:
            for field, value in filters.items():
                if value is not None and value != "":
                    if field == "year" and isinstance(value, dict):
                        # Range filter: {"gte": 2020, "lte": 2024}
                        filter_clauses.append({"range": {"year": value}})
                    else:
                        filter_clauses.append({"term": {field: value}})

        body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    **({"filter": filter_clauses} if filter_clauses else {}),
                },
            },
            "size": top_k,
            "_source": True,
        }

        try:
            response = await self.client.search(index=index_name, body=body)

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append({
                    "chunk_id": hit["_id"],
                    "score": hit["_score"],
                    "text": source.get("text", ""),
                    "chunk_type": source.get("chunk_type", ""),
                    "source_type": source.get("source_type", ""),
                    "act_name": source.get("act_name"),
                    "act_short_name": source.get("act_short_name"),
                    "section_number": source.get("section_number"),
                    "section_title": source.get("section_title"),
                    "chapter": source.get("chapter"),
                    "case_name": source.get("case_name"),
                    "court": source.get("court"),
                    "year": source.get("year"),
                    "citation": source.get("citation"),
                    "domain": source.get("domain"),
                    "jurisdiction": source.get("jurisdiction"),
                    "status": source.get("status"),
                })

            return results

        except Exception as e:
            logger.error(
                "es_search_error",
                index=index_name,
                query=query[:100],
                error=str(e),
            )
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone Indexing Function
# ═══════════════════════════════════════════════════════════════════════════════


async def run_es_indexing() -> dict:
    """
    Index all legal chunks into Elasticsearch.

    Chunks are generated from the database via the chunker,
    then bulk-indexed into ES indices.

    Returns stats about the indexing run.
    """
    stats = {
        "section_chunks": 0,
        "judgment_chunks": 0,
        "sections_indexed": 0,
        "judgments_indexed": 0,
        "duration_seconds": 0,
    }

    start = time.time()

    # Step 1: Chunk all data
    logger.info("es_indexing_step", step="chunking")
    from data.processors.chunker import chunk_all

    all_chunks = await chunk_all()

    if not all_chunks:
        logger.warning("es_no_chunks_to_index")
        return stats

    # Separate by type
    section_chunks = [c for c in all_chunks if c.source_type == "act"]
    judgment_chunks = [c for c in all_chunks if c.source_type == "judgment"]
    stats["section_chunks"] = len(section_chunks)
    stats["judgment_chunks"] = len(judgment_chunks)

    # Step 2: Connect and create indices
    logger.info("es_indexing_step", step="connect_and_create")
    indexer = ElasticsearchIndexer()
    await indexer.connect()
    await indexer.create_indices()

    # Step 3: Index sections
    if section_chunks:
        logger.info("es_indexing_step", step="index_sections", count=len(section_chunks))
        stats["sections_indexed"] = await indexer.index_chunks(INDEX_SECTIONS, section_chunks)

    # Step 4: Index judgments
    if judgment_chunks:
        logger.info("es_indexing_step", step="index_judgments", count=len(judgment_chunks))
        stats["judgments_indexed"] = await indexer.index_chunks(INDEX_JUDGMENTS, judgment_chunks)

    # Print stats
    for idx_name in [INDEX_SECTIONS, INDEX_JUDGMENTS]:
        try:
            idx_stats = await indexer.get_index_stats(idx_name)
            logger.info("es_index_stats", **idx_stats)
        except Exception:
            pass

    await indexer.close()

    stats["duration_seconds"] = round(time.time() - start, 2)
    logger.info("es_indexing_complete", **stats)

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Run the Elasticsearch indexing pipeline."""
    import asyncio

    print("\n" + "=" * 60)
    print("NyayaMitra — Elasticsearch BM25 Indexing")
    print("=" * 60)

    stats = await run_es_indexing()

    print(f"\n{'='*60}")
    print("ES Indexing Results:")
    print(f"  Section chunks:    {stats['section_chunks']}")
    print(f"  Judgment chunks:   {stats['judgment_chunks']}")
    print(f"  Sections indexed:  {stats['sections_indexed']}")
    print(f"  Judgments indexed:  {stats['judgments_indexed']}")
    print(f"  Duration:          {stats['duration_seconds']}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())