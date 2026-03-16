"""
NyayaMitra — Embedding & Indexing Pipeline.

Takes chunked legal documents, embeds them using BGE-large,
and indexes them into both Qdrant (dense/semantic) and
Elasticsearch (sparse/BM25) for hybrid retrieval.

Pipeline:
    1. Load chunks from the chunker
    2. Embed each chunk using sentence-transformers (BGE-large-en-v1.5)
    3. Upsert vectors + metadata into Qdrant collections
    4. Bulk-index text + metadata into Elasticsearch indices
    5. Mark indexed items in PostgreSQL (is_indexed=True, indexed_at=now)

Usage:
    python -m data.embeddings.indexer
    python -m data.embeddings.indexer --qdrant-only
    python -m data.embeddings.indexer --es-only
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings

logger = structlog.get_logger()

# Collection/index names — shared across Qdrant and Elasticsearch
COLLECTION_SECTIONS = "legal_sections"
COLLECTION_JUDGMENTS = "legal_judgments"


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding Model
# ═══════════════════════════════════════════════════════════════════════════════


class LegalEmbedder:
    """
    Wraps the sentence-transformers embedding model.

    Uses BGE-large-en-v1.5 which produces 1024-dim embeddings.
    BGE requires a query prefix for asymmetric search.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.model = None
        self.dimension = settings.EMBEDDING_DIMENSION

    def load(self) -> None:
        """Load the embedding model into memory."""
        from sentence_transformers import SentenceTransformer

        logger.info("loading_embedding_model", model=self.model_name)
        start = time.time()

        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        duration = round(time.time() - start, 2)
        logger.info(
            "embedding_model_loaded",
            model=self.model_name,
            dimension=self.dimension,
            duration_seconds=duration,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document texts.

        For BGE models, documents don't need a prefix.
        Returns a list of embedding vectors.
        """
        if not self.model:
            self.load()

        embeddings = self.model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query.

        BGE models require the prefix "Represent this sentence for searching
        relevant passages:" for queries (asymmetric search).
        """
        if not self.model:
            self.load()

        # BGE query prefix for better retrieval
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
        )

        return embedding.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Qdrant Indexer
# ═══════════════════════════════════════════════════════════════════════════════


class QdrantIndexer:
    """
    Manages Qdrant collections and upserts embedded chunks.
    """

    def __init__(self):
        self.client = None

    async def connect(self) -> None:
        """Connect to Qdrant."""
        from qdrant_client import QdrantClient

        logger.info(
            "connecting_to_qdrant",
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
        )

        # Verify connection
        collections = self.client.get_collections()
        logger.info(
            "qdrant_connected",
            existing_collections=[c.name for c in collections.collections],
        )

    def create_collections(self, dimension: int) -> None:
        """
        Create Qdrant collections for sections and judgments.

        Uses cosine similarity (normalized vectors from BGE).
        """
        from qdrant_client.models import Distance, VectorParams

        for collection_name in [COLLECTION_SECTIONS, COLLECTION_JUDGMENTS]:
            # Check if collection already exists
            existing = [c.name for c in self.client.get_collections().collections]

            if collection_name in existing:
                logger.info("collection_exists", collection=collection_name)
                continue

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )

            logger.info(
                "collection_created",
                collection=collection_name,
                dimension=dimension,
            )

    def index_chunks(
        self,
        collection_name: str,
        chunks: list,
        embeddings: list[list[float]],
    ) -> int:
        """
        Upsert chunks with their embeddings into a Qdrant collection.

        Args:
            collection_name: Target collection
            chunks: List of LegalChunk objects
            embeddings: Corresponding embedding vectors

        Returns:
            Number of chunks indexed.
        """
        from qdrant_client.models import PointStruct

        if not chunks:
            return 0

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    **chunk.to_payload(),
                },
            )
            points.append(point)

        # Upsert in batches of 100
        batch_size = 100
        indexed = 0

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )
            indexed += len(batch)
            logger.info(
                "batch_indexed",
                collection=collection_name,
                batch_size=len(batch),
                total_indexed=indexed,
            )

        return indexed

    def get_collection_info(self, collection_name: str) -> dict:
        """Get collection stats."""
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.name,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PostgreSQL is_indexed Marker
# ═══════════════════════════════════════════════════════════════════════════════


async def mark_indexed_in_db(chunks: list) -> int:
    """
    Mark sections and judgments as indexed in PostgreSQL.

    Sets is_indexed=True and indexed_at=now() for each record
    whose source_id appears in the indexed chunks.

    Args:
        chunks: List of LegalChunk objects that were successfully indexed.

    Returns:
        Number of records updated.
    """
    from sqlalchemy import update

    from app.database import async_session
    from app.models.legal import Section, Judgment

    now = datetime.utcnow()
    updated = 0

    # Collect source IDs by type
    section_ids = [c.source_id for c in chunks if c.source_type == "act" and c.source_id]
    judgment_ids = [c.source_id for c in chunks if c.source_type == "judgment" and c.source_id]

    # Deduplicate (multiple chunks can reference the same judgment)
    section_ids = list(set(section_ids))
    judgment_ids = list(set(judgment_ids))

    async with async_session() as session:
        # Mark sections
        if section_ids:
            stmt = (
                update(Section)
                .where(Section.id.in_(section_ids))
                .values(is_indexed=True, indexed_at=now)
            )
            result = await session.execute(stmt)
            updated += result.rowcount
            logger.info("sections_marked_indexed", count=result.rowcount)

        # Mark judgments
        if judgment_ids:
            stmt = (
                update(Judgment)
                .where(Judgment.id.in_(judgment_ids))
                .values(is_indexed=True, indexed_at=now)
            )
            result = await session.execute(stmt)
            updated += result.rowcount
            logger.info("judgments_marked_indexed", count=result.rowcount)

        await session.commit()

    return updated


# ═══════════════════════════════════════════════════════════════════════════════
# Main Indexing Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


async def run_indexing_pipeline(
    qdrant: bool = True,
    elasticsearch: bool = True,
) -> dict:
    """
    Full indexing pipeline:
        1. Chunk all legal data from PostgreSQL
        2. Embed chunks with BGE-large-en-v1.5
        3. Index into Qdrant (dense vectors)
        4. Index into Elasticsearch (BM25 text)
        5. Mark indexed records in PostgreSQL

    Args:
        qdrant: Whether to index into Qdrant (default True).
        elasticsearch: Whether to index into Elasticsearch (default True).

    Returns:
        Stats dict about the indexing run.
    """
    stats = {
        "total_chunks": 0,
        "section_chunks": 0,
        "judgment_chunks": 0,
        "qdrant_sections_indexed": 0,
        "qdrant_judgments_indexed": 0,
        "es_sections_indexed": 0,
        "es_judgments_indexed": 0,
        "db_records_marked": 0,
        "duration_seconds": 0,
    }

    start = time.time()

    # ── Step 1: Chunk all data ───────────────────────────────────────────
    logger.info("indexing_step", step="chunking")
    from data.processors.chunker import chunk_all

    all_chunks = await chunk_all()
    stats["total_chunks"] = len(all_chunks)

    if not all_chunks:
        logger.warning("no_chunks_to_index")
        return stats

    # Separate into sections and judgments
    section_chunks = [c for c in all_chunks if c.source_type == "act"]
    judgment_chunks = [c for c in all_chunks if c.source_type == "judgment"]
    stats["section_chunks"] = len(section_chunks)
    stats["judgment_chunks"] = len(judgment_chunks)

    logger.info(
        "chunks_ready",
        total=len(all_chunks),
        sections=len(section_chunks),
        judgments=len(judgment_chunks),
    )

    # ── Step 2: Embed all chunks ─────────────────────────────────────────
    section_embeddings = []
    judgment_embeddings = []

    if qdrant:
        logger.info("indexing_step", step="embedding", total=len(all_chunks))
        embedder = LegalEmbedder()
        embedder.load()

        if section_chunks:
            section_texts = [c.text for c in section_chunks]
            logger.info("embedding_sections", count=len(section_texts))
            section_embeddings = embedder.embed_documents(section_texts)

        if judgment_chunks:
            judgment_texts = [c.text for c in judgment_chunks]
            logger.info("embedding_judgments", count=len(judgment_texts))
            judgment_embeddings = embedder.embed_documents(judgment_texts)

    # ── Step 3: Index into Qdrant (dense) ────────────────────────────────
    if qdrant:
        logger.info("indexing_step", step="qdrant_indexing")
        qdrant_indexer = QdrantIndexer()
        await qdrant_indexer.connect()
        qdrant_indexer.create_collections(
            dimension=embedder.dimension if qdrant else settings.EMBEDDING_DIMENSION
        )

        if section_chunks:
            stats["qdrant_sections_indexed"] = qdrant_indexer.index_chunks(
                COLLECTION_SECTIONS, section_chunks, section_embeddings
            )

        if judgment_chunks:
            stats["qdrant_judgments_indexed"] = qdrant_indexer.index_chunks(
                COLLECTION_JUDGMENTS, judgment_chunks, judgment_embeddings
            )

        # Print Qdrant collection stats
        for col_name in [COLLECTION_SECTIONS, COLLECTION_JUDGMENTS]:
            try:
                info = qdrant_indexer.get_collection_info(col_name)
                logger.info("qdrant_collection_stats", **info)
            except Exception:
                pass

    # ── Step 4: Index into Elasticsearch (BM25) ──────────────────────────
    if elasticsearch:
        logger.info("indexing_step", step="elasticsearch_indexing")
        try:
            from data.embeddings.es_indexer import ElasticsearchIndexer

            es_indexer = ElasticsearchIndexer()
            await es_indexer.connect()
            await es_indexer.create_indices(recreate=True)

            if section_chunks:
                stats["es_sections_indexed"] = await es_indexer.index_chunks(
                    COLLECTION_SECTIONS, section_chunks
                )

            if judgment_chunks:
                stats["es_judgments_indexed"] = await es_indexer.index_chunks(
                    COLLECTION_JUDGMENTS, judgment_chunks
                )

            # Print ES index stats
            for idx_name in [COLLECTION_SECTIONS, COLLECTION_JUDGMENTS]:
                try:
                    idx_stats = await es_indexer.get_index_stats(idx_name)
                    logger.info("es_index_stats", **idx_stats)
                except Exception:
                    pass

            await es_indexer.close()

        except Exception as e:
            logger.error(
                "elasticsearch_indexing_failed",
                error=str(e),
                message="Elasticsearch indexing failed but Qdrant indexing succeeded. Continuing.",
            )

    # ── Step 5: Mark indexed in PostgreSQL ────────────────────────────────
    logger.info("indexing_step", step="mark_indexed_in_db")
    try:
        all_indexed_chunks = section_chunks + judgment_chunks
        stats["db_records_marked"] = await mark_indexed_in_db(all_indexed_chunks)
    except Exception as e:
        logger.error("mark_indexed_failed", error=str(e))

    stats["duration_seconds"] = round(time.time() - start, 2)

    logger.info("indexing_complete", **stats)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Run the indexing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NyayaMitra — Embedding & Indexing Pipeline",
    )
    parser.add_argument(
        "--qdrant-only",
        action="store_true",
        help="Only index into Qdrant (skip Elasticsearch)",
    )
    parser.add_argument(
        "--es-only",
        action="store_true",
        help="Only index into Elasticsearch (skip Qdrant embedding)",
    )
    args = parser.parse_args()

    # Determine which backends to index into
    do_qdrant = True
    do_es = True

    if args.qdrant_only:
        do_es = False
    elif args.es_only:
        do_qdrant = False

    print("\n" + "=" * 60)
    print("NyayaMitra — Embedding & Indexing Pipeline")
    print(f"  Qdrant:        {'enabled' if do_qdrant else 'disabled'}")
    print(f"  Elasticsearch: {'enabled' if do_es else 'disabled'}")
    print("=" * 60)

    stats = await run_indexing_pipeline(qdrant=do_qdrant, elasticsearch=do_es)

    print(f"\n{'='*60}")
    print("Indexing Results:")
    print(f"  Total chunks:           {stats['total_chunks']}")
    print(f"  Section chunks:         {stats['section_chunks']}")
    print(f"  Judgment chunks:        {stats['judgment_chunks']}")
    if do_qdrant:
        print(f"  Qdrant sections:        {stats['qdrant_sections_indexed']}")
        print(f"  Qdrant judgments:        {stats['qdrant_judgments_indexed']}")
    if do_es:
        print(f"  ES sections:            {stats['es_sections_indexed']}")
        print(f"  ES judgments:            {stats['es_judgments_indexed']}")
    print(f"  DB records marked:      {stats['db_records_marked']}")
    print(f"  Duration:               {stats['duration_seconds']}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())