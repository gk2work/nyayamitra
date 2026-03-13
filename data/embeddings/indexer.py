"""
NyayaMitra — Embedding & Indexing Pipeline.

Takes chunked legal documents, embeds them using BGE-large,
and indexes them into Qdrant for semantic retrieval.

Pipeline:
    1. Load chunks from the chunker
    2. Embed each chunk using sentence-transformers (BGE-large-en-v1.5)
    3. Upsert vectors + metadata into Qdrant collections
    4. Mark indexed items in PostgreSQL

Usage:
    python -m data.embeddings.indexer
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings

logger = structlog.get_logger()

# Qdrant collection names
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
# Main Indexing Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


async def run_indexing_pipeline() -> dict:
    """
    Full indexing pipeline:
        1. Chunk all legal data
        2. Embed chunks with BGE
        3. Index into Qdrant

    Returns stats about the indexing run.
    """
    stats = {
        "total_chunks": 0,
        "section_chunks": 0,
        "judgment_chunks": 0,
        "sections_indexed": 0,
        "judgments_indexed": 0,
        "duration_seconds": 0,
    }

    start = time.time()

    # Step 1: Chunk all data
    logger.info("indexing_step_1", step="chunking")
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

    # Step 2: Embed all chunks
    logger.info("indexing_step_2", step="embedding", total=len(all_chunks))
    embedder = LegalEmbedder()
    embedder.load()

    # Embed sections
    if section_chunks:
        section_texts = [c.text for c in section_chunks]
        logger.info("embedding_sections", count=len(section_texts))
        section_embeddings = embedder.embed_documents(section_texts)
    else:
        section_embeddings = []

    # Embed judgments
    if judgment_chunks:
        judgment_texts = [c.text for c in judgment_chunks]
        logger.info("embedding_judgments", count=len(judgment_texts))
        judgment_embeddings = embedder.embed_documents(judgment_texts)
    else:
        judgment_embeddings = []

    # Step 3: Index into Qdrant
    logger.info("indexing_step_3", step="qdrant_indexing")
    indexer = QdrantIndexer()
    await indexer.connect()
    indexer.create_collections(dimension=embedder.dimension)

    # Index sections
    if section_chunks:
        stats["sections_indexed"] = indexer.index_chunks(
            COLLECTION_SECTIONS, section_chunks, section_embeddings
        )

    # Index judgments
    if judgment_chunks:
        stats["judgments_indexed"] = indexer.index_chunks(
            COLLECTION_JUDGMENTS, judgment_chunks, judgment_embeddings
        )

    stats["duration_seconds"] = round(time.time() - start, 2)

    # Print collection stats
    for col_name in [COLLECTION_SECTIONS, COLLECTION_JUDGMENTS]:
        try:
            info = indexer.get_collection_info(col_name)
            logger.info("collection_stats", **info)
        except Exception:
            pass

    logger.info("indexing_complete", **stats)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Run the indexing pipeline."""
    print("\n" + "=" * 60)
    print("NyayaMitra — Embedding & Indexing Pipeline")
    print("=" * 60)

    stats = await run_indexing_pipeline()

    print(f"\n{'='*60}")
    print("Indexing Results:")
    print(f"  Total chunks:      {stats['total_chunks']}")
    print(f"  Section chunks:    {stats['section_chunks']}")
    print(f"  Judgment chunks:   {stats['judgment_chunks']}")
    print(f"  Sections indexed:  {stats['sections_indexed']}")
    print(f"  Judgments indexed:  {stats['judgments_indexed']}")
    print(f"  Duration:          {stats['duration_seconds']}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())