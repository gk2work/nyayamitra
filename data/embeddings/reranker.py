"""
NyayaMitra — Cross-Encoder Re-ranker.

Standalone module wrapping the cross-encoder model for re-ranking
retrieval results. Separated from the retrieval service for clean
testability and reuse.

The cross-encoder scores (query, document) pairs jointly, producing
more accurate relevance scores than bi-encoder cosine similarity.
Trade-off: slower than bi-encoders, so applied only to top-K
candidates after initial retrieval + RRF fusion.

Model: cross-encoder/ms-marco-MiniLM-L-12-v2
    - Trained on MS MARCO passage ranking
    - Input: (query, passage) pair
    - Output: relevance score (higher = more relevant)
    - Max sequence length: 512 tokens

Usage:
    from data.embeddings.reranker import LegalReranker

    reranker = LegalReranker()
    reranker.load()
    scored = reranker.score_pairs("What is bail?", ["Section 436...", "Section 302..."])
    reranked = reranker.rerank(query, results, top_k=10)

CLI:
    python -m data.embeddings.reranker
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Re-ranker
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RerankResult:
    """A single re-ranked result with the original data and new score."""

    text: str
    ce_score: float
    original_score: float = 0.0
    original_rank: int = 0
    metadata: dict | None = None


class LegalReranker:
    """
    Cross-encoder re-ranker for legal retrieval results.

    Loads ms-marco-MiniLM-L-12-v2 (or a configured alternative)
    and scores (query, document) pairs for re-ranking.

    Thread-safe after loading. The model is loaded lazily on first use.

    Attributes:
        model_name: HuggingFace model ID for the cross-encoder.
        max_length: Maximum token length for input pairs (default 512).
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.max_length = max_length
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """
        Load the cross-encoder model into memory.

        Uses sentence-transformers CrossEncoder class which handles
        tokenization, padding, and inference internally.
        """
        if self._loaded:
            return

        from sentence_transformers import CrossEncoder

        logger.info("reranker_loading", model=self.model_name)
        start = time.time()

        self.model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
        )

        duration = round(time.time() - start, 2)
        logger.info(
            "reranker_loaded",
            model=self.model_name,
            max_length=self.max_length,
            duration_seconds=duration,
        )
        self._loaded = True

    def score_pairs(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 32,
    ) -> list[float]:
        """
        Score a list of (query, document) pairs.

        Args:
            query: The search query.
            documents: List of document texts to score against the query.
            batch_size: Batch size for inference.

        Returns:
            List of relevance scores (one per document), higher = more relevant.
        """
        if not self._loaded:
            self.load()

        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int | None = None,
        score_attr: str = "score",
        text_attr: str = "text",
    ) -> list:
        """
        Re-rank a list of retrieval result objects.

        Works with any object that has a text attribute and a score attribute.
        Mutates the score attribute in place with the cross-encoder score
        and returns the list sorted by the new score.

        Args:
            query: The original user query.
            results: List of result objects (RetrievalResult or similar).
            top_k: Number of results to return (default from config).
            score_attr: Name of the score attribute to overwrite.
            text_attr: Name of the text attribute to read.

        Returns:
            Re-ranked and truncated list of result objects.
        """
        if not results:
            return []

        if not self._loaded:
            self.load()

        top_k = top_k or settings.RETRIEVAL_FINAL_TOP_K

        # Extract texts
        documents = [getattr(r, text_attr, "") for r in results]

        # Score with cross-encoder
        scores = self.score_pairs(query, documents)

        # Assign new scores
        for result, ce_score in zip(results, scores):
            setattr(result, score_attr, ce_score)

        # Sort by new score descending
        results.sort(key=lambda r: getattr(r, score_attr, 0), reverse=True)

        return results[:top_k]

    def rerank_with_details(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        metadata_list: list[dict] | None = None,
    ) -> list[RerankResult]:
        """
        Re-rank documents and return detailed RerankResult objects.

        Useful for evaluation and debugging — shows the original rank,
        original score, and cross-encoder score side by side.

        Args:
            query: The search query.
            documents: List of document texts.
            top_k: Number of results to return.
            metadata_list: Optional metadata dicts parallel to documents.

        Returns:
            List of RerankResult objects sorted by cross-encoder score.
        """
        if not documents:
            return []

        top_k = top_k or settings.RETRIEVAL_FINAL_TOP_K

        scores = self.score_pairs(query, documents)

        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                text=doc,
                ce_score=score,
                original_rank=i + 1,
                metadata=metadata_list[i] if metadata_list and i < len(metadata_list) else None,
            ))

        results.sort(key=lambda r: r.ce_score, reverse=True)
        return results[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point — Interactive Testing
# ═══════════════════════════════════════════════════════════════════════════════


SAMPLE_QUERIES = [
    {
        "query": "Can police arrest me without a warrant?",
        "documents": [
            "Code of Criminal Procedure, 1973 - Section 41: When police may arrest without warrant\n\nAny police officer may without an order from a Magistrate and without a warrant, arrest any person who has been concerned in any cognizable offence.",
            "Code of Criminal Procedure, 1973 - Section 436: In what cases bail to be taken\n\nWhen any person other than a person accused of a non-bailable offence is arrested or detained without warrant by an officer in charge of a police station, such person shall be released on bail.",
            "Indian Penal Code, 1860 - Section 302: Punishment for murder\n\nWhoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
            "Code of Criminal Procedure, 1973 - Section 41A: Notice of appearance before police officer\n\nThe police officer shall, in all cases where the arrest of a person is not required under section 41, issue a notice directing the person to appear before him.",
            "Constitution of India - Article 21: Protection of life and personal liberty\n\nNo person shall be deprived of his life or personal liberty except according to procedure established by law.",
        ],
    },
    {
        "query": "What are my rights as a tenant?",
        "documents": [
            "Transfer of Property Act, 1882 - Section 106: Duration of certain leases\n\nA lease of immoveable property for any purpose other than agriculture or manufacturing shall be deemed to be a lease from month to month, terminable by fifteen days' notice.",
            "Transfer of Property Act, 1882 - Section 54: Sale defined\n\nSale is a transfer of ownership in exchange for a price paid or promised.",
            "Real Estate (Regulation and Development) Act, 2016 - Section 18: Return of amount and compensation\n\nIf the promoter fails to complete or give possession, he shall be liable to return the amount with interest.",
            "Indian Penal Code, 1860 - Section 420: Cheating and dishonestly inducing delivery of property\n\nWhoever cheats and thereby dishonestly induces the person to deliver any property shall be punished.",
            "Hindu Marriage Act, 1955 - Section 13: Divorce\n\nAny marriage may be dissolved by a decree of divorce on the ground of cruelty or desertion.",
        ],
    },
]


async def main():
    """Run interactive re-ranking test on sample queries."""
    print("\n" + "=" * 70)
    print("  NyayaMitra — Cross-Encoder Re-ranker Test")
    print("=" * 70)

    reranker = LegalReranker()
    reranker.load()

    for sample in SAMPLE_QUERIES:
        query = sample["query"]
        documents = sample["documents"]

        print(f"\n{'─' * 70}")
        print(f"  Query: {query}")
        print(f"{'─' * 70}")

        results = reranker.rerank_with_details(
            query, documents, top_k=len(documents),
        )

        for i, r in enumerate(results, 1):
            # Show first 80 chars of the document
            preview = r.text.split("\n")[0][:80]
            rank_change = r.original_rank - i
            arrow = "↑" if rank_change > 0 else ("↓" if rank_change < 0 else "=")
            print(
                f"  {i}. [CE: {r.ce_score:+.4f}] "
                f"(was #{r.original_rank} {arrow}) "
                f"{preview}..."
            )

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())