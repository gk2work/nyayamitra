"""
NyayaMitra — Legal-Aware Document Chunker.

Converts acts/sections and judgments into chunks optimized for
embedding and retrieval. Legal documents need domain-specific
chunking — not generic fixed-size windows.

Chunking Strategy:
    - Acts: One chunk per section (atomic legal unit)
    - Judgments: Separate chunks for headnote, ratio, facts
    - Each chunk carries full metadata for filtering

Usage:
    python -m data.processors.chunker

Output:
    List of chunks ready for embedding, each with:
    - text: The content to embed
    - metadata: act/section/case info for filtering and display
    - chunk_type: "section", "headnote", "ratio", "facts"
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.database import async_session
from app.models.legal import Act, Section, Judgment

logger = structlog.get_logger()


@dataclass
class LegalChunk:
    """A single chunk ready for embedding and indexing."""

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    chunk_type: str = ""
    # "section", "headnote", "ratio_decidendi", "facts", "full_judgment"

    # Source metadata (stored as Qdrant payload for filtering)
    source_type: str = ""  # "act" or "judgment"
    source_id: str = ""  # UUID of the act/section/judgment

    # Act/Section metadata
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

    # Common metadata
    domain: str | None = None
    jurisdiction: str | None = None
    status: str | None = None

    def to_payload(self) -> dict:
        """Convert to Qdrant payload dict (for storage alongside the vector)."""
        payload = {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "domain": self.domain or "general",
            "jurisdiction": self.jurisdiction or "central",
        }
        if self.act_name:
            payload["act_name"] = self.act_name
        if self.act_short_name:
            payload["act_short_name"] = self.act_short_name
        if self.section_number:
            payload["section_number"] = self.section_number
        if self.section_title:
            payload["section_title"] = self.section_title
        if self.chapter:
            payload["chapter"] = self.chapter
        if self.case_name:
            payload["case_name"] = self.case_name
        if self.court:
            payload["court"] = self.court
        if self.year:
            payload["year"] = self.year
        if self.citation:
            payload["citation"] = self.citation
        if self.status:
            payload["status"] = self.status
        return payload


async def chunk_sections() -> list[LegalChunk]:
    """
    Create one chunk per section from all acts in the database.

    Each section becomes a single chunk with the format:
        "{Act Name} - Section {Number}: {Title}\n\n{Section Text}"

    This preserves the full context of each legal provision.
    """
    chunks = []

    async with async_session() as session:
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        stmt = select(Act).options(selectinload(Act.sections))
        result = await session.execute(stmt)
        acts = result.scalars().all()

        for act in acts:
            for section in act.sections:
                # Build the chunk text with context
                text_parts = []

                # Header: "Indian Penal Code, 1860 - Section 302: Punishment for murder"
                header = f"{act.name} - Section {section.section_number}"
                if section.title:
                    header += f": {section.title}"
                text_parts.append(header)

                # Chapter context if available
                if section.chapter:
                    text_parts.append(f"[{section.chapter}]")

                # Section text
                text_parts.append("")
                text_parts.append(section.text)

                # Explanation if present
                if section.explanation:
                    text_parts.append(f"\nExplanation: {section.explanation}")

                # Proviso if present
                if section.proviso:
                    text_parts.append(f"\nProviso: {section.proviso}")

                chunk_text = "\n".join(text_parts)

                chunk = LegalChunk(
                    text=chunk_text,
                    chunk_type="section",
                    source_type="act",
                    source_id=str(section.id),
                    act_name=act.name,
                    act_short_name=act.short_name,
                    section_number=section.section_number,
                    section_title=section.title,
                    chapter=section.chapter,
                    domain=act.domain,
                    jurisdiction=act.jurisdiction,
                    status=section.status,
                )
                chunks.append(chunk)

    logger.info("sections_chunked", count=len(chunks))
    return chunks


async def chunk_judgments() -> list[LegalChunk]:
    """
    Create multiple chunks per judgment.

    Each judgment produces up to 3 chunks:
        1. Headnote chunk — the summary/headnote
        2. Ratio chunk — the binding legal principle
        3. Facts chunk — statement of facts (if available)

    This allows the retrieval engine to match on different aspects
    of the same case depending on the query.
    """
    chunks = []

    async with async_session() as session:
        from sqlalchemy import select

        stmt = select(Judgment)
        result = await session.execute(stmt)
        judgments = result.scalars().all()

        for j in judgments:
            base_header = f"{j.case_name} ({j.year}) - {j.court}"
            if j.citation_scc:
                base_header += f" [{j.citation_scc}]"

            citation = j.citation_scc or j.citation_air or ""

            # Chunk 1: Headnote
            if j.headnote:
                chunk = LegalChunk(
                    text=f"{base_header}\n\nHeadnote:\n{j.headnote}",
                    chunk_type="headnote",
                    source_type="judgment",
                    source_id=str(j.id),
                    case_name=j.case_name,
                    court=j.court,
                    year=j.year,
                    citation=citation,
                    domain=j.domain,
                    jurisdiction=j.jurisdiction,
                )
                chunks.append(chunk)

            # Chunk 2: Ratio Decidendi
            if j.ratio_decidendi:
                chunk = LegalChunk(
                    text=f"{base_header}\n\nRatio Decidendi:\n{j.ratio_decidendi}",
                    chunk_type="ratio_decidendi",
                    source_type="judgment",
                    source_id=str(j.id),
                    case_name=j.case_name,
                    court=j.court,
                    year=j.year,
                    citation=citation,
                    domain=j.domain,
                    jurisdiction=j.jurisdiction,
                )
                chunks.append(chunk)

            # Chunk 3: Facts (if available)
            if j.facts:
                chunk = LegalChunk(
                    text=f"{base_header}\n\nFacts:\n{j.facts}",
                    chunk_type="facts",
                    source_type="judgment",
                    source_id=str(j.id),
                    case_name=j.case_name,
                    court=j.court,
                    year=j.year,
                    citation=citation,
                    domain=j.domain,
                    jurisdiction=j.jurisdiction,
                )
                chunks.append(chunk)

    logger.info("judgments_chunked", count=len(chunks))
    return chunks


async def chunk_all() -> list[LegalChunk]:
    """
    Chunk all legal data in the database.

    Returns all chunks (sections + judgments) ready for embedding.
    """
    logger.info("chunking_start")

    section_chunks = await chunk_sections()
    judgment_chunks = await chunk_judgments()

    all_chunks = section_chunks + judgment_chunks

    logger.info(
        "chunking_complete",
        total=len(all_chunks),
        sections=len(section_chunks),
        judgments=len(judgment_chunks),
    )

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run the chunker and display results."""
    chunks = await chunk_all()

    print(f"\n{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'='*60}")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ({chunk.chunk_type}) ---")
        print(f"Source: {chunk.act_name or chunk.case_name}")
        if chunk.section_number:
            print(f"Section: {chunk.section_number}")
        print(f"Domain: {chunk.domain}")
        print(f"Text preview: {chunk.text[:200]}...")
        print()

    if len(chunks) > 5:
        print(f"... and {len(chunks) - 5} more chunks")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())