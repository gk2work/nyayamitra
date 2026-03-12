"""
NyayaMitra — SQLAlchemy Database Models for Legal Data.

These models define the PostgreSQL schema for storing:
- Acts (Central and State legislation)
- Sections (individual provisions within acts)
- Judgments (SC and HC decisions)
- Metadata for tracking ingestion state

All scrapers and processors write to these tables.
The retrieval pipeline reads from them.

Usage:
    from app.models.legal import Act, Section, Judgment
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Acts & Sections
# ═══════════════════════════════════════════════════════════════════════════════


class Act(Base):
    """
    An Act of Parliament or State Legislature.

    Examples: Indian Penal Code 1860, Consumer Protection Act 2019,
              Maharashtra Rent Control Act 1999
    """

    __tablename__ = "acts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(500), nullable=False, index=True)
    short_name = Column(String(100), nullable=True)
    # e.g., "IPC", "CrPC", "CPA"
    year = Column(Integer, nullable=False)
    act_number = Column(String(50), nullable=True)
    # e.g., "Act No. 45 of 1860"

    # Classification
    domain = Column(
        String(50),
        nullable=False,
        default="general",
        index=True,
    )
    # criminal, property, family, labor, consumer, constitutional, ip
    jurisdiction = Column(String(100), nullable=False, default="central", index=True)
    # "central" or state name like "Maharashtra"

    # Status
    status = Column(String(30), nullable=False, default="active", index=True)
    # active, repealed, partially_repealed, amended
    enforcement_date = Column(Date, nullable=True)
    repeal_date = Column(Date, nullable=True)
    replaced_by = Column(String(500), nullable=True)
    # e.g., "Bharatiya Nyaya Sanhita, 2023" for IPC

    # Source
    source_url = Column(Text, nullable=True)
    raw_text = Column(Text, nullable=True)
    # Full act text for CPT training data

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_scraped_at = Column(DateTime, nullable=True)

    # Relationships
    sections = relationship("Section", back_populates="act", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("name", "year", "jurisdiction", name="uq_act_name_year_jurisdiction"),
        Index("ix_acts_domain_jurisdiction", "domain", "jurisdiction"),
    )

    def __repr__(self) -> str:
        return f"<Act(name='{self.name}', year={self.year}, status='{self.status}')>"


class Section(Base):
    """
    A section (provision) within an Act.

    This is the atomic unit of Indian legislation.
    Each section gets its own embedding in the vector database.

    Examples: Section 302 IPC (Murder), Section 498A IPC (Cruelty)
    """

    __tablename__ = "sections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    act_id = Column(UUID(as_uuid=True), ForeignKey("acts.id", ondelete="CASCADE"), nullable=False)

    # Section identification
    section_number = Column(String(50), nullable=False)
    # e.g., "302", "498A", "41(1)(b)", "Schedule I"
    title = Column(String(500), nullable=True)
    # e.g., "Punishment for murder"
    part = Column(String(200), nullable=True)
    # e.g., "Part II"
    chapter = Column(String(200), nullable=True)
    # e.g., "Chapter XVI - Of Offences Affecting the Human Body"

    # Content
    text = Column(Text, nullable=False)
    # Full text of the section
    explanation = Column(Text, nullable=True)
    # Explanation or illustration if present
    proviso = Column(Text, nullable=True)
    # Proviso text if present

    # Status
    status = Column(String(30), nullable=False, default="active")
    # active, repealed, amended, substituted
    amendment_notes = Column(Text, nullable=True)
    # History of amendments to this section
    effective_date = Column(Date, nullable=True)

    # Indexing metadata
    is_indexed = Column(Boolean, default=False, nullable=False)
    # Whether this section has been embedded and indexed in Qdrant
    indexed_at = Column(DateTime, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    act = relationship("Act", back_populates="sections")

    __table_args__ = (
        UniqueConstraint("act_id", "section_number", name="uq_section_act_number"),
        Index("ix_sections_act_id", "act_id"),
        Index("ix_sections_status", "status"),
        Index("ix_sections_indexed", "is_indexed"),
    )

    def __repr__(self) -> str:
        return f"<Section(number='{self.section_number}', title='{self.title}')>"


# ═══════════════════════════════════════════════════════════════════════════════
# Judgments
# ═══════════════════════════════════════════════════════════════════════════════


class Judgment(Base):
    """
    A judicial decision from the Supreme Court or High Courts.

    Each judgment may interpret multiple sections of multiple acts.
    Key components (headnote, ratio, facts) become separate embeddings.

    Examples: D.K. Basu v. State of WB (1997), Lalita Kumari v. State of UP (2014)
    """

    __tablename__ = "judgments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Case identification
    case_name = Column(String(500), nullable=False, index=True)
    # e.g., "D.K. Basu v. State of West Bengal"
    case_number = Column(String(200), nullable=True)
    # e.g., "Writ Petition (Crl.) No. 539 of 1986"

    # Court
    court = Column(String(100), nullable=False, index=True)
    # "Supreme Court", "Bombay High Court", "Delhi High Court", etc.
    court_type = Column(String(20), nullable=False, default="SC")
    # SC, HC, Tribunal
    bench = Column(String(500), nullable=True)
    # e.g., "Justice A.S. Anand, Justice M. Srinivasan"
    bench_size = Column(Integer, nullable=True)
    # 1, 2, 3, 5 (constitution bench), 7, 9, etc.

    # Date
    judgment_date = Column(Date, nullable=True, index=True)
    year = Column(Integer, nullable=False, index=True)

    # Citations
    citation_scc = Column(String(100), nullable=True)
    # e.g., "(1997) 1 SCC 416"
    citation_air = Column(String(100), nullable=True)
    # e.g., "AIR 1997 SC 610"
    indian_kanoon_id = Column(String(100), nullable=True, unique=True)
    # Indian Kanoon document ID for deduplication

    # Classification
    domain = Column(String(50), nullable=True)
    # criminal, property, family, etc.
    jurisdiction = Column(String(100), nullable=True)
    # State for HC judgments

    # Content (extracted components)
    headnote = Column(Text, nullable=True)
    # Official summary/headnote
    facts = Column(Text, nullable=True)
    # Statement of facts
    issues = Column(Text, nullable=True)
    # Issues framed by the court
    ratio_decidendi = Column(Text, nullable=True)
    # The binding legal principle
    obiter_dicta = Column(Text, nullable=True)
    # Non-binding observations
    order = Column(Text, nullable=True)
    # Final order/direction
    full_text = Column(Text, nullable=True)
    # Complete judgment text (for CPT training)

    # Sections interpreted
    sections_interpreted = Column(Text, nullable=True)
    # JSON list: [{"act": "IPC", "section": "302"}, ...]

    # Status
    is_overruled = Column(Boolean, default=False, nullable=False)
    overruled_by = Column(String(500), nullable=True)
    # Case name that overruled this judgment

    # Indexing
    is_indexed = Column(Boolean, default=False, nullable=False)
    indexed_at = Column(DateTime, nullable=True)

    # Source
    source_url = Column(Text, nullable=True)
    source = Column(String(50), nullable=True)
    # "indian_kanoon", "sci", "hc_website"

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_judgments_court_year", "court", "year"),
        Index("ix_judgments_domain", "domain"),
        Index("ix_judgments_indexed", "is_indexed"),
        Index("ix_judgments_overruled", "is_overruled"),
    )

    def __repr__(self) -> str:
        return f"<Judgment(case='{self.case_name}', year={self.year}, court='{self.court}')>"


# ═══════════════════════════════════════════════════════════════════════════════
# Ingestion Tracking
# ═══════════════════════════════════════════════════════════════════════════════


class IngestionLog(Base):
    """
    Tracks data ingestion runs for incremental scraping.

    Each scraper logs its last successful run here.
    On the next run, it only fetches data newer than last_success_at.
    """

    __tablename__ = "ingestion_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(50), nullable=False, index=True)
    # "indian_kanoon", "india_code", "sci"
    task = Column(String(100), nullable=False)
    # "scrape_sc_judgments", "scrape_central_acts", etc.

    # Run tracking
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False, default="running")
    # running, success, failed

    # Counts
    items_fetched = Column(Integer, default=0)
    items_new = Column(Integer, default=0)
    items_updated = Column(Integer, default=0)
    items_failed = Column(Integer, default=0)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # For incremental runs
    last_success_at = Column(DateTime, nullable=True)
    # Timestamp of the last successfully scraped item

    __table_args__ = (
        Index("ix_ingestion_source_task", "source", "task"),
    )

    def __repr__(self) -> str:
        return f"<IngestionLog(source='{self.source}', task='{self.task}', status='{self.status}')>"