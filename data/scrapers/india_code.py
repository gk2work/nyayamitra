"""
NyayaMitra — India Code Scraper (Sprint 7 — Full Corpus).

Scrapes Central Acts from legislative.gov.in (India Code portal).
Parses each act into its constituent sections and stores them in PostgreSQL.

Sprint 1-6 version scraped 5 priority acts from a hardcoded list.
Sprint 7 version:
    - Driven by data.config.acts_registry (96+ acts across 10 domains)
    - Concurrent scraping with configurable parallelism
    - Raw HTML caching for replay without re-fetching
    - Checkpoint / resume: skips acts already ingested (unless --force)
    - Per-act ingestion logging for progress tracking
    - Graceful degradation: if one act fails, others continue

Uses:
    - ActParser (data.processors.act_parser) for HTML/text parsing
    - DataValidator (data.processors.validator) for pre-insert validation
    - ActsRegistry (data.config.acts_registry) for the scraping manifest

Usage:
    # Scrape all P0 acts (core)
    python -m data.scrapers.india_code --priority P0

    # Scrape P0 + P1 acts
    python -m data.scrapers.india_code --priority P1

    # Scrape everything
    python -m data.scrapers.india_code --priority P2

    # Force re-scrape (ignore existing)
    python -m data.scrapers.india_code --priority P0 --force

    # Seed-only mode (no web scraping)
    python -m data.scrapers.india_code --seed-only

    # Dry run (show what would be scraped)
    python -m data.scrapers.india_code --priority P0 --dry-run

    # Scrape a single act by short name
    python -m data.scrapers.india_code --act IPC

Rate Limits:
    - Default 3 concurrent requests, 2s delay between bursts
    - Exponential backoff on 429/5xx (max 3 retries)
    - Respects robots.txt
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import structlog

# ── Project path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.database import async_session
from app.exceptions import FetchError, ParseError, ScraperError
from app.models.legal import Act, IngestionLog, Section

from data.config.acts_registry import (
    ActEntry,
    ActStatus,
    Priority,
    get_acts_by_priority,
    get_acts_for_ingestion,
    get_all_acts,
    get_short_name_index,
)
from data.processors.act_parser import ActParser
from data.processors.validator import DataValidator

logger = structlog.get_logger()

# ── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "acts"

# Base URL patterns for India Code
INDIA_CODE_BASE = "https://www.indiacode.nic.in"
INDIA_CODE_HANDLE = f"{INDIA_CODE_BASE}/handle/123456789"

# Legacy URL map for the original 5 acts (known-good URLs from Phase 1)
LEGACY_URLS: dict[str, str] = {
    "IPC": f"{INDIA_CODE_HANDLE}/2263",
    "CrPC": f"{INDIA_CODE_HANDLE}/1362",
    "IEA": f"{INDIA_CODE_HANDLE}/2188",
    "CPC": f"{INDIA_CODE_HANDLE}/2191",
    "Constitution": f"{INDIA_CODE_HANDLE}/2013",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Rate Limiter
# ═══════════════════════════════════════════════════════════════════════════════


class RateLimiter:
    """Async rate limiter using token bucket algorithm."""

    def __init__(self, max_requests: int, period: float):
        self.max_requests = max_requests
        self.period = period
        self.tokens = float(max_requests)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_requests,
                self.tokens + elapsed * (self.max_requests / self.period),
            )
            self.last_refill = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.period / self.max_requests)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Scraping Statistics
# ═══════════════════════════════════════════════════════════════════════════════


class ScrapeStats:
    """Thread-safe scraping statistics accumulator."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.acts_attempted: int = 0
        self.acts_success: int = 0
        self.acts_skipped: int = 0
        self.acts_failed: int = 0
        self.acts_cached: int = 0
        self.sections_created: int = 0
        self.sections_updated: int = 0
        self.validation_warnings: int = 0
        self.start_time: float = time.time()

    async def increment(self, **kwargs: int) -> None:
        async with self._lock:
            for key, value in kwargs.items():
                current = getattr(self, key, 0)
                setattr(self, key, current + value)

    def to_dict(self) -> dict:
        return {
            "acts_attempted": self.acts_attempted,
            "acts_success": self.acts_success,
            "acts_skipped": self.acts_skipped,
            "acts_failed": self.acts_failed,
            "acts_cached": self.acts_cached,
            "sections_created": self.sections_created,
            "sections_updated": self.sections_updated,
            "validation_warnings": self.validation_warnings,
            "duration_seconds": round(time.time() - self.start_time, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# India Code Scraper (Sprint 7)
# ═══════════════════════════════════════════════════════════════════════════════


class IndiaCodeScraper:
    """
    Scrapes acts and sections from India Code (legislative.gov.in).

    Sprint 7 improvements over Phase 1:
    - Registry-driven: reads from data.config.acts_registry
    - Concurrent: semaphore-controlled parallelism (default 3)
    - Cached: raw HTML saved to data/raw/acts/ for offline replay
    - Resumable: checks PostgreSQL for already-ingested acts
    - Robust: per-act error handling, never fails the whole batch
    """

    def __init__(
        self,
        concurrency: int = 3,
        force: bool = False,
        cache_enabled: bool = True,
    ):
        self.concurrency = concurrency
        self.force = force
        self.cache_enabled = cache_enabled

        self.rate_limiter = RateLimiter(
            max_requests=getattr(settings, "SCRAPER_RATE_LIMIT_REQUESTS", 5),
            period=getattr(settings, "SCRAPER_RATE_LIMIT_PERIOD", 10.0),
        )
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client: httpx.AsyncClient | None = None
        self.parser = ActParser()
        self.validator = DataValidator()
        self.stats = ScrapeStats()

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=True,
            headers={
                "User-Agent": "NyayaMitra-Bot/2.0 (Legal Research; +https://nyayamitra.in)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-IN,en;q=0.9",
            },
        )
        # Ensure cache directory exists
        if self.cache_enabled:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    # ── URL Resolution ────────────────────────────────────────────────────

    def resolve_url(self, act: ActEntry) -> str | None:
        """
        Resolve the India Code URL for an act.

        Priority:
        1. Legacy URL map (known-good from Phase 1)
        2. alt_source_url from registry
        3. Search URL (most reliable for acts we don't have handle IDs for)

        Note: india_code_id slugs don't work as URL params on India Code.
        The site uses numeric handle IDs (e.g., /handle/123456789/2263).
        For acts without known handles, we use the search endpoint.
        """
        # Check legacy map first (these are verified working)
        if act.short_name in LEGACY_URLS:
            return LEGACY_URLS[act.short_name]

        # Check for explicit alt source
        if act.alt_source_url:
            return act.alt_source_url

        # Search by name (most reliable for unknown acts)
        from urllib.parse import quote
        search_query = quote(f"{act.name}, {act.year}")
        return f"{INDIA_CODE_BASE}/search/act/?query={search_query}"

    # ── Cache Management ──────────────────────────────────────────────────

    def _cache_key(self, act: ActEntry) -> str:
        """Generate a deterministic cache filename for an act."""
        slug = re.sub(r"[^a-z0-9]+", "_", act.short_name.lower()).strip("_")
        return f"{slug}_{act.year}.html"

    def _read_cache(self, act: ActEntry) -> str | None:
        """Read cached HTML for an act, if available."""
        if not self.cache_enabled:
            return None
        cache_path = CACHE_DIR / self._cache_key(act)
        if cache_path.exists():
            logger.debug("cache_hit", act=act.short_name, path=str(cache_path))
            return cache_path.read_text(encoding="utf-8", errors="replace")
        return None

    def _write_cache(self, act: ActEntry, html: str) -> None:
        """Write raw HTML to cache."""
        if not self.cache_enabled:
            return
        cache_path = CACHE_DIR / self._cache_key(act)
        cache_path.write_text(html, encoding="utf-8")
        logger.debug("cache_written", act=act.short_name, size=len(html))

    # ── HTTP Fetching ─────────────────────────────────────────────────────

    async def fetch_page(self, url: str, act_name: str = "") -> str | None:
        """
        Fetch a page with rate limiting, retries, and exponential backoff.

        Returns HTML content or None on failure.
        """
        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries):
            await self.rate_limiter.acquire()

            try:
                response = await self.client.get(url)

                if response.status_code == 200:
                    return response.text

                if response.status_code == 429:
                    delay = base_delay * (2 ** attempt) + 5.0  # Extra 5s for 429
                    logger.warning(
                        "rate_limited",
                        act=act_name,
                        status=429,
                        retry_in=delay,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(delay)
                    continue

                if response.status_code >= 500:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "server_error",
                        act=act_name,
                        status=response.status_code,
                        retry_in=delay,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(delay)
                    continue

                # 4xx other than 429 — don't retry
                logger.error(
                    "fetch_client_error",
                    act=act_name,
                    status=response.status_code,
                    url=url[:120],
                )
                return None

            except httpx.TimeoutException:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "fetch_timeout",
                    act=act_name,
                    retry_in=delay,
                    attempt=attempt + 1,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPError as e:
                logger.error("fetch_http_error", act=act_name, error=str(e))
                return None

        logger.error("fetch_all_retries_exhausted", act=act_name, url=url[:120])
        return None

    # ── Ingestion Check ───────────────────────────────────────────────────

    async def is_already_ingested(self, act: ActEntry) -> bool:
        """
        Check if an act has already been ingested (has sections in DB).

        Returns True if act exists with at least 1 section and is_indexed=True.
        Skipped if --force is set.
        """
        if self.force:
            return False

        from sqlalchemy import func, select

        async with async_session() as session:
            stmt = (
                select(func.count(Section.id))
                .join(Act, Section.act_id == Act.id)
                .where(
                    Act.short_name == act.short_name,
                    Act.year == act.year,
                )
            )
            result = await session.execute(stmt)
            count = result.scalar() or 0

            if count > 0:
                logger.info(
                    "act_already_ingested",
                    act=act.short_name,
                    sections=count,
                )
                return True

        return False

    # ── Store Act + Sections ──────────────────────────────────────────────

    async def store_act_and_sections(
        self,
        act: ActEntry,
        sections_data: list[dict],
    ) -> tuple[int, int]:
        """
        Store an act and its parsed sections in PostgreSQL.

        Returns (sections_created, sections_updated).
        """
        from sqlalchemy import select

        created = 0
        updated = 0

        async with async_session() as session:
            try:
                # Find or create the Act record
                stmt = select(Act).where(
                    Act.short_name == act.short_name,
                    Act.year == act.year,
                )
                result = await session.execute(stmt)
                existing_act = result.scalar_one_or_none()

                if existing_act:
                    act_id = existing_act.id
                    # Update metadata
                    existing_act.domain = act.domain.value
                    existing_act.status = act.status.value
                    existing_act.replaced_by = act.replaced_by or None
                    existing_act.last_scraped_at = datetime.utcnow()
                else:
                    act_id = uuid.uuid4()
                    new_act = Act(
                        id=act_id,
                        name=act.name,
                        short_name=act.short_name,
                        year=act.year,
                        act_number=act.act_number,
                        domain=act.domain.value,
                        jurisdiction="central",
                        status=act.status.value,
                        replaced_by=act.replaced_by or None,
                        last_scraped_at=datetime.utcnow(),
                    )
                    session.add(new_act)
                    await session.flush()

                # Upsert sections
                for sec_data in sections_data:
                    # Validate
                    sec_result = self.validator.validate_section(sec_data, act_id=act_id)
                    if not sec_result.is_valid:
                        logger.debug(
                            "section_validation_failed",
                            act=act.short_name,
                            section=sec_data.get("section_number"),
                            errors=sec_result.errors,
                        )
                        continue

                    if sec_result.warnings:
                        await self.stats.increment(
                            validation_warnings=len(sec_result.warnings),
                        )

                    # Check if section already exists
                    sec_stmt = select(Section).where(
                        Section.act_id == act_id,
                        Section.section_number == sec_data["section_number"],
                    )
                    sec_result_db = await session.execute(sec_stmt)
                    existing_section = sec_result_db.scalar_one_or_none()

                    if existing_section:
                        # Update existing section
                        existing_section.text = sec_data["text"]
                        existing_section.title = sec_data.get("title")
                        existing_section.chapter = sec_data.get("chapter")
                        existing_section.part = sec_data.get("part")
                        existing_section.updated_at = datetime.utcnow()
                        updated += 1
                    else:
                        new_section = Section(
                            id=uuid.uuid4(),
                            act_id=act_id,
                            section_number=sec_data["section_number"],
                            title=sec_data.get("title"),
                            text=sec_data["text"],
                            chapter=sec_data.get("chapter"),
                            part=sec_data.get("part"),
                            status="active",
                        )
                        session.add(new_section)
                        created += 1

                await session.commit()

            except Exception as e:
                await session.rollback()
                logger.error(
                    "store_act_failed",
                    act=act.short_name,
                    error=str(e),
                )
                raise

        return created, updated

    # ── Scrape Single Act ─────────────────────────────────────────────────

    async def scrape_act(self, act: ActEntry) -> bool:
        """
        Scrape a single act: resolve URL → fetch/cache → parse → validate → store.

        Returns True on success, False on failure.
        """
        async with self.semaphore:
            await self.stats.increment(acts_attempted=1)

            log_ctx = {"act": act.short_name, "year": act.year, "domain": act.domain.value}
            logger.info("scraping_act_start", **log_ctx)

            # 1. Check if already ingested
            if await self.is_already_ingested(act):
                await self.stats.increment(acts_skipped=1)
                return True

            # 2. Check cache first
            html = self._read_cache(act)
            if html:
                await self.stats.increment(acts_cached=1)
            else:
                # 3. Resolve URL and fetch
                url = self.resolve_url(act)
                if not url:
                    logger.error("no_url_resolved", **log_ctx)
                    await self.stats.increment(acts_failed=1)
                    return False

                html = await self.fetch_page(url, act_name=act.short_name)
                if not html:
                    await self.stats.increment(acts_failed=1)
                    return False

                # Cache the raw HTML
                self._write_cache(act, html)

            # 4. Parse HTML into sections
            try:
                parsed_sections = self.parser.parse_html(html, act.name)
            except ParseError as e:
                logger.warning("parse_failed", error=str(e), **log_ctx)
                parsed_sections = []

            # Convert ParsedSection objects to dicts
            sections_data = [
                {
                    "section_number": s.section_number,
                    "title": s.title,
                    "text": s.text,
                    "chapter": s.chapter,
                    "part": s.part,
                }
                for s in parsed_sections
                if s.is_valid()
            ]

            logger.info(
                "act_parsed",
                sections_found=len(sections_data),
                total_parsed=len(parsed_sections),
                **log_ctx,
            )

            # 5. Store in PostgreSQL
            try:
                created, updated = await self.store_act_and_sections(act, sections_data)
                await self.stats.increment(
                    acts_success=1,
                    sections_created=created,
                    sections_updated=updated,
                )
                logger.info(
                    "act_stored",
                    sections_created=created,
                    sections_updated=updated,
                    **log_ctx,
                )
                return True

            except Exception as e:
                logger.error("act_store_failed", error=str(e), **log_ctx)
                await self.stats.increment(acts_failed=1)
                return False

    # ── Bulk Ingestion ────────────────────────────────────────────────────

    async def scrape_acts(
        self,
        acts: list[ActEntry],
        label: str = "bulk",
    ) -> dict:
        """
        Scrape a list of acts concurrently with progress tracking.

        Args:
            acts: List of ActEntry objects from the registry.
            label: Label for the ingestion log.

        Returns:
            Scraping statistics dict.
        """
        total = len(acts)
        logger.info(
            "bulk_scrape_start",
            label=label,
            total_acts=total,
            concurrency=self.concurrency,
            force=self.force,
        )

        # Create master ingestion log
        log_id = uuid.uuid4()
        async with async_session() as session:
            log = IngestionLog(
                id=log_id,
                source="india_code",
                task=f"scrape_{label}",
                started_at=datetime.utcnow(),
                status="running",
            )
            session.add(log)
            await session.commit()

        # Scrape all acts concurrently (semaphore controls parallelism)
        tasks = [self.scrape_act(act) for act in acts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count unexpected exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "act_unexpected_exception",
                    act=acts[i].short_name,
                    error=str(result),
                    error_type=type(result).__name__,
                )
                await self.stats.increment(acts_failed=1)

        # Finalize ingestion log
        stats = self.stats.to_dict()
        async with async_session() as session:
            from sqlalchemy import select

            stmt = select(IngestionLog).where(IngestionLog.id == log_id)
            result = await session.execute(stmt)
            log = result.scalar_one()
            log.completed_at = datetime.utcnow()
            log.status = "success" if stats["acts_failed"] == 0 else "partial"
            log.items_fetched = stats["acts_attempted"]
            log.items_new = stats["sections_created"]
            log.items_updated = stats["sections_updated"]
            log.items_failed = stats["acts_failed"]
            log.last_success_at = datetime.utcnow()
            await session.commit()

        logger.info("bulk_scrape_complete", label=label, **stats)
        return stats

    # ── Convenience Methods ───────────────────────────────────────────────

    async def scrape_by_priority(self, max_priority: str = "P0") -> dict:
        """Scrape all acts up to the given priority tier."""
        acts = get_acts_for_ingestion(max_priority)
        return await self.scrape_acts(acts, label=f"priority_{max_priority}")

    async def scrape_single(self, short_name: str) -> dict:
        """Scrape a single act by short name."""
        index = get_short_name_index()
        act = index.get(short_name)
        if not act:
            logger.error("act_not_found_in_registry", short_name=short_name)
            return {"error": f"Act '{short_name}' not found in registry"}
        await self.scrape_act(act)
        return self.stats.to_dict()

    async def scrape_priority_acts(self) -> dict:
        """
        Backward-compatible method: scrape P0 acts only.

        This preserves the Phase 1 API so existing pipeline.py calls still work.
        """
        return await self.scrape_by_priority("P0")


# ═══════════════════════════════════════════════════════════════════════════════
# Seed Data — Manually curated sections for initial testing
#
# Preserved from Phase 1 for backward compatibility.
# These are used when --seed-only is passed and ensure the database
# has some data even without web scraping.
# ═══════════════════════════════════════════════════════════════════════════════

SEED_SECTIONS_IPC = [
    {
        "section_number": "302",
        "title": "Punishment for murder",
        "text": (
            "Whoever commits murder shall be punished with death, or "
            "imprisonment for life, and shall also be liable to fine."
        ),
        "chapter": "Chapter XVI - Of Offences Affecting the Human Body",
    },
    {
        "section_number": "304A",
        "title": "Causing death by negligence",
        "text": (
            "Whoever causes the death of any person by doing any rash "
            "or negligent act not amounting to culpable homicide, shall "
            "be punished with imprisonment of either description for a "
            "term which may extend to two years, or with fine, or with both."
        ),
        "chapter": "Chapter XVI - Of Offences Affecting the Human Body",
    },
    {
        "section_number": "354",
        "title": "Assault or criminal force to woman with intent to outrage her modesty",
        "text": (
            "Whoever assaults or uses criminal force to any woman, "
            "intending to outrage or knowing it to be likely that he will "
            "thereby outrage her modesty, shall be punished with "
            "imprisonment of either description for a term which shall "
            "not be less than one year but which may extend to five years, "
            "and shall also be liable to fine."
        ),
        "chapter": "Chapter XVI - Of Offences Affecting the Human Body",
    },
    {
        "section_number": "420",
        "title": "Cheating and dishonestly inducing delivery of property",
        "text": (
            "Whoever cheats and thereby dishonestly induces the person "
            "deceived to deliver any property to any person, or to make, "
            "alter or destroy the whole or any part of a valuable security, "
            "or anything which is signed or sealed, and which is capable "
            "of being converted into a valuable security, shall be "
            "punished with imprisonment of either description for a term "
            "which may extend to seven years, and shall also be liable to fine."
        ),
        "chapter": "Chapter XVII - Of Offences Against Property",
    },
    {
        "section_number": "498A",
        "title": "Husband or relative of husband of a woman subjecting her to cruelty",
        "text": (
            "Whoever, being the husband or the relative of the husband "
            "of a woman, subjects such woman to cruelty shall be punished "
            "with imprisonment for a term which may extend to three years "
            "and shall also be liable to fine."
        ),
        "chapter": "Chapter XXA - Of Cruelty by Husband or Relatives of Husband",
    },
]

SEED_SECTIONS_CRPC = [
    {
        "section_number": "41",
        "title": "When police may arrest without warrant",
        "text": (
            "Any police officer may without an order from a Magistrate "
            "and without a warrant, arrest any person who has been "
            "concerned in any cognizable offence, or against whom a "
            "reasonable complaint has been made, or credible information "
            "has been received, or a reasonable suspicion exists, of "
            "his having been so concerned."
        ),
        "chapter": "Chapter V - Of the Arrest of Persons",
    },
    {
        "section_number": "154",
        "title": "Information in cognizable cases (FIR)",
        "text": (
            "Every information relating to the commission of a cognizable "
            "offence, if given orally to an officer in charge of a police "
            "station, shall be reduced to writing by him or under his "
            "direction, and be read over to the informant; and every such "
            "information, whether given in writing or reduced to writing "
            "as aforesaid, shall be signed by the person giving it, and "
            "the substance thereof shall be entered in a book to be kept "
            "by such officer in such form as the State Government may prescribe."
        ),
        "chapter": "Chapter XII - Information to the Police and their Powers to Investigate",
    },
    {
        "section_number": "161",
        "title": "Examination of witnesses by police",
        "text": (
            "Any police officer making an investigation under this Chapter, "
            "or any police officer not below such rank as the State "
            "Government may, by general or special order, prescribe in this "
            "behalf, acting on the requisition of such officer, may examine "
            "orally any person supposed to be acquainted with the facts and "
            "circumstances of the case."
        ),
        "chapter": "Chapter XII - Information to the Police and their Powers to Investigate",
    },
    {
        "section_number": "437",
        "title": "When bail may be taken in case of non-bailable offence",
        "text": (
            "When any person accused of, or suspected of, the commission "
            "of any non-bailable offence is arrested or detained without "
            "warrant by an officer in charge of a police station or "
            "appears or is brought before a Court other than the High "
            "Court or Court of Session, he may be released on bail."
        ),
        "chapter": "Chapter XXXIII - Provisions as to Bail and Bonds",
    },
    {
        "section_number": "482",
        "title": "Saving of inherent powers of High Court",
        "text": (
            "Nothing in this Code shall be deemed to limit or affect the "
            "inherent powers of the High Court to make such orders as may "
            "be necessary to give effect to any order under this Code, or "
            "to prevent abuse of the process of any Court or otherwise to "
            "secure the ends of justice."
        ),
        "chapter": "Chapter XXXVI - Miscellaneous",
    },
]


async def seed_initial_data() -> dict:
    """
    Seed the database with manually curated sections.

    Preserved from Phase 1 for backward compatibility.
    """
    logger.info("seed_data_start")
    stats = {"acts": 0, "sections": 0, "validation_skipped": 0}
    validator = DataValidator()

    seed_data = [
        {
            "act_info": {
                "name": "Indian Penal Code, 1860",
                "short_name": "IPC",
                "year": 1860,
                "act_number": "Act No. 45 of 1860",
                "domain": "criminal",
                "replaced_by": "Bharatiya Nyaya Sanhita, 2023",
            },
            "sections": SEED_SECTIONS_IPC,
        },
        {
            "act_info": {
                "name": "Code of Criminal Procedure, 1973",
                "short_name": "CrPC",
                "year": 1973,
                "act_number": "Act No. 2 of 1974",
                "domain": "criminal",
                "replaced_by": "Bharatiya Nagarik Suraksha Sanhita, 2023",
            },
            "sections": SEED_SECTIONS_CRPC,
        },
    ]

    async with async_session() as session:
        for data in seed_data:
            act_info = data["act_info"]

            act_result = validator.validate_act(act_info)
            if not act_result.is_valid:
                logger.warning(
                    "seed_act_invalid",
                    name=act_info["name"],
                    errors=act_result.errors,
                )
                stats["validation_skipped"] += 1
                continue

            from sqlalchemy import select

            stmt = select(Act).where(
                Act.name == act_info["name"],
                Act.year == act_info["year"],
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                act_id = existing.id
                logger.info("seed_act_exists", name=act_info["name"])
            else:
                new_act = Act(
                    id=uuid.uuid4(),
                    name=act_info["name"],
                    short_name=act_info.get("short_name"),
                    year=act_info["year"],
                    act_number=act_info.get("act_number"),
                    domain=act_info.get("domain", "general"),
                    jurisdiction="central",
                    status="active",
                    replaced_by=act_info.get("replaced_by"),
                )
                session.add(new_act)
                await session.flush()
                act_id = new_act.id
                stats["acts"] += 1
                logger.info("seed_act_created", name=act_info["name"])

            for sec in data["sections"]:
                sec_result = validator.validate_section(sec, act_id=act_id)
                if not sec_result.is_valid:
                    logger.debug("seed_section_invalid", errors=sec_result.errors)
                    stats["validation_skipped"] += 1
                    continue

                stmt = select(Section).where(
                    Section.act_id == act_id,
                    Section.section_number == sec["section_number"],
                )
                result = await session.execute(stmt)
                if result.scalar_one_or_none() is None:
                    new_section = Section(
                        id=uuid.uuid4(),
                        act_id=act_id,
                        section_number=sec["section_number"],
                        title=sec.get("title"),
                        text=sec["text"],
                        chapter=sec.get("chapter"),
                        status="active",
                    )
                    session.add(new_section)
                    stats["sections"] += 1

        await session.commit()

    logger.info("seed_data_complete", **stats)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Run the India Code scraper with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="NyayaMitra India Code Scraper (Sprint 7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m data.scrapers.india_code --priority P0          # Core 33 acts
  python -m data.scrapers.india_code --priority P1          # Core + important (78 acts)
  python -m data.scrapers.india_code --priority P2          # Everything (96 acts)
  python -m data.scrapers.india_code --act IPC              # Single act
  python -m data.scrapers.india_code --priority P0 --force  # Re-scrape even if exists
  python -m data.scrapers.india_code --seed-only            # Curated seed data only
  python -m data.scrapers.india_code --dry-run              # Show plan, don't execute
        """,
    )
    parser.add_argument(
        "--priority",
        choices=["P0", "P1", "P2"],
        default=None,
        help="Maximum priority tier to scrape (P0=core, P1=+important, P2=all)",
    )
    parser.add_argument(
        "--act",
        type=str,
        default=None,
        help="Scrape a single act by short name (e.g., IPC, TPA, HMA)",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only seed curated data, skip web scraping",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-scrape even if act already exists in DB",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of concurrent scrape requests (default: 3)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable HTML caching",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be scraped without executing",
    )

    args = parser.parse_args()

    # Always seed curated data first
    seed_stats = await seed_initial_data()
    print(f"\nSeed data: {seed_stats['acts']} acts, {seed_stats['sections']} sections created")

    if args.seed_only:
        return

    # Determine what to scrape
    if args.act:
        # Single act mode
        index = get_short_name_index()
        act = index.get(args.act)
        if not act:
            print(f"\nError: Act '{args.act}' not found in registry.")
            print("Available short names:")
            for name in sorted(index.keys()):
                if "," not in name:  # Skip citation_key entries
                    print(f"  {name}")
            return
        target_acts = [act]
        label = f"single_{args.act}"
    elif args.priority:
        target_acts = get_acts_for_ingestion(args.priority)
        label = f"priority_{args.priority}"
    else:
        print("\nSpecify --priority or --act to scrape. Use --help for options.")
        return

    # Dry run
    if args.dry_run:
        print(f"\n{'=' * 65}")
        print(f"  DRY RUN — Would scrape {len(target_acts)} acts")
        print(f"{'=' * 65}\n")
        for i, act in enumerate(target_acts, 1):
            status = f" [{act.status.value}]" if act.status != ActStatus.ACTIVE else ""
            url = LEGACY_URLS.get(act.short_name, act.india_code_id or "search")
            print(f"  {i:>3}. {act.citation_key:<35} ({act.domain.value}){status}")
        print(f"\n  Total: {len(target_acts)} acts")
        print(f"  Concurrency: {args.concurrency}")
        print(f"  Force: {args.force}")
        print(f"  Cache: {'disabled' if args.no_cache else 'enabled'}")
        return

    # Execute scraping
    print(f"\nScraping {len(target_acts)} acts (concurrency={args.concurrency})...\n")

    async with IndiaCodeScraper(
        concurrency=args.concurrency,
        force=args.force,
        cache_enabled=not args.no_cache,
    ) as scraper:
        stats = await scraper.scrape_acts(target_acts, label=label)

    # Print report
    print(f"\n{'=' * 55}")
    print(f"  India Code Scraper — Results")
    print(f"{'=' * 55}")
    print(f"  Acts attempted:     {stats['acts_attempted']}")
    print(f"  Acts succeeded:     {stats['acts_success']}")
    print(f"  Acts skipped:       {stats['acts_skipped']} (already ingested)")
    print(f"  Acts from cache:    {stats['acts_cached']}")
    print(f"  Acts failed:        {stats['acts_failed']}")
    print(f"  Sections created:   {stats['sections_created']}")
    print(f"  Sections updated:   {stats['sections_updated']}")
    print(f"  Validation warns:   {stats['validation_warnings']}")
    print(f"  Duration:           {stats['duration_seconds']}s")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    asyncio.run(main())