"""
NyayaMitra — Indian Kanoon API Client (Sprint 7 — Full Corpus).

Fetches Supreme Court and High Court judgments from indiankanoon.org.
Indian Kanoon is the largest free repository of Indian legal documents.

Sprint 1-6 version:
    - Fetched SC judgments only, small batches, simple pagination.

Sprint 7 version:
    - Registry-driven: reads courts from data.config.courts_registry
    - Multi-court: SC + all 25 HCs + tribunals
    - Year-by-year pagination: fetches each court for each year separately
    - Raw JSON caching: every API response saved for offline replay
    - Resume support: tracks progress via ingestion_logs, skips completed court-years
    - Concurrent: semaphore-controlled parallelism across courts
    - Target: 100K+ judgments (SC 15K + HC 62K from P0 courts alone)

Usage:
    # Seed landmark judgments (no API key needed)
    python -m data.scrapers.indian_kanoon --seed-only

    # Scrape SC + P0 HCs (default)
    python -m data.scrapers.indian_kanoon --scrape --priority P0

    # Scrape all courts including P1 HCs
    python -m data.scrapers.indian_kanoon --scrape --priority P1

    # Scrape a single court
    python -m data.scrapers.indian_kanoon --scrape --court SC
    python -m data.scrapers.indian_kanoon --scrape --court "DEL HC"

    # Specific year range
    python -m data.scrapers.indian_kanoon --scrape --priority P0 --start-year 2020 --end-year 2025

    # Force re-scrape (ignore checkpoint)
    python -m data.scrapers.indian_kanoon --scrape --priority P0 --force

    # Dry run
    python -m data.scrapers.indian_kanoon --scrape --priority P0 --dry-run

API Rate Limits:
    - ~100 requests/min on free tier (we use ~30/min to be safe)
    - Exponential backoff on 429 responses
    - 2-3 second minimum delay between requests
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from datetime import datetime, date
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
from app.models.legal import Judgment, IngestionLog

from data.config.courts_registry import (
    CourtEntry,
    CourtType,
    SUPREME_COURT,
    ALL_HIGH_COURTS,
    TRIBUNALS,
    get_courts_for_ingestion,
    get_court_by_short_name,
)
from data.processors.judgment_parser import JudgmentParser
from data.processors.validator import DataValidator

logger = structlog.get_logger()

# ── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "judgments"
IK_BASE_URL = "https://api.indiankanoon.org"
IK_RESULTS_PER_PAGE = 10  # Indian Kanoon returns 10 docs per page


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
    """Thread-safe scraping statistics."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.courts_attempted: int = 0
        self.courts_completed: int = 0
        self.court_years_scraped: int = 0
        self.court_years_skipped: int = 0
        self.pages_fetched: int = 0
        self.judgments_fetched: int = 0
        self.judgments_new: int = 0
        self.judgments_existing: int = 0
        self.judgments_invalid: int = 0
        self.api_errors: int = 0
        self.store_errors: int = 0
        self.start_time: float = time.time()

    async def increment(self, **kwargs: int) -> None:
        async with self._lock:
            for key, value in kwargs.items():
                current = getattr(self, key, 0)
                setattr(self, key, current + value)

    def to_dict(self) -> dict:
        return {
            "courts_attempted": self.courts_attempted,
            "courts_completed": self.courts_completed,
            "court_years_scraped": self.court_years_scraped,
            "court_years_skipped": self.court_years_skipped,
            "pages_fetched": self.pages_fetched,
            "judgments_fetched": self.judgments_fetched,
            "judgments_new": self.judgments_new,
            "judgments_existing": self.judgments_existing,
            "judgments_invalid": self.judgments_invalid,
            "api_errors": self.api_errors,
            "store_errors": self.store_errors,
            "duration_seconds": round(time.time() - self.start_time, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Indian Kanoon Client (Sprint 7)
# ═══════════════════════════════════════════════════════════════════════════════


class IndianKanoonClient:
    """
    Client for the Indian Kanoon API — Sprint 7 full-corpus version.

    Improvements over Phase 1:
    - Multi-court: scrapes SC + any number of HCs from the registry
    - Year-by-year: paginate each court per year for systematic coverage
    - Cached: raw API responses saved to data/raw/judgments/
    - Resumable: checkpoint via ingestion_logs per (court, year)
    - Concurrent: semaphore across courts (but sequential within a court)

    API docs: https://api.indiankanoon.org/doc/
    """

    def __init__(
        self,
        concurrency: int = 2,
        force: bool = False,
        cache_enabled: bool = True,
        min_delay: float = 2.0,
    ):
        self.concurrency = concurrency
        self.force = force
        self.cache_enabled = cache_enabled
        self.min_delay = min_delay

        self.token = getattr(settings, "INDIAN_KANOON_API_TOKEN", "")
        self.rate_limiter = RateLimiter(max_requests=20, period=60.0)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client: httpx.AsyncClient | None = None
        self.parser = JudgmentParser()
        self.validator = DataValidator()
        self.stats = ScrapeStats()

    async def __aenter__(self):
        headers = {
            "User-Agent": "NyayaMitra-Bot/2.0 (Legal Research; +https://nyayamitra.in)",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Token {self.token}"

        self.client = httpx.AsyncClient(timeout=45.0, headers=headers)

        if self.cache_enabled:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    # ── Cache Management ──────────────────────────────────────────────────

    def _cache_path(self, court_id: str, year: int, page: int) -> Path:
        court_dir = CACHE_DIR / court_id
        court_dir.mkdir(parents=True, exist_ok=True)
        return court_dir / f"{year}_page{page:04d}.json"

    def _read_cache(self, court_id: str, year: int, page: int) -> dict | None:
        if not self.cache_enabled:
            return None
        path = self._cache_path(court_id, year, page)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _write_cache(self, court_id: str, year: int, page: int, data: dict) -> None:
        if not self.cache_enabled:
            return
        path = self._cache_path(court_id, year, page)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # ── Checkpoint Management ─────────────────────────────────────────────

    async def is_court_year_done(self, court_id: str, year: int) -> bool:
        """Check if a (court, year) pair has already been fully scraped."""
        if self.force:
            return False

        from sqlalchemy import select

        async with async_session() as session:
            stmt = select(IngestionLog).where(
                IngestionLog.source == "indian_kanoon",
                IngestionLog.task == f"court_{court_id}_year_{year}",
                IngestionLog.status == "success",
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none() is not None

    async def mark_court_year_done(
        self,
        court_id: str,
        year: int,
        items_fetched: int,
        items_new: int,
    ) -> None:
        """Record that a (court, year) has been fully scraped."""
        async with async_session() as session:
            log = IngestionLog(
                id=uuid.uuid4(),
                source="indian_kanoon",
                task=f"court_{court_id}_year_{year}",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                status="success",
                items_fetched=items_fetched,
                items_new=items_new,
                last_success_at=datetime.utcnow(),
            )
            session.add(log)
            await session.commit()

    # ── API Calls ─────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        page: int = 0,
    ) -> dict | None:
        """
        Search Indian Kanoon API.

        The query already includes court and date filters.
        """
        if not self.token:
            logger.warning("no_api_token", hint="Set INDIAN_KANOON_API_TOKEN in .env")
            return None

        await self.rate_limiter.acquire()

        # Enforce minimum delay between requests
        await asyncio.sleep(self.min_delay)

        max_retries = 3
        base_delay = 3.0

        for attempt in range(max_retries):
            try:
                resp = await self.client.post(
                    f"{IK_BASE_URL}/search/",
                    data={"formInput": query, "pagenum": page},
                )

                if resp.status_code == 200:
                    await self.stats.increment(pages_fetched=1)
                    return resp.json()

                if resp.status_code == 429:
                    delay = base_delay * (2 ** attempt) + 10.0
                    logger.warning(
                        "ik_rate_limited",
                        retry_in=delay,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code >= 500:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "ik_server_error",
                        status=resp.status_code,
                        retry_in=delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                # 4xx other than 429
                logger.error("ik_client_error", status=resp.status_code)
                await self.stats.increment(api_errors=1)
                return None

            except httpx.TimeoutException:
                delay = base_delay * (2 ** attempt)
                logger.warning("ik_timeout", retry_in=delay, attempt=attempt + 1)
                await asyncio.sleep(delay)

            except httpx.HTTPError as e:
                logger.error("ik_http_error", error=str(e))
                await self.stats.increment(api_errors=1)
                return None

        logger.error("ik_all_retries_exhausted")
        await self.stats.increment(api_errors=1)
        return None

    async def get_document(self, doc_id: str) -> dict | None:
        """Fetch a specific document by Indian Kanoon doc ID."""
        if not self.token:
            return None

        await self.rate_limiter.acquire()
        await asyncio.sleep(self.min_delay)

        try:
            resp = await self.client.post(f"{IK_BASE_URL}/doc/{doc_id}/")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("ik_doc_error", doc_id=doc_id, error=str(e))
            await self.stats.increment(api_errors=1)
            return None

    # ── Parsing ───────────────────────────────────────────────────────────

    def parse_judgment(self, doc: dict, court_entry: CourtEntry) -> dict | None:
        """
        Parse an Indian Kanoon API response into a judgment dict.

        Enriches with court metadata from the registry.
        """
        try:
            parsed = self.parser.parse_indian_kanoon_doc(doc)
            doc_id = str(doc.get("tid", ""))

            # Determine court type
            if court_entry.court_type == CourtType.SUPREME_COURT:
                court_type = "SC"
                court_name = "Supreme Court"
            elif court_entry.court_type == CourtType.HIGH_COURT:
                court_type = "HC"
                court_name = court_entry.name
            else:
                court_type = "TRIBUNAL"
                court_name = court_entry.name

            return {
                "case_name": parsed.case_name,
                "court": court_name,
                "court_type": court_type,
                "court_short_name": court_entry.short_name,
                "judgment_date": parsed.judgment_date,
                "year": parsed.year or datetime.now().year,
                "citation_scc": parsed.citation_scc,
                "citation_air": parsed.citation_air,
                "indian_kanoon_id": doc_id,
                "domain": parsed.domain if hasattr(parsed, "domain") else None,
                "headnote": parsed.headnote,
                "facts": parsed.facts,
                "ratio_decidendi": parsed.ratio_decidendi,
                "sections_interpreted": parsed.sections_interpreted,
                "full_text": parsed.full_text,
                "bench": getattr(parsed, "bench", None),
                "bench_size": getattr(parsed, "bench_size", None),
                "source": "indian_kanoon",
                "source_url": f"https://indiankanoon.org/doc/{doc_id}/" if doc_id else None,
            }
        except ParseError as e:
            logger.debug("parse_error", error=str(e))
            return None
        except Exception as e:
            logger.debug("parse_error_unexpected", error=str(e))
            return None

    # ── Storage ───────────────────────────────────────────────────────────

    async def store_judgment(self, judgment_data: dict) -> bool:
        """
        Store a judgment in PostgreSQL with validation and deduplication.

        Deduplication key: (case_name, court, year) OR indian_kanoon_id.
        Returns True if stored (new), False if duplicate/error.
        """
        from sqlalchemy import select

        # Validate
        val_result = self.validator.validate_judgment(judgment_data)
        if not val_result.is_valid:
            await self.stats.increment(judgments_invalid=1)
            return False

        async with async_session() as session:
            try:
                # Dedup by indian_kanoon_id (most reliable)
                ik_id = judgment_data.get("indian_kanoon_id")
                if ik_id:
                    stmt = select(Judgment).where(Judgment.indian_kanoon_id == ik_id)
                    result = await session.execute(stmt)
                    if result.scalar_one_or_none():
                        await self.stats.increment(judgments_existing=1)
                        return False

                # Fallback dedup by case_name + year
                stmt = select(Judgment).where(
                    Judgment.case_name == judgment_data["case_name"],
                    Judgment.year == judgment_data["year"],
                )
                result = await session.execute(stmt)
                if result.scalar_one_or_none():
                    await self.stats.increment(judgments_existing=1)
                    return False

                # Insert new judgment
                new_judgment = Judgment(
                    id=uuid.uuid4(),
                    case_name=judgment_data["case_name"],
                    court=judgment_data.get("court"),
                    court_type=judgment_data.get("court_type"),
                    bench=judgment_data.get("bench"),
                    bench_size=judgment_data.get("bench_size"),
                    judgment_date=judgment_data.get("judgment_date"),
                    year=judgment_data["year"],
                    citation_scc=judgment_data.get("citation_scc"),
                    citation_air=judgment_data.get("citation_air"),
                    indian_kanoon_id=ik_id,
                    domain=judgment_data.get("domain"),
                    headnote=judgment_data.get("headnote"),
                    facts=judgment_data.get("facts"),
                    ratio_decidendi=judgment_data.get("ratio_decidendi"),
                    full_text=judgment_data.get("full_text"),
                    sections_interpreted=judgment_data.get("sections_interpreted"),
                    source=judgment_data.get("source", "indian_kanoon"),
                    source_url=judgment_data.get("source_url"),
                )
                session.add(new_judgment)
                await session.commit()
                await self.stats.increment(judgments_new=1)
                return True

            except Exception as e:
                await session.rollback()
                logger.error(
                    "store_judgment_error",
                    case=judgment_data.get("case_name", "?")[:60],
                    error=str(e),
                )
                await self.stats.increment(store_errors=1)
                return False

    # ── Scrape One Court-Year ─────────────────────────────────────────────

    async def scrape_court_year(
        self,
        court: CourtEntry,
        year: int,
        max_judgments: int = 0,
    ) -> int:
        """
        Scrape all judgments for a single court and year.

        Paginates through Indian Kanoon search results until exhausted
        or max_judgments reached.

        Returns the number of new judgments stored.
        """
        court_id = court.indian_kanoon_id
        log_ctx = {"court": court.short_name, "year": year}

        # Check checkpoint
        if await self.is_court_year_done(court_id, year):
            logger.info("court_year_skipped", reason="already_done", **log_ctx)
            await self.stats.increment(court_years_skipped=1)
            return 0

        # Build search query with court and year filter
        query = f"doctypes: judgments fromdate: {year}-01-01 todate: {year}-12-31"
        if court.indian_kanoon_doc_type == "highcourt":
            query = f"{query} court: {court_id}"

        new_count = 0
        page = 0
        consecutive_empty = 0
        max_pages = 500  # Safety limit: 500 pages × 10 = 5000 judgments/year max

        logger.info("court_year_start", pages_max=max_pages, **log_ctx)

        while page < max_pages:
            # Check cache first
            cached = self._read_cache(court_id, year, page)
            if cached:
                result = cached
            else:
                result = await self.search(query, page=page)
                if result:
                    self._write_cache(court_id, year, page, result)

            if not result or "docs" not in result:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                page += 1
                continue

            docs = result.get("docs", [])
            if not docs:
                break

            consecutive_empty = 0

            for doc in docs:
                parsed = self.parse_judgment(doc, court)
                if parsed:
                    await self.stats.increment(judgments_fetched=1)
                    was_new = await self.store_judgment(parsed)
                    if was_new:
                        new_count += 1

            # Check if we've hit the max for this court-year
            if max_judgments > 0 and new_count >= max_judgments:
                logger.info("court_year_max_reached", new=new_count, **log_ctx)
                break

            page += 1

        # Mark court-year as done
        fetched_total = page * IK_RESULTS_PER_PAGE
        await self.mark_court_year_done(court_id, year, fetched_total, new_count)
        await self.stats.increment(court_years_scraped=1)

        logger.info(
            "court_year_complete",
            pages=page,
            new_judgments=new_count,
            **log_ctx,
        )

        return new_count

    # ── Scrape One Court (All Years) ──────────────────────────────────────

    async def scrape_court(
        self,
        court: CourtEntry,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> int:
        """
        Scrape all years for a single court, sequentially.

        Sequential within a court to avoid overwhelming the API.
        Concurrency is across courts (via semaphore in scrape_all).
        """
        async with self.semaphore:
            await self.stats.increment(courts_attempted=1)

            sy = start_year or court.year_range_start
            ey = end_year or court.year_range_end
            target = court.target_judgments

            logger.info(
                "court_scrape_start",
                court=court.short_name,
                years=f"{sy}-{ey}",
                target=target,
            )

            total_new = 0
            years = list(range(ey, sy - 1, -1))  # Most recent first

            # Budget judgments roughly equally across years
            years_count = len(years)
            per_year_budget = max(target // years_count, 100) if years_count else 500

            for year in years:
                try:
                    new = await self.scrape_court_year(
                        court,
                        year,
                        max_judgments=per_year_budget,
                    )
                    total_new += new
                except Exception as e:
                    logger.error(
                        "court_year_error",
                        court=court.short_name,
                        year=year,
                        error=str(e),
                    )

                # If we've hit the overall target for this court, stop
                if total_new >= target:
                    logger.info(
                        "court_target_reached",
                        court=court.short_name,
                        total_new=total_new,
                        target=target,
                    )
                    break

            await self.stats.increment(courts_completed=1)

            logger.info(
                "court_scrape_complete",
                court=court.short_name,
                total_new=total_new,
                target=target,
            )

            return total_new

    # ── Scrape Multiple Courts ────────────────────────────────────────────

    async def scrape_courts(
        self,
        courts: list[CourtEntry],
        start_year: int | None = None,
        end_year: int | None = None,
        label: str = "bulk",
    ) -> dict:
        """
        Scrape multiple courts concurrently.

        Courts run in parallel (controlled by semaphore), but within
        each court, years are processed sequentially to avoid
        overwhelming Indian Kanoon.
        """
        total = len(courts)
        target_total = sum(c.target_judgments for c in courts)

        logger.info(
            "multi_court_scrape_start",
            label=label,
            courts=total,
            target_judgments=target_total,
            concurrency=self.concurrency,
        )

        # Create master ingestion log
        log_id = uuid.uuid4()
        async with async_session() as session:
            log = IngestionLog(
                id=log_id,
                source="indian_kanoon",
                task=f"scrape_{label}",
                started_at=datetime.utcnow(),
                status="running",
            )
            session.add(log)
            await session.commit()

        # Launch all courts concurrently
        tasks = [
            self.scrape_court(court, start_year, end_year)
            for court in courts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "court_scrape_exception",
                    court=courts[i].short_name,
                    error=str(result),
                )

        # Finalize master log
        stats = self.stats.to_dict()
        async with async_session() as session:
            from sqlalchemy import select

            stmt = select(IngestionLog).where(IngestionLog.id == log_id)
            result = await session.execute(stmt)
            log = result.scalar_one()
            log.completed_at = datetime.utcnow()
            log.status = "success" if stats["store_errors"] == 0 else "partial"
            log.items_fetched = stats["judgments_fetched"]
            log.items_new = stats["judgments_new"]
            log.items_failed = stats["store_errors"] + stats["api_errors"]
            log.last_success_at = datetime.utcnow()
            await session.commit()

        logger.info("multi_court_scrape_complete", label=label, **stats)
        return stats

    # ── Convenience Methods ───────────────────────────────────────────────

    async def scrape_by_priority(
        self,
        max_priority: str = "P0",
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict:
        """Scrape all courts up to the given priority tier."""
        courts = get_courts_for_ingestion(max_priority)
        return await self.scrape_courts(
            courts,
            start_year=start_year,
            end_year=end_year,
            label=f"priority_{max_priority}",
        )

    async def scrape_court_judgments(
        self,
        court: str = "supremecourt",
        query: str = "",
        max_pages: int = 5,
    ) -> dict:
        """
        Backward-compatible method from Phase 1.

        Preserved so existing pipeline.py calls still work.
        """
        logger.info(
            "legacy_scrape_court_judgments",
            court=court,
            max_pages=max_pages,
        )

        # Map legacy court string to registry entry
        court_entry = None
        if court == "supremecourt":
            court_entry = SUPREME_COURT
        else:
            for hc in ALL_HIGH_COURTS:
                if hc.indian_kanoon_id == court:
                    court_entry = hc
                    break

        if not court_entry:
            logger.error("legacy_court_not_found", court=court)
            return self.stats.to_dict()

        # Simple page-based scrape (legacy behavior)
        for page in range(max_pages):
            search_query = f"{query} doctypes: judgments" if query else "doctypes: judgments"
            result = await self.search(search_query, page=page)

            if not result or "docs" not in result:
                break

            docs = result.get("docs", [])
            if not docs:
                break

            for doc in docs:
                parsed = self.parse_judgment(doc, court_entry)
                if parsed:
                    await self.stats.increment(judgments_fetched=1)
                    await self.store_judgment(parsed)

        return self.stats.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# Seed Data — Landmark SC Judgments
#
# Preserved from Phase 1 for backward compatibility.
# ═══════════════════════════════════════════════════════════════════════════════

LANDMARK_JUDGMENTS = [
    {
        "case_name": "D.K. Basu v. State of West Bengal",
        "case_number": "Writ Petition (Crl.) No. 539 of 1986",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench": "Justice A.S. Anand, Justice M. Srinivasan",
        "bench_size": 2,
        "judgment_date": "1997-01-18",
        "year": 1997,
        "citation_scc": "(1997) 1 SCC 416",
        "citation_air": "AIR 1997 SC 610",
        "domain": "criminal",
        "headnote": (
            "The Supreme Court laid down 11 mandatory guidelines to be "
            "followed in all cases of arrest and detention to prevent "
            "custodial violence and torture."
        ),
        "ratio_decidendi": (
            "Custodial death is one of the worst crimes in a civilized "
            "society. The rights under Articles 21 and 22(1) of the "
            "Constitution are available to every citizen and must be "
            "scrupulously protected."
        ),
        "sections_interpreted": json.dumps([
            {"act": "Constitution of India", "section": "21"},
            {"act": "Constitution of India", "section": "22"},
            {"act": "CrPC", "section": "41"},
        ]),
        "source": "indian_kanoon",
    },
    {
        "case_name": "Lalita Kumari v. Govt. of UP",
        "case_number": "Writ Petition (Crl.) No. 68 of 2008",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 5,
        "judgment_date": "2013-11-12",
        "year": 2013,
        "citation_scc": "(2014) 2 SCC 1",
        "domain": "criminal",
        "headnote": (
            "Registration of FIR under Section 154 CrPC is mandatory "
            "when information discloses commission of a cognizable "
            "offence. Police cannot refuse to register an FIR."
        ),
        "ratio_decidendi": (
            "Registration of FIR is mandatory under Section 154 CrPC "
            "if the information discloses a cognizable offence. The "
            "police officer cannot conduct a preliminary inquiry before "
            "registering FIR in such cases."
        ),
        "sections_interpreted": json.dumps([
            {"act": "CrPC", "section": "154"},
            {"act": "CrPC", "section": "155"},
        ]),
        "source": "indian_kanoon",
    },
    {
        "case_name": "Arnesh Kumar v. State of Bihar",
        "case_number": "Criminal Appeal No. 1277 of 2014",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "2014-07-02",
        "year": 2014,
        "citation_scc": "(2014) 8 SCC 273",
        "domain": "criminal",
        "headnote": (
            "Automatic arrest under Section 498A IPC is not warranted. "
            "Police must follow the checklist under Section 41(1)(b)(ii) "
            "CrPC before making an arrest."
        ),
        "ratio_decidendi": (
            "In offences punishable with imprisonment up to seven years, "
            "police must satisfy themselves about the necessity of arrest "
            "under the parameters in Section 41 CrPC."
        ),
        "sections_interpreted": json.dumps([
            {"act": "IPC", "section": "498A"},
            {"act": "CrPC", "section": "41"},
        ]),
        "source": "indian_kanoon",
    },
    {
        "case_name": "Maneka Gandhi v. Union of India",
        "case_number": "Writ Petition No. 231 of 1977",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 7,
        "judgment_date": "1978-01-25",
        "year": 1978,
        "citation_scc": "(1978) 1 SCC 248",
        "citation_air": "AIR 1978 SC 597",
        "domain": "constitutional",
        "headnote": (
            "Article 21 protection is not limited to mere animal existence. "
            "The right to life includes the right to live with dignity."
        ),
        "ratio_decidendi": (
            "Article 21 requires that any procedure depriving a person of "
            "life or liberty must be fair, just and reasonable. Articles "
            "14, 19 and 21 are not mutually exclusive."
        ),
        "sections_interpreted": json.dumps([
            {"act": "Constitution of India", "section": "14"},
            {"act": "Constitution of India", "section": "19"},
            {"act": "Constitution of India", "section": "21"},
        ]),
        "source": "indian_kanoon",
    },
    {
        "case_name": "Vishaka v. State of Rajasthan",
        "case_number": "Writ Petition (Crl.) No. 666-70 of 1992",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 3,
        "judgment_date": "1997-08-13",
        "year": 1997,
        "citation_scc": "(1997) 6 SCC 241",
        "citation_air": "AIR 1997 SC 3011",
        "domain": "labour",
        "headnote": (
            "In the absence of legislation, the Supreme Court laid down "
            "guidelines for prevention of sexual harassment at workplace."
        ),
        "ratio_decidendi": (
            "Every employer has a duty to provide a safe working "
            "environment. Sexual harassment violates Articles 14, 15, "
            "19 and 21 of the Constitution."
        ),
        "sections_interpreted": json.dumps([
            {"act": "Constitution of India", "section": "14"},
            {"act": "Constitution of India", "section": "21"},
        ]),
        "source": "indian_kanoon",
    },
    {
        "case_name": "K.S. Puttaswamy v. Union of India",
        "case_number": "Writ Petition (Civil) No. 494 of 2012",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 9,
        "judgment_date": "2017-08-24",
        "year": 2017,
        "citation_scc": "(2017) 10 SCC 1",
        "domain": "constitutional",
        "headnote": (
            "Right to privacy is a fundamental right under Article 21 "
            "of the Constitution of India."
        ),
        "ratio_decidendi": (
            "Privacy is a constitutionally protected right which emerges "
            "from the guarantee of life and personal liberty in Article "
            "21. It is a natural right that inheres in all natural persons."
        ),
        "sections_interpreted": json.dumps([
            {"act": "Constitution of India", "section": "21"},
        ]),
        "source": "indian_kanoon",
    },
    {
        "case_name": "Navtej Singh Johar v. Union of India",
        "case_number": "Writ Petition (Criminal) No. 76 of 2016",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 5,
        "judgment_date": "2018-09-06",
        "year": 2018,
        "citation_scc": "(2018) 10 SCC 1",
        "domain": "constitutional",
        "headnote": (
            "Section 377 IPC was read down to decriminalise consensual "
            "sexual acts between adults in private."
        ),
        "ratio_decidendi": (
            "Consensual sexual acts of adults in private cannot be "
            "criminalised. Section 377 insofar as it criminalises "
            "consensual sexual conduct between adults is unconstitutional."
        ),
        "sections_interpreted": json.dumps([
            {"act": "IPC", "section": "377"},
            {"act": "Constitution of India", "section": "14"},
            {"act": "Constitution of India", "section": "21"},
        ]),
        "source": "indian_kanoon",
    },
    {
        "case_name": "Joseph Shine v. Union of India",
        "case_number": "Writ Petition (Criminal) No. 194 of 2017",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 5,
        "judgment_date": "2018-09-27",
        "year": 2018,
        "citation_scc": "(2019) 3 SCC 39",
        "domain": "criminal",
        "headnote": (
            "Section 497 IPC (adultery) struck down as unconstitutional. "
            "The provision treated women as property of their husbands."
        ),
        "ratio_decidendi": (
            "Section 497 IPC denudes women of their agency and treats "
            "them as chattel of their husbands. The provision violates "
            "Articles 14 and 21 of the Constitution."
        ),
        "sections_interpreted": json.dumps([
            {"act": "IPC", "section": "497"},
            {"act": "CrPC", "section": "198"},
            {"act": "Constitution of India", "section": "14"},
            {"act": "Constitution of India", "section": "21"},
        ]),
        "source": "indian_kanoon",
    },
]


async def seed_landmark_judgments() -> dict:
    """
    Seed the database with curated landmark SC judgments.

    Preserved from Phase 1 for backward compatibility.
    """
    logger.info("seed_landmark_judgments_start")
    stats = {"new": 0, "existing": 0, "errors": 0}
    validator = DataValidator()

    async with async_session() as session:
        from sqlalchemy import select

        for jdata in LANDMARK_JUDGMENTS:
            val_result = validator.validate_judgment(jdata)
            if not val_result.is_valid:
                logger.warning(
                    "seed_judgment_invalid",
                    case=jdata.get("case_name"),
                    errors=val_result.errors,
                )
                continue

            # Dedup
            stmt = select(Judgment).where(
                Judgment.case_name == jdata["case_name"],
                Judgment.year == jdata["year"],
            )
            result = await session.execute(stmt)
            if result.scalar_one_or_none():
                stats["existing"] += 1
                continue

            try:
                # Convert date string to date object for PostgreSQL
                jdate = jdata.get("judgment_date")
                if isinstance(jdate, str):
                    try:
                        jdate = datetime.strptime(jdate, "%Y-%m-%d").date()
                    except ValueError:
                        jdate = None

                new_j = Judgment(
                    id=uuid.uuid4(),
                    case_name=jdata["case_name"],
                    case_number=jdata.get("case_number"),
                    court=jdata.get("court"),
                    court_type=jdata.get("court_type"),
                    bench=jdata.get("bench"),
                    bench_size=jdata.get("bench_size"),
                    judgment_date=jdate,
                    year=jdata["year"],
                    citation_scc=jdata.get("citation_scc"),
                    citation_air=jdata.get("citation_air"),
                    domain=jdata.get("domain"),
                    headnote=jdata.get("headnote"),
                    facts=jdata.get("facts"),
                    ratio_decidendi=jdata.get("ratio_decidendi"),
                    sections_interpreted=jdata.get("sections_interpreted"),
                    source=jdata.get("source", "indian_kanoon"),
                    source_url=jdata.get("source_url"),
                )
                session.add(new_j)
                stats["new"] += 1
            except Exception as e:
                logger.error("seed_judgment_error", case=jdata["case_name"], error=str(e))
                stats["errors"] += 1

        await session.commit()

    logger.info("seed_landmark_judgments_complete", **stats)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Run the Indian Kanoon client with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="NyayaMitra Indian Kanoon Client (Sprint 7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m data.scrapers.indian_kanoon --seed-only
  python -m data.scrapers.indian_kanoon --scrape --priority P0
  python -m data.scrapers.indian_kanoon --scrape --priority P1
  python -m data.scrapers.indian_kanoon --scrape --court SC
  python -m data.scrapers.indian_kanoon --scrape --court "DEL HC"
  python -m data.scrapers.indian_kanoon --scrape --priority P0 --start-year 2022
  python -m data.scrapers.indian_kanoon --scrape --priority P0 --dry-run
        """,
    )
    parser.add_argument("--seed-only", action="store_true", help="Only seed landmark judgments")
    parser.add_argument("--scrape", action="store_true", help="Scrape from Indian Kanoon API")
    parser.add_argument("--priority", choices=["P0", "P1", "P2"], default=None,
                        help="Court priority tier (P0=SC+top10 HCs, P1=all HCs, P2=+tribunals)")
    parser.add_argument("--court", type=str, default=None,
                        help="Single court short name (e.g., SC, 'DEL HC', 'BOM HC')")
    parser.add_argument("--start-year", type=int, default=None, help="Start year (default: from registry)")
    parser.add_argument("--end-year", type=int, default=None, help="End year (default: from registry)")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent courts (default: 2)")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoints, re-scrape everything")
    parser.add_argument("--no-cache", action="store_true", help="Disable API response caching")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")

    args = parser.parse_args()

    # Always seed first
    seed_stats = await seed_landmark_judgments()
    print(f"\nSeed data: {seed_stats['new']} new, {seed_stats['existing']} existing landmark judgments")

    if args.seed_only:
        return

    if not args.scrape:
        print("\nUse --scrape to fetch from Indian Kanoon API. See --help for options.")
        return

    # Check API token
    token = getattr(settings, "INDIAN_KANOON_API_TOKEN", "")
    if not token:
        print("\nERROR: Set INDIAN_KANOON_API_TOKEN in .env to use the API")
        print("Get a token from: https://api.indiankanoon.org/")
        return

    # Determine target courts
    if args.court:
        court_entry = get_court_by_short_name(args.court)
        if not court_entry:
            print(f"\nError: Court '{args.court}' not found.")
            print("Available courts:")
            for c in get_courts_for_ingestion("P2"):
                print(f"  {c.short_name:<12} {c.name}")
            return
        target_courts = [court_entry]
        label = f"single_{args.court.replace(' ', '_')}"
    elif args.priority:
        target_courts = get_courts_for_ingestion(args.priority)
        label = f"priority_{args.priority}"
    else:
        print("\nSpecify --priority or --court. See --help.")
        return

    # Dry run
    if args.dry_run:
        total_target = sum(c.target_judgments for c in target_courts)
        print(f"\n{'=' * 70}")
        print(f"  DRY RUN — Would scrape {len(target_courts)} courts, ~{total_target:,} target judgments")
        print(f"{'=' * 70}\n")
        print(f"  {'Court':<35} {'Type':<6} {'Years':<12} {'Target':>8}")
        print("  " + "─" * 65)
        for c in target_courts:
            sy = args.start_year or c.year_range_start
            ey = args.end_year or c.year_range_end
            print(f"  {c.name:<35} {c.court_type.value:<6} {sy}-{ey:<6} {c.target_judgments:>8,}")
        print("  " + "─" * 65)
        print(f"  {'TOTAL':<35} {'':6} {'':12} {total_target:>8,}")
        print(f"\n  Concurrency: {args.concurrency}")
        print(f"  Force: {args.force}")
        print(f"  Cache: {'disabled' if args.no_cache else 'enabled'}")
        return

    # Execute
    print(f"\nScraping {len(target_courts)} courts (concurrency={args.concurrency})...\n")

    async with IndianKanoonClient(
        concurrency=args.concurrency,
        force=args.force,
        cache_enabled=not args.no_cache,
    ) as client:
        stats = await client.scrape_courts(
            target_courts,
            start_year=args.start_year,
            end_year=args.end_year,
            label=label,
        )

    # Print report
    print(f"\n{'=' * 60}")
    print(f"  Indian Kanoon Scraper — Results")
    print(f"{'=' * 60}")
    print(f"  Courts attempted:     {stats['courts_attempted']}")
    print(f"  Courts completed:     {stats['courts_completed']}")
    print(f"  Court-years scraped:  {stats['court_years_scraped']}")
    print(f"  Court-years skipped:  {stats['court_years_skipped']} (checkpoint)")
    print(f"  Pages fetched:        {stats['pages_fetched']}")
    print(f"  Judgments fetched:    {stats['judgments_fetched']}")
    print(f"  Judgments new:        {stats['judgments_new']}")
    print(f"  Judgments existing:   {stats['judgments_existing']} (dedup)")
    print(f"  Judgments invalid:    {stats['judgments_invalid']}")
    print(f"  API errors:           {stats['api_errors']}")
    print(f"  Store errors:         {stats['store_errors']}")
    print(f"  Duration:             {stats['duration_seconds']}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())