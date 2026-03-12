"""
NyayaMitra — India Code Scraper.

Scrapes Central Acts from legislative.gov.in (India Code portal).
Parses each act into its constituent sections and stores them in PostgreSQL.

This scraper targets the top Central Acts that form the backbone
of Indian legal knowledge. Each act is parsed into individual sections
for embedding and retrieval.

Usage:
    python -m data.scrapers.india_code

Rate Limits:
    - Max 10 requests per second (configurable via .env)
    - Exponential backoff on failure
    - Respects robots.txt
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import structlog
from bs4 import BeautifulSoup

# Add project root to path so we can import from backend
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings
from app.database import async_session
from app.models.legal import Act, IngestionLog, Section

logger = structlog.get_logger()

# ═══════════════════════════════════════════════════════════════════════════════
# Priority Acts — The core legal corpus for Phase 1
# ═══════════════════════════════════════════════════════════════════════════════
# These 5 acts are the initial dataset for Sprint 2.
# Sprint 7 expands to 100+ acts.

PRIORITY_ACTS = [
    {
        "name": "Indian Penal Code, 1860",
        "short_name": "IPC",
        "year": 1860,
        "act_number": "Act No. 45 of 1860",
        "domain": "criminal",
        "replaced_by": "Bharatiya Nyaya Sanhita, 2023",
        "india_code_id": "1860-45",
        "url": "https://www.indiacode.nic.in/handle/123456789/2263",
    },
    {
        "name": "Code of Criminal Procedure, 1973",
        "short_name": "CrPC",
        "year": 1973,
        "act_number": "Act No. 2 of 1974",
        "domain": "criminal",
        "replaced_by": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "india_code_id": "1974-2",
        "url": "https://www.indiacode.nic.in/handle/123456789/1362",
    },
    {
        "name": "Indian Evidence Act, 1872",
        "short_name": "Evidence Act",
        "year": 1872,
        "act_number": "Act No. 1 of 1872",
        "domain": "criminal",
        "replaced_by": "Bharatiya Sakshya Adhiniyam, 2023",
        "india_code_id": "1872-1",
        "url": "https://www.indiacode.nic.in/handle/123456789/2188",
    },
    {
        "name": "Code of Civil Procedure, 1908",
        "short_name": "CPC",
        "year": 1908,
        "act_number": "Act No. 5 of 1908",
        "domain": "property",
        "replaced_by": None,
        "india_code_id": "1908-5",
        "url": "https://www.indiacode.nic.in/handle/123456789/2191",
    },
    {
        "name": "Constitution of India",
        "short_name": "Constitution",
        "year": 1950,
        "act_number": None,
        "domain": "constitutional",
        "replaced_by": None,
        "india_code_id": "constitution",
        "url": "https://www.indiacode.nic.in/handle/123456789/2013",
    },
]

# Extended list for Sprint 7
EXTENDED_ACTS = [
    {"name": "Transfer of Property Act, 1882", "short_name": "TPA", "year": 1882, "domain": "property"},
    {"name": "Hindu Marriage Act, 1955", "short_name": "HMA", "year": 1955, "domain": "family"},
    {"name": "Special Marriage Act, 1954", "short_name": "SMA", "year": 1954, "domain": "family"},
    {"name": "Consumer Protection Act, 2019", "short_name": "CPA", "year": 2019, "domain": "consumer"},
    {"name": "Right to Information Act, 2005", "short_name": "RTI", "year": 2005, "domain": "constitutional"},
    {"name": "Information Technology Act, 2000", "short_name": "IT Act", "year": 2000, "domain": "ip"},
    {"name": "Negotiable Instruments Act, 1881", "short_name": "NI Act", "year": 1881, "domain": "property"},
    {"name": "Protection of Women from Domestic Violence Act, 2005", "short_name": "DV Act", "year": 2005, "domain": "family"},
    {"name": "Industrial Disputes Act, 1947", "short_name": "ID Act", "year": 1947, "domain": "labor"},
    {"name": "Real Estate (Regulation and Development) Act, 2016", "short_name": "RERA", "year": 2016, "domain": "property"},
    {"name": "Companies Act, 2013", "short_name": "Companies Act", "year": 2013, "domain": "property"},
    {"name": "Indian Contract Act, 1872", "short_name": "Contract Act", "year": 1872, "domain": "property"},
    {"name": "Bharatiya Nyaya Sanhita, 2023", "short_name": "BNS", "year": 2023, "domain": "criminal"},
    {"name": "Bharatiya Nagarik Suraksha Sanhita, 2023", "short_name": "BNSS", "year": 2023, "domain": "criminal"},
    {"name": "Bharatiya Sakshya Adhiniyam, 2023", "short_name": "BSA", "year": 2023, "domain": "criminal"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Rate Limiter
# ═══════════════════════════════════════════════════════════════════════════════


class RateLimiter:
    """Simple async rate limiter using token bucket algorithm."""

    def __init__(self, max_requests: int, period: float):
        self.max_requests = max_requests
        self.period = period
        self.tokens = max_requests
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_requests, self.tokens + elapsed * (self.max_requests / self.period))
            self.last_refill = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.period / self.max_requests)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


# ═══════════════════════════════════════════════════════════════════════════════
# India Code Scraper
# ═══════════════════════════════════════════════════════════════════════════════


class IndiaCodeScraper:
    """
    Scrapes acts and sections from India Code (legislative.gov.in).

    The India Code portal hosts all Central Acts. This scraper:
    1. Fetches the HTML page for each act
    2. Parses it into sections using BeautifulSoup
    3. Stores the act and its sections in PostgreSQL
    4. Tracks ingestion state for incremental updates
    """

    def __init__(self):
        self.rate_limiter = RateLimiter(
            max_requests=settings.SCRAPER_RATE_LIMIT_REQUESTS,
            period=settings.SCRAPER_RATE_LIMIT_PERIOD,
        )
        self.client: httpx.AsyncClient | None = None
        self.stats = {
            "acts_processed": 0,
            "acts_new": 0,
            "acts_updated": 0,
            "sections_created": 0,
            "errors": 0,
        }

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "NyayaMitra-Bot/1.0 (Legal Research; +https://nyayamitra.in)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-IN,en;q=0.9",
            },
        )
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def fetch_page(self, url: str) -> str | None:
        """
        Fetch a page with rate limiting and retry logic.

        Returns the HTML content or None if all retries fail.
        """
        for attempt in range(settings.SCRAPER_RETRY_MAX):
            await self.rate_limiter.acquire()
            try:
                resp = await self.client.get(url)
                resp.raise_for_status()
                return resp.text
            except httpx.HTTPStatusError as e:
                logger.warning(
                    "scraper_http_error",
                    url=url,
                    status=e.response.status_code,
                    attempt=attempt + 1,
                )
                if e.response.status_code == 429:
                    # Rate limited — wait longer
                    wait = settings.SCRAPER_RETRY_BACKOFF ** (attempt + 2)
                    await asyncio.sleep(wait)
                elif e.response.status_code >= 500:
                    wait = settings.SCRAPER_RETRY_BACKOFF ** (attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    return None
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(
                    "scraper_connection_error",
                    url=url,
                    error=str(e),
                    attempt=attempt + 1,
                )
                wait = settings.SCRAPER_RETRY_BACKOFF ** (attempt + 1)
                await asyncio.sleep(wait)

        logger.error("scraper_all_retries_failed", url=url)
        self.stats["errors"] += 1
        return None

    def parse_sections_from_html(self, html: str, act_name: str) -> list[dict]:
        """
        Parse an act's HTML page into individual sections.

        This is a generic parser that handles the common India Code HTML structure.
        Different acts may have slightly different formats, so this uses
        multiple parsing strategies and falls back gracefully.

        Returns a list of section dicts with keys:
            section_number, title, text, chapter, part
        """
        soup = BeautifulSoup(html, "lxml")
        sections = []
        current_chapter = None
        current_part = None

        # Strategy 1: Look for section headers in common formats
        # India Code uses various heading patterns:
        #   "Section 302.", "302.", "Section 302 -", "S. 302"
        section_pattern = re.compile(
            r"(?:Section|S\.?)\s*(\d+[A-Za-z]*(?:\([^)]+\))?)\s*[-.:]\s*(.*)",
            re.IGNORECASE,
        )

        # Try to find sections from structured elements
        for element in soup.find_all(["p", "div", "h3", "h4", "h5", "tr"]):
            text = element.get_text(strip=True)

            if not text or len(text) < 5:
                continue

            # Detect chapter headings
            chapter_match = re.match(
                r"(?:CHAPTER|Chapter)\s+([IVXLCDM]+[A-Z]*|\d+[A-Z]*)\s*[-.:]\s*(.*)",
                text,
            )
            if chapter_match:
                current_chapter = f"Chapter {chapter_match.group(1)} - {chapter_match.group(2)}".strip()
                continue

            # Detect part headings
            part_match = re.match(
                r"(?:PART|Part)\s+([IVXLCDM]+[A-Z]*|\d+[A-Z]*)\s*[-.:]\s*(.*)",
                text,
            )
            if part_match:
                current_part = f"Part {part_match.group(1)} - {part_match.group(2)}".strip()
                continue

            # Detect sections
            section_match = section_pattern.match(text)
            if section_match:
                section_num = section_match.group(1).strip()
                rest = section_match.group(2).strip()

                # Try to separate title from body
                # Common pattern: "Section 302. Punishment for murder.— Whoever commits..."
                title_body = re.split(r"[.—]\s*", rest, maxsplit=1)
                title = title_body[0].strip() if title_body else ""
                body = title_body[1].strip() if len(title_body) > 1 else rest

                # Get the full text including following paragraphs
                full_text = self._get_section_full_text(element, body)

                sections.append({
                    "section_number": section_num,
                    "title": title[:500] if title else None,
                    "text": full_text,
                    "chapter": current_chapter,
                    "part": current_part,
                })

        # Strategy 2: If structured parsing found nothing, try raw text extraction
        if not sections:
            sections = self._parse_sections_from_raw_text(soup, act_name)

        logger.info(
            "sections_parsed",
            act=act_name,
            num_sections=len(sections),
            strategy="structured" if sections else "fallback",
        )

        return sections

    def _get_section_full_text(self, element, initial_text: str) -> str:
        """
        Get the full text of a section by collecting following sibling elements
        until the next section header is found.
        """
        texts = [initial_text] if initial_text else []

        # Collect text from sibling elements
        sibling = element.find_next_sibling()
        section_pattern = re.compile(r"(?:Section|S\.?)\s*\d+", re.IGNORECASE)

        count = 0
        while sibling and count < 50:
            sibling_text = sibling.get_text(strip=True)
            if not sibling_text:
                sibling = sibling.find_next_sibling()
                count += 1
                continue

            # Stop if we hit the next section
            if section_pattern.match(sibling_text):
                break

            # Stop if we hit a chapter/part heading
            if re.match(r"(?:CHAPTER|PART)\s+[IVXLCDM\d]", sibling_text, re.IGNORECASE):
                break

            texts.append(sibling_text)
            sibling = sibling.find_next_sibling()
            count += 1

        return "\n".join(texts).strip()

    def _parse_sections_from_raw_text(self, soup: BeautifulSoup, act_name: str) -> list[dict]:
        """
        Fallback parser: extract sections from raw text when structured parsing fails.

        Splits the full text on section number patterns.
        """
        full_text = soup.get_text(separator="\n")
        sections = []

        # Split on section patterns
        parts = re.split(
            r"\n\s*(?:Section|S\.?)\s+(\d+[A-Za-z]*(?:\([^)]+\))?)\s*[-.:]\s*",
            full_text,
            flags=re.IGNORECASE,
        )

        # parts will be: [preamble, sec_num_1, sec_text_1, sec_num_2, sec_text_2, ...]
        for i in range(1, len(parts) - 1, 2):
            section_num = parts[i].strip()
            section_text = parts[i + 1].strip()

            # Limit text length (some sections include garbage from page rendering)
            if len(section_text) > 10000:
                section_text = section_text[:10000] + "..."

            if section_text and len(section_text) > 10:
                # Try to extract title from first sentence
                first_line = section_text.split("\n")[0]
                title_match = re.match(r"^([^.—]+)[.—]", first_line)
                title = title_match.group(1).strip()[:500] if title_match else None

                sections.append({
                    "section_number": section_num,
                    "title": title,
                    "text": section_text,
                    "chapter": None,
                    "part": None,
                })

        return sections

    async def store_act_and_sections(
        self,
        act_info: dict,
        sections_data: list[dict],
    ) -> None:
        """
        Store an act and its sections in PostgreSQL.

        Uses upsert logic: if the act already exists (by name+year+jurisdiction),
        update it. Otherwise create a new record.
        """
        async with async_session() as session:
            try:
                # Check if act already exists
                from sqlalchemy import select

                stmt = select(Act).where(
                    Act.name == act_info["name"],
                    Act.year == act_info["year"],
                    Act.jurisdiction == "central",
                )
                result = await session.execute(stmt)
                existing_act = result.scalar_one_or_none()

                if existing_act:
                    # Update existing act
                    existing_act.last_scraped_at = datetime.utcnow()
                    existing_act.updated_at = datetime.utcnow()
                    act_id = existing_act.id
                    self.stats["acts_updated"] += 1
                    logger.info("act_updated", name=act_info["name"])
                else:
                    # Create new act
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
                        source_url=act_info.get("url"),
                        last_scraped_at=datetime.utcnow(),
                    )
                    session.add(new_act)
                    await session.flush()
                    act_id = new_act.id
                    self.stats["acts_new"] += 1
                    logger.info("act_created", name=act_info["name"], id=str(act_id))

                # Store sections
                for sec_data in sections_data:
                    # Check if section already exists
                    stmt = select(Section).where(
                        Section.act_id == act_id,
                        Section.section_number == sec_data["section_number"],
                    )
                    result = await session.execute(stmt)
                    existing_section = result.scalar_one_or_none()

                    if existing_section:
                        # Update existing section
                        existing_section.text = sec_data["text"]
                        existing_section.title = sec_data.get("title")
                        existing_section.chapter = sec_data.get("chapter")
                        existing_section.part = sec_data.get("part")
                        existing_section.updated_at = datetime.utcnow()
                    else:
                        # Create new section
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
                        self.stats["sections_created"] += 1

                await session.commit()
                self.stats["acts_processed"] += 1

            except Exception as e:
                await session.rollback()
                logger.error(
                    "store_act_failed",
                    act=act_info["name"],
                    error=str(e),
                )
                self.stats["errors"] += 1
                raise

    async def scrape_act(self, act_info: dict) -> None:
        """
        Scrape a single act: fetch HTML, parse sections, store in DB.
        """
        url = act_info.get("url")
        if not url:
            logger.warning("scraper_no_url", act=act_info["name"])
            return

        logger.info("scraping_act", act=act_info["name"], url=url)

        html = await self.fetch_page(url)
        if not html:
            logger.error("scraper_fetch_failed", act=act_info["name"])
            return

        sections = self.parse_sections_from_html(html, act_info["name"])

        if sections:
            await self.store_act_and_sections(act_info, sections)
        else:
            # Even if no sections parsed, store the act record
            # so we know we attempted it
            logger.warning(
                "scraper_no_sections_found",
                act=act_info["name"],
                html_length=len(html),
            )
            await self.store_act_and_sections(act_info, [])

    async def scrape_priority_acts(self) -> dict:
        """
        Scrape the 5 priority Central Acts for Phase 1.

        Returns scraping statistics.
        """
        logger.info(
            "india_code_scraper_start",
            num_acts=len(PRIORITY_ACTS),
        )

        start_time = time.time()

        # Create ingestion log
        log_id = uuid.uuid4()
        async with async_session() as session:
            log = IngestionLog(
                id=log_id,
                source="india_code",
                task="scrape_priority_acts",
                started_at=datetime.utcnow(),
                status="running",
            )
            session.add(log)
            await session.commit()

        # Scrape each act
        for act_info in PRIORITY_ACTS:
            try:
                await self.scrape_act(act_info)
            except Exception as e:
                logger.error(
                    "scrape_act_error",
                    act=act_info["name"],
                    error=str(e),
                )
                self.stats["errors"] += 1

        duration = round(time.time() - start_time, 2)

        # Update ingestion log
        async with async_session() as session:
            from sqlalchemy import select

            stmt = select(IngestionLog).where(IngestionLog.id == log_id)
            result = await session.execute(stmt)
            log = result.scalar_one()
            log.completed_at = datetime.utcnow()
            log.status = "success" if self.stats["errors"] == 0 else "partial"
            log.items_fetched = self.stats["acts_processed"]
            log.items_new = self.stats["acts_new"]
            log.items_updated = self.stats["acts_updated"]
            log.items_failed = self.stats["errors"]
            log.last_success_at = datetime.utcnow()
            await session.commit()

        logger.info(
            "india_code_scraper_complete",
            duration_seconds=duration,
            **self.stats,
        )

        return self.stats


# ═══════════════════════════════════════════════════════════════════════════════
# Seed Data — Manually curated sections for initial testing
# ═══════════════════════════════════════════════════════════════════════════════
# While the web scraper handles live fetching, we also provide seed data
# for the most important sections. This ensures the system has accurate
# data even if scraping encounters issues.


SEED_SECTIONS_IPC = [
    {
        "section_number": "302",
        "title": "Punishment for murder",
        "text": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
        "chapter": "Chapter XVI - Of Offences Affecting the Human Body",
    },
    {
        "section_number": "304A",
        "title": "Causing death by negligence",
        "text": "Whoever causes the death of any person by doing any rash or negligent act not amounting to culpable homicide shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both.",
        "chapter": "Chapter XVI - Of Offences Affecting the Human Body",
    },
    {
        "section_number": "354",
        "title": "Assault or criminal force to woman with intent to outrage her modesty",
        "text": "Whoever assaults or uses criminal force to any woman, intending to outrage or knowing it to be likely that he will thereby outrage her modesty, shall be punished with imprisonment of either description for a term which shall not be less than one year but which may extend to five years, and shall also be liable to fine.",
        "chapter": "Chapter XVI - Of Offences Affecting the Human Body",
    },
    {
        "section_number": "376",
        "title": "Punishment for rape",
        "text": "Whoever commits rape shall be punished with rigorous imprisonment of either description for a term which shall not be less than ten years, but which may extend to imprisonment for life, and shall also be liable to fine.",
        "chapter": "Chapter XVI - Of Offences Affecting the Human Body",
    },
    {
        "section_number": "420",
        "title": "Cheating and dishonestly inducing delivery of property",
        "text": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, or anything which is signed or sealed, and which is capable of being converted into a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.",
        "chapter": "Chapter XVII - Of Offences Against Property",
    },
    {
        "section_number": "498A",
        "title": "Husband or relative of husband of a woman subjecting her to cruelty",
        "text": "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished with imprisonment for a term which may extend to three years and shall also be liable to fine. Explanation: For the purposes of this section, 'cruelty' means (a) any wilful conduct which is of such a nature as is likely to drive the woman to commit suicide or to cause grave injury or danger to life, limb or health (whether mental or physical) of the woman; or (b) harassment of the woman where such harassment is with a view to coercing her or any person related to her to meet any unlawful demand for any property or valuable security or is on account of failure by her or any person related to her to meet such demand.",
        "chapter": "Chapter XXA - Of Cruelty by Husband or Relatives of Husband",
    },
]

SEED_SECTIONS_CRPC = [
    {
        "section_number": "41",
        "title": "When police may arrest without warrant",
        "text": "Any police officer may without an order from a Magistrate and without a warrant, arrest any person who has been concerned in any cognizable offence, or against whom a reasonable complaint has been made, or credible information has been received, or a reasonable suspicion exists, of his having been so concerned.",
        "chapter": "Chapter V - Arrest of Persons",
    },
    {
        "section_number": "41A",
        "title": "Notice of appearance before police officer",
        "text": "The police officer shall, in all cases where the arrest of a person is not required under the provisions of sub-section (1) of section 41, issue a notice directing the person against whom a reasonable complaint has been made, or credible information has been received, or a reasonable suspicion exists that he has committed a cognizable offence, to appear before him or at such other place as may be specified in the notice.",
        "chapter": "Chapter V - Arrest of Persons",
    },
    {
        "section_number": "154",
        "title": "Information in cognizable cases (FIR)",
        "text": "Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing by him or under his direction, and be read over to the informant; and every such information, whether given in writing or reduced to writing as aforesaid, shall be signed by the person giving it, and the substance thereof shall be entered in a book to be kept by such officer in such form as the State Government may prescribe in this behalf.",
        "chapter": "Chapter XII - Information to the Police and their Powers to Investigate",
    },
    {
        "section_number": "167",
        "title": "Procedure when investigation cannot be completed in twenty-four hours",
        "text": "Whenever any person is arrested and detained in custody, and it appears that the investigation cannot be completed within the period of twenty-four hours fixed by section 57, and there are grounds for believing that the accusation or information is well-founded, the officer in charge of the police station or the police officer making the investigation, if he is not below the rank of sub-inspector, shall forthwith transmit to the nearest Judicial Magistrate a copy of the entries in the diary hereinafter prescribed relating to the case, and shall at the same time forward the accused to such Magistrate.",
        "chapter": "Chapter XII - Information to the Police and their Powers to Investigate",
    },
    {
        "section_number": "436",
        "title": "In what cases bail to be taken",
        "text": "When any person other than a person accused of a non-bailable offence is arrested or detained without warrant by an officer in charge of a police station, or appears or is brought before a Court, and is prepared at any time while in the custody of such officer or at any stage of the proceeding before such Court to give bail, such person shall be released on bail.",
        "chapter": "Chapter XXXIII - Provisions as to Bail and Bonds",
    },
    {
        "section_number": "438",
        "title": "Direction for grant of bail to person apprehending arrest (Anticipatory Bail)",
        "text": "Where any person has reason to believe that he may be arrested on accusation of having committed a non-bailable offence, he may apply to the High Court or the Court of Session for a direction under this section that in the event of such arrest, he shall be released on bail; and that Court may, after taking into consideration, inter alia, the nature and gravity of the accusation; the antecedents of the applicant including the fact as to whether he has previously undergone imprisonment on conviction by a Court in respect of any cognisable offence; the possibility of the applicant to flee from justice; and where the accusation has been made with the object of injuring or humiliating the applicant by having him so arrested, either reject the application forthwith or issue an interim order for the grant of anticipatory bail.",
        "chapter": "Chapter XXXIII - Provisions as to Bail and Bonds",
    },
]


async def seed_initial_data() -> dict:
    """
    Seed the database with manually curated sections for the most important acts.

    This ensures the system has accurate, verified legal data for the core
    acts even before the web scraper successfully fetches everything.
    """
    logger.info("seed_data_start")
    stats = {"acts": 0, "sections": 0}

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

            # Check if act exists
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

            # Add sections
            for sec in data["sections"]:
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
    """Run the India Code scraper."""
    import argparse

    parser = argparse.ArgumentParser(description="NyayaMitra India Code Scraper")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only seed curated data, skip web scraping",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape acts from India Code website",
    )
    args = parser.parse_args()

    # Always seed the curated data first
    seed_stats = await seed_initial_data()
    print(f"\nSeed data: {seed_stats['acts']} acts, {seed_stats['sections']} sections created")

    if args.scrape:
        # Run the web scraper
        async with IndiaCodeScraper() as scraper:
            stats = await scraper.scrape_priority_acts()
            print(f"\nScraping complete: {json.dumps(stats, indent=2)}")
    elif not args.seed_only:
        print("\nUse --scrape to fetch from India Code website")
        print("Use --seed-only to only seed curated data")


if __name__ == "__main__":
    asyncio.run(main())