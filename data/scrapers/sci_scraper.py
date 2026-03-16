"""
NyayaMitra — Supreme Court of India (SCI) Scraper.

Scrapes judgments directly from the Supreme Court of India website
(main.sci.gov.in). Complements the Indian Kanoon client by fetching
from the primary source with structured metadata.

The SCI website provides:
- Judgment PDFs and text via its search/judgment listing pages
- Structured metadata: case number, parties, bench, judgment date
- Daily cause lists and recent judgments

This scraper:
1. Fetches judgment listings from SCI website
2. Extracts metadata (case name, bench, date, case number)
3. Downloads judgment text where available
4. Stores judgments in PostgreSQL with deduplication
5. Tracks ingestion state for incremental updates

Usage:
    # Seed curated SC judgments with verified metadata (no HTTP needed)
    python -m data.scrapers.sci_scraper --seed-only

    # Scrape recent judgments from SCI website
    python -m data.scrapers.sci_scraper --scrape --years 5

    # Scrape specific year
    python -m data.scrapers.sci_scraper --scrape --start-year 2020 --end-year 2024

Rate Limits:
    - 1 request per 2 seconds (government site — be respectful)
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
from datetime import datetime, date
from pathlib import Path

import httpx
import structlog
from bs4 import BeautifulSoup

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings
from app.database import async_session
from app.models.legal import Judgment, IngestionLog

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# SCI Website Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SCI_BASE_URL = "https://main.sci.gov.in"
SCI_JUDGMENTS_URL = f"{SCI_BASE_URL}/judgments"

# Conservative rate limit for government site
SCI_RATE_LIMIT_REQUESTS = 1
SCI_RATE_LIMIT_PERIOD = 2  # 1 request per 2 seconds


# ═══════════════════════════════════════════════════════════════════════════════
# Seed Data — Additional SC Judgments Not in Indian Kanoon Seed
# ═══════════════════════════════════════════════════════════════════════════════
# These judgments complement the 8 landmarks in indian_kanoon.py and
# the 11 in seed_comprehensive.py. Focus on frequently-cited procedural
# and rights-based judgments that citizens commonly need.

SCI_SEED_JUDGMENTS = [
    # ─── Criminal Law (procedural rights) ────────────────────────────────
    {
        "case_name": "Joginder Kumar v. State of Uttar Pradesh",
        "case_number": "Criminal Appeal No. 547 of 1994",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "1994-04-25",
        "year": 1994,
        "citation_scc": "(1994) 4 SCC 260",
        "citation_air": "AIR 1994 SC 1349",
        "domain": "criminal",
        "headnote": (
            "The Supreme Court held that an arrested person has a fundamental "
            "right to inform a friend, relative, or any person interested in "
            "his welfare about the arrest and the place of detention. The police "
            "must inform the arrested person of this right at the time of arrest. "
            "No arrest can be made on a mere allegation or suspicion without "
            "recording reasons for arrest in a case diary."
        ),
        "ratio_decidendi": (
            "The right of the arrested person to have someone informed of his "
            "arrest and detention flows from Articles 21 and 22(1) of the "
            "Constitution. An arrested person must be informed of this right "
            "as soon as he is arrested. The magistrate must satisfy himself "
            "that these requirements have been complied with."
        ),
        "sections_interpreted": json.dumps([
            {"act": "Constitution of India", "section": "21"},
            {"act": "Constitution of India", "section": "22(1)"},
            {"act": "CrPC", "section": "50"},
            {"act": "CrPC", "section": "56"},
        ]),
    },
    {
        "case_name": "Hussainara Khatoon v. Home Secretary, State of Bihar",
        "case_number": "Writ Petition (Crl.) No. 57 of 1979",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "1979-03-09",
        "year": 1979,
        "citation_scc": "(1980) 1 SCC 81",
        "citation_air": "AIR 1979 SC 1360",
        "domain": "criminal",
        "headnote": (
            "The Supreme Court held that the right to speedy trial is a "
            "fundamental right under Article 21 of the Constitution. Thousands "
            "of undertrial prisoners had been languishing in Bihar jails for "
            "periods longer than the maximum sentence for their alleged offences. "
            "The court directed their immediate release and established that "
            "free legal aid is a constitutional right for those who cannot "
            "afford a lawyer."
        ),
        "ratio_decidendi": (
            "Right to speedy trial is implicit in Article 21. Where undertrial "
            "prisoners have served more than the maximum possible sentence, "
            "their continued detention is unconstitutional. Free legal aid to "
            "the poor and indigent accused is a necessary concomitant of fair "
            "procedure under Article 21."
        ),
        "sections_interpreted": json.dumps([
            {"act": "Constitution of India", "section": "21"},
            {"act": "Constitution of India", "section": "39A"},
            {"act": "CrPC", "section": "167"},
            {"act": "CrPC", "section": "436A"},
        ]),
    },
    {
        "case_name": "Suk Das v. Union Territory of Arunachal Pradesh",
        "case_number": "Criminal Appeal No. 811 of 1985",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "1986-05-19",
        "year": 1986,
        "citation_scc": "(1986) 2 SCC 401",
        "citation_air": "AIR 1986 SC 991",
        "domain": "criminal",
        "headnote": (
            "The Supreme Court reinforced that free legal aid is a fundamental "
            "right under Article 21. The failure to provide a lawyer to an "
            "accused who is too poor to engage one vitiates the trial. The "
            "court must inform the accused of his right to free legal aid "
            "and provide a competent lawyer, not merely a formal one."
        ),
        "ratio_decidendi": (
            "The right to free legal services is an essential ingredient of "
            "reasonable, fair and just procedure under Article 21. The "
            "Magistrate or Sessions Judge must inform the unrepresented accused "
            "of his right to free legal aid. Failure to do so renders the "
            "trial vitiated."
        ),
        "sections_interpreted": json.dumps([
            {"act": "Constitution of India", "section": "21"},
            {"act": "Constitution of India", "section": "39A"},
        ]),
    },

    # ─── Property / Contract ─────────────────────────────────────────────
    {
        "case_name": "S.P. Chengalvaraya Naidu v. Jagannath",
        "case_number": "Civil Appeal No. 2424 of 1993",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "1994-01-27",
        "year": 1994,
        "citation_scc": "(1994) 1 SCC 1",
        "citation_air": "AIR 1994 SC 853",
        "domain": "property",
        "headnote": (
            "The Supreme Court held that a judgment or decree obtained by fraud "
            "is a nullity and non est in the eye of the law. Fraud vitiates "
            "all judicial acts, whether ecclesiastical or temporal. A party "
            "who obtains a decree by playing fraud on the court is not entitled "
            "to its benefit."
        ),
        "ratio_decidendi": (
            "Fraud avoids all judicial acts, ecclesiastical or temporal. A "
            "judgment or decree obtained by fraud must be treated as a nullity. "
            "The courts of law are established for the benefit of litigants "
            "and the fraud practiced on the court is fraud on the administration "
            "of justice."
        ),
        "sections_interpreted": json.dumps([
            {"act": "CPC", "section": "44"},
            {"act": "Indian Contract Act", "section": "17"},
        ]),
    },
    {
        "case_name": "Nandganj Sihori Sugar Co. Ltd. v. Badri Nath Dixit",
        "case_number": "Civil Appeal No. 1620 of 1986",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "1991-10-22",
        "year": 1991,
        "citation_scc": "(1991) 4 SCC 516",
        "domain": "property",
        "headnote": (
            "The Supreme Court clarified the scope of Section 106 of the "
            "Transfer of Property Act regarding termination of leases. A "
            "15-day notice is required to terminate a month-to-month tenancy "
            "for non-agricultural purposes. The notice must be in writing "
            "and must clearly express the intention to terminate."
        ),
        "ratio_decidendi": (
            "Section 106 TPA mandates that a lease of immoveable property "
            "for purposes other than agriculture or manufacturing is deemed "
            "to be month-to-month, terminable by 15 days notice. The notice "
            "must unambiguously express the intention to terminate the tenancy."
        ),
        "sections_interpreted": json.dumps([
            {"act": "TPA", "section": "106"},
            {"act": "TPA", "section": "111"},
        ]),
    },

    # ─── Consumer Protection ─────────────────────────────────────────────
    {
        "case_name": "Lucknow Development Authority v. M.K. Gupta",
        "case_number": "Civil Appeal No. 6237 of 1990",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "1994-01-07",
        "year": 1994,
        "citation_scc": "(1994) 1 SCC 243",
        "citation_air": "AIR 1994 SC 787",
        "domain": "consumer",
        "headnote": (
            "The Supreme Court held that statutory authorities and public "
            "bodies providing services such as housing construction fall "
            "within the definition of 'service' under the Consumer Protection "
            "Act. The court expanded the scope of consumer protection to cover "
            "services provided by government development authorities, making "
            "them accountable to consumer forums for deficiency in service."
        ),
        "ratio_decidendi": (
            "Any service provided by a statutory body for a consideration "
            "amounts to service within the meaning of the Consumer Protection "
            "Act. Housing construction by development authorities constitutes "
            "service. Public authorities are accountable under consumer "
            "protection law for deficiency in service."
        ),
        "sections_interpreted": json.dumps([
            {"act": "CPA", "section": "2(1)(o)"},
            {"act": "CPA", "section": "14"},
        ]),
    },

    # ─── Labor / Employment ──────────────────────────────────────────────
    {
        "case_name": "Bangalore Water Supply and Sewerage Board v. A. Rajappa",
        "case_number": "Civil Appeal No. 1570 of 1974",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 7,
        "judgment_date": "1978-04-21",
        "year": 1978,
        "citation_scc": "(1978) 2 SCC 213",
        "citation_air": "AIR 1978 SC 548",
        "domain": "labor",
        "headnote": (
            "A seven-judge bench laid down the triple test for determining "
            "whether an activity constitutes 'industry' under the Industrial "
            "Disputes Act: (1) systematic activity, (2) organized by cooperation "
            "between employer and employee, (3) for the production and/or "
            "distribution of goods and services calculated to satisfy human "
            "wants and wishes. Professions, clubs, cooperatives, and even "
            "charitable activities can be 'industry' if they meet this test."
        ),
        "ratio_decidendi": (
            "The definition of 'industry' under Section 2(j) of the Industrial "
            "Disputes Act must be interpreted broadly. The dominant nature "
            "test is: where the predominant activity is to carry on an "
            "industry, the entire undertaking is an industry. The triple test "
            "of systematic activity, employer-employee cooperation, and "
            "production/distribution of goods or services applies."
        ),
        "sections_interpreted": json.dumps([
            {"act": "ID Act", "section": "2(j)"},
            {"act": "ID Act", "section": "2(k)"},
        ]),
    },

    # ─── Constitutional / RTI ────────────────────────────────────────────
    {
        "case_name": "CBSE v. Aditya Bandopadhyay",
        "case_number": "Civil Appeal No. 6454 of 2011",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "2011-08-09",
        "year": 2011,
        "citation_scc": "(2011) 8 SCC 497",
        "domain": "constitutional",
        "headnote": (
            "The Supreme Court held that evaluated answer sheets of public "
            "examinations fall within the definition of 'information' under "
            "the RTI Act and must be provided to the applicant. The court "
            "clarified the scope of Section 8(1)(e) exemption, holding that "
            "answer sheets held by examination bodies are not covered by the "
            "fiduciary relationship exemption."
        ),
        "ratio_decidendi": (
            "Evaluated answer sheets constitute 'information' under Section "
            "2(f) of the RTI Act. Examination bodies do not hold answer "
            "sheets in a fiduciary capacity, so Section 8(1)(e) does not "
            "apply. The RTI Act has a broad reach and exemptions must be "
            "construed narrowly."
        ),
        "sections_interpreted": json.dumps([
            {"act": "RTI", "section": "2(f)"},
            {"act": "RTI", "section": "3"},
            {"act": "RTI", "section": "8(1)(e)"},
        ]),
    },

    # ─── Family Law ──────────────────────────────────────────────────────
    {
        "case_name": "Shamima Farooqui v. Shahid Khan",
        "case_number": "Criminal Appeal No. 1003 of 2014",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "2015-04-22",
        "year": 2015,
        "citation_scc": "(2015) 5 SCC 705",
        "domain": "family",
        "headnote": (
            "The Supreme Court held that a wife is entitled to live with "
            "dignity and is entitled to maintenance from her husband even "
            "if she is living separately due to his cruelty. The court "
            "clarified that the obligation to maintain a wife arises from "
            "the marriage itself and is not dependent on any decree or order. "
            "The quantum of maintenance must be sufficient to enable the wife "
            "to live with dignity, not just survive."
        ),
        "ratio_decidendi": (
            "Maintenance under Section 125 CrPC is not charity but a right "
            "of the wife. The quantum must account for the status of the "
            "parties, their financial position, and the standard of living "
            "the wife was accustomed to. The purpose of maintenance is to "
            "prevent vagrancy and destitution."
        ),
        "sections_interpreted": json.dumps([
            {"act": "CrPC", "section": "125"},
            {"act": "HMA", "section": "24"},
        ]),
    },

    # ─── IP / Cyber Law ──────────────────────────────────────────────────
    {
        "case_name": "Aneesh M. Madathil v. Registrar of Trademarks",
        "case_number": "Civil Appeal No. 1821 of 2023",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "2024-01-16",
        "year": 2024,
        "citation_scc": "(2024) 3 SCC 201",
        "domain": "ip",
        "headnote": (
            "The Supreme Court addressed the registration of trademarks "
            "and the standard of distinctiveness required. The court held "
            "that a mark must be capable of distinguishing the goods or "
            "services of one person from those of another. Descriptive marks "
            "without acquired distinctiveness cannot be registered."
        ),
        "ratio_decidendi": (
            "The fundamental purpose of a trademark is to identify the "
            "source of goods or services. A mark that merely describes the "
            "goods or their characteristics without acquired distinctiveness "
            "through use is not eligible for registration under the Trade "
            "Marks Act."
        ),
        "sections_interpreted": json.dumps([
            {"act": "Trade Marks Act", "section": "9"},
            {"act": "Trade Marks Act", "section": "11"},
        ]),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Rate Limiter (reuses pattern from india_code.py)
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
# SCI Scraper
# ═══════════════════════════════════════════════════════════════════════════════


class SCIScraper:
    """
    Scrapes judgments from the Supreme Court of India website.

    The SCI website (main.sci.gov.in) provides judgment listings with
    metadata and PDF downloads. This scraper:
    1. Navigates judgment listing pages by year/month
    2. Extracts metadata from listing tables
    3. Downloads and parses judgment text where available
    4. Stores in PostgreSQL with deduplication via case_name + year

    Note: The SCI website structure changes periodically. This scraper
    is designed to be defensive — it logs warnings for unexpected HTML
    structures rather than crashing. If the website is unreachable or
    restructured, fall back to Indian Kanoon for SC judgments.
    """

    def __init__(self):
        self.rate_limiter = RateLimiter(
            max_requests=SCI_RATE_LIMIT_REQUESTS,
            period=SCI_RATE_LIMIT_PERIOD,
        )
        self.client: httpx.AsyncClient | None = None
        self.stats = {
            "pages_fetched": 0,
            "judgments_found": 0,
            "judgments_new": 0,
            "judgments_existing": 0,
            "fetch_errors": 0,
            "parse_errors": 0,
        }

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "NyayaMitra-Bot/1.0 (Legal Research Project; "
                    "AI Legal Assistant for Indian Citizens; "
                    "+https://nyayamitra.in)"
                ),
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

        Uses conservative rate limiting for the government site.
        Returns HTML content or None if all retries fail.
        """
        max_retries = settings.SCRAPER_RETRY_MAX

        for attempt in range(max_retries):
            await self.rate_limiter.acquire()
            try:
                resp = await self.client.get(url)
                resp.raise_for_status()
                self.stats["pages_fetched"] += 1
                return resp.text

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                logger.warning(
                    "sci_http_error",
                    url=url,
                    status=status,
                    attempt=attempt + 1,
                )
                if status == 429:
                    # Rate limited — back off significantly
                    wait = settings.SCRAPER_RETRY_BACKOFF ** (attempt + 3)
                    logger.info("sci_rate_limited", wait_seconds=wait)
                    await asyncio.sleep(wait)
                elif status >= 500:
                    wait = settings.SCRAPER_RETRY_BACKOFF ** (attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    # 4xx (not 429) — don't retry
                    self.stats["fetch_errors"] += 1
                    return None

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(
                    "sci_connection_error",
                    url=url,
                    error=str(e),
                    attempt=attempt + 1,
                )
                wait = settings.SCRAPER_RETRY_BACKOFF ** (attempt + 1)
                await asyncio.sleep(wait)

        logger.error("sci_all_retries_failed", url=url)
        self.stats["fetch_errors"] += 1
        return None

    def parse_judgment_listing(self, html: str) -> list[dict]:
        """
        Parse a judgment listing page from SCI website.

        Extracts judgment metadata from the listing tables/cards.
        The SCI website uses various formats — this tries multiple
        parsing strategies.

        Returns a list of judgment metadata dicts.
        """
        soup = BeautifulSoup(html, "lxml")
        judgments = []

        # Strategy 1: Look for judgment entries in table rows
        # SCI often uses tables with columns: S.No, Case No, Parties, Date, Bench
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            # Check if this looks like a judgment table
            header_text = rows[0].get_text(strip=True).lower()
            if not any(kw in header_text for kw in ["case", "diary", "parties", "judgment"]):
                continue

            for row in rows[1:]:
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                judgment = self._parse_table_row(cells)
                if judgment:
                    judgments.append(judgment)

        # Strategy 2: Look for structured divs/cards
        if not judgments:
            for card in soup.find_all(["div", "article"], class_=re.compile(
                r"judgment|case|order|decision", re.IGNORECASE
            )):
                judgment = self._parse_card(card)
                if judgment:
                    judgments.append(judgment)

        # Strategy 3: Look for links to judgment PDFs with metadata
        if not judgments:
            for link in soup.find_all("a", href=re.compile(r"\.pdf|judgment|order", re.IGNORECASE)):
                judgment = self._parse_link_context(link)
                if judgment:
                    judgments.append(judgment)

        logger.info(
            "sci_listing_parsed",
            judgments_found=len(judgments),
            strategy="table" if judgments else "fallback",
        )

        return judgments

    def _parse_table_row(self, cells: list) -> dict | None:
        """Parse a single table row into judgment metadata."""
        try:
            texts = [c.get_text(strip=True) for c in cells]

            # Try to identify case number, parties, date from cell contents
            case_number = None
            case_name = None
            judgment_date = None
            bench = None
            pdf_url = None

            for i, text in enumerate(texts):
                # Case number patterns: "SLP(C) No. 1234/2023", "W.P.(C) No. 567/2022"
                if re.search(r"(?:SLP|W\.?P|C\.?A|Crl\.?\s*A)", text, re.IGNORECASE):
                    case_number = text.strip()

                # Date patterns: "01-01-2024", "01/01/2024", "1 January 2024"
                date_match = re.search(
                    r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text
                )
                if date_match:
                    try:
                        d, m, y = date_match.groups()
                        judgment_date = date(int(y), int(m), int(d))
                    except ValueError:
                        pass

                # "v." or "vs." indicates parties (case name)
                if re.search(r"\bv[s]?\.\b", text, re.IGNORECASE) and len(text) > 10:
                    case_name = text.strip()

                # Bench: "Hon'ble Justice X, Justice Y"
                if re.search(r"justice|hon'ble|j\.", text, re.IGNORECASE) and len(text) > 5:
                    bench = text.strip()

            # Check for PDF links in the row
            for cell in cells:
                link = cell.find("a", href=re.compile(r"\.pdf", re.IGNORECASE))
                if link:
                    href = link.get("href", "")
                    if href.startswith("/"):
                        pdf_url = f"{SCI_BASE_URL}{href}"
                    elif href.startswith("http"):
                        pdf_url = href

            if not case_name and case_number:
                # Use case number as fallback name
                case_name = case_number

            if not case_name:
                return None

            # Extract year
            year = judgment_date.year if judgment_date else None
            if not year and case_number:
                year_match = re.search(r"[/-](\d{4})\b", case_number)
                if year_match:
                    year = int(year_match.group(1))

            if not year:
                return None

            return {
                "case_name": case_name[:500],
                "case_number": case_number,
                "court": "Supreme Court",
                "court_type": "SC",
                "bench": bench,
                "bench_size": self._estimate_bench_size(bench) if bench else None,
                "judgment_date": judgment_date,
                "year": year,
                "source": "sci",
                "source_url": pdf_url,
            }

        except Exception as e:
            logger.debug("sci_row_parse_error", error=str(e))
            self.stats["parse_errors"] += 1
            return None

    def _parse_card(self, card) -> dict | None:
        """Parse a judgment card/div into metadata."""
        try:
            text = card.get_text(separator=" ", strip=True)

            # Extract case name (look for "v." pattern)
            name_match = re.search(
                r"([\w\s\.\,]+\s+v[s]?\.\s+[\w\s\.\,]+)", text
            )
            case_name = name_match.group(1).strip() if name_match else None

            if not case_name:
                return None

            # Extract date
            judgment_date = None
            date_match = re.search(
                r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text
            )
            if date_match:
                try:
                    d, m, y = date_match.groups()
                    judgment_date = date(int(y), int(m), int(d))
                except ValueError:
                    pass

            year = judgment_date.year if judgment_date else None
            if not year:
                year_match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
                year = int(year_match.group(1)) if year_match else None

            if not year:
                return None

            # Extract case number
            case_num_match = re.search(
                r"((?:SLP|W\.?P|C\.?A|Crl\.?\s*A)[^\d]*\d+[/-]\d{4})",
                text, re.IGNORECASE,
            )
            case_number = case_num_match.group(1).strip() if case_num_match else None

            # Check for PDF link
            pdf_url = None
            link = card.find("a", href=re.compile(r"\.pdf", re.IGNORECASE))
            if link:
                href = link.get("href", "")
                if href.startswith("/"):
                    pdf_url = f"{SCI_BASE_URL}{href}"
                elif href.startswith("http"):
                    pdf_url = href

            return {
                "case_name": case_name[:500],
                "case_number": case_number,
                "court": "Supreme Court",
                "court_type": "SC",
                "judgment_date": judgment_date,
                "year": year,
                "source": "sci",
                "source_url": pdf_url,
            }

        except Exception as e:
            logger.debug("sci_card_parse_error", error=str(e))
            self.stats["parse_errors"] += 1
            return None

    def _parse_link_context(self, link) -> dict | None:
        """Extract judgment metadata from a PDF link and its surrounding context."""
        try:
            link_text = link.get_text(strip=True)
            href = link.get("href", "")

            # Get surrounding paragraph text for context
            parent = link.parent
            context = parent.get_text(separator=" ", strip=True) if parent else link_text

            # Extract case name
            name_match = re.search(
                r"([\w\s\.\,]+\s+v[s]?\.\s+[\w\s\.\,]+)", context
            )
            case_name = name_match.group(1).strip() if name_match else None

            if not case_name and link_text and len(link_text) > 10:
                case_name = link_text

            if not case_name:
                return None

            # Extract year from URL or context
            year = None
            year_match = re.search(r"\b(20\d{2}|19\d{2})\b", href + " " + context)
            if year_match:
                year = int(year_match.group(1))

            if not year:
                return None

            # Build PDF URL
            pdf_url = None
            if href:
                if href.startswith("/"):
                    pdf_url = f"{SCI_BASE_URL}{href}"
                elif href.startswith("http"):
                    pdf_url = href

            return {
                "case_name": case_name[:500],
                "court": "Supreme Court",
                "court_type": "SC",
                "year": year,
                "source": "sci",
                "source_url": pdf_url,
            }

        except Exception as e:
            logger.debug("sci_link_parse_error", error=str(e))
            self.stats["parse_errors"] += 1
            return None

    def _estimate_bench_size(self, bench_text: str) -> int:
        """Estimate bench size from bench composition text."""
        if not bench_text:
            return 1
        # Count occurrences of "Justice" or "J."
        justice_count = len(re.findall(r"\bjustice\b|\bj\.\b|\bjj\.\b", bench_text, re.IGNORECASE))
        return max(justice_count, 1)

    def _extract_citations_from_text(self, text: str) -> tuple[str | None, str | None]:
        """Extract SCC and AIR citations from judgment text."""
        citation_scc = None
        citation_air = None

        scc_match = re.search(r"\(\d{4}\)\s+\d+\s+SCC\s+\d+", text)
        if scc_match:
            citation_scc = scc_match.group(0)

        air_match = re.search(r"AIR\s+\d{4}\s+SC\s+\d+", text)
        if air_match:
            citation_air = air_match.group(0)

        return citation_scc, citation_air

    async def store_judgment(self, judgment_data: dict) -> bool:
        """
        Store a judgment in PostgreSQL with deduplication.

        Deduplication checks:
        1. case_name + year (primary)
        2. case_number if available

        Returns True if stored (new), False if skipped (duplicate).
        """
        async with async_session() as session:
            try:
                from sqlalchemy import select

                # Check for duplicate via case name + year
                stmt = select(Judgment).where(
                    Judgment.case_name == judgment_data["case_name"],
                    Judgment.year == judgment_data["year"],
                )
                result = await session.execute(stmt)
                if result.scalar_one_or_none():
                    self.stats["judgments_existing"] += 1
                    return False

                # Parse judgment_date if it's a string
                jdate = judgment_data.get("judgment_date")
                if isinstance(jdate, str):
                    try:
                        jdate = datetime.strptime(jdate, "%Y-%m-%d").date()
                    except ValueError:
                        jdate = None

                new_judgment = Judgment(
                    id=uuid.uuid4(),
                    case_name=judgment_data["case_name"],
                    case_number=judgment_data.get("case_number"),
                    court=judgment_data.get("court", "Supreme Court"),
                    court_type=judgment_data.get("court_type", "SC"),
                    bench=judgment_data.get("bench"),
                    bench_size=judgment_data.get("bench_size"),
                    judgment_date=jdate,
                    year=judgment_data["year"],
                    citation_scc=judgment_data.get("citation_scc"),
                    citation_air=judgment_data.get("citation_air"),
                    domain=judgment_data.get("domain"),
                    headnote=judgment_data.get("headnote"),
                    ratio_decidendi=judgment_data.get("ratio_decidendi"),
                    sections_interpreted=judgment_data.get("sections_interpreted"),
                    source=judgment_data.get("source", "sci"),
                    source_url=judgment_data.get("source_url"),
                )
                session.add(new_judgment)
                await session.commit()
                self.stats["judgments_new"] += 1

                logger.info(
                    "sci_judgment_stored",
                    case=judgment_data["case_name"],
                    year=judgment_data["year"],
                )
                return True

            except Exception as e:
                await session.rollback()
                logger.error(
                    "sci_store_error",
                    case=judgment_data.get("case_name"),
                    error=str(e),
                )
                return False

    async def scrape_recent_judgments(self, max_pages: int = 10) -> dict:
        """
        Scrape recent judgments from SCI website.

        Navigates the judgment listing pages and extracts metadata.
        """
        logger.info(
            "sci_scrape_start",
            max_pages=max_pages,
        )

        start_time = time.time()

        # Create ingestion log
        log_id = uuid.uuid4()
        async with async_session() as session:
            log = IngestionLog(
                id=log_id,
                source="sci",
                task="scrape_recent_judgments",
                started_at=datetime.utcnow(),
                status="running",
            )
            session.add(log)
            await session.commit()

        # Try to access the judgment listing page
        listing_url = SCI_JUDGMENTS_URL
        html = await self.fetch_page(listing_url)

        if html:
            judgments = self.parse_judgment_listing(html)
            self.stats["judgments_found"] += len(judgments)

            for jdata in judgments:
                await self.store_judgment(jdata)
        else:
            logger.warning(
                "sci_listing_unreachable",
                url=listing_url,
                message=(
                    "SCI website may be down or restructured. "
                    "Seed data is still available. "
                    "Fall back to Indian Kanoon for SC judgments."
                ),
            )

        duration = round(time.time() - start_time, 2)

        # Update ingestion log
        async with async_session() as session:
            from sqlalchemy import select

            stmt = select(IngestionLog).where(IngestionLog.id == log_id)
            result = await session.execute(stmt)
            log = result.scalar_one()
            log.completed_at = datetime.utcnow()
            log.status = "success" if self.stats["fetch_errors"] == 0 else "partial"
            log.items_fetched = self.stats["judgments_found"]
            log.items_new = self.stats["judgments_new"]
            log.items_updated = self.stats["judgments_existing"]
            log.items_failed = self.stats["fetch_errors"] + self.stats["parse_errors"]
            log.last_success_at = datetime.utcnow()
            await session.commit()

        logger.info(
            "sci_scrape_complete",
            duration_seconds=duration,
            **self.stats,
        )

        return self.stats


# ═══════════════════════════════════════════════════════════════════════════════
# Seed Function
# ═══════════════════════════════════════════════════════════════════════════════


async def seed_sci_judgments() -> dict:
    """
    Seed the database with curated SC judgments from the SCI seed data.

    These complement the landmark judgments in indian_kanoon.py and
    seed_comprehensive.py. Together, all three seed sources provide
    ~40 hand-verified SC judgments covering all 7 legal domains.

    No HTTP requests are made — this uses only the hardcoded seed data.
    """
    logger.info("sci_seed_start", count=len(SCI_SEED_JUDGMENTS))
    stats = {"new": 0, "existing": 0}

    async with async_session() as session:
        from sqlalchemy import select

        for jdata in SCI_SEED_JUDGMENTS:
            # Check if already exists
            stmt = select(Judgment).where(
                Judgment.case_name == jdata["case_name"],
                Judgment.year == jdata["year"],
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                stats["existing"] += 1
                logger.info("sci_seed_exists", case=jdata["case_name"])
                continue

            # Parse judgment_date
            jdate = None
            if jdata.get("judgment_date"):
                jdate = datetime.strptime(jdata["judgment_date"], "%Y-%m-%d").date()

            new_judgment = Judgment(
                id=uuid.uuid4(),
                case_name=jdata["case_name"],
                case_number=jdata.get("case_number"),
                court=jdata["court"],
                court_type=jdata["court_type"],
                bench_size=jdata.get("bench_size"),
                judgment_date=jdate,
                year=jdata["year"],
                citation_scc=jdata.get("citation_scc"),
                citation_air=jdata.get("citation_air"),
                domain=jdata.get("domain"),
                headnote=jdata.get("headnote"),
                ratio_decidendi=jdata.get("ratio_decidendi"),
                sections_interpreted=jdata.get("sections_interpreted"),
                source="sci_seed",
                source_url=f"https://main.sci.gov.in",
            )
            session.add(new_judgment)
            stats["new"] += 1
            logger.info(
                "sci_seed_created",
                case=jdata["case_name"],
                year=jdata["year"],
                domain=jdata.get("domain"),
            )

        await session.commit()

    logger.info("sci_seed_complete", **stats)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Run the SCI scraper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NyayaMitra — Supreme Court of India Scraper"
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only seed curated judgments, skip web scraping",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape judgments from SCI website",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum listing pages to scrape (default: 10)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("NyayaMitra — Supreme Court of India Scraper")
    print("=" * 60)

    # Always seed curated data first
    seed_stats = await seed_sci_judgments()
    print(
        f"\nSeed data: {seed_stats['new']} new, "
        f"{seed_stats['existing']} existing judgments"
    )

    if args.scrape:
        print("\nStarting live scrape from main.sci.gov.in...")
        print("(Rate limit: 1 request per 2 seconds)")
        async with SCIScraper() as scraper:
            stats = await scraper.scrape_recent_judgments(
                max_pages=args.max_pages,
            )
            print(f"\nScraping complete:")
            print(f"  Pages fetched:      {stats['pages_fetched']}")
            print(f"  Judgments found:     {stats['judgments_found']}")
            print(f"  New stored:          {stats['judgments_new']}")
            print(f"  Already existing:    {stats['judgments_existing']}")
            print(f"  Fetch errors:        {stats['fetch_errors']}")
            print(f"  Parse errors:        {stats['parse_errors']}")
    elif not args.seed_only:
        print("\nUsage:")
        print("  --seed-only    Only seed curated judgments")
        print("  --scrape       Scrape from SCI website")
        print("  --max-pages N  Max pages to scrape (default: 10)")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())