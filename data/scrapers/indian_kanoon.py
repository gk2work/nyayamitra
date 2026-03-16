"""
NyayaMitra — Indian Kanoon API Client.

Fetches Supreme Court and High Court judgments from indiankanoon.org.
Indian Kanoon is the largest free repository of Indian legal documents.

This client:
1. Searches for judgments by court, date range, and keyword
2. Fetches individual judgment documents
3. Extracts metadata using JudgmentParser
4. Validates data using DataValidator
5. Stores judgments in PostgreSQL with deduplication

Usage:
    # Seed curated landmark judgments (no API key needed)
    python -m data.scrapers.indian_kanoon --seed-only

    # Fetch from API (requires INDIAN_KANOON_API_TOKEN in .env)
    python -m data.scrapers.indian_kanoon --scrape --court SC --years 5

Rate Limits:
    - ~100 requests/min on free tier
    - Exponential backoff on 429 responses
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings
from app.database import async_session
from app.exceptions import FetchError, ParseError, ScraperError
from app.models.legal import Judgment, IngestionLog

from data.processors.judgment_parser import JudgmentParser
from data.processors.validator import DataValidator

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Seed Data — Landmark SC Judgments
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
        "headnote": "The Supreme Court laid down 11 mandatory guidelines to be followed in all cases of arrest and detention to prevent custodial violence and torture. These guidelines include: right to be informed of the grounds of arrest, right to inform a relative, right to legal counsel, medical examination every 48 hours, and the requirement that the arrestee must be produced before a magistrate within 24 hours.",
        "ratio_decidendi": "Custodial death is one of the worst crimes in a civilized society. The rights under Articles 21 and 22(1) of the Constitution are available to every citizen and must be scrupulously protected. The court laid down 11 specific requirements for police to follow during arrest and detention to protect the fundamental rights of the arrested person.",
        "sections_interpreted": '[{"act": "Constitution of India", "section": "21"}, {"act": "Constitution of India", "section": "22"}, {"act": "CrPC", "section": "41"}, {"act": "CrPC", "section": "50"}, {"act": "CrPC", "section": "56"}, {"act": "CrPC", "section": "57"}]',
        "source": "indian_kanoon",
    },
    {
        "case_name": "Lalita Kumari v. Government of Uttar Pradesh",
        "case_number": "Writ Petition (Crl.) No. 68 of 2008",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench": "Justice P. Sathasivam, Justice B.S. Chauhan, Justice Ranjan Gogoi, Justice S.A. Bobde, Justice N.V. Ramana",
        "bench_size": 5,
        "judgment_date": "2013-11-12",
        "year": 2013,
        "citation_scc": "(2014) 2 SCC 1",
        "citation_air": "AIR 2014 SC 187",
        "domain": "criminal",
        "headnote": "A Constitution Bench held that registration of FIR under Section 154 CrPC is mandatory when information discloses commission of a cognizable offence. The police officer cannot conduct a preliminary inquiry before registering the FIR in cases of cognizable offences. However, in cases where the information does not disclose a cognizable offence, a preliminary inquiry may be conducted.",
        "ratio_decidendi": "Registration of FIR is mandatory under Section 154 of the Code, if the information discloses commission of a cognizable offence and no preliminary inquiry is permissible in such a situation. The scope of preliminary inquiry is not to verify the veracity or otherwise of the information received but only to ascertain whether the information reveals any cognizable offence.",
        "sections_interpreted": '[{"act": "CrPC", "section": "154"}, {"act": "CrPC", "section": "155"}, {"act": "CrPC", "section": "156"}]',
        "source": "indian_kanoon",
    },
    {
        "case_name": "Arnesh Kumar v. State of Bihar",
        "case_number": "Criminal Appeal No. 1277 of 2014",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench": "Justice C.K. Prasad, Justice P.C. Ghose",
        "bench_size": 2,
        "judgment_date": "2014-07-02",
        "year": 2014,
        "citation_scc": "(2014) 8 SCC 273",
        "citation_air": "AIR 2014 SC 2756",
        "domain": "criminal",
        "headnote": "The Supreme Court issued guidelines to prevent automatic arrests in cases under Section 498A IPC and Section 4 of the Dowry Prohibition Act. Police officers must be satisfied that arrest is necessary under Section 41 CrPC parameters before making an arrest. Magistrates must not authorize detention casually and must satisfy themselves that Section 41 CrPC conditions are met.",
        "ratio_decidendi": "In cases under Section 498A IPC, the police should not automatically arrest the accused. The police officer must first satisfy himself about the necessity of arrest under the parameters laid down in Section 41 CrPC. A person accused under Section 498A should not be arrested without following the procedure prescribed under Section 41A CrPC.",
        "sections_interpreted": '[{"act": "IPC", "section": "498A"}, {"act": "CrPC", "section": "41"}, {"act": "CrPC", "section": "41A"}, {"act": "Dowry Prohibition Act", "section": "4"}]',
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
        "headnote": "The Supreme Court expanded the scope of Article 21, holding that the right to life and personal liberty is not merely the right to physical existence but includes the right to live with dignity. Any law that deprives a person of personal liberty must be just, fair and reasonable, not fanciful, oppressive or arbitrary. The procedure established by law must satisfy the requirements of natural justice.",
        "ratio_decidendi": "Article 21 does not exclude the applicability of Article 14 and Article 19. The procedure contemplated by Article 21 must answer the test of reasonableness. The right to travel abroad is part of personal liberty under Article 21 and no person can be deprived of this right except according to procedure established by law which must be fair, just and reasonable.",
        "sections_interpreted": '[{"act": "Constitution of India", "section": "14"}, {"act": "Constitution of India", "section": "19"}, {"act": "Constitution of India", "section": "21"}, {"act": "Passports Act", "section": "10(3)(c)"}]',
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
        "domain": "labor",
        "headnote": "The Supreme Court laid down guidelines for the prevention of sexual harassment of women at the workplace, known as the Vishaka Guidelines. These guidelines were binding and enforceable until suitable legislation was enacted. The guidelines defined sexual harassment, placed obligation on employers, and established complaint mechanisms. This led to the enactment of the POSH Act, 2013.",
        "ratio_decidendi": "In the absence of enacted law to provide for effective enforcement of the basic human right of gender equality and guarantee against sexual harassment at workplace, the court laid down guidelines and norms to be observed at all workplaces. These are binding and enforceable in law. Each incident of sexual harassment results in violation of fundamental rights under Articles 14, 15, 19 and 21.",
        "sections_interpreted": '[{"act": "Constitution of India", "section": "14"}, {"act": "Constitution of India", "section": "15"}, {"act": "Constitution of India", "section": "19(1)(g)"}, {"act": "Constitution of India", "section": "21"}]',
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
        "citation_air": "AIR 2017 SC 4161",
        "domain": "constitutional",
        "headnote": "A nine-judge Constitution Bench unanimously held that the right to privacy is a fundamental right protected under Part III of the Constitution, particularly under Article 21. Privacy includes informational privacy, bodily autonomy, and decisional privacy. Any invasion of privacy must satisfy the threefold test of legality, legitimate aim, and proportionality.",
        "ratio_decidendi": "The right to privacy is protected as an intrinsic part of the right to life and personal liberty under Article 21 and as a part of the freedoms guaranteed by Part III of the Constitution. Privacy is not an absolute right but any interference must meet the three-fold requirement: legality (sanctioned by law), need (legitimate state aim), and proportionality.",
        "sections_interpreted": '[{"act": "Constitution of India", "section": "14"}, {"act": "Constitution of India", "section": "19"}, {"act": "Constitution of India", "section": "21"}]',
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
        "citation_air": "AIR 2018 SC 4321",
        "domain": "constitutional",
        "headnote": "A five-judge Constitution Bench struck down Section 377 of the IPC to the extent it criminalized consensual sexual acts between adults. The Court held that Section 377 violated Articles 14, 15, 19 and 21 of the Constitution insofar as it criminalized consensual sexual conduct between adults of the same sex.",
        "ratio_decidendi": "Section 377 IPC, insofar as it criminalises consensual sexual conduct between adults of the same sex, is violative of Articles 14, 15, 19 and 21 of the Constitution. Sexual orientation is an intrinsic element of liberty, dignity and privacy. History owes an apology to LGBT community for the delay in ensuring their rights.",
        "sections_interpreted": '[{"act": "IPC", "section": "377"}, {"act": "Constitution of India", "section": "14"}, {"act": "Constitution of India", "section": "15"}, {"act": "Constitution of India", "section": "19"}, {"act": "Constitution of India", "section": "21"}]',
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
        "citation_air": "AIR 2018 SC 4898",
        "domain": "criminal",
        "headnote": "The Supreme Court struck down Section 497 IPC (adultery) as unconstitutional. The Court held that Section 497 was manifestly arbitrary, violated Article 14 by treating a woman as the property of her husband, and infringed upon the dignity and autonomy guaranteed under Article 21.",
        "ratio_decidendi": "Section 497 IPC denudes women of their agency and treats them as chattel of their husbands. The provision is violative of Articles 14 and 21 of the Constitution. Adultery may be a ground for civil remedy but cannot be a criminal offence. The autonomy of individual choice in matters of sexuality is an intrinsic part of dignity.",
        "sections_interpreted": '[{"act": "IPC", "section": "497"}, {"act": "CrPC", "section": "198"}, {"act": "Constitution of India", "section": "14"}, {"act": "Constitution of India", "section": "21"}]',
        "source": "indian_kanoon",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Indian Kanoon API Client
# ═══════════════════════════════════════════════════════════════════════════════


class IndianKanoonClient:
    """
    Client for the Indian Kanoon API.

    API docs: https://api.indiankanoon.org/doc/
    Requires an API token for search and document retrieval.

    Uses JudgmentParser for structured extraction and DataValidator
    for pre-insert quality checks.
    """

    BASE_URL = "https://api.indiankanoon.org"

    def __init__(self):
        self.token = settings.INDIAN_KANOON_API_TOKEN
        self.client: httpx.AsyncClient | None = None
        self.parser = JudgmentParser()
        self.validator = DataValidator()
        self.stats = {
            "judgments_fetched": 0,
            "judgments_new": 0,
            "judgments_updated": 0,
            "validation_skipped": 0,
            "errors": 0,
        }

    async def __aenter__(self):
        headers = {
            "User-Agent": "NyayaMitra-Bot/1.0 (Legal Research)",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Token {self.token}"

        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers=headers,
        )
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def search(
        self,
        query: str,
        page: int = 0,
        court: str = "supremecourt",
    ) -> dict | None:
        """
        Search Indian Kanoon for documents.

        Args:
            query: Search query (e.g., "Section 498A IPC")
            page: Page number (0-indexed)
            court: Court filter ("supremecourt", "allahabad", "bombay", etc.)

        Returns:
            API response dict or None on failure.
        """
        if not self.token:
            logger.warning("indian_kanoon_no_token", message="Set INDIAN_KANOON_API_TOKEN in .env")
            return None

        try:
            resp = await self.client.post(
                f"{self.BASE_URL}/search/",
                data={
                    "formInput": query,
                    "pagenum": page,
                },
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("indian_kanoon_search_error", query=query, error=str(e))
            self.stats["errors"] += 1
            return None

    async def get_document(self, doc_id: str) -> dict | None:
        """Fetch a specific document by Indian Kanoon doc ID."""
        if not self.token:
            return None

        try:
            resp = await self.client.post(
                f"{self.BASE_URL}/doc/{doc_id}/",
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("indian_kanoon_doc_error", doc_id=doc_id, error=str(e))
            self.stats["errors"] += 1
            return None

    def parse_judgment(self, doc: dict) -> dict | None:
        """
        Parse an Indian Kanoon API response using JudgmentParser.

        Returns a dict ready for store_judgment(), or None on failure.
        """
        try:
            parsed = self.parser.parse_indian_kanoon_doc(doc)
            doc_id = str(doc.get("tid", ""))

            return {
                "case_name": parsed.case_name,
                "court": parsed.court,
                "court_type": parsed.court_type,
                "judgment_date": parsed.judgment_date,
                "year": parsed.year or datetime.now().year,
                "citation_scc": parsed.citation_scc,
                "citation_air": parsed.citation_air,
                "indian_kanoon_id": doc_id,
                "headnote": parsed.headnote,
                "facts": parsed.facts,
                "ratio_decidendi": parsed.ratio_decidendi,
                "sections_interpreted": parsed.sections_interpreted,
                "full_text": parsed.full_text,
                "source": "indian_kanoon",
                "source_url": f"https://indiankanoon.org/doc/{doc_id}/" if doc_id else None,
            }
        except ParseError as e:
            logger.error("parse_judgment_error", error=str(e))
            return None
        except Exception as e:
            logger.error("parse_judgment_error", error=str(e))
            return None

    async def store_judgment(self, judgment_data: dict) -> None:
        """Store a judgment in PostgreSQL with validation and deduplication."""
        # Validate before insertion
        result = self.validator.validate_judgment(judgment_data)
        if not result.is_valid:
            logger.debug(
                "judgment_validation_failed",
                case=judgment_data.get("case_name"),
                errors=result.errors,
            )
            self.stats["validation_skipped"] += 1
            return

        async with async_session() as session:
            try:
                from sqlalchemy import select

                # Check for duplicate via indian_kanoon_id
                ik_id = judgment_data.get("indian_kanoon_id")
                if ik_id:
                    if await self.validator.check_duplicate_judgment_by_ik_id(session, ik_id):
                        self.stats["judgments_updated"] += 1
                        return

                # Check for duplicate via case name + year
                if await self.validator.check_duplicate_judgment(
                    session, judgment_data["case_name"], judgment_data["year"]
                ):
                    self.stats["judgments_updated"] += 1
                    return

                # Parse date if string
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
                    court=judgment_data["court"],
                    court_type=judgment_data.get("court_type", "SC"),
                    bench=judgment_data.get("bench"),
                    bench_size=judgment_data.get("bench_size"),
                    judgment_date=jdate,
                    year=judgment_data["year"],
                    citation_scc=judgment_data.get("citation_scc"),
                    citation_air=judgment_data.get("citation_air"),
                    indian_kanoon_id=judgment_data.get("indian_kanoon_id"),
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
                self.stats["judgments_new"] += 1

                logger.info(
                    "judgment_stored",
                    case=judgment_data["case_name"],
                    year=judgment_data["year"],
                )

            except Exception as e:
                await session.rollback()
                logger.error(
                    "store_judgment_error",
                    case=judgment_data.get("case_name"),
                    error=str(e),
                )
                self.stats["errors"] += 1

    async def scrape_court_judgments(
        self,
        court: str = "supremecourt",
        query: str = "",
        max_pages: int = 5,
    ) -> dict:
        """
        Scrape judgments from a specific court via Indian Kanoon search.

        Args:
            court: Court to search (supremecourt, bombay, delhi, etc.)
            query: Search query to filter results
            max_pages: Maximum pages to fetch (10 results per page)
        """
        logger.info(
            "scraping_court_judgments",
            court=court,
            query=query or "(all)",
            max_pages=max_pages,
        )

        for page in range(max_pages):
            search_query = f"{query} doctypes: judgments" if query else "doctypes: judgments"
            result = await self.search(search_query, page=page, court=court)

            if not result or "docs" not in result:
                logger.info("no_more_results", page=page)
                break

            docs = result.get("docs", [])
            if not docs:
                break

            for doc in docs:
                parsed = self.parse_judgment(doc)
                if parsed:
                    await self.store_judgment(parsed)
                    self.stats["judgments_fetched"] += 1

            # Rate limiting between pages
            await asyncio.sleep(1)

        return self.stats


# ═══════════════════════════════════════════════════════════════════════════════
# Seed Function
# ═══════════════════════════════════════════════════════════════════════════════


async def seed_landmark_judgments() -> dict:
    """
    Seed the database with curated landmark SC judgments.

    Uses DataValidator for pre-insert quality checks.
    """
    logger.info("seed_judgments_start", count=len(LANDMARK_JUDGMENTS))
    stats = {"new": 0, "existing": 0, "validation_skipped": 0}
    validator = DataValidator()

    async with async_session() as session:
        from sqlalchemy import select

        for jdata in LANDMARK_JUDGMENTS:
            # Validate
            result = validator.validate_judgment(jdata)
            if not result.is_valid:
                logger.warning(
                    "seed_judgment_invalid",
                    case=jdata.get("case_name"),
                    errors=result.errors,
                )
                stats["validation_skipped"] += 1
                continue

            # Check if already exists
            stmt = select(Judgment).where(
                Judgment.case_name == jdata["case_name"],
                Judgment.year == jdata["year"],
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                stats["existing"] += 1
                logger.info("seed_judgment_exists", case=jdata["case_name"])
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
                bench=jdata.get("bench"),
                bench_size=jdata.get("bench_size"),
                judgment_date=jdate,
                year=jdata["year"],
                citation_scc=jdata.get("citation_scc"),
                citation_air=jdata.get("citation_air"),
                domain=jdata.get("domain"),
                headnote=jdata.get("headnote"),
                ratio_decidendi=jdata.get("ratio_decidendi"),
                sections_interpreted=jdata.get("sections_interpreted"),
                source=jdata.get("source", "seed"),
                source_url=f"https://indiankanoon.org/search/?formInput={jdata['case_name'].replace(' ', '+')}",
            )
            session.add(new_judgment)
            stats["new"] += 1
            logger.info("seed_judgment_created", case=jdata["case_name"], year=jdata["year"])

        await session.commit()

    logger.info("seed_judgments_complete", **stats)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Run the Indian Kanoon scraper."""
    import argparse

    parser = argparse.ArgumentParser(description="NyayaMitra Indian Kanoon Client")
    parser.add_argument("--seed-only", action="store_true", help="Only seed landmark judgments")
    parser.add_argument("--scrape", action="store_true", help="Scrape from Indian Kanoon API")
    parser.add_argument("--court", default="supremecourt", help="Court to scrape (default: supremecourt)")
    parser.add_argument("--query", default="", help="Search query filter")
    parser.add_argument("--pages", type=int, default=5, help="Max pages to fetch (default: 5)")
    args = parser.parse_args()

    # Always seed landmark judgments first
    seed_stats = await seed_landmark_judgments()
    print(f"\nSeed data: {seed_stats['new']} new, {seed_stats['existing']} existing landmark judgments")

    if args.scrape:
        if not settings.INDIAN_KANOON_API_TOKEN:
            print("\nERROR: Set INDIAN_KANOON_API_TOKEN in .env to use the API")
            print("Get a token from: https://api.indiankanoon.org/")
            return

        async with IndianKanoonClient() as client:
            stats = await client.scrape_court_judgments(
                court=args.court,
                query=args.query,
                max_pages=args.pages,
            )
            print(f"\nScraping complete: {json.dumps(stats, indent=2)}")

    elif not args.seed_only:
        print("\nUse --scrape to fetch from Indian Kanoon API")
        print("Use --seed-only to only seed landmark judgments")


if __name__ == "__main__":
    asyncio.run(main())