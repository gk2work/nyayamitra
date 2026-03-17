"""
NyayaMitra — Judgment Parser Processor (Sprint 7 — Full Corpus).

Extracts structured metadata and components from raw judgment text.

Sprint 7 improvements over Phase 1:
    - Expanded ACT_ABBREVIATIONS: covers all 96+ acts in the registry
    - HC format support: handles diverse HC judgment structures
    - Domain auto-detection: infers legal domain from sections cited
    - Overruled detection: flags "overruled by" / "no longer good law"
    - Additional citation formats: SCALE, ALL, SCC Online, Cri LJ
    - More date formats: handles regional HC date styles
    - Parse statistics: tracks extraction quality per judgment

Supports:
    - Indian Kanoon API responses (HTML with doc/headline fields)
    - SCI website HTML
    - HC judgment HTML (varying formats across 25 HCs)
    - Plain text from PDF extraction

Usage:
    from data.processors.judgment_parser import JudgmentParser

    parser = JudgmentParser()
    result = parser.parse_indian_kanoon_doc(api_response)
    result = parser.parse_html(html_content)
    result = parser.parse_text(plain_text)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.exceptions import ParseError

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Act Name Abbreviation Mapping (Sprint 7: expanded to match acts_registry)
# ═══════════════════════════════════════════════════════════════════════════════

ACT_ABBREVIATIONS: dict[str, str] = {
    # ── Criminal ──────────────────────────────────────────────────────────
    "IPC": "Indian Penal Code, 1860",
    "Indian Penal Code": "Indian Penal Code, 1860",
    "BNS": "Bharatiya Nyaya Sanhita, 2023",
    "Bharatiya Nyaya Sanhita": "Bharatiya Nyaya Sanhita, 2023",
    "CrPC": "Code of Criminal Procedure, 1973",
    "Cr.P.C.": "Code of Criminal Procedure, 1973",
    "Cr.P.C": "Code of Criminal Procedure, 1973",
    "Code of Criminal Procedure": "Code of Criminal Procedure, 1973",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "Evidence Act": "Indian Evidence Act, 1872",
    "Indian Evidence Act": "Indian Evidence Act, 1872",
    "IEA": "Indian Evidence Act, 1872",
    "BSA": "Bharatiya Sakshya Adhiniyam, 2023",
    "NDPS Act": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "NDPS": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "Prevention of Corruption Act": "Prevention of Corruption Act, 1988",
    "PCA": "Prevention of Corruption Act, 1988",
    "PC Act": "Prevention of Corruption Act, 1988",
    "SC/ST Act": "SC/ST (Prevention of Atrocities) Act, 1989",
    "Atrocities Act": "SC/ST (Prevention of Atrocities) Act, 1989",
    "POCSO Act": "Protection of Children from Sexual Offences Act, 2012",
    "POCSO": "Protection of Children from Sexual Offences Act, 2012",
    "JJ Act": "Juvenile Justice Act, 2015",
    "UAPA": "Unlawful Activities (Prevention) Act, 1967",
    "Dowry Prohibition Act": "Dowry Prohibition Act, 1961",
    "Arms Act": "Arms Act, 1959",
    "PMLA": "Prevention of Money Laundering Act, 2002",
    # ── Property / Civil ──────────────────────────────────────────────────
    "CPC": "Code of Civil Procedure, 1908",
    "C.P.C.": "Code of Civil Procedure, 1908",
    "C.P.C": "Code of Civil Procedure, 1908",
    "Code of Civil Procedure": "Code of Civil Procedure, 1908",
    "TPA": "Transfer of Property Act, 1882",
    "Transfer of Property Act": "Transfer of Property Act, 1882",
    "Registration Act": "Registration Act, 1908",
    "Stamp Act": "Indian Stamp Act, 1899",
    "Indian Stamp Act": "Indian Stamp Act, 1899",
    "RERA": "Real Estate (Regulation and Development) Act, 2016",
    "Easements Act": "Indian Easements Act, 1882",
    "Specific Relief Act": "Specific Relief Act, 1963",
    "Benami Act": "Benami Transactions (Prohibition) Act, 1988",
    "LARR Act": "Land Acquisition Act, 2013",
    "Land Acquisition Act": "Land Acquisition Act, 2013",
    "Contract Act": "Indian Contract Act, 1872",
    "Indian Contract Act": "Indian Contract Act, 1872",
    "Sale of Goods Act": "Sale of Goods Act, 1930",
    "Limitation Act": "Limitation Act, 1963",
    "Partition Act": "Partition Act, 1893",
    # ── Family ────────────────────────────────────────────────────────────
    "HMA": "Hindu Marriage Act, 1955",
    "Hindu Marriage Act": "Hindu Marriage Act, 1955",
    "HSA": "Hindu Succession Act, 1956",
    "Hindu Succession Act": "Hindu Succession Act, 1956",
    "SMA": "Special Marriage Act, 1954",
    "Special Marriage Act": "Special Marriage Act, 1954",
    "DV Act": "Protection of Women from Domestic Violence Act, 2005",
    "Domestic Violence Act": "Protection of Women from Domestic Violence Act, 2005",
    "HMGA": "Hindu Minority and Guardianship Act, 1956",
    "HAMA": "Hindu Adoptions and Maintenance Act, 1956",
    "Guardians and Wards Act": "Guardians and Wards Act, 1890",
    "Indian Divorce Act": "Indian Divorce Act, 1869",
    "DMMA": "Dissolution of Muslim Marriages Act, 1939",
    "Indian Succession Act": "Indian Succession Act, 1925",
    "Senior Citizens Act": "Maintenance and Welfare of Parents and Senior Citizens Act, 2007",
    "PCMA": "Prohibition of Child Marriage Act, 2006",
    # ── Labour ────────────────────────────────────────────────────────────
    "ID Act": "Industrial Disputes Act, 1947",
    "Industrial Disputes Act": "Industrial Disputes Act, 1947",
    "POSH Act": "POSH Act, 2013",
    "Gratuity Act": "Payment of Gratuity Act, 1972",
    "EPF Act": "Employees' Provident Funds Act, 1952",
    "ESI Act": "Employees' State Insurance Act, 1948",
    "Minimum Wages Act": "Minimum Wages Act, 1948",
    "Maternity Benefit Act": "Maternity Benefit Act, 1961",
    "Wages Code": "Code on Wages, 2019",
    "IR Code": "Industrial Relations Code, 2020",
    "SS Code": "Code on Social Security, 2020",
    "OSH Code": "Occupational Safety Code, 2020",
    "Child Labour Act": "Child Labour Act, 1986",
    "Contract Labour Act": "Contract Labour Act, 1970",
    "CLRA Act": "Contract Labour Act, 1970",
    # ── Consumer ──────────────────────────────────────────────────────────
    "CPA": "Consumer Protection Act, 2019",
    "Consumer Protection Act": "Consumer Protection Act, 2019",
    "FSSA": "Food Safety and Standards Act, 2006",
    "Competition Act": "Competition Act, 2002",
    # ── Constitutional ────────────────────────────────────────────────────
    "Constitution": "Constitution of India",
    "Constitution of India": "Constitution of India",
    "RTI Act": "Right to Information Act, 2005",
    "Right to Information Act": "Right to Information Act, 2005",
    "Arbitration Act": "Arbitration and Conciliation Act, 1996",
    "LSA Act": "Legal Services Authorities Act, 1987",
    "RPA": "Representation of the People Act, 1951",
    "Contempt Act": "Contempt of Courts Act, 1971",
    # ── IP / Cyber ────────────────────────────────────────────────────────
    "IT Act": "Information Technology Act, 2000",
    "Information Technology Act": "Information Technology Act, 2000",
    "Copyright Act": "Copyright Act, 1957",
    "Patents Act": "Patents Act, 1970",
    "Trade Marks Act": "Trade Marks Act, 1999",
    "Designs Act": "Designs Act, 2000",
    "DPDP Act": "Digital Personal Data Protection Act, 2023",
    # ── Corporate ─────────────────────────────────────────────────────────
    "Companies Act": "Companies Act, 2013",
    "LLP Act": "Limited Liability Partnership Act, 2008",
    "IBC": "Insolvency and Bankruptcy Code, 2016",
    "Insolvency and Bankruptcy Code": "Insolvency and Bankruptcy Code, 2016",
    "NI Act": "Negotiable Instruments Act, 1881",
    "Negotiable Instruments Act": "Negotiable Instruments Act, 1881",
    "Partnership Act": "Indian Partnership Act, 1932",
    "SEBI Act": "SEBI Act, 1992",
    "FEMA": "Foreign Exchange Management Act, 1999",
    # ── Taxation ──────────────────────────────────────────────────────────
    "Income Tax Act": "Income Tax Act, 1961",
    "CGST Act": "Central Goods and Services Tax Act, 2017",
    "IGST Act": "Integrated Goods and Services Tax Act, 2017",
    # ── Environmental ─────────────────────────────────────────────────────
    "EPA": "Environment (Protection) Act, 1986",
    "Environment Act": "Environment (Protection) Act, 1986",
    "NGT Act": "National Green Tribunal Act, 2010",
    "Wildlife Act": "Wildlife Protection Act, 1972",
    "Forest Act": "Forest (Conservation) Act, 1980",
}

# Domain inference from act references
ACT_DOMAIN_MAP: dict[str, str] = {
    "Indian Penal Code": "criminal",
    "Bharatiya Nyaya Sanhita": "criminal",
    "Code of Criminal Procedure": "criminal",
    "Bharatiya Nagarik Suraksha Sanhita": "criminal",
    "Indian Evidence Act": "criminal",
    "Bharatiya Sakshya Adhiniyam": "criminal",
    "NDPS": "criminal",
    "POCSO": "criminal",
    "Transfer of Property Act": "property",
    "Registration Act": "property",
    "RERA": "property",
    "Indian Contract Act": "property",
    "Hindu Marriage Act": "family",
    "Hindu Succession Act": "family",
    "Special Marriage Act": "family",
    "Domestic Violence Act": "family",
    "Industrial Disputes Act": "labour",
    "POSH Act": "labour",
    "Consumer Protection Act": "consumer",
    "Constitution of India": "constitutional",
    "Right to Information Act": "constitutional",
    "Information Technology Act": "ip",
    "Copyright Act": "ip",
    "Patents Act": "ip",
    "Companies Act": "corporate",
    "Insolvency and Bankruptcy Code": "corporate",
    "Negotiable Instruments Act": "corporate",
    "Income Tax Act": "taxation",
    "Environment Act": "environmental",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Regex Patterns
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section references ────────────────────────────────────────────────────
SECTION_REF_PATTERNS = [
    # "Section 302 of the Indian Penal Code" / "Section 302 of IPC"
    re.compile(
        r"(?:Section|S\.?|Sec\.?)\s+(\d+[A-Za-z]*(?:\([^)]*\))*)\s+of\s+(?:the\s+)?"
        r"([A-Za-z][A-Za-z\s\.,]+?)(?:\s*,\s*\d{4})?(?=\s*[,;.\)\]]|\s+(?:and|read|r/w|with)|\s*$)",
        re.IGNORECASE,
    ),
    # "under S. 41 CrPC" / "u/s 302 IPC"
    re.compile(
        r"(?:under|u/s|u/S)\s+(?:Section|S\.?)?\s*(\d+[A-Za-z]*(?:\([^)]*\))*)"
        r"\s+(?:of\s+)?([A-Z][A-Za-z\s\.]+?)(?:\s*,\s*\d{4})?(?=[\s,;.\)])",
        re.IGNORECASE,
    ),
    # "Article 21 of the Constitution"
    re.compile(
        r"(?:Article|Art\.?)\s+(\d+[A-Za-z]*(?:\([^)]*\))*)\s+of\s+(?:the\s+)?"
        r"(Constitution[A-Za-z\s]*)",
        re.IGNORECASE,
    ),
    # "Section 302 IPC" (compact, no "of")
    re.compile(
        r"(?:Section|S\.?|Sec\.?)\s+(\d+[A-Za-z]*(?:\([^)]*\))*)\s+"
        r"(IPC|CrPC|CPC|BNS|BNSS|BSA|TPA|HMA|NI Act|IT Act|NDPS Act|POCSO|RERA|IBC)",
        re.IGNORECASE,
    ),
    # Standalone "Article 21" (assume Constitution)
    re.compile(
        r"(?:Article|Art\.?)\s+(\d+[A-Za-z]*(?:\([^)]*\))*)",
        re.IGNORECASE,
    ),
]

# ── Citation patterns ─────────────────────────────────────────────────────
SCC_CITATION_PATTERN = re.compile(r"\(\d{4}\)\s+\d+\s+SCC\s+\d+")
AIR_CITATION_PATTERN = re.compile(r"AIR\s+\d{4}\s+(?:SC|[A-Za-z]+)\s+\d+")

# Sprint 7: additional citation formats
SCC_ONLINE_PATTERN = re.compile(r"\d{4}\s+SCC\s+OnLine\s+(?:SC|[A-Z]+)\s+\d+")
SCALE_CITATION_PATTERN = re.compile(r"\(\d{4}\)\s+\d+\s+SCALE\s+\d+")
CRI_LJ_PATTERN = re.compile(r"\d{4}\s+Cri\.?\s*L\.?J\.?\s+\d+")
ALL_CITATION_PATTERNS = [
    SCC_CITATION_PATTERN,
    AIR_CITATION_PATTERN,
    SCC_ONLINE_PATTERN,
    SCALE_CITATION_PATTERN,
    CRI_LJ_PATTERN,
]

# ── Date patterns ─────────────────────────────────────────────────────────
DATE_PATTERNS = [
    # "13th November, 2013" / "13 November 2013"
    re.compile(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s*,?\s*(\d{4})",
        re.IGNORECASE,
    ),
    # "November 13, 2013" (US style, some HCs use this)
    re.compile(
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{1,2})\s*,?\s*(\d{4})",
        re.IGNORECASE,
    ),
    # "01-01-2024" / "01/01/2024"
    re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})"),
    # "2024-01-01" (ISO)
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    # "01.01.2024" (dot separator, common in some HC formats)
    re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})"),
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# ── Bench / case number ───────────────────────────────────────────────────
BENCH_PATTERN = re.compile(
    r"(?:BENCH|CORAM|Before|HON'BLE)[:\s]*(.*?)(?:\n|$)",
    re.IGNORECASE,
)

CASE_NUMBER_PATTERN = re.compile(
    r"((?:Writ Petition|W\.?P\.?|SLP|C\.?A\.?|Crl\.?\s*A\.?|Criminal Appeal|Civil Appeal|"
    r"Special Leave Petition|R\.?S\.?A\.?|F\.?A\.?|M\.?A\.?|O\.?P\.?|C\.?R\.?P\.?|"
    r"S\.?A\.?|L\.?P\.?A\.?|Review Petition|Contempt Petition|T\.?C\.?)"
    r"[^)]*(?:\([^)]*\))?\s*No\.?\s*\d+[^,\n]*(?:of|/)\s*\d{4})",
    re.IGNORECASE,
)

# ── Component section headers ─────────────────────────────────────────────
COMPONENT_PATTERNS = {
    "headnote": re.compile(
        r"(?:^|\n)\s*(?:HEAD\s*NOTE|HEADNOTE|SUMMARY|HELD)\s*[:\-—]?\s*",
        re.IGNORECASE,
    ),
    "facts": re.compile(
        r"(?:^|\n)\s*(?:FACTS|STATEMENT\s+OF\s+FACTS|FACTUAL\s+BACKGROUND|"
        r"BRIEF\s+FACTS|FACTS\s+OF\s+THE\s+CASE)\s*[:\-—]?\s*",
        re.IGNORECASE,
    ),
    "issues": re.compile(
        r"(?:^|\n)\s*(?:ISSUES?\s+(?:FRAMED|FOR\s+CONSIDERATION)|"
        r"QUESTIONS?\s+(?:OF\s+LAW|FOR\s+DETERMINATION)|POINT\s+FOR\s+CONSIDERATION)"
        r"\s*[:\-—]?\s*",
        re.IGNORECASE,
    ),
    "ratio": re.compile(
        r"(?:^|\n)\s*(?:RATIO\s+DECIDENDI|RATIO|REASONING|ANALYSIS\s+AND\s+REASONING|"
        r"DISCUSSION|ANALYSIS|OUR\s+VIEW|FINDINGS)\s*[:\-—]?\s*",
        re.IGNORECASE,
    ),
    "order": re.compile(
        r"(?:^|\n)\s*(?:ORDER|DIRECTION|DIRECTIONS|RESULT|CONCLUSION|"
        r"OPERATIVE\s+PART|DISPOSITION|ACCORDINGLY)\s*[:\-—]?\s*",
        re.IGNORECASE,
    ),
}

# ── Overruled detection ───────────────────────────────────────────────────
OVERRULED_PATTERNS = [
    re.compile(r"overruled\s+(?:in|by)\s+", re.IGNORECASE),
    re.compile(r"no\s+longer\s+good\s+law", re.IGNORECASE),
    re.compile(r"stood\s+overruled", re.IGNORECASE),
    re.compile(r"expressly\s+overruled", re.IGNORECASE),
    re.compile(r"impliedly\s+overruled", re.IGNORECASE),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParsedJudgment:
    """Structured components extracted from a raw judgment."""

    case_name: str = ""
    case_number: str | None = None
    court: str = "Supreme Court"
    court_type: str = "SC"
    bench: str | None = None
    bench_size: int | None = None
    judgment_date: date | None = None
    year: int = 0
    citation_scc: str | None = None
    citation_air: str | None = None
    citation_other: str | None = None   # Sprint 7: SCC Online, SCALE, Cri LJ
    domain: str | None = None           # Sprint 7: auto-detected domain
    headnote: str | None = None
    facts: str | None = None
    issues: str | None = None
    ratio_decidendi: str | None = None
    obiter_dicta: str | None = None
    order: str | None = None
    sections_interpreted: str | None = None  # JSON string
    full_text: str | None = None
    is_overruled: bool = False          # Sprint 7: overruled detection
    overruled_note: str | None = None   # Sprint 7: context

    def is_valid(self) -> bool:
        """Check minimum validity: must have case name and year."""
        return bool(self.case_name and self.year > 0)


@dataclass
class JudgmentParseStats:
    """Quality metrics for a single judgment parse."""

    case_name: str = ""
    has_headnote: bool = False
    has_ratio: bool = False
    has_facts: bool = False
    has_order: bool = False
    has_citation: bool = False
    has_bench: bool = False
    has_date: bool = False
    sections_count: int = 0
    domain_detected: str | None = None
    text_length: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Judgment Parser
# ═══════════════════════════════════════════════════════════════════════════════


class JudgmentParser:
    """
    Extracts structured metadata and components from raw judgment text.

    Sprint 7 enhancements:
    - Expanded abbreviation map (100+ entries matching acts registry)
    - HC-specific case number patterns (RSA, FA, MA, OP, CRP, LPA)
    - Domain auto-detection from sections interpreted
    - Overruled judgment detection
    - Additional citation formats (SCC Online, SCALE, Cri LJ)
    - More date formats (dot-separated, US month-first)
    - Parse stats tracking

    Thread-safe and stateless — create one instance and reuse.
    """

    def __init__(self):
        self.last_parse_stats: JudgmentParseStats | None = None

    def parse_html(self, html: str, metadata: dict | None = None) -> ParsedJudgment:
        """Parse judgment from HTML content."""
        if not html or not html.strip():
            raise ParseError(source="judgment_parser", reason="Empty HTML input")

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
        except ImportError:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            raise ParseError(source="judgment_parser", reason=f"HTML parsing failed: {e}")

        text = soup.get_text(separator="\n")
        result = self._extract_all(text, metadata)

        # Try structured HTML elements
        headline_el = soup.find(class_=re.compile(
            r"headline|headnote|summary", re.IGNORECASE,
        ))
        if headline_el and not result.headnote:
            result.headnote = headline_el.get_text(strip=True)[:3000]

        return result

    def parse_text(self, text: str, metadata: dict | None = None) -> ParsedJudgment:
        """Parse judgment from plain text."""
        if not text or not text.strip():
            raise ParseError(source="judgment_parser", reason="Empty text input")
        return self._extract_all(text, metadata)

    def parse_indian_kanoon_doc(self, doc: dict) -> ParsedJudgment:
        """
        Parse an Indian Kanoon API response document.

        Handles the specific structure of IK API responses.
        """
        result = ParsedJudgment()

        title = doc.get("title", "")
        doc_text = doc.get("doc", "")
        doc_id = str(doc.get("tid", ""))

        # Case name from title: "Case Name on Date, Court"
        if " on " in title:
            result.case_name = title.split(" on ")[0].strip()[:500]
        else:
            result.case_name = title[:500]

        # Court
        court_name = doc.get("docsource", "Supreme Court")
        result.court = court_name
        result.court_type = self._detect_court_type(court_name)

        # Date
        date_str = doc.get("publishdate", "")
        if date_str:
            result.judgment_date = self._parse_date_string(date_str)
            if result.judgment_date:
                result.year = result.judgment_date.year

        # Year fallback from title
        if not result.year:
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", title)
            if year_match:
                result.year = int(year_match.group(1))

        # Citations
        if doc_text:
            result.citation_scc = self._extract_citation(doc_text, SCC_CITATION_PATTERN)
            result.citation_air = self._extract_citation(doc_text, AIR_CITATION_PATTERN)
            result.citation_other = self._extract_other_citations(doc_text)

        # Headnote
        result.headnote = doc.get("headline", "")
        if not result.headnote and doc_text:
            result.headnote = doc_text[:1000].strip()

        # Full text
        result.full_text = doc_text[:50000] if doc_text else None

        # Sections interpreted + domain
        if doc_text:
            result.sections_interpreted = self.extract_sections_interpreted(doc_text)
            result.domain = self._detect_domain(result.sections_interpreted)

        # Components
        if doc_text:
            components = self._split_components(doc_text)
            if not result.headnote and components.get("headnote"):
                result.headnote = components["headnote"]
            result.facts = components.get("facts")
            result.issues = components.get("issues")
            result.ratio_decidendi = components.get("ratio")
            result.order = components.get("order")

        # Overruled check
        if doc_text:
            self._detect_overruled(result, doc_text)

        # Case number
        if doc_text:
            result.case_number = self._extract_case_number(doc_text)

        # Bench
        if doc_text:
            bench_text, bench_size = self._extract_bench(doc_text)
            result.bench = bench_text
            result.bench_size = bench_size

        self._track_stats(result)
        return result

    # ── Core Extraction ───────────────────────────────────────────────────

    def _extract_all(self, text: str, metadata: dict | None = None) -> ParsedJudgment:
        """Extract all components from raw judgment text."""
        metadata = metadata or {}
        result = ParsedJudgment()

        # Case name
        result.case_name = metadata.get("case_name", "") or self._extract_case_name(text)

        # Case number
        result.case_number = metadata.get("case_number") or self._extract_case_number(text)

        # Court
        result.court = metadata.get("court", "Supreme Court")
        result.court_type = metadata.get("court_type") or self._detect_court_type(result.court)

        # Bench
        bench_text, bench_size = self._extract_bench(text)
        result.bench = metadata.get("bench") or bench_text
        result.bench_size = metadata.get("bench_size") or bench_size

        # Date
        result.judgment_date = metadata.get("judgment_date") or self._extract_date(text)
        if result.judgment_date:
            result.year = result.judgment_date.year
        elif metadata.get("year"):
            result.year = metadata["year"]
        else:
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text[:500])
            result.year = int(year_match.group(1)) if year_match else 0

        # Citations
        result.citation_scc = self._extract_citation(text, SCC_CITATION_PATTERN)
        result.citation_air = self._extract_citation(text, AIR_CITATION_PATTERN)
        result.citation_other = self._extract_other_citations(text)

        # Sections interpreted + domain
        result.sections_interpreted = self.extract_sections_interpreted(text)
        result.domain = metadata.get("domain") or self._detect_domain(
            result.sections_interpreted,
        )

        # Components
        components = self._split_components(text)
        result.headnote = components.get("headnote")
        result.facts = components.get("facts")
        result.issues = components.get("issues")
        result.ratio_decidendi = components.get("ratio")
        result.order = components.get("order")

        # Fallback headnote
        if not result.headnote and text:
            result.headnote = text[:1000].strip()

        # Full text
        result.full_text = text[:50000] if text else None

        # Overruled
        self._detect_overruled(result, text)

        self._track_stats(result)

        logger.info(
            "judgment_parsed",
            case=result.case_name[:80] if result.case_name else "unknown",
            year=result.year,
            court_type=result.court_type,
            domain=result.domain,
            has_headnote=bool(result.headnote),
            has_ratio=bool(result.ratio_decidendi),
            sections_count=len(json.loads(result.sections_interpreted))
            if result.sections_interpreted else 0,
        )

        return result

    # ── Individual Extractors ─────────────────────────────────────────────

    def _extract_case_name(self, text: str) -> str:
        """Extract case name (X v. Y pattern) from first 500 chars."""
        header = text[:500]
        match = re.search(
            r"([\w\s\.\,\(\)]+?)\s+v[s]?\.\s+([\w\s\.\,\(\)]+?)(?:\n|on\s+\d|\s*$)",
            header,
        )
        if match:
            pet = re.sub(r"\s+", " ", match.group(1).strip())
            resp = re.sub(r"\s+", " ", match.group(2).strip())
            return f"{pet} v. {resp}"[:500]

        first_line = text.split("\n")[0].strip()
        return first_line[:500] if len(first_line) > 10 else ""

    def _extract_case_number(self, text: str) -> str | None:
        """Extract case number from judgment text."""
        match = CASE_NUMBER_PATTERN.search(text[:2000])
        return match.group(1).strip() if match else None

    def _extract_bench(self, text: str) -> tuple[str | None, int | None]:
        """Extract bench composition and estimate size."""
        match = BENCH_PATTERN.search(text[:3000])
        if not match:
            return None, None

        bench_text = re.sub(r"\s+", " ", match.group(1).strip())[:500]

        justice_count = len(re.findall(
            r"\bjustice\b|\bj\.\b|\bjj\.\b|\bhon'ble\b",
            bench_text,
            re.IGNORECASE,
        ))
        bench_size = max(justice_count, 1) if bench_text else None

        return bench_text, bench_size

    def _extract_date(self, text: str) -> date | None:
        """Extract judgment date from text header."""
        header = text[:1500]

        for pattern in DATE_PATTERNS:
            match = pattern.search(header)
            if not match:
                continue

            groups = match.groups()
            try:
                if len(groups) == 3 and groups[1].isalpha():
                    # "13 November 2013"
                    day = int(groups[0])
                    month = MONTH_MAP.get(groups[1].lower(), 0)
                    year = int(groups[2])
                    if month and 1 <= day <= 31 and 1900 < year < 2030:
                        return date(year, month, day)

                elif len(groups) == 3 and groups[0].isalpha():
                    # "November 13, 2013" (US style)
                    month = MONTH_MAP.get(groups[0].lower(), 0)
                    day = int(groups[1])
                    year = int(groups[2])
                    if month and 1 <= day <= 31 and 1900 < year < 2030:
                        return date(year, month, day)

                elif len(groups) == 3 and len(groups[0]) == 4:
                    # ISO "2024-01-01"
                    y, m, d = int(groups[0]), int(groups[1]), int(groups[2])
                    if 1900 < y < 2030:
                        return date(y, m, d)

                elif len(groups) == 3:
                    # "01-01-2024" / "01.01.2024"
                    d, m, y = int(groups[0]), int(groups[1]), int(groups[2])
                    if y > 1900 and 1 <= m <= 12 and 1 <= d <= 31:
                        return date(y, m, d)

            except (ValueError, TypeError):
                continue

        return None

    def _parse_date_string(self, date_str: str) -> date | None:
        """Parse a date string in common formats."""
        for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y",
                    "%d %B %Y", "%d.%m.%Y", "%Y/%m/%d"]:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None

    def _extract_citation(self, text: str, pattern: re.Pattern) -> str | None:
        """Extract first citation matching the given pattern."""
        match = pattern.search(text)
        return match.group(0) if match else None

    def _extract_other_citations(self, text: str) -> str | None:
        """Extract non-SCC/AIR citations (SCC Online, SCALE, Cri LJ)."""
        for pattern in [SCC_ONLINE_PATTERN, SCALE_CITATION_PATTERN, CRI_LJ_PATTERN]:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None

    def _detect_court_type(self, court: str) -> str:
        """Detect court type from court name."""
        court_lower = court.lower()
        if "supreme" in court_lower:
            return "SC"
        if "high" in court_lower:
            return "HC"
        if "tribunal" in court_lower or "commission" in court_lower:
            return "TRIBUNAL"
        if "district" in court_lower or "sessions" in court_lower:
            return "DISTRICT"
        return "SC"

    # ── Sections Interpreted ──────────────────────────────────────────────

    def extract_sections_interpreted(self, text: str) -> str | None:
        """
        Extract all section/article references from judgment text.

        Returns JSON string of [{act, section}, ...] or None.
        """
        sections: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for pattern in SECTION_REF_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()

                if len(groups) >= 2:
                    section_num = groups[0].strip()
                    act_ref = groups[1].strip().rstrip(".,;")
                elif len(groups) == 1:
                    section_num = groups[0].strip()
                    act_ref = "Constitution of India"
                else:
                    continue

                act_name = self._resolve_act_name(act_ref)

                key = (act_name, section_num)
                if key not in seen:
                    seen.add(key)
                    sections.append({"act": act_name, "section": section_num})

        return json.dumps(sections) if sections else None

    def _resolve_act_name(self, act_ref: str) -> str:
        """Resolve act reference to canonical name."""
        act_ref_clean = act_ref.strip().rstrip(".,;")

        # Exact match
        if act_ref_clean in ACT_ABBREVIATIONS:
            return ACT_ABBREVIATIONS[act_ref_clean]

        # Case-insensitive exact
        for abbr, full_name in ACT_ABBREVIATIONS.items():
            if abbr.lower() == act_ref_clean.lower():
                return full_name

        # Partial match
        for abbr, full_name in ACT_ABBREVIATIONS.items():
            if abbr.lower() in act_ref_clean.lower():
                return full_name

        return act_ref_clean

    # ── Domain Detection ──────────────────────────────────────────────────

    def _detect_domain(self, sections_json: str | None) -> str | None:
        """
        Auto-detect legal domain from the acts referenced in the judgment.

        Counts domain hits from sections_interpreted and returns the
        most common domain.
        """
        if not sections_json:
            return None

        try:
            sections = json.loads(sections_json)
        except (json.JSONDecodeError, TypeError):
            return None

        domain_counts: dict[str, int] = {}
        for sec in sections:
            act = sec.get("act", "")
            for act_key, domain in ACT_DOMAIN_MAP.items():
                if act_key.lower() in act.lower():
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    break

        if not domain_counts:
            return None

        return max(domain_counts, key=domain_counts.get)

    # ── Overruled Detection ───────────────────────────────────────────────

    def _detect_overruled(self, result: ParsedJudgment, text: str) -> None:
        """Detect if the judgment text indicates overruling."""
        for pattern in OVERRULED_PATTERNS:
            match = pattern.search(text[:5000])
            if match:
                # Get surrounding context (80 chars each side)
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 80)
                context = text[start:end].strip()

                # Only flag if it's about this judgment being overruled
                # (not this judgment overruling something else)
                if "is overruled" in context.lower() or "stood overruled" in context.lower():
                    result.is_overruled = True
                    result.overruled_note = context[:300]
                    break

    # ── Component Splitting ───────────────────────────────────────────────

    def _split_components(self, text: str) -> dict[str, str | None]:
        """Split judgment text into components based on section headers."""
        components: dict[str, str | None] = {
            "headnote": None,
            "facts": None,
            "issues": None,
            "ratio": None,
            "order": None,
        }

        boundaries: list[tuple[int, str]] = []
        for comp_name, pattern in COMPONENT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                boundaries.append((match.end(), comp_name))

        if not boundaries:
            return components

        boundaries.sort(key=lambda x: x[0])

        for i, (start_pos, comp_name) in enumerate(boundaries):
            end_pos = (
                boundaries[i + 1][0] - 200 if i + 1 < len(boundaries)
                else min(start_pos + 5000, len(text))
            )
            # Ensure end_pos is after start_pos
            end_pos = max(end_pos, start_pos + 10)

            component_text = text[start_pos:end_pos].strip()
            if len(component_text) > 5000:
                component_text = component_text[:5000] + "..."

            if component_text:
                components[comp_name] = component_text

        return components

    # ── Stats Tracking ────────────────────────────────────────────────────

    def _track_stats(self, result: ParsedJudgment) -> None:
        """Track parse quality metrics."""
        self.last_parse_stats = JudgmentParseStats(
            case_name=result.case_name[:80],
            has_headnote=bool(result.headnote),
            has_ratio=bool(result.ratio_decidendi),
            has_facts=bool(result.facts),
            has_order=bool(result.order),
            has_citation=bool(result.citation_scc or result.citation_air or result.citation_other),
            has_bench=bool(result.bench),
            has_date=bool(result.judgment_date),
            sections_count=len(json.loads(result.sections_interpreted))
            if result.sections_interpreted else 0,
            domain_detected=result.domain,
            text_length=len(result.full_text) if result.full_text else 0,
        )