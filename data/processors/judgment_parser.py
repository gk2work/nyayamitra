"""
NyayaMitra — Judgment Parser Processor.

Standalone processor that takes raw judgment text (from Indian Kanoon HTML,
SCI PDFs, or plain text) and extracts structured components:
- Case name and case number
- Bench composition and size
- Judgment date and year
- SCC and AIR citations
- Headnote, ratio decidendi, facts, issues, order
- Sections interpreted (as JSON list)

Supports:
- Indian Kanoon HTML/API response format
- Plain text from PDF extraction
- SCI website HTML

Usage:
    from data.processors.judgment_parser import JudgmentParser

    parser = JudgmentParser()
    result = parser.parse_text(judgment_text)
    result = parser.parse_html(html_content)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import structlog
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.exceptions import ParseError

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Act Name Abbreviation Mapping
# ═══════════════════════════════════════════════════════════════════════════════
# Maps common abbreviations used in judgments to canonical act names.
# Used when extracting "sections interpreted" from judgment text.

ACT_ABBREVIATIONS: dict[str, str] = {
    # Criminal
    "IPC": "Indian Penal Code, 1860",
    "Indian Penal Code": "Indian Penal Code, 1860",
    "CrPC": "Code of Criminal Procedure, 1973",
    "Cr.P.C.": "Code of Criminal Procedure, 1973",
    "Cr.P.C": "Code of Criminal Procedure, 1973",
    "Code of Criminal Procedure": "Code of Criminal Procedure, 1973",
    "Evidence Act": "Indian Evidence Act, 1872",
    "Indian Evidence Act": "Indian Evidence Act, 1872",
    "BNS": "Bharatiya Nyaya Sanhita, 2023",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "BSA": "Bharatiya Sakshya Adhiniyam, 2023",
    "NDPS Act": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "Dowry Prohibition Act": "Dowry Prohibition Act, 1961",
    # Property / Civil
    "CPC": "Code of Civil Procedure, 1908",
    "C.P.C.": "Code of Civil Procedure, 1908",
    "TPA": "Transfer of Property Act, 1882",
    "Transfer of Property Act": "Transfer of Property Act, 1882",
    "Registration Act": "Indian Registration Act, 1908",
    "Contract Act": "Indian Contract Act, 1872",
    "Indian Contract Act": "Indian Contract Act, 1872",
    "Specific Relief Act": "Specific Relief Act, 1963",
    "RERA": "Real Estate (Regulation and Development) Act, 2016",
    "Stamp Act": "Indian Stamp Act, 1899",
    # Family
    "HMA": "Hindu Marriage Act, 1955",
    "Hindu Marriage Act": "Hindu Marriage Act, 1955",
    "SMA": "Special Marriage Act, 1954",
    "Special Marriage Act": "Special Marriage Act, 1954",
    "DV Act": "Protection of Women from Domestic Violence Act, 2005",
    "Domestic Violence Act": "Protection of Women from Domestic Violence Act, 2005",
    "Hindu Succession Act": "Hindu Succession Act, 1956",
    "Guardians and Wards Act": "Guardians and Wards Act, 1890",
    # Consumer
    "CPA": "Consumer Protection Act, 2019",
    "Consumer Protection Act": "Consumer Protection Act, 2019",
    # Labor
    "ID Act": "Industrial Disputes Act, 1947",
    "Industrial Disputes Act": "Industrial Disputes Act, 1947",
    "POSH Act": "Sexual Harassment of Women at Workplace Act, 2013",
    "Minimum Wages Act": "Minimum Wages Act, 1948",
    "Payment of Wages Act": "Payment of Wages Act, 1936",
    "Factories Act": "Factories Act, 1948",
    # Constitutional
    "Constitution": "Constitution of India",
    "Constitution of India": "Constitution of India",
    "RTI": "Right to Information Act, 2005",
    "RTI Act": "Right to Information Act, 2005",
    "Right to Information Act": "Right to Information Act, 2005",
    # IP / IT
    "IT Act": "Information Technology Act, 2000",
    "Information Technology Act": "Information Technology Act, 2000",
    "Copyright Act": "Copyright Act, 1957",
    "Patents Act": "Patents Act, 1970",
    "Trade Marks Act": "Trade Marks Act, 1999",
    "Trademarks Act": "Trade Marks Act, 1999",
    # Other commonly cited
    "Arbitration Act": "Arbitration and Conciliation Act, 1996",
    "Companies Act": "Companies Act, 2013",
    "NI Act": "Negotiable Instruments Act, 1881",
    "Negotiable Instruments Act": "Negotiable Instruments Act, 1881",
    "IBC": "Insolvency and Bankruptcy Code, 2016",
    "Insolvency and Bankruptcy Code": "Insolvency and Bankruptcy Code, 2016",
    "POCSO Act": "Protection of Children from Sexual Offences Act, 2012",
    "SC/ST Act": "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989",
    "Passports Act": "Passports Act, 1967",
    "Arms Act": "Arms Act, 1959",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Regex Patterns for Extraction
# ═══════════════════════════════════════════════════════════════════════════════

# Section references: "Section 302 of the IPC", "under S. 41 CrPC", "Article 21"
SECTION_REF_PATTERNS = [
    # "Section 302 of the Indian Penal Code" / "Section 302 of IPC"
    re.compile(
        r"(?:Section|S\.?|Sec\.?)\s+(\d+[A-Za-z]*(?:\([^)]*\))*)\s+of\s+(?:the\s+)?([A-Za-z][A-Za-z\s\.,]+?)(?:\s*,\s*\d{4})?(?=\s*[,;.\)\]]|\s+(?:and|read|r/w|with)|\s*$)",
        re.IGNORECASE,
    ),
    # "under S. 41 CrPC" / "u/s 302 IPC"
    re.compile(
        r"(?:under|u/s|u/S)\s+(?:Section|S\.?)?\s*(\d+[A-Za-z]*(?:\([^)]*\))*)\s+(?:of\s+)?([A-Z][A-Za-z\s\.]+?)(?:\s*,\s*\d{4})?(?=[\s,;.\)])",
        re.IGNORECASE,
    ),
    # "Article 21 of the Constitution"
    re.compile(
        r"(?:Article|Art\.?)\s+(\d+[A-Za-z]*(?:\([^)]*\))*)\s+of\s+(?:the\s+)?(Constitution[A-Za-z\s]*)",
        re.IGNORECASE,
    ),
    # Standalone "Article 21" (assume Constitution)
    re.compile(
        r"(?:Article|Art\.?)\s+(\d+[A-Za-z]*(?:\([^)]*\))*)",
        re.IGNORECASE,
    ),
]

# Citation patterns
SCC_CITATION_PATTERN = re.compile(r"\(\d{4}\)\s+\d+\s+SCC\s+\d+")
AIR_CITATION_PATTERN = re.compile(r"AIR\s+\d{4}\s+SC\s+\d+")

# Date patterns in judgments
DATE_PATTERNS = [
    # "13th November, 2013" / "13 November 2013"
    re.compile(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*(\d{4})",
        re.IGNORECASE,
    ),
    # "01-01-2024" / "01/01/2024"
    re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})"),
    # "2024-01-01" (ISO)
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Bench composition: "Justice A.S. Anand, Justice M. Srinivasan"
BENCH_PATTERN = re.compile(
    r"(?:BENCH|CORAM|Before)[:\s]*(.*?)(?:\n|$)",
    re.IGNORECASE,
)

# Case number: "Writ Petition (Crl.) No. 539 of 1986"
CASE_NUMBER_PATTERN = re.compile(
    r"((?:Writ Petition|W\.?P\.?|SLP|C\.?A\.?|Crl\.?\s*A\.?|Criminal Appeal|Civil Appeal|Special Leave Petition)"
    r"[^)]*(?:\([^)]*\))?\s*No\.?\s*\d+[^,\n]*(?:of|/)\s*\d{4})",
    re.IGNORECASE,
)

# Component section headers in judgment text
COMPONENT_PATTERNS = {
    "headnote": re.compile(r"(?:^|\n)\s*(?:HEAD\s*NOTE|HEADNOTE|SUMMARY)\s*[:\-]?\s*", re.IGNORECASE),
    "facts": re.compile(r"(?:^|\n)\s*(?:FACTS|STATEMENT\s+OF\s+FACTS|FACTUAL\s+BACKGROUND)\s*[:\-]?\s*", re.IGNORECASE),
    "issues": re.compile(r"(?:^|\n)\s*(?:ISSUES?\s+(?:FRAMED|FOR\s+CONSIDERATION)|QUESTIONS?\s+(?:OF\s+LAW|FOR\s+DETERMINATION))\s*[:\-]?\s*", re.IGNORECASE),
    "ratio": re.compile(r"(?:^|\n)\s*(?:RATIO\s+DECIDENDI|RATIO|REASONING|ANALYSIS\s+AND\s+REASONING|DISCUSSION)\s*[:\-]?\s*", re.IGNORECASE),
    "order": re.compile(r"(?:^|\n)\s*(?:ORDER|DIRECTION|RESULT|CONCLUSION|OPERATIVE\s+PART)\s*[:\-]?\s*", re.IGNORECASE),
}


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
    headnote: str | None = None
    facts: str | None = None
    issues: str | None = None
    ratio_decidendi: str | None = None
    obiter_dicta: str | None = None
    order: str | None = None
    sections_interpreted: str | None = None  # JSON string
    full_text: str | None = None

    def is_valid(self) -> bool:
        """Check minimum validity: must have case name and year."""
        return bool(self.case_name and self.year > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Judgment Parser
# ═══════════════════════════════════════════════════════════════════════════════


class JudgmentParser:
    """
    Extracts structured metadata and components from raw judgment text.

    Handles multiple formats:
    - Indian Kanoon API responses (HTML with doc/headline fields)
    - SCI website HTML
    - Plain text from PDF extraction

    Thread-safe and stateless — create one instance and reuse.
    """

    def parse_html(self, html: str, metadata: dict | None = None) -> ParsedJudgment:
        """
        Parse judgment from HTML content.

        Args:
            html: Raw HTML string.
            metadata: Optional pre-extracted metadata (court, date hints).

        Returns:
            ParsedJudgment with extracted components.

        Raises:
            ParseError: If HTML cannot be parsed.
        """
        if not html or not html.strip():
            raise ParseError(source="judgment_parser", reason="Empty HTML input")

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as e:
            raise ParseError(source="judgment_parser", reason=f"HTML parsing failed: {e}")

        text = soup.get_text(separator="\n")
        result = self._extract_all(text, metadata)

        # Try to get headline from structured HTML (Indian Kanoon format)
        headline_el = soup.find(class_=re.compile(r"headline|headnote|summary", re.IGNORECASE))
        if headline_el and not result.headnote:
            result.headnote = headline_el.get_text(strip=True)[:3000]

        return result

    def parse_text(self, text: str, metadata: dict | None = None) -> ParsedJudgment:
        """
        Parse judgment from plain text.

        Args:
            text: Raw judgment text.
            metadata: Optional pre-extracted metadata.

        Returns:
            ParsedJudgment with extracted components.

        Raises:
            ParseError: If text cannot be parsed.
        """
        if not text or not text.strip():
            raise ParseError(source="judgment_parser", reason="Empty text input")

        return self._extract_all(text, metadata)

    def parse_indian_kanoon_doc(self, doc: dict) -> ParsedJudgment:
        """
        Parse an Indian Kanoon API response document.

        This is a convenience method that handles the specific structure
        of Indian Kanoon API responses with known field names.

        Args:
            doc: Dict from Indian Kanoon API (with title, doc, headline, etc.).

        Returns:
            ParsedJudgment with extracted components.
        """
        result = ParsedJudgment()

        # Extract from known API fields
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

        # Headnote
        result.headnote = doc.get("headline", "")
        if not result.headnote and doc_text:
            result.headnote = doc_text[:1000].strip()

        # Full text
        result.full_text = doc_text[:50000] if doc_text else None

        # Sections interpreted
        if doc_text:
            result.sections_interpreted = self.extract_sections_interpreted(doc_text)

        # Parse components from full text
        if doc_text:
            components = self._split_components(doc_text)
            if not result.headnote and components.get("headnote"):
                result.headnote = components["headnote"]
            result.facts = components.get("facts")
            result.issues = components.get("issues")
            result.ratio_decidendi = components.get("ratio")
            result.order = components.get("order")

        return result

    # ─── Core Extraction ─────────────────────────────────────────────────

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

        # Sections interpreted
        result.sections_interpreted = self.extract_sections_interpreted(text)

        # Components
        components = self._split_components(text)
        result.headnote = components.get("headnote")
        result.facts = components.get("facts")
        result.issues = components.get("issues")
        result.ratio_decidendi = components.get("ratio")
        result.order = components.get("order")

        # Full text (capped)
        result.full_text = text[:50000] if text else None

        # If no headnote extracted, use first 1000 chars
        if not result.headnote and text:
            result.headnote = text[:1000].strip()

        logger.info(
            "judgment_parsed",
            case=result.case_name[:80] if result.case_name else "unknown",
            year=result.year,
            has_headnote=bool(result.headnote),
            has_ratio=bool(result.ratio_decidendi),
            has_facts=bool(result.facts),
            sections_count=len(json.loads(result.sections_interpreted)) if result.sections_interpreted else 0,
        )

        return result

    # ─── Individual Extractors ───────────────────────────────────────────

    def _extract_case_name(self, text: str) -> str:
        """Extract case name from judgment text (look for 'v.' or 'vs.' pattern)."""
        # Look in first 500 chars for "X v. Y" pattern
        header = text[:500]
        match = re.search(
            r"([\w\s\.\,\(\)]+?)\s+v[s]?\.\s+([\w\s\.\,\(\)]+?)(?:\n|on\s+\d|\s*$)",
            header,
        )
        if match:
            petitioner = match.group(1).strip()
            respondent = match.group(2).strip()
            # Clean up excessive whitespace
            petitioner = re.sub(r"\s+", " ", petitioner)
            respondent = re.sub(r"\s+", " ", respondent)
            return f"{petitioner} v. {respondent}"[:500]

        # Fallback: first line that looks like a title
        first_line = text.split("\n")[0].strip()
        if len(first_line) > 10:
            return first_line[:500]

        return ""

    def _extract_case_number(self, text: str) -> str | None:
        """Extract case number from judgment text."""
        match = CASE_NUMBER_PATTERN.search(text[:1000])
        return match.group(1).strip() if match else None

    def _extract_bench(self, text: str) -> tuple[str | None, int | None]:
        """Extract bench composition and estimate size."""
        match = BENCH_PATTERN.search(text[:2000])
        if not match:
            return None, None

        bench_text = match.group(1).strip()

        # Clean up bench text
        bench_text = re.sub(r"\s+", " ", bench_text)
        bench_text = bench_text[:500]

        # Count justices
        justice_count = len(re.findall(
            r"\bjustice\b|\bj\.\b|\bjj\.\b",
            bench_text,
            re.IGNORECASE,
        ))
        bench_size = max(justice_count, 1) if bench_text else None

        return bench_text, bench_size

    def _extract_date(self, text: str) -> date | None:
        """Extract judgment date from text."""
        header = text[:1000]

        for pattern in DATE_PATTERNS:
            match = pattern.search(header)
            if not match:
                continue

            groups = match.groups()
            try:
                if len(groups) == 3 and groups[1].isalpha():
                    # "13 November 2013" format
                    day = int(groups[0])
                    month = MONTH_MAP.get(groups[1].lower(), 0)
                    year = int(groups[2])
                    if month:
                        return date(year, month, day)

                elif len(groups) == 3 and len(groups[0]) == 4:
                    # ISO format "2024-01-01"
                    return date(int(groups[0]), int(groups[1]), int(groups[2]))

                elif len(groups) == 3:
                    # "01-01-2024" or "01/01/2024"
                    d, m, y = int(groups[0]), int(groups[1]), int(groups[2])
                    if y > 1900:
                        return date(y, m, d)

            except (ValueError, TypeError):
                continue

        return None

    def _parse_date_string(self, date_str: str) -> date | None:
        """Parse a date string in common formats."""
        for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y", "%d %B %Y"]:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None

    def _extract_citation(self, text: str, pattern: re.Pattern) -> str | None:
        """Extract a citation matching the given pattern."""
        match = pattern.search(text)
        return match.group(0) if match else None

    def _detect_court_type(self, court: str) -> str:
        """Detect court type from court name."""
        court_lower = court.lower()
        if "supreme" in court_lower:
            return "SC"
        if "high" in court_lower:
            return "HC"
        if "tribunal" in court_lower:
            return "Tribunal"
        return "SC"  # default assumption

    # ─── Sections Interpreted Extraction ─────────────────────────────────

    def extract_sections_interpreted(self, text: str) -> str | None:
        """
        Extract all section/article references from judgment text.

        Scans for patterns like "Section 302 of IPC", "Article 21",
        "u/s 498A IPC" and resolves act abbreviations to canonical names.

        Returns:
            JSON string of [{act, section}, ...] or None if none found.
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
                    # Standalone Article pattern — assume Constitution
                    section_num = groups[0].strip()
                    act_ref = "Constitution of India"
                else:
                    continue

                # Resolve act abbreviation
                act_name = self._resolve_act_name(act_ref)

                key = (act_name, section_num)
                if key not in seen:
                    seen.add(key)
                    sections.append({"act": act_name, "section": section_num})

        if not sections:
            return None

        return json.dumps(sections)

    def _resolve_act_name(self, act_ref: str) -> str:
        """
        Resolve an act reference to its canonical name.

        Tries exact match first, then partial match against known abbreviations.
        """
        act_ref_clean = act_ref.strip().rstrip(".,;")

        # Exact match
        if act_ref_clean in ACT_ABBREVIATIONS:
            return ACT_ABBREVIATIONS[act_ref_clean]

        # Case-insensitive match
        for abbr, full_name in ACT_ABBREVIATIONS.items():
            if abbr.lower() == act_ref_clean.lower():
                return full_name

        # Partial match: check if the reference contains a known abbreviation
        for abbr, full_name in ACT_ABBREVIATIONS.items():
            if abbr.lower() in act_ref_clean.lower():
                return full_name

        # No match found — return as-is
        return act_ref_clean

    # ─── Component Splitting ─────────────────────────────────────────────

    def _split_components(self, text: str) -> dict[str, str | None]:
        """
        Split judgment text into components (headnote, facts, issues,
        ratio, order) based on section headers.

        Returns dict with keys: headnote, facts, issues, ratio, order.
        Values are the extracted text or None if not found.
        """
        components: dict[str, str | None] = {
            "headnote": None,
            "facts": None,
            "issues": None,
            "ratio": None,
            "order": None,
        }

        # Find all component boundaries
        boundaries: list[tuple[int, str]] = []
        for comp_name, pattern in COMPONENT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                boundaries.append((match.end(), comp_name))

        if not boundaries:
            return components

        # Sort by position in text
        boundaries.sort(key=lambda x: x[0])

        # Extract text between boundaries
        for i, (start_pos, comp_name) in enumerate(boundaries):
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][0]
                # Step back to before the next header
                next_pattern = COMPONENT_PATTERNS[boundaries[i + 1][1]]
                next_match = next_pattern.search(text[boundaries[i + 1][0] - 200:boundaries[i + 1][0] + 10])
                if next_match:
                    end_pos = boundaries[i + 1][0] - 200 + next_match.start()
            else:
                end_pos = min(start_pos + 5000, len(text))

            component_text = text[start_pos:end_pos].strip()

            # Cap at reasonable length
            if len(component_text) > 5000:
                component_text = component_text[:5000] + "..."

            if component_text:
                components[comp_name] = component_text

        return components