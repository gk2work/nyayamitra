"""
NyayaMitra — Act Parser Processor (Sprint 7 — Full Corpus).

Standalone, reusable processor that converts raw act HTML or plain text
into a structured list of sections. Decoupled from scrapers for
testability and reuse.

Sprint 7 improvements over Phase 1:
    - Constitution support: parses Articles (not "Section") + Schedules
    - Sub-section handling: preserves (1), (2), (a), (b) numbering
    - Amendment markers: detects "[Substituted by Act X of YYYY]" notes
    - Encoding robustness: handles India Code's mixed encodings
    - Schedule parsing: captures Schedules as separate sections
    - Repealed section detection: marks omitted/repealed sections
    - Statistics: reports parsing quality metrics per act

Supports:
    - India Code HTML format (legislative.gov.in)
    - Plain text pasted from PDFs
    - Constitution of India (Article-based, not Section-based)
    - Acts with Schedules, Appendices, Forms

Parsing strategies (tried in order):
    1. Structured HTML: section headers in tables, divs, headings
    2. Regex-based: split text on "Section NNN" / "Article NNN" patterns
    3. Coarse fallback: numbered pattern splitting

Usage:
    from data.processors.act_parser import ActParser

    parser = ActParser()
    sections = parser.parse_html(html, "Indian Penal Code, 1860")
    sections = parser.parse_text(raw_text, "Constitution of India")
    stats = parser.last_parse_stats
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.exceptions import ParseError

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParsedSection:
    """A single section/article extracted from an act."""

    section_number: str
    title: str | None = None
    text: str = ""
    chapter: str | None = None
    part: str | None = None
    explanation: str | None = None
    proviso: str | None = None
    is_article: bool = False         # True for Constitution articles
    is_schedule: bool = False        # True for schedule entries
    is_repealed: bool = False        # Section has been omitted/repealed
    amendment_note: str | None = None  # "[Substituted by Act X of YYYY]"

    def is_valid(self) -> bool:
        """Check minimum validity: must have a number and some text."""
        if not self.section_number:
            return False
        # Repealed sections are valid even with minimal text
        if self.is_repealed:
            return True
        return bool(self.text and len(self.text.strip()) >= 10)


@dataclass
class ParseStats:
    """Parsing quality metrics for a single act."""

    act_name: str = ""
    strategy_used: str = ""
    total_extracted: int = 0
    valid_sections: int = 0
    invalid_dropped: int = 0
    repealed_found: int = 0
    articles_found: int = 0
    schedules_found: int = 0
    chapters_found: int = 0
    parts_found: int = 0
    provisos_found: int = 0
    explanations_found: int = 0
    amendments_found: int = 0
    avg_section_length: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section patterns ──────────────────────────────────────────────────────

# Matches: "Section 302", "Section 41A", "Section 41(1)(b)(ii)", "S. 302"
SECTION_HEADER_PATTERN = re.compile(
    r"(?:Section|S\.?)\s*(\d+[A-Za-z]*(?:\([^)]*\))*)\s*[-.:—]\s*(.*)",
    re.IGNORECASE,
)

# For splitting raw text on section boundaries
SECTION_SPLIT_PATTERN = re.compile(
    r"\n\s*(?:Section|S\.?)\s+(\d+[A-Za-z]*(?:\([^)]*\))*)\s*[-.:—]\s*",
    re.IGNORECASE,
)

# ── Article patterns (Constitution) ───────────────────────────────────────

# Matches: "Article 14", "Article 21", "Article 368(1)"
ARTICLE_HEADER_PATTERN = re.compile(
    r"(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\([^)]*\))*)\s*[-.:—]\s*(.*)",
    re.IGNORECASE,
)

ARTICLE_SPLIT_PATTERN = re.compile(
    r"\n\s*(?:Article|Art\.?)\s+(\d+[A-Za-z]*(?:\([^)]*\))*)\s*[-.:—]\s*",
    re.IGNORECASE,
)

# ── Structural patterns ───────────────────────────────────────────────────

CHAPTER_PATTERN = re.compile(
    r"(?:CHAPTER|Chapter)\s+([IVXLCDM]+[A-Z]*|\d+[A-Z]*)\s*[-.:—]\s*(.*)",
)

PART_PATTERN = re.compile(
    r"(?:PART|Part)\s+([IVXLCDM]+[A-Z]*|\d+[A-Z]*)\s*[-.:—]\s*(.*)",
)

SCHEDULE_HEADER_PATTERN = re.compile(
    r"(?:THE\s+)?(?:SCHEDULE|Schedule|FIRST SCHEDULE|SECOND SCHEDULE|THIRD SCHEDULE|"
    r"FOURTH SCHEDULE|FIFTH SCHEDULE|SIXTH SCHEDULE|SEVENTH SCHEDULE|EIGHTH SCHEDULE|"
    r"NINTH SCHEDULE|TENTH SCHEDULE|ELEVENTH SCHEDULE|TWELFTH SCHEDULE)"
    r"\s*([IVXLCDM]*\d*[A-Z]*)",
    re.IGNORECASE,
)

# ── Content patterns ──────────────────────────────────────────────────────

PROVISO_PATTERN = re.compile(
    r"(?:^|\n)\s*Provided\s+that\b",
    re.IGNORECASE,
)

EXPLANATION_PATTERN = re.compile(
    r"(?:^|\n)\s*Explanation\.?\s*[-.:—]?\s*",
    re.IGNORECASE,
)

ILLUSTRATION_PATTERN = re.compile(
    r"(?:^|\n)\s*Illustration\.?\s*[-.:—]?\s*",
    re.IGNORECASE,
)

# ── Amendment / Repeal markers ────────────────────────────────────────────

AMENDMENT_PATTERN = re.compile(
    r"\[(?:Substituted|Inserted|Added|Amended)\s+by\s+(?:Act|the\s+\w+\s+Act)"
    r"[^]]*\]",
    re.IGNORECASE,
)

REPEALED_PATTERN = re.compile(
    r"(?:\[Omitted|Repealed|Rep\.\s+by|Omitted\s+by|Section\s+\d+\s+omitted)",
    re.IGNORECASE,
)

# ── Encoding artifacts ────────────────────────────────────────────────────

ENCODING_FIXES = {
    "\xa0": " ",        # Non-breaking space
    "\u200b": "",       # Zero-width space
    "\u200c": "",       # Zero-width non-joiner
    "\u200d": "",       # Zero-width joiner
    "\ufeff": "",       # BOM
    "\u2013": "–",      # En-dash
    "\u2014": "—",      # Em-dash
    "\u2018": "'",      # Left single quote
    "\u2019": "'",      # Right single quote
    "\u201c": '"',      # Left double quote
    "\u201d": '"',      # Right double quote
    "\u2026": "...",    # Ellipsis
    "\r\n": "\n",       # Windows newline
    "\r": "\n",         # Old Mac newline
}


# ═══════════════════════════════════════════════════════════════════════════════
# Act Parser
# ═══════════════════════════════════════════════════════════════════════════════


class ActParser:
    """
    Parses raw act HTML or plain text into structured sections.

    Sprint 7 enhancements:
    - Constitution mode: auto-detects and parses Articles instead of Sections
    - Schedule parsing: captures schedules as separate entries
    - Repealed/omitted section detection
    - Amendment note extraction
    - Encoding fix pipeline
    - Parse statistics tracking

    Thread-safe and stateless — create one instance and reuse.
    """

    def __init__(self):
        self.last_parse_stats: ParseStats | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def parse_html(self, html: str, act_name: str = "") -> list[ParsedSection]:
        """
        Parse an act's HTML page into individual sections.

        Auto-detects whether to use Section or Article parsing based
        on the act name and content.
        """
        if not html or not html.strip():
            raise ParseError(
                source="act_parser",
                reason="Empty HTML input",
                document_hint=act_name,
            )

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
        except ImportError:
            # Fallback to html.parser if lxml not available
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            raise ParseError(
                source="act_parser",
                reason=f"HTML parsing failed: {e}",
                document_hint=act_name,
            )

        is_constitution = self._is_constitution(act_name, html)

        # Strategy 1: Structured HTML elements
        sections = self._parse_structured(soup, act_name, is_constitution)

        # Strategy 2: Fall back to raw text extraction
        if not sections:
            full_text = soup.get_text(separator="\n")
            full_text = self._fix_encoding(full_text)
            sections = self._parse_regex(full_text, act_name, is_constitution)

        # Strategy 3: Coarse fallback
        if not sections:
            full_text = soup.get_text(separator="\n")
            full_text = self._fix_encoding(full_text)
            sections = self._parse_coarse(full_text, act_name, is_constitution)

        # Post-process: detect amendments, repeals
        for s in sections:
            self._detect_amendment(s)
            self._detect_repeal(s)

        # Filter and compute stats
        valid = [s for s in sections if s.is_valid()]
        strategy = "structured" if sections and len(valid) > 0 else "regex"
        if not sections:
            strategy = "coarse"

        self.last_parse_stats = self._compute_stats(
            act_name, strategy, sections, valid,
        )

        logger.info(
            "act_parsed",
            act=act_name,
            strategy=strategy,
            total=len(sections),
            valid=len(valid),
            is_constitution=is_constitution,
        )

        return valid

    def parse_text(self, text: str, act_name: str = "") -> list[ParsedSection]:
        """
        Parse plain text of an act into individual sections.
        """
        if not text or not text.strip():
            raise ParseError(
                source="act_parser",
                reason="Empty text input",
                document_hint=act_name,
            )

        text = self._fix_encoding(text)
        is_constitution = self._is_constitution(act_name, text)

        sections = self._parse_regex(text, act_name, is_constitution)
        if not sections:
            sections = self._parse_coarse(text, act_name, is_constitution)

        for s in sections:
            self._detect_amendment(s)
            self._detect_repeal(s)

        valid = [s for s in sections if s.is_valid()]

        self.last_parse_stats = self._compute_stats(
            act_name, "regex", sections, valid,
        )

        logger.info(
            "act_text_parsed",
            act=act_name,
            total=len(sections),
            valid=len(valid),
        )

        return valid

    # ── Auto-Detection ────────────────────────────────────────────────────

    def _is_constitution(self, act_name: str, content: str) -> bool:
        """Detect if the act is the Constitution of India."""
        name_lower = act_name.lower()
        if "constitution" in name_lower:
            return True
        # Check content for Article headers (more Articles than Sections)
        article_count = len(ARTICLE_HEADER_PATTERN.findall(content[:10000]))
        section_count = len(SECTION_HEADER_PATTERN.findall(content[:10000]))
        return article_count > section_count and article_count >= 3

    # ── Strategy 1: Structured HTML Parsing ───────────────────────────────

    def _parse_structured(
        self,
        soup,
        act_name: str,
        is_constitution: bool,
    ) -> list[ParsedSection]:
        """
        Parse sections from structured HTML elements.

        Walks through elements looking for section/article headers,
        tracking chapter/part context.
        """
        sections: list[ParsedSection] = []
        current_chapter: str | None = None
        current_part: str | None = None

        header_pattern = ARTICLE_HEADER_PATTERN if is_constitution else SECTION_HEADER_PATTERN

        for element in soup.find_all(
            ["p", "div", "h2", "h3", "h4", "h5", "h6", "tr", "span", "li"],
        ):
            text = element.get_text(strip=True)
            if not text or len(text) < 3:
                continue

            # Track chapter context
            ch_match = CHAPTER_PATTERN.match(text)
            if ch_match:
                current_chapter = f"Chapter {ch_match.group(1)} - {ch_match.group(2)}".strip()
                continue

            # Track part context
            pt_match = PART_PATTERN.match(text)
            if pt_match:
                current_part = f"Part {pt_match.group(1)} - {pt_match.group(2)}".strip()
                continue

            # Track schedule
            sch_match = SCHEDULE_HEADER_PATTERN.match(text)
            if sch_match:
                schedule_text = self._collect_section_text(element, text)
                schedule_num = sch_match.group(0).strip()
                sections.append(ParsedSection(
                    section_number=schedule_num,
                    title=schedule_num,
                    text=self._clean_text(schedule_text),
                    chapter=current_chapter,
                    part=current_part,
                    is_schedule=True,
                ))
                continue

            # Detect section/article header
            sec_match = header_pattern.match(text)
            if sec_match:
                section_num = sec_match.group(1).strip()
                rest = sec_match.group(2).strip()

                title, body = self._split_title_body(rest)
                full_text = self._collect_section_text(element, body)
                full_text, proviso = self._extract_proviso(full_text)
                full_text, explanation = self._extract_explanation(full_text)

                sections.append(ParsedSection(
                    section_number=section_num,
                    title=title[:500] if title else None,
                    text=self._clean_text(full_text),
                    chapter=current_chapter,
                    part=current_part,
                    explanation=explanation,
                    proviso=proviso,
                    is_article=is_constitution,
                ))

        return sections

    def _collect_section_text(self, element, initial_text: str) -> str:
        """
        Collect full section text from following siblings until the next
        section/chapter/part/schedule header.
        """
        texts = [initial_text] if initial_text else []
        sibling = element.find_next_sibling()
        count = 0

        while sibling and count < 80:
            sibling_text = sibling.get_text(strip=True)
            if not sibling_text:
                sibling = sibling.find_next_sibling()
                count += 1
                continue

            # Stop at next section/article header
            if SECTION_HEADER_PATTERN.match(sibling_text):
                break
            if ARTICLE_HEADER_PATTERN.match(sibling_text):
                break
            if CHAPTER_PATTERN.match(sibling_text):
                break
            if PART_PATTERN.match(sibling_text):
                break
            if SCHEDULE_HEADER_PATTERN.match(sibling_text):
                break

            texts.append(sibling_text)
            sibling = sibling.find_next_sibling()
            count += 1

        return "\n".join(texts).strip()

    # ── Strategy 2: Regex-Based Splitting ─────────────────────────────────

    def _parse_regex(
        self,
        text: str,
        act_name: str,
        is_constitution: bool,
    ) -> list[ParsedSection]:
        """
        Split text on Section/Article boundaries using regex.
        """
        sections: list[ParsedSection] = []
        current_chapter: str | None = None
        current_part: str | None = None

        split_pattern = ARTICLE_SPLIT_PATTERN if is_constitution else SECTION_SPLIT_PATTERN

        parts = split_pattern.split(text)

        # Scan preamble for chapter/part
        if parts:
            self._scan_structural_headers(parts[0], None, None)
            for line in parts[0].split("\n"):
                line = line.strip()
                ch = CHAPTER_PATTERN.match(line)
                if ch:
                    current_chapter = f"Chapter {ch.group(1)} - {ch.group(2)}".strip()
                pt = PART_PATTERN.match(line)
                if pt:
                    current_part = f"Part {pt.group(1)} - {pt.group(2)}".strip()

        # Process section pairs
        for i in range(1, len(parts) - 1, 2):
            section_num = parts[i].strip()
            section_text = parts[i + 1].strip()

            if not section_text or len(section_text) < 5:
                continue

            # Cap very long sections
            if len(section_text) > 15000:
                section_text = section_text[:15000] + "..."

            # Update chapter/part from inline headings
            for line in section_text.split("\n")[:10]:  # Only check first 10 lines
                line = line.strip()
                ch = CHAPTER_PATTERN.match(line)
                if ch:
                    current_chapter = f"Chapter {ch.group(1)} - {ch.group(2)}".strip()
                pt = PART_PATTERN.match(line)
                if pt:
                    current_part = f"Part {pt.group(1)} - {pt.group(2)}".strip()

            title, body = self._split_title_body(section_text)
            body_or_full = body if body else section_text
            body_or_full, proviso = self._extract_proviso(body_or_full)
            body_or_full, explanation = self._extract_explanation(body_or_full)

            sections.append(ParsedSection(
                section_number=section_num,
                title=title[:500] if title else None,
                text=self._clean_text(body_or_full),
                chapter=current_chapter,
                part=current_part,
                explanation=explanation,
                proviso=proviso,
                is_article=is_constitution,
            ))

        return sections

    # ── Strategy 3: Coarse Fallback ───────────────────────────────────────

    def _parse_coarse(
        self,
        text: str,
        act_name: str,
        is_constitution: bool,
    ) -> list[ParsedSection]:
        """
        Coarse fallback: try to split on any numbered pattern.

        Used when both structured and regex parsing fail (unusual formats).
        Tries patterns like "1.", "1)", "(1)", "1 -" as section delimiters.
        """
        sections: list[ParsedSection] = []

        # Try splitting on "N." at start of line where N is 1-999
        coarse_pattern = re.compile(
            r"\n\s*(\d{1,3})\.\s+([A-Z])",
        )

        matches = list(coarse_pattern.finditer(text))
        if len(matches) < 3:
            # Not enough matches for this to be a valid split
            return []

        for i, match in enumerate(matches):
            num = match.group(1)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else min(start + 5000, len(text))

            chunk = text[start:end].strip()
            if len(chunk) < 20:
                continue

            title, body = self._split_title_body(chunk[len(num) + 1:].strip())

            sections.append(ParsedSection(
                section_number=num,
                title=title[:500] if title else None,
                text=self._clean_text(body if body else chunk),
            ))

        logger.info(
            "coarse_parse_used",
            act=act_name,
            sections_found=len(sections),
        )

        return sections

    # ── Post-Processing ───────────────────────────────────────────────────

    def _detect_amendment(self, section: ParsedSection) -> None:
        """Detect and extract amendment notes from section text."""
        match = AMENDMENT_PATTERN.search(section.text)
        if match:
            section.amendment_note = match.group(0)

    def _detect_repeal(self, section: ParsedSection) -> None:
        """Detect if a section has been omitted or repealed."""
        combined = (section.text or "") + " " + (section.title or "")
        if REPEALED_PATTERN.search(combined):
            section.is_repealed = True
            # Check if the text is just the repeal notice
            clean = re.sub(r"\[.*?\]", "", section.text).strip()
            if len(clean) < 20:
                section.is_repealed = True

    def _scan_structural_headers(
        self, text: str, chapter: str | None, part: str | None,
    ) -> tuple[str | None, str | None]:
        """Scan text for chapter/part headers and return updated context."""
        for line in text.split("\n"):
            line = line.strip()
            ch = CHAPTER_PATTERN.match(line)
            if ch:
                chapter = f"Chapter {ch.group(1)} - {ch.group(2)}".strip()
            pt = PART_PATTERN.match(line)
            if pt:
                part = f"Part {pt.group(1)} - {pt.group(2)}".strip()
        return chapter, part

    # ── Text Processing Helpers ───────────────────────────────────────────

    def _split_title_body(self, text: str) -> tuple[str | None, str]:
        """
        Split section text into title and body.

        Common pattern: "Punishment for murder.— Whoever commits..."
        """
        # Try splitting on ".—" or ".-" or ".--" (Indian legal formatting)
        for delimiter in [".—", ".-", ".--", ". —", ".–"]:
            if delimiter in text:
                parts = text.split(delimiter, 1)
                title = parts[0].strip()
                body = parts[1].strip() if len(parts) > 1 else ""
                if title and len(title) < 300:
                    return title, body

        # Try splitting on first period + space + capital letter
        match = re.match(r"^([^.]+\.)\s+([A-Z].*)", text, re.DOTALL)
        if match and len(match.group(1)) < 200:
            return match.group(1).rstrip(".").strip(), match.group(2).strip()

        return None, text

    def _extract_proviso(self, text: str) -> tuple[str, str | None]:
        """Extract proviso ("Provided that...") from section text."""
        match = PROVISO_PATTERN.search(text)
        if not match:
            return text, None
        main_text = text[:match.start()].strip()
        proviso_text = text[match.start():].strip()
        return main_text, proviso_text

    def _extract_explanation(self, text: str) -> tuple[str, str | None]:
        """Extract explanation from section text."""
        match = EXPLANATION_PATTERN.search(text)
        if not match:
            return text, None
        main_text = text[:match.start()].strip()
        explanation_text = text[match.start():].strip()
        return main_text, explanation_text

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding artifacts from India Code pages."""
        for bad, good in ENCODING_FIXES.items():
            text = text.replace(bad, good)
        # Remove any remaining control characters (except \n, \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text

    def _clean_text(self, text: str) -> str:
        """Clean section text: normalize whitespace, strip artifacts."""
        if not text:
            return ""

        text = self._fix_encoding(text)

        # Strip any remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Normalize whitespace (preserve paragraph breaks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    # ── Statistics ────────────────────────────────────────────────────────

    def _compute_stats(
        self,
        act_name: str,
        strategy: str,
        all_sections: list[ParsedSection],
        valid_sections: list[ParsedSection],
    ) -> ParseStats:
        """Compute parsing quality metrics."""
        stats = ParseStats(
            act_name=act_name,
            strategy_used=strategy,
            total_extracted=len(all_sections),
            valid_sections=len(valid_sections),
            invalid_dropped=len(all_sections) - len(valid_sections),
        )

        for s in valid_sections:
            if s.is_repealed:
                stats.repealed_found += 1
            if s.is_article:
                stats.articles_found += 1
            if s.is_schedule:
                stats.schedules_found += 1
            if s.proviso:
                stats.provisos_found += 1
            if s.explanation:
                stats.explanations_found += 1
            if s.amendment_note:
                stats.amendments_found += 1

        chapters = set()
        parts = set()
        for s in valid_sections:
            if s.chapter:
                chapters.add(s.chapter)
            if s.part:
                parts.add(s.part)
        stats.chapters_found = len(chapters)
        stats.parts_found = len(parts)

        if valid_sections:
            total_len = sum(len(s.text) for s in valid_sections)
            stats.avg_section_length = total_len // len(valid_sections)

        return stats