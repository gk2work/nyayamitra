"""
NyayaMitra — Act Parser Processor.

Standalone, reusable processor that converts raw act HTML or plain text
into a structured list of sections. This decouples parsing logic from
the scraper, making it testable and reusable across different data sources.

Supports:
- India Code HTML format (legislative.gov.in)
- Plain text pasted from PDFs
- Raw text dumps from other sources

Parsing strategies (tried in order):
1. Structured HTML: look for section headers in table rows, divs, headings
2. Regex-based: split text on "Section NNN" patterns
3. Fallback: coarse splitting on any numbered patterns

Usage:
    from data.processors.act_parser import ActParser

    parser = ActParser()
    sections = parser.parse_html(html, "Indian Penal Code, 1860")
    sections = parser.parse_text(raw_text, "Indian Penal Code, 1860")
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.exceptions import ParseError

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParsedSection:
    """A single section extracted from an act."""

    section_number: str
    title: str | None = None
    text: str = ""
    chapter: str | None = None
    part: str | None = None
    explanation: str | None = None
    proviso: str | None = None

    def is_valid(self) -> bool:
        """Check minimum validity: must have a section number and some text."""
        return bool(self.section_number and self.text and len(self.text.strip()) >= 10)


# ═══════════════════════════════════════════════════════════════════════════════
# Section Pattern Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Matches: "Section 302", "Section 41A", "Section 41(1)(b)", "S. 302"
SECTION_HEADER_PATTERN = re.compile(
    r"(?:Section|S\.?)\s*(\d+[A-Za-z]*(?:\([^)]*\))*)\s*[-.:]\s*(.*)",
    re.IGNORECASE,
)

# For splitting raw text on section boundaries
SECTION_SPLIT_PATTERN = re.compile(
    r"\n\s*(?:Section|S\.?)\s+(\d+[A-Za-z]*(?:\([^)]*\))*)\s*[-.:]\s*",
    re.IGNORECASE,
)

# Chapter heading: "CHAPTER XVI - Of Offences Affecting the Human Body"
CHAPTER_PATTERN = re.compile(
    r"(?:CHAPTER|Chapter)\s+([IVXLCDM]+[A-Z]*|\d+[A-Z]*)\s*[-.:]\s*(.*)",
)

# Part heading: "PART II - General Exceptions"
PART_PATTERN = re.compile(
    r"(?:PART|Part)\s+([IVXLCDM]+[A-Z]*|\d+[A-Z]*)\s*[-.:]\s*(.*)",
)

# Schedule heading
SCHEDULE_PATTERN = re.compile(
    r"(?:SCHEDULE|Schedule)\s*([IVXLCDM]*\d*[A-Z]*)",
    re.IGNORECASE,
)

# Proviso pattern within section text
PROVISO_PATTERN = re.compile(
    r"(?:^|\n)\s*Provided\s+that\b",
    re.IGNORECASE,
)

# Explanation pattern within section text
EXPLANATION_PATTERN = re.compile(
    r"(?:^|\n)\s*Explanation\.?\s*[-.:—]?\s*",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Act Parser
# ═══════════════════════════════════════════════════════════════════════════════


class ActParser:
    """
    Parses raw act HTML or plain text into structured sections.

    Tries multiple parsing strategies in order of reliability:
    1. Structured HTML parsing (tables, headings, divs)
    2. Regex-based splitting on section patterns
    3. Coarse fallback for unusual formats

    Thread-safe and stateless — create one instance and reuse.
    """

    def parse_html(self, html: str, act_name: str = "") -> list[ParsedSection]:
        """
        Parse an act's HTML page into individual sections.

        Args:
            html: Raw HTML string from India Code or similar source.
            act_name: Name of the act (for logging context).

        Returns:
            List of ParsedSection objects.

        Raises:
            ParseError: If HTML cannot be parsed at all.
        """
        if not html or not html.strip():
            raise ParseError(
                source="act_parser",
                reason="Empty HTML input",
                document_hint=act_name,
            )

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as e:
            raise ParseError(
                source="act_parser",
                reason=f"HTML parsing failed: {e}",
                document_hint=act_name,
            )

        # Strategy 1: Structured HTML elements
        sections = self._parse_structured(soup, act_name)

        # Strategy 2: Fall back to raw text extraction
        if not sections:
            full_text = soup.get_text(separator="\n")
            sections = self._parse_regex(full_text, act_name)

        # Filter out invalid sections
        valid_sections = [s for s in sections if s.is_valid()]

        logger.info(
            "act_parsed",
            act=act_name,
            total_extracted=len(sections),
            valid=len(valid_sections),
            strategy="structured" if len(sections) != len(valid_sections) or sections else "regex",
        )

        return valid_sections

    def parse_text(self, text: str, act_name: str = "") -> list[ParsedSection]:
        """
        Parse plain text of an act into individual sections.

        Args:
            text: Raw text string (from PDF extraction, copy-paste, etc.).
            act_name: Name of the act (for logging context).

        Returns:
            List of ParsedSection objects.

        Raises:
            ParseError: If text cannot be parsed at all.
        """
        if not text or not text.strip():
            raise ParseError(
                source="act_parser",
                reason="Empty text input",
                document_hint=act_name,
            )

        sections = self._parse_regex(text, act_name)
        valid_sections = [s for s in sections if s.is_valid()]

        logger.info(
            "act_text_parsed",
            act=act_name,
            total_extracted=len(sections),
            valid=len(valid_sections),
        )

        return valid_sections

    # ─── Strategy 1: Structured HTML Parsing ─────────────────────────────

    def _parse_structured(self, soup: BeautifulSoup, act_name: str) -> list[ParsedSection]:
        """
        Parse sections from structured HTML elements.

        Walks through elements looking for section headers, tracking
        the current chapter/part context as it goes.
        """
        sections: list[ParsedSection] = []
        current_chapter: str | None = None
        current_part: str | None = None

        for element in soup.find_all(["p", "div", "h3", "h4", "h5", "tr", "span"]):
            text = element.get_text(strip=True)

            if not text or len(text) < 5:
                continue

            # Track chapter context
            chapter_match = CHAPTER_PATTERN.match(text)
            if chapter_match:
                current_chapter = f"Chapter {chapter_match.group(1)} - {chapter_match.group(2)}".strip()
                continue

            # Track part context
            part_match = PART_PATTERN.match(text)
            if part_match:
                current_part = f"Part {part_match.group(1)} - {part_match.group(2)}".strip()
                continue

            # Detect section headers
            section_match = SECTION_HEADER_PATTERN.match(text)
            if section_match:
                section_num = section_match.group(1).strip()
                rest = section_match.group(2).strip()

                # Separate title from body text
                title, body = self._split_title_body(rest)

                # Collect full section text from following siblings
                full_text = self._collect_section_text(element, body)

                # Extract proviso and explanation
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
                ))

        return sections

    def _collect_section_text(self, element, initial_text: str) -> str:
        """
        Collect the full text of a section by reading following sibling
        elements until the next section, chapter, or part header is found.
        """
        texts = [initial_text] if initial_text else []

        sibling = element.find_next_sibling()
        count = 0

        while sibling and count < 50:
            sibling_text = sibling.get_text(strip=True)
            if not sibling_text:
                sibling = sibling.find_next_sibling()
                count += 1
                continue

            # Stop at next section header
            if SECTION_HEADER_PATTERN.match(sibling_text):
                break

            # Stop at chapter or part heading
            if CHAPTER_PATTERN.match(sibling_text) or PART_PATTERN.match(sibling_text):
                break

            texts.append(sibling_text)
            sibling = sibling.find_next_sibling()
            count += 1

        return "\n".join(texts).strip()

    # ─── Strategy 2: Regex-Based Splitting ───────────────────────────────

    def _parse_regex(self, text: str, act_name: str) -> list[ParsedSection]:
        """
        Parse sections by splitting raw text on section number patterns.

        This is the fallback when structured HTML parsing finds nothing.
        Splits the full text on "Section NNN" boundaries.
        """
        sections: list[ParsedSection] = []
        current_chapter: str | None = None
        current_part: str | None = None

        # Split on section patterns
        # Result: [preamble, sec_num_1, sec_text_1, sec_num_2, sec_text_2, ...]
        parts = SECTION_SPLIT_PATTERN.split(text)

        # Scan preamble for chapter/part context
        if parts:
            preamble = parts[0]
            for line in preamble.split("\n"):
                line = line.strip()
                ch_match = CHAPTER_PATTERN.match(line)
                if ch_match:
                    current_chapter = f"Chapter {ch_match.group(1)} - {ch_match.group(2)}".strip()
                pt_match = PART_PATTERN.match(line)
                if pt_match:
                    current_part = f"Part {pt_match.group(1)} - {pt_match.group(2)}".strip()

        # Process section pairs
        for i in range(1, len(parts) - 1, 2):
            section_num = parts[i].strip()
            section_text = parts[i + 1].strip()

            if not section_text or len(section_text) < 10:
                continue

            # Cap very long sections (likely includes page noise)
            if len(section_text) > 10000:
                section_text = section_text[:10000] + "..."

            # Update chapter/part from inline headings within the section text
            for line in section_text.split("\n"):
                line = line.strip()
                ch_match = CHAPTER_PATTERN.match(line)
                if ch_match:
                    current_chapter = f"Chapter {ch_match.group(1)} - {ch_match.group(2)}".strip()
                pt_match = PART_PATTERN.match(line)
                if pt_match:
                    current_part = f"Part {pt_match.group(1)} - {pt_match.group(2)}".strip()

            # Extract title from first line
            title, body = self._split_title_body(section_text)

            # Extract proviso and explanation
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
            ))

        return sections

    # ─── Text Processing Helpers ─────────────────────────────────────────

    def _split_title_body(self, text: str) -> tuple[str | None, str]:
        """
        Split section text into title and body.

        Common pattern: "Punishment for murder.— Whoever commits..."
        The title is the text before the first sentence-ending marker.
        """
        # Try splitting on ".—" or ".-" (common in Indian legal formatting)
        for delimiter in [".—", ".-", ".--", ". —"]:
            if delimiter in text:
                parts = text.split(delimiter, 1)
                title = parts[0].strip()
                body = parts[1].strip() if len(parts) > 1 else ""
                return title, body

        # Try splitting on first period followed by space and capital letter
        match = re.match(r"^([^.]+\.)\s+([A-Z].*)", text, re.DOTALL)
        if match and len(match.group(1)) < 200:
            return match.group(1).rstrip(".").strip(), match.group(2).strip()

        # No clear title — return full text as body
        return None, text

    def _extract_proviso(self, text: str) -> tuple[str, str | None]:
        """
        Extract proviso from section text.

        Provisos start with "Provided that" and modify the main section.
        Returns (text_without_proviso, proviso_text_or_none).
        """
        match = PROVISO_PATTERN.search(text)
        if not match:
            return text, None

        # Everything from "Provided that" onwards is the proviso
        proviso_start = match.start()
        main_text = text[:proviso_start].strip()
        proviso_text = text[proviso_start:].strip()

        return main_text, proviso_text

    def _extract_explanation(self, text: str) -> tuple[str, str | None]:
        """
        Extract explanation from section text.

        Explanations start with "Explanation" and clarify the main section.
        Returns (text_without_explanation, explanation_text_or_none).
        """
        match = EXPLANATION_PATTERN.search(text)
        if not match:
            return text, None

        explanation_start = match.start()
        main_text = text[:explanation_start].strip()
        explanation_text = text[explanation_start:].strip()

        return main_text, explanation_text

    def _clean_text(self, text: str) -> str:
        """
        Clean section text: normalize whitespace, strip HTML artifacts,
        fix common encoding issues.
        """
        if not text:
            return ""

        # Strip any remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Normalize whitespace (but preserve paragraph breaks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix common encoding artifacts
        text = text.replace("\xa0", " ")  # Non-breaking space
        text = text.replace("\u200b", "")  # Zero-width space
        text = text.replace("\ufeff", "")  # BOM

        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()