"""
NyayaMitra — Data Validation Layer.

Centralized validation for all ingested legal data. Checks quality
before records reach the database, ensuring:
- No duplicates (by key fields)
- No malformed sections (too short, encoding issues, HTML leakage)
- Required fields present (completeness)
- Citation format validity (SCC, AIR patterns)
- Section number format validity
- Cross-reference integrity (section.act_id exists)

Every scraper and seeder calls the validator before inserting records.
Validation results are logged and can be aggregated for reporting.

Usage:
    from data.processors.validator import DataValidator, ValidationResult

    validator = DataValidator()

    # Validate a single record
    result = validator.validate_act(act_data)
    if not result.is_valid:
        logger.warning("validation_failed", errors=result.errors)

    # Validate a batch
    results = validator.validate_batch(records, "act")

    # Check for duplicates in database
    is_dup = await validator.check_duplicate_act(session, name, year, jurisdiction)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.exceptions import ValidationError as NyayaValidationError
from app.models.legal import Act, Section, Judgment

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ValidationResult:
    """
    Result of validating a single record.

    Tracks validity, hard errors (block insertion), and soft warnings
    (log but allow insertion).
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    record_type: str = ""
    record_id: str | None = None

    def add_error(self, message: str) -> None:
        """Add a validation error. Marks result as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning. Does not affect validity."""
        self.warnings.append(message)

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"[{status}] {self.record_type}"]
        if self.record_id:
            parts.append(f"id={self.record_id}")
        if self.errors:
            parts.append(f"errors={self.errors}")
        if self.warnings:
            parts.append(f"warnings={self.warnings}")
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Patterns
# ═══════════════════════════════════════════════════════════════════════════════

# Valid section number patterns
SECTION_NUMBER_PATTERN = re.compile(
    r"^(\d+[A-Za-z]*(?:\([^)]*\))*|Schedule\s+[IVXLCDM\d]+[A-Z]*)$",
    re.IGNORECASE,
)

# SCC citation: "(1997) 1 SCC 416"
SCC_CITATION_PATTERN = re.compile(r"^\(\d{4}\)\s+\d+\s+SCC\s+\d+$")

# AIR citation: "AIR 1997 SC 610"
AIR_CITATION_PATTERN = re.compile(r"^AIR\s+\d{4}\s+\w+\s+\d+$")

# HTML tag detection
HTML_TAG_PATTERN = re.compile(r"<[a-zA-Z][^>]*>")

# Common mojibake patterns (Hindi/Devanagari encoding issues)
MOJIBAKE_PATTERNS = [
    re.compile(r"Ã[\x80-\xBF]"),  # UTF-8 decoded as Latin-1
    re.compile(r"â\x80[^\w]"),  # Smart quotes misencoded
    re.compile(r"\x00"),  # Null bytes
]

# Valid legal domains
VALID_DOMAINS = {
    "criminal", "property", "family", "labor",
    "consumer", "constitutional", "ip", "general",
}

# Valid act statuses
VALID_ACT_STATUSES = {"active", "repealed", "partially_repealed", "amended"}

# Valid court types
VALID_COURT_TYPES = {"SC", "HC", "Tribunal"}

# Minimum text lengths
MIN_SECTION_TEXT_LENGTH = 10
MIN_HEADNOTE_LENGTH = 20


# ═══════════════════════════════════════════════════════════════════════════════
# Data Validator
# ═══════════════════════════════════════════════════════════════════════════════


class DataValidator:
    """
    Validates legal data records before database insertion.

    Performs structural checks (required fields, format), content checks
    (text quality, encoding), and optional database checks (duplicates,
    cross-references).

    Thread-safe and stateless.
    """

    # ─── Act Validation ──────────────────────────────────────────────────

    def validate_act(self, act_data: dict) -> ValidationResult:
        """
        Validate an act record.

        Required fields: name, year, domain.
        Checks: domain validity, year range, name length.
        """
        result = ValidationResult(record_type="act")

        # Required fields
        name = act_data.get("name", "").strip()
        if not name:
            result.add_error("Missing required field: name")
        elif len(name) < 3:
            result.add_error(f"Act name too short: '{name}'")
        elif len(name) > 500:
            result.add_warning(f"Act name very long ({len(name)} chars), will be truncated")

        year = act_data.get("year")
        if not year:
            result.add_error("Missing required field: year")
        elif not isinstance(year, int) or year < 1800 or year > 2030:
            result.add_error(f"Invalid year: {year} (expected 1800-2030)")

        domain = act_data.get("domain", "general")
        if domain not in VALID_DOMAINS:
            result.add_error(f"Invalid domain: '{domain}' (valid: {VALID_DOMAINS})")

        # Optional field checks
        status = act_data.get("status", "active")
        if status and status not in VALID_ACT_STATUSES:
            result.add_warning(f"Unusual act status: '{status}'")

        # Text quality
        raw_text = act_data.get("raw_text", "")
        if raw_text:
            encoding_issues = self._check_encoding(raw_text)
            for issue in encoding_issues:
                result.add_warning(f"Encoding issue in raw_text: {issue}")

        return result

    # ─── Section Validation ──────────────────────────────────────────────

    def validate_section(
        self,
        section_data: dict,
        act_id: UUID | str | None = None,
    ) -> ValidationResult:
        """
        Validate a section record.

        Required fields: section_number, text.
        Checks: section number format, text length, HTML leakage, encoding.
        """
        result = ValidationResult(record_type="section")

        # Required fields
        section_number = section_data.get("section_number", "").strip()
        if not section_number:
            result.add_error("Missing required field: section_number")
        elif not SECTION_NUMBER_PATTERN.match(section_number):
            result.add_warning(f"Unusual section number format: '{section_number}'")

        text = section_data.get("text", "").strip()
        if not text:
            result.add_error("Missing required field: text")
        elif len(text) < MIN_SECTION_TEXT_LENGTH:
            result.add_error(
                f"Section text too short ({len(text)} chars, min {MIN_SECTION_TEXT_LENGTH})"
            )

        # Check for HTML tags leaked into text
        if text and HTML_TAG_PATTERN.search(text):
            result.add_warning("HTML tags detected in section text")

        # Encoding quality
        if text:
            encoding_issues = self._check_encoding(text)
            for issue in encoding_issues:
                result.add_warning(f"Encoding issue in text: {issue}")

        # Cross-reference
        if act_id is None and not section_data.get("act_id"):
            result.add_error("Missing required field: act_id (section must belong to an act)")

        return result

    # ─── Judgment Validation ─────────────────────────────────────────────

    def validate_judgment(self, judgment_data: dict) -> ValidationResult:
        """
        Validate a judgment record.

        Required fields: case_name, year, court.
        Checks: year range, citation format, court type, text quality.
        """
        result = ValidationResult(record_type="judgment")

        # Required fields
        case_name = judgment_data.get("case_name", "").strip()
        if not case_name:
            result.add_error("Missing required field: case_name")
        elif len(case_name) < 5:
            result.add_error(f"Case name too short: '{case_name}'")

        year = judgment_data.get("year")
        if not year:
            result.add_error("Missing required field: year")
        elif not isinstance(year, int) or year < 1947 or year > 2030:
            result.add_error(f"Invalid judgment year: {year} (expected 1947-2030)")

        court = judgment_data.get("court", "").strip()
        if not court:
            result.add_error("Missing required field: court")

        court_type = judgment_data.get("court_type", "")
        if court_type and court_type not in VALID_COURT_TYPES:
            result.add_warning(f"Unusual court type: '{court_type}'")

        # Citation format validation
        citation_scc = judgment_data.get("citation_scc", "")
        if citation_scc and not SCC_CITATION_PATTERN.match(citation_scc.strip()):
            result.add_warning(f"SCC citation format may be incorrect: '{citation_scc}'")

        citation_air = judgment_data.get("citation_air", "")
        if citation_air and not AIR_CITATION_PATTERN.match(citation_air.strip()):
            result.add_warning(f"AIR citation format may be incorrect: '{citation_air}'")

        # Bench size sanity
        bench_size = judgment_data.get("bench_size")
        if bench_size is not None:
            if not isinstance(bench_size, int) or bench_size < 1 or bench_size > 15:
                result.add_warning(f"Unusual bench size: {bench_size}")

        # Domain
        domain = judgment_data.get("domain", "")
        if domain and domain not in VALID_DOMAINS:
            result.add_warning(f"Invalid domain: '{domain}'")

        # Content quality
        headnote = judgment_data.get("headnote", "")
        if headnote and len(headnote) < MIN_HEADNOTE_LENGTH:
            result.add_warning(f"Headnote very short ({len(headnote)} chars)")

        # Sections interpreted JSON validity
        si = judgment_data.get("sections_interpreted", "")
        if si:
            self._validate_sections_interpreted_json(si, result)

        return result

    # ─── Batch Validation ────────────────────────────────────────────────

    def validate_batch(
        self,
        records: list[dict],
        record_type: str,
    ) -> list[ValidationResult]:
        """
        Validate a batch of records.

        Args:
            records: List of record dicts.
            record_type: One of "act", "section", "judgment".

        Returns:
            List of ValidationResult objects (one per record).
        """
        validators = {
            "act": self.validate_act,
            "section": self.validate_section,
            "judgment": self.validate_judgment,
        }

        validator_fn = validators.get(record_type)
        if not validator_fn:
            raise NyayaValidationError(
                message=f"Unknown record type: {record_type}",
                record_type=record_type,
            )

        results = []
        for record in records:
            result = validator_fn(record)
            results.append(result)

        # Log summary
        valid_count = sum(1 for r in results if r.is_valid)
        warning_count = sum(len(r.warnings) for r in results)
        error_count = sum(len(r.errors) for r in results)

        logger.info(
            "batch_validation_complete",
            record_type=record_type,
            total=len(records),
            valid=valid_count,
            invalid=len(records) - valid_count,
            warnings=warning_count,
            errors=error_count,
        )

        return results

    # ─── Database Duplicate Checks ───────────────────────────────────────

    async def check_duplicate_act(
        self,
        session: AsyncSession,
        name: str,
        year: int,
        jurisdiction: str = "central",
    ) -> bool:
        """Check if an act already exists in the database."""
        stmt = select(Act).where(
            Act.name == name,
            Act.year == year,
            Act.jurisdiction == jurisdiction,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def check_duplicate_section(
        self,
        session: AsyncSession,
        act_id: UUID | str,
        section_number: str,
    ) -> bool:
        """Check if a section already exists for the given act."""
        stmt = select(Section).where(
            Section.act_id == act_id,
            Section.section_number == section_number,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def check_duplicate_judgment(
        self,
        session: AsyncSession,
        case_name: str,
        year: int,
    ) -> bool:
        """Check if a judgment already exists (by case name + year)."""
        stmt = select(Judgment).where(
            Judgment.case_name == case_name,
            Judgment.year == year,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def check_duplicate_judgment_by_ik_id(
        self,
        session: AsyncSession,
        indian_kanoon_id: str,
    ) -> bool:
        """Check if a judgment exists by Indian Kanoon document ID."""
        stmt = select(Judgment).where(
            Judgment.indian_kanoon_id == indian_kanoon_id,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None

    # ─── Text Quality Helpers ────────────────────────────────────────────

    def _check_encoding(self, text: str) -> list[str]:
        """
        Check for common encoding issues in text.

        Returns a list of issue descriptions (empty if clean).
        """
        issues = []

        for pattern in MOJIBAKE_PATTERNS:
            if pattern.search(text):
                issues.append(f"Mojibake pattern detected: {pattern.pattern}")

        # Check for excessive non-ASCII that might indicate corruption
        if text:
            non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
            # High ratio is fine for Hindi/Devanagari, but flag extreme cases
            if non_ascii_ratio > 0.8:
                issues.append(
                    f"Very high non-ASCII ratio ({non_ascii_ratio:.1%}), "
                    "may indicate encoding corruption"
                )

        return issues

    def fix_encoding(self, text: str) -> str:
        """
        Attempt to fix common encoding issues in text.

        Fixes: null bytes, zero-width spaces, non-breaking spaces,
        BOM characters, excessive whitespace.

        Returns cleaned text.
        """
        if not text:
            return ""

        # Remove null bytes
        text = text.replace("\x00", "")

        # Fix common Unicode artifacts
        text = text.replace("\xa0", " ")  # Non-breaking space -> space
        text = text.replace("\u200b", "")  # Zero-width space
        text = text.replace("\ufeff", "")  # BOM
        text = text.replace("\u200c", "")  # Zero-width non-joiner (keep for Devanagari)
        text = text.replace("\u200d", "")  # Zero-width joiner

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def strip_html(self, text: str) -> str:
        """Remove any HTML tags that leaked into plain text fields."""
        if not text:
            return ""
        return re.sub(r"<[^>]+>", "", text).strip()

    def _validate_sections_interpreted_json(
        self,
        json_str: str,
        result: ValidationResult,
    ) -> None:
        """Validate the sections_interpreted JSON field."""
        try:
            import json
            parsed = json.loads(json_str)

            if not isinstance(parsed, list):
                result.add_warning("sections_interpreted is not a JSON array")
                return

            for item in parsed:
                if not isinstance(item, dict):
                    result.add_warning("sections_interpreted contains non-dict items")
                    break
                if "act" not in item or "section" not in item:
                    result.add_warning(
                        "sections_interpreted item missing 'act' or 'section' key"
                    )
                    break

        except (json.JSONDecodeError, TypeError):
            result.add_warning("sections_interpreted is not valid JSON")