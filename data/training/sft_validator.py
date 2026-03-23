"""
NyayaMitra — SFT Dataset Validator (Sprint 8).

Validates SFT pairs for format compliance, citation validity, and
quality before they enter the final training dataset.

Validation checks:
    1. Format compliance — all required template sections present
    2. Citation validity — cited sections exist in the acts DB
    3. Case validity — cited case names exist in the judgments DB
    4. Length bounds — response and instruction within token limits
    5. Disclaimer present — standard disclaimer text exists
    6. Domain consistency — domain tag matches cited content
    7. Deduplication — no near-duplicate questions (by embedding similarity)
    8. Content quality — no empty sections, no placeholder text

Usage:
    # Validate all raw pairs
    python -m data.training.sft_validator

    # Validate a specific file
    python -m data.training.sft_validator --input data/datasets/sft/raw/headnote_pairs.jsonl

    # Skip dedup (faster, useful during development)
    python -m data.training.sft_validator --skip-dedup

    # Strict mode (reject on warnings too)
    python -m data.training.sft_validator --strict

    # Export validation report
    python -m data.training.sft_validator --report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from data.training.sft_config import (
    SFTPair,
    SFT_PATHS,
    REQUIRED_SECTIONS,
    OPTIONAL_SECTIONS,
    RESPONSE_SECTIONS,
    STANDARD_DISCLAIMER,
    QUALITY_THRESHOLDS,
    estimate_token_count,
)

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ValidationResult:
    """Result of validating a single SFT pair."""

    pair_id: str = ""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class ValidationReport:
    """Aggregate validation report for a batch of pairs."""

    total: int = 0
    valid: int = 0
    invalid: int = 0
    warnings_only: int = 0
    error_counts: dict[str, int] = field(default_factory=lambda: Counter())
    warning_counts: dict[str, int] = field(default_factory=lambda: Counter())
    duplicates_removed: int = 0
    per_pair: list[ValidationResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def valid_rate(self) -> float:
        return self.valid / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "warnings_only": self.warnings_only,
            "valid_rate": round(self.valid_rate, 4),
            "duplicates_removed": self.duplicates_removed,
            "top_errors": dict(self.error_counts.most_common(10)),
            "top_warnings": dict(self.warning_counts.most_common(10)),
            "duration_seconds": self.duration_seconds,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Citation Cache — pre-load valid sections and cases from DB
# ═══════════════════════════════════════════════════════════════════════════════


class CitationCache:
    """
    Pre-loads valid section IDs and case names from PostgreSQL
    for fast citation validation without per-pair DB queries.
    """

    def __init__(self):
        self.valid_sections: set[str] = set()     # "IPC/302", "CrPC/41", etc.
        self.valid_cases: set[str] = set()         # lowercase case names
        self.act_short_names: set[str] = set()     # "IPC", "CrPC", etc.
        self._loaded = False

    async def load(self) -> None:
        """Load all valid sections and cases from PostgreSQL."""
        if self._loaded:
            return

        from app.database import async_session
        from app.models.legal import Act, Section, Judgment
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        async with async_session() as session:
            # Load all sections
            result = await session.execute(
                select(Act).options(selectinload(Act.sections))
            )
            acts = result.scalars().all()

            for act in acts:
                short = act.short_name or ""
                self.act_short_names.add(short)
                self.act_short_names.add(act.name)
                for sec in act.sections:
                    # Store multiple forms: "IPC/302", "Indian Penal Code, 1860/302"
                    self.valid_sections.add(f"{short}/{sec.section_number}")
                    self.valid_sections.add(f"{act.name}/{sec.section_number}")

            # Load all case names
            result = await session.execute(select(Judgment.case_name))
            for row in result.scalars().all():
                if row:
                    self.valid_cases.add(row.lower().strip())

        self._loaded = True
        logger.info(
            "citation_cache_loaded",
            sections=len(self.valid_sections),
            cases=len(self.valid_cases),
            acts=len(self.act_short_names),
        )

    def section_exists(self, section_ref: str) -> bool:
        """Check if a section reference like 'IPC/302' is valid."""
        if not self._loaded:
            return True  # Assume valid if cache not loaded
        return section_ref in self.valid_sections

    def case_exists(self, case_name: str) -> bool:
        """Check if a case name exists in the DB (fuzzy match)."""
        if not self._loaded:
            return True
        name_lower = case_name.lower().strip()
        # Exact match
        if name_lower in self.valid_cases:
            return True
        # Substring match (handle "D.K. Basu" matching "D.K. Basu v. State of WB")
        for valid_case in self.valid_cases:
            if name_lower in valid_case or valid_case in name_lower:
                return True
        return False

    def act_exists(self, act_ref: str) -> bool:
        """Check if an act name/abbreviation is known."""
        if not self._loaded:
            return True
        return act_ref in self.act_short_names


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Validators
# ═══════════════════════════════════════════════════════════════════════════════


def validate_format(pair: SFTPair, result: ValidationResult) -> None:
    """Check that the response has all required template sections."""
    response = pair.response

    for section in REQUIRED_SECTIONS:
        if f"[{section}]" not in response:
            result.add_error(f"missing_section:{section}")

    for section in OPTIONAL_SECTIONS:
        if f"[{section}]" not in response:
            result.add_warning(f"missing_optional:{section}")


def validate_length(pair: SFTPair, result: ValidationResult) -> None:
    """Check instruction and response length bounds."""
    inst_tokens = estimate_token_count(pair.instruction)
    resp_tokens = estimate_token_count(pair.response)

    if inst_tokens < QUALITY_THRESHOLDS.min_instruction_tokens:
        result.add_error(f"instruction_too_short:{inst_tokens}")

    if inst_tokens > QUALITY_THRESHOLDS.max_instruction_tokens:
        result.add_warning(f"instruction_too_long:{inst_tokens}")

    if resp_tokens < QUALITY_THRESHOLDS.min_response_tokens:
        result.add_error(f"response_too_short:{resp_tokens}")

    if resp_tokens > QUALITY_THRESHOLDS.max_response_tokens:
        result.add_warning(f"response_too_long:{resp_tokens}")


def validate_disclaimer(pair: SFTPair, result: ValidationResult) -> None:
    """Check that the disclaimer is present."""
    if "[DISCLAIMER]" not in pair.response:
        result.add_error("missing_disclaimer_section")
        return

    # Check that some disclaimer text follows the tag
    disclaimer_idx = pair.response.find("[DISCLAIMER]")
    after = pair.response[disclaimer_idx + len("[DISCLAIMER]"):].strip()
    if len(after) < 20:
        result.add_error("empty_disclaimer")


def validate_citations(
    pair: SFTPair,
    cache: CitationCache,
    result: ValidationResult,
) -> None:
    """Validate that cited sections and cases exist in the database."""
    # Validate section citations
    invalid_sections = []
    for ref in pair.cited_sections:
        if not cache.section_exists(ref):
            invalid_sections.append(ref)

    if invalid_sections:
        result.add_warning(f"unverified_sections:{','.join(invalid_sections)}")

    # Validate case citations
    invalid_cases = []
    for case in pair.cited_cases:
        if not cache.case_exists(case):
            invalid_cases.append(case)

    if invalid_cases:
        result.add_warning(f"unverified_cases:{','.join(invalid_cases[:3])}")

    # Check that APPLICABLE_LAW section actually has content
    law_idx = pair.response.find("[APPLICABLE_LAW]")
    if law_idx >= 0:
        # Find next section tag
        next_section = None
        for sec in RESPONSE_SECTIONS:
            if sec == "APPLICABLE_LAW":
                continue
            idx = pair.response.find(f"[{sec}]", law_idx + 1)
            if idx >= 0 and (next_section is None or idx < next_section):
                next_section = idx

        if next_section:
            law_content = pair.response[law_idx + len("[APPLICABLE_LAW]"):next_section].strip()
        else:
            law_content = pair.response[law_idx + len("[APPLICABLE_LAW]"):].strip()

        if len(law_content) < 10:
            result.add_warning("empty_applicable_law")


def validate_content_quality(pair: SFTPair, result: ValidationResult) -> None:
    """Check for placeholder text, empty content, and obvious issues."""
    response = pair.response.lower()

    # Check for placeholder/template text
    placeholders = [
        "lorem ipsum", "todo", "fixme", "placeholder",
        "insert here", "example text", "[your ", "{your ",
    ]
    for p in placeholders:
        if p in response:
            result.add_error(f"placeholder_text:{p}")

    # Check instruction quality
    instruction = pair.instruction.strip()
    if not instruction:
        result.add_error("empty_instruction")
    elif instruction.endswith("..."):
        result.add_warning("truncated_instruction")

    # Check for very repetitive response (same sentence repeated)
    sentences = [s.strip() for s in pair.response.split(".") if len(s.strip()) > 20]
    if len(sentences) > 3:
        unique = set(s.lower() for s in sentences)
        if len(unique) < len(sentences) * 0.5:
            result.add_warning("repetitive_response")

    # Domain should be set
    if not pair.domain:
        result.add_warning("missing_domain")


def validate_confidence(pair: SFTPair, result: ValidationResult) -> None:
    """Validate confidence section has a valid level."""
    conf_idx = pair.response.find("[CONFIDENCE]")
    if conf_idx < 0:
        return  # Already caught by format validator

    after = pair.response[conf_idx + len("[CONFIDENCE]"):].strip()
    first_line = after.split("\n")[0].strip().lower()

    if not any(level in first_line for level in ["high", "medium", "low"]):
        result.add_warning("invalid_confidence_level")


# ═══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════════════


def deduplicate_pairs(
    pairs: list[SFTPair],
    threshold: float = QUALITY_THRESHOLDS.dedup_similarity_threshold,
) -> tuple[list[SFTPair], int]:
    """
    Remove near-duplicate pairs by instruction text similarity.

    Uses a simple character n-gram Jaccard similarity as a fast proxy
    for embedding similarity. For full embedding-based dedup, use
    the embedding model (slower but more accurate).

    Returns (deduplicated pairs, count of duplicates removed).
    """
    def ngram_set(text: str, n: int = 3) -> set[str]:
        text = text.lower().strip()
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # Build n-gram sets
    pair_ngrams = [(p, ngram_set(p.instruction)) for p in pairs]

    kept: list[SFTPair] = []
    removed = 0

    for i, (pair, ngrams) in enumerate(pair_ngrams):
        is_dup = False
        for kept_pair, kept_ngrams in [(k, ngram_set(k.instruction)) for k in kept[-200:]]:
            sim = jaccard(ngrams, kept_ngrams)
            if sim >= threshold:
                is_dup = True
                break

        if is_dup:
            removed += 1
        else:
            kept.append(pair)

    return kept, removed


# ═══════════════════════════════════════════════════════════════════════════════
# Main Validation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


async def validate_pairs(
    pairs: list[SFTPair],
    skip_dedup: bool = False,
    strict: bool = False,
) -> tuple[list[SFTPair], ValidationReport]:
    """
    Run all validation checks on a list of SFT pairs.

    Args:
        pairs: Pairs to validate.
        skip_dedup: Skip deduplication (faster).
        strict: Treat warnings as errors.

    Returns:
        (valid_pairs, report)
    """
    start = time.time()
    report = ValidationReport(total=len(pairs))

    # Load citation cache
    cache = CitationCache()
    try:
        await cache.load()
    except Exception as e:
        logger.warning("citation_cache_failed", error=str(e))
        # Continue without citation validation

    # Validate each pair
    for pair in pairs:
        result = ValidationResult(pair_id=pair.pair_id)

        validate_format(pair, result)
        validate_length(pair, result)
        validate_disclaimer(pair, result)
        validate_citations(pair, cache, result)
        validate_content_quality(pair, result)
        validate_confidence(pair, result)

        # In strict mode, warnings become errors
        if strict and result.warnings:
            for w in result.warnings:
                result.add_error(f"strict:{w}")

        report.per_pair.append(result)

        # Count errors and warnings
        for e in result.errors:
            error_type = e.split(":")[0]
            report.error_counts[error_type] += 1
        for w in result.warnings:
            warning_type = w.split(":")[0]
            report.warning_counts[warning_type] += 1

    # Separate valid and invalid
    valid_pairs = []
    for pair, result in zip(pairs, report.per_pair):
        if result.valid:
            if result.warnings:
                report.warnings_only += 1
            valid_pairs.append(pair)
            report.valid += 1
        else:
            report.invalid += 1

    # Deduplication
    if not skip_dedup and valid_pairs:
        logger.info("dedup_start", pairs=len(valid_pairs))
        valid_pairs, dupes = deduplicate_pairs(valid_pairs)
        report.duplicates_removed = dupes
        report.valid -= dupes
        logger.info("dedup_complete", kept=len(valid_pairs), removed=dupes)

    report.duration_seconds = round(time.time() - start, 2)
    return valid_pairs, report


# ═══════════════════════════════════════════════════════════════════════════════
# File Operations
# ═══════════════════════════════════════════════════════════════════════════════


def load_pairs_from_jsonl(path: Path) -> list[SFTPair]:
    """Load SFTPairs from a JSONL file."""
    pairs = []
    if not path.exists():
        return pairs

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                pairs.append(SFTPair.from_dict(data))

    return pairs


def save_pairs_to_jsonl(pairs: list[SFTPair], path: Path) -> None:
    """Save SFTPairs to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")


def save_report(report: ValidationReport, path: Path) -> None:
    """Save validation report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def print_report(report: ValidationReport) -> None:
    """Print validation report to console."""
    print(f"\n{'═' * 60}")
    print(f"  SFT Validation Report")
    print(f"{'═' * 60}\n")
    print(f"  Total pairs:        {report.total:,}")
    print(f"  Valid:              {report.valid:,} ({report.valid_rate:.1%})")
    print(f"  Invalid:            {report.invalid:,}")
    print(f"  Warnings only:      {report.warnings_only:,}")
    print(f"  Duplicates removed: {report.duplicates_removed:,}")
    print(f"  Duration:           {report.duration_seconds}s")

    if report.error_counts:
        print(f"\n  Top errors:")
        for err, count in report.error_counts.most_common(10):
            print(f"    {err:<35} {count:>5}")

    if report.warning_counts:
        print(f"\n  Top warnings:")
        for warn, count in report.warning_counts.most_common(10):
            print(f"    {warn:<35} {count:>5}")

    # Verdict
    target = QUALITY_THRESHOLDS.min_total_pairs
    print(f"\n  Target: {target:,} valid pairs")
    if report.valid >= target:
        print(f"  Verdict: PASS ({report.valid:,} >= {target:,})")
    else:
        gap = target - report.valid
        print(f"  Verdict: GAP — need {gap:,} more valid pairs")

    print(f"\n{'═' * 60}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra SFT Dataset Validator (Sprint 8)",
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Specific JSONL file to validate (default: all raw files)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for validated pairs")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Skip deduplication (faster)")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    parser.add_argument("--report", action="store_true",
                        help="Save validation report as JSON")

    args = parser.parse_args()

    # Load pairs
    if args.input:
        input_path = Path(args.input)
        pairs = load_pairs_from_jsonl(input_path)
        print(f"\n  Loaded {len(pairs)} pairs from {input_path}")
    else:
        # Load all raw files + accepted annotation pairs
        pairs = []
        sources = [
            SFT_PATHS.headnote_raw,
            SFT_PATHS.synthetic_raw,
            SFT_PATHS.nalsa_raw,
            SFT_PATHS.procedural_raw,
            SFT_PATHS.annotation_dir / "accepted.jsonl",
        ]
        for src in sources:
            if src.exists():
                loaded = load_pairs_from_jsonl(src)
                print(f"  Loaded {len(loaded):,} pairs from {src.name}")
                pairs.extend(loaded)

        if not pairs:
            print("\n  No pairs found. Run the extraction pipelines first.\n")
            return

        print(f"\n  Total loaded: {len(pairs):,} pairs")

    # Validate
    valid_pairs, report = await validate_pairs(
        pairs,
        skip_dedup=args.skip_dedup,
        strict=args.strict,
    )

    # Print report
    print_report(report)

    # Save validated pairs
    output_path = Path(args.output) if args.output else SFT_PATHS.validated_all
    save_pairs_to_jsonl(valid_pairs, output_path)
    print(f"  Validated pairs saved to: {output_path}")

    # Save report
    if args.report:
        report_path = SFT_PATHS.audit_dir / "validation_report.json"
        save_report(report, report_path)
        print(f"  Report saved to: {report_path}")

    print()


if __name__ == "__main__":
    asyncio.run(main())