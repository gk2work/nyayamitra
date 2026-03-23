"""
NyayaMitra — Headnote Extraction Pipeline (Sprint 8).

Automatically generates SFT instruction-response pairs from SC/HC
judgment headnotes and ratio decidendi stored in PostgreSQL.

This is the highest-volume source of training data (target: 25,000 pairs).
For each judgment with a headnote + ratio, we generate 1-5 question
variants and build a structured response matching the mandatory
SFT template.

Strategy:
    1. Stream judgments from PostgreSQL (only those with headnote + ratio)
    2. For each judgment, extract: sections interpreted, citation, domain
    3. Generate diverse question phrasings using templates
    4. Build structured response in SFT format
    5. Validate and export as JSONL

Question generation styles:
    - "What did the court hold in {case}?"         (case_outcome)
    - "What is the law on {topic}?"                (rights)
    - "What are the rights regarding {topic}?"     (rights)
    - "Explain the legal position on {topic}"      (section_explanation)
    - "How does {case} apply to {topic}?"          (case_outcome)

Usage:
    # Generate all headnote pairs
    python -m data.training.headnote_extractor

    # Limit to specific domain
    python -m data.training.headnote_extractor --domain criminal

    # Limit count (for testing)
    python -m data.training.headnote_extractor --limit 100

    # Dry run (show sample pairs without saving)
    python -m data.training.headnote_extractor --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import async_session
from app.models.legal import Judgment

from data.training.sft_config import (
    SFT_SYSTEM_PROMPT,
    SFTPair,
    SFT_PATHS,
    STANDARD_DISCLAIMER,
    CONFIDENCE_LEVELS,
    QUESTION_TEMPLATES,
    format_applicable_law_section,
    format_precedent_section,
    format_sft_pair,
    estimate_token_count,
)

logger = structlog.get_logger()

# Minimum quality thresholds for source judgments
MIN_HEADNOTE_LENGTH = 80       # chars — skip thin headnotes
MIN_RATIO_LENGTH = 50          # chars — skip empty ratios
MAX_PAIRS_PER_JUDGMENT = 5     # Don't over-generate from one judgment


# ═══════════════════════════════════════════════════════════════════════════════
# Topic Extraction
# ═══════════════════════════════════════════════════════════════════════════════


def extract_topic_from_headnote(headnote: str, case_name: str) -> str:
    """
    Extract a concise topic phrase from a headnote for question generation.

    Examples:
        "The Supreme Court laid down 11 mandatory guidelines..."
        → "guidelines for arrest and detention"

        "Registration of FIR under Section 154 CrPC is mandatory..."
        → "mandatory FIR registration"
    """
    # Try to extract the core issue from first sentence
    first_sentence = headnote.split(".")[0].strip()

    # Remove "The Supreme Court held that" / "The Court observed that" etc.
    cleaned = re.sub(
        r"^(?:The\s+)?(?:Supreme\s+Court|Court|Hon'ble\s+Court|Bench)\s+"
        r"(?:held|observed|ruled|laid\s+down|declared|struck\s+down|upheld)"
        r"\s+(?:that\s+)?",
        "",
        first_sentence,
        flags=re.IGNORECASE,
    ).strip()

    # If cleaned is too long, truncate at a natural boundary
    if len(cleaned) > 100:
        # Try to cut at a comma or conjunction
        for delimiter in [",", " and ", " or ", " which ", " that "]:
            idx = cleaned.find(delimiter, 30)
            if 30 < idx < 100:
                cleaned = cleaned[:idx].strip()
                break
        else:
            cleaned = cleaned[:100].strip()

    # If extraction failed, use case name as topic
    if len(cleaned) < 10:
        # Extract topic from case name parties
        parts = case_name.split(" v. ")
        if len(parts) == 2:
            cleaned = f"the rights of {parts[0].strip()}"
        else:
            cleaned = "the legal issue in this case"

    return cleaned.rstrip(".,;:")


def extract_sections_list(sections_json: str | None) -> list[dict]:
    """Parse sections_interpreted JSON into a list of {act, section} dicts."""
    if not sections_json:
        return []
    try:
        sections = json.loads(sections_json)
        return [s for s in sections if s.get("act") and s.get("section")]
    except (json.JSONDecodeError, TypeError):
        return []


def determine_confidence(bench_size: int | None, court_type: str) -> str:
    """
    Determine confidence level based on bench size and court type.

    Constitution Bench (5+) or SC = High
    Division Bench (2-3) SC = High
    Single judge HC = Medium
    Unknown = Medium
    """
    if court_type == "SC":
        if bench_size and bench_size >= 5:
            return "High"
        return "High"
    elif court_type == "HC":
        if bench_size and bench_size >= 2:
            return "Medium"
        return "Medium"
    return "Medium"


# ═══════════════════════════════════════════════════════════════════════════════
# Question Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_questions(
    case_name: str,
    headnote: str,
    domain: str,
    sections: list[dict],
    topic: str,
) -> list[dict]:
    """
    Generate diverse question phrasings for a single judgment.

    Returns a list of {"question": str, "query_type": str} dicts.
    Generates 1-5 questions depending on available data.
    """
    questions = []

    # 1. Always: "What did the court hold in {case}?"
    questions.append({
        "question": f"What did the Supreme Court hold in {case_name}?",
        "query_type": "case_outcome",
    })

    # 2. Topic-based rights question
    if topic and len(topic) > 10:
        questions.append({
            "question": f"What is the law on {topic}?",
            "query_type": "rights",
        })

    # 3. Section-specific question (if sections available)
    if sections:
        primary = sections[0]
        act = primary.get("act", "")
        sec = primary.get("section", "")
        if act and sec:
            questions.append({
                "question": f"What is the interpretation of Section {sec} of {act}?",
                "query_type": "section_explanation",
            })

    # 4. Scenario-based question (rephrase the topic as a citizen query)
    if domain == "criminal" and topic:
        questions.append({
            "question": f"What are my rights regarding {topic}?",
            "query_type": "rights",
        })
    elif domain == "property" and topic:
        questions.append({
            "question": f"What does the law say about {topic} in property matters?",
            "query_type": "rights",
        })
    elif domain == "family" and topic:
        questions.append({
            "question": f"What legal remedies are available for {topic}?",
            "query_type": "rights",
        })
    elif domain == "labour" and topic:
        questions.append({
            "question": f"What are the employee rights regarding {topic}?",
            "query_type": "rights",
        })
    elif domain == "consumer" and topic:
        questions.append({
            "question": f"What consumer protection is available for {topic}?",
            "query_type": "rights",
        })
    elif domain == "constitutional" and topic:
        questions.append({
            "question": f"What does the Constitution say about {topic}?",
            "query_type": "rights",
        })

    # 5. "Is {case} still good law?" (occasionally)
    if len(questions) < MAX_PAIRS_PER_JUDGMENT:
        questions.append({
            "question": f"What is the current legal position after {case_name}?",
            "query_type": "case_outcome",
        })

    return questions[:MAX_PAIRS_PER_JUDGMENT]


# ═══════════════════════════════════════════════════════════════════════════════
# Response Builder
# ═══════════════════════════════════════════════════════════════════════════════


def build_response(
    headnote: str,
    ratio: str,
    case_name: str,
    year: int,
    court: str,
    citation_scc: str | None,
    citation_air: str | None,
    sections: list[dict],
    domain: str,
    bench_size: int | None,
    court_type: str,
) -> str:
    """
    Build a structured SFT response from judgment components.

    Follows the mandatory template: [APPLICABLE_LAW], [PRECEDENT],
    [LEGAL_POSITION], [PROCEDURE], [JURISDICTION_NOTE], [CONFIDENCE],
    [DISCLAIMER].
    """
    # [APPLICABLE_LAW]
    if sections:
        law_lines = []
        for s in sections:
            law_lines.append({
                "act": s.get("act", ""),
                "section": s.get("section", ""),
                "text": "",  # We don't have section text in the judgment record
            })
        applicable_law = format_applicable_law_section(law_lines)
    else:
        applicable_law = "Refer to the specific provisions discussed in the judgment."

    # [PRECEDENT]
    citation = citation_scc or citation_air or ""
    precedent_data = [{
        "case": case_name,
        "year": year,
        "court": court or "Supreme Court",
        "citation": citation,
        "relevance": _truncate(ratio, 200) if ratio else _truncate(headnote, 200),
    }]
    precedent = format_precedent_section(precedent_data)

    # [LEGAL_POSITION]
    legal_position = ""
    if ratio:
        legal_position = _truncate(ratio, 600)
    elif headnote:
        legal_position = _truncate(headnote, 600)

    # [PROCEDURE] — most judgment pairs don't have procedural steps
    procedure = "Not applicable for this query."

    # [JURISDICTION_NOTE]
    if court_type == "SC":
        jurisdiction_note = (
            "This is a Supreme Court judgment and is binding on all courts across India."
        )
    elif court_type == "HC":
        jurisdiction_note = (
            f"This is a High Court judgment ({court}) and is binding within "
            f"its territorial jurisdiction. Other High Courts may take a different view."
        )
    else:
        jurisdiction_note = "Verify the binding authority of this decision in your jurisdiction."

    # [CONFIDENCE]
    confidence = determine_confidence(bench_size, court_type)

    return format_sft_pair(
        instruction="",  # Not used in output
        applicable_law=applicable_law,
        precedent=precedent,
        legal_position=legal_position,
        procedure=procedure,
        jurisdiction_note=jurisdiction_note,
        confidence=confidence,
    )


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len chars at a sentence boundary."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    # Try to cut at last period
    last_period = truncated.rfind(".")
    if last_period > max_len * 0.6:
        return truncated[:last_period + 1]
    return truncated.rstrip() + "..."


# ═══════════════════════════════════════════════════════════════════════════════
# Main Extraction Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


async def extract_headnote_pairs(
    domain_filter: str | None = None,
    limit: int | None = None,
) -> list[SFTPair]:
    """
    Extract SFT pairs from all qualifying judgments in PostgreSQL.

    Args:
        domain_filter: Only process judgments in this domain.
        limit: Maximum number of judgments to process (not pairs).

    Returns:
        List of SFTPair objects ready for validation and export.
    """
    from sqlalchemy import select, func, and_

    pairs: list[SFTPair] = []
    processed = 0
    skipped_short = 0
    skipped_no_sections = 0

    logger.info(
        "headnote_extraction_start",
        domain_filter=domain_filter,
        limit=limit,
    )

    # Stream judgments in pages
    page_size = 500
    offset = 0

    while True:
        async with async_session() as session:
            stmt = select(Judgment).where(
                Judgment.headnote.isnot(None),
            ).order_by(Judgment.id).offset(offset).limit(page_size)

            if domain_filter:
                stmt = stmt.where(Judgment.domain == domain_filter)

            result = await session.execute(stmt)
            judgments = result.scalars().all()

        if not judgments:
            break

        for j in judgments:
            if limit and processed >= limit:
                break

            # Quality filter: skip thin headnotes
            headnote = (j.headnote or "").strip()
            ratio = (j.ratio_decidendi or "").strip()

            if len(headnote) < MIN_HEADNOTE_LENGTH:
                skipped_short += 1
                continue

            # Extract metadata
            sections = extract_sections_list(j.sections_interpreted)
            topic = extract_topic_from_headnote(headnote, j.case_name)
            domain = j.domain or "general"

            # Generate questions
            questions = generate_questions(
                case_name=j.case_name,
                headnote=headnote,
                domain=domain,
                sections=sections,
                topic=topic,
            )

            # Build response (same for all questions from this judgment)
            response = build_response(
                headnote=headnote,
                ratio=ratio,
                case_name=j.case_name,
                year=j.year,
                court=j.court or "Supreme Court",
                citation_scc=j.citation_scc,
                citation_air=j.citation_air,
                sections=sections,
                domain=domain,
                bench_size=j.bench_size,
                court_type=j.court_type or "SC",
            )

            # Create SFTPair for each question variant
            for i, q in enumerate(questions):
                pair_id = f"headnote_{j.case_name[:40]}_{i}".replace(" ", "_").lower()
                # Make ID deterministic using hash
                pair_hash = hashlib.md5(
                    f"{j.case_name}:{q['question']}".encode()
                ).hexdigest()[:10]
                pair_id = f"hn_{pair_hash}"

                pair = SFTPair(
                    pair_id=pair_id,
                    source="headnote",
                    instruction=q["question"],
                    response=response,
                    system_prompt=SFT_SYSTEM_PROMPT,
                    domain=domain,
                    query_type=q["query_type"],
                    jurisdiction="central" if j.court_type == "SC" else "state",
                    status="draft",
                    source_judgment_id=str(j.id),
                    cited_sections=[
                        f"{s['act']}/{s['section']}" for s in sections
                    ],
                    cited_cases=[j.case_name],
                )
                pairs.append(pair)

            processed += 1

        offset += page_size

        if limit and processed >= limit:
            break

        logger.info(
            "headnote_extraction_progress",
            processed=processed,
            pairs_generated=len(pairs),
            offset=offset,
        )

    logger.info(
        "headnote_extraction_complete",
        judgments_processed=processed,
        judgments_skipped_short=skipped_short,
        total_pairs=len(pairs),
    )

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════


def export_pairs(pairs: list[SFTPair], output_path: Path | None = None) -> Path:
    """Export pairs as JSONL."""
    path = output_path or SFT_PATHS.headnote_raw
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    logger.info("headnote_pairs_exported", path=str(path), count=len(pairs))
    return path


def print_summary(pairs: list[SFTPair]) -> None:
    """Print extraction summary."""
    from collections import Counter

    print(f"\n{'═' * 60}")
    print(f"  Headnote Extraction — Summary")
    print(f"{'═' * 60}\n")
    print(f"  Total pairs: {len(pairs)}")

    # By domain
    domains = Counter(p.domain for p in pairs)
    print(f"\n  By domain:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        pct = count / len(pairs) * 100 if pairs else 0
        print(f"    {domain:<20} {count:>6} ({pct:.1f}%)")

    # By query type
    qtypes = Counter(p.query_type for p in pairs)
    print(f"\n  By query type:")
    for qt, count in sorted(qtypes.items(), key=lambda x: -x[1]):
        print(f"    {qt:<20} {count:>6}")

    # By source judgment (unique judgments)
    unique_judgments = len(set(p.source_judgment_id for p in pairs if p.source_judgment_id))
    avg_per_judgment = len(pairs) / unique_judgments if unique_judgments else 0
    print(f"\n  Unique source judgments: {unique_judgments}")
    print(f"  Avg pairs per judgment: {avg_per_judgment:.1f}")

    # Response length stats
    if pairs:
        lengths = [estimate_token_count(p.response) for p in pairs]
        print(f"\n  Response length (est. tokens):")
        print(f"    Min: {min(lengths)}")
        print(f"    Max: {max(lengths)}")
        print(f"    Avg: {sum(lengths) // len(lengths)}")

    # Sample pairs
    print(f"\n  Sample pairs (first 3):")
    for p in pairs[:3]:
        print(f"    [{p.pair_id}] ({p.domain})")
        print(f"      Q: {p.instruction[:80]}...")
        print(f"      A: {p.response[:100]}...")
        print()

    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra Headnote Extraction Pipeline (Sprint 8)",
    )
    parser.add_argument("--domain", type=str, default=None,
                        help="Only extract from this domain")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum judgments to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sample pairs without saving")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path")

    args = parser.parse_args()

    start = time.time()

    pairs = await extract_headnote_pairs(
        domain_filter=args.domain,
        limit=args.limit,
    )

    duration = round(time.time() - start, 2)

    if args.dry_run:
        print_summary(pairs)
        print(f"  (Dry run — {len(pairs)} pairs generated in {duration}s, not saved)")
    else:
        output_path = Path(args.output) if args.output else None
        path = export_pairs(pairs, output_path)
        print_summary(pairs)
        print(f"  Exported to: {path}")
        print(f"  Duration: {duration}s")


if __name__ == "__main__":
    asyncio.run(main())