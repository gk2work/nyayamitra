"""
NyayaMitra — Synthetic SFT Pair Generator (Sprint 8).

Uses the Anthropic API (Claude) to generate draft instruction-response
pairs from act sections stored in PostgreSQL. Each section is sent as
context, and Claude generates realistic citizen questions + NyayaMitra-
format structured responses.

These are DRAFTS — every pair gets status "needs_review" and must be
verified by a human annotator before entering the final dataset.

Target: 15,000 draft pairs across all domains.

Pipeline:
    1. Stream sections from PostgreSQL (with act metadata)
    2. For each section, build a generation prompt with context
    3. Call Anthropic API to generate Q&A pairs
    4. Parse API response into SFTPair objects
    5. Validate format compliance
    6. Export as JSONL with status "needs_review"

Usage:
    # Generate from all sections
    python -m data.training.synthetic_generator

    # Specific domain
    python -m data.training.synthetic_generator --domain criminal

    # Limit (for testing / budget control)
    python -m data.training.synthetic_generator --limit 50

    # Dry run (show prompts without calling API)
    python -m data.training.synthetic_generator --dry-run

    # Custom batch size
    python -m data.training.synthetic_generator --batch-size 5
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

import httpx
import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.database import async_session
from app.models.legal import Act, Section

from data.training.sft_config import (
    SFT_SYSTEM_PROMPT,
    SFTPair,
    SFT_PATHS,
    STANDARD_DISCLAIMER,
    RESPONSE_SECTIONS,
    LEGAL_DOMAINS,
    validate_response_format,
    estimate_token_count,
)

logger = structlog.get_logger()

# ── Anthropic API Config ──────────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS_PER_CALL = 4000

# Rate limiting: stay well within Anthropic limits
API_DELAY_SECONDS = 1.5       # Delay between API calls
MAX_RETRIES = 3               # Retries on failure
RETRY_BACKOFF = 3.0           # Exponential backoff base

# Generation config
PAIRS_PER_SECTION = 3         # How many Q&A pairs to request per section
MAX_SECTIONS_PER_BATCH = 1    # Sections per API call (1 for quality)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Engineering
# ═══════════════════════════════════════════════════════════════════════════════

GENERATION_PROMPT_TEMPLATE = """You are helping build a training dataset for NyayaMitra, an AI legal assistant for Indian citizens. Given a section of Indian law below, generate {num_pairs} realistic question-answer pairs.

SECTION CONTEXT:
Act: {act_name} ({act_short_name}), {act_year}
Section: {section_number}
Title: {section_title}
Domain: {domain}
Chapter: {chapter}

Section Text:
{section_text}

{related_context}

INSTRUCTIONS:
1. Generate {num_pairs} different questions a citizen might ask about this legal provision.
2. Questions should be in natural language — how a real person (not a lawyer) would ask.
3. Mix question styles: some about rights, some about procedure, some about specific scenarios.
4. Each answer MUST follow this exact structured format:

[APPLICABLE_LAW]
- Section {{number}} of {{Act Name, Year}}: Brief explanation of what it provides

[PRECEDENT]
Cite any relevant landmark case if known, or write "No specific precedent directly on point."

[LEGAL_POSITION]
Plain-language explanation of the current legal position (2-3 sentences)

[PROCEDURE]
Step-by-step guidance if applicable, or "Not applicable for this query."

[JURISDICTION_NOTE]
Note if this is central law or has state variations

[CONFIDENCE]
High/Medium/Low — {{reasoning}}

[DISCLAIMER]
This is legal information, not legal advice. For case-specific advice, consult a qualified advocate. Laws are subject to amendments and judicial interpretation. Verify current status before acting.

IMPORTANT RULES:
- ONLY cite Section {section_number} of {act_short_name} and any sections mentioned in the text.
- Do NOT invent section numbers or case names.
- Keep answers concise but complete (150-400 words).
- Use simple language a non-lawyer can understand.

Respond with a JSON array of objects, each with "question" and "answer" keys:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

Return ONLY the JSON array, no other text."""

RELATED_CONTEXT_TEMPLATE = """
RELATED SECTIONS (for cross-reference only — do not cite unless genuinely relevant):
{related_sections}

INTERPRETING JUDGMENTS:
{interpreting_judgments}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Context Builder
# ═══════════════════════════════════════════════════════════════════════════════


async def build_section_context(
    section: Section,
    act: Act,
) -> dict:
    """
    Build the full context dict for a section, including related
    sections and interpreting judgments from Neo4j / PostgreSQL.
    """
    context = {
        "act_name": act.name,
        "act_short_name": act.short_name or "",
        "act_year": act.year,
        "section_number": section.section_number,
        "section_title": section.title or "Untitled",
        "section_text": (section.text or "")[:3000],  # Cap context
        "domain": act.domain or "general",
        "chapter": section.chapter or "",
        "related_sections": "",
        "interpreting_judgments": "",
    }

    # Try to get related sections (same act, adjacent numbers)
    try:
        from sqlalchemy import select

        async with async_session() as session:
            stmt = (
                select(Section)
                .where(Section.act_id == act.id)
                .where(Section.id != section.id)
                .limit(3)
            )
            result = await session.execute(stmt)
            related = result.scalars().all()

            if related:
                related_lines = []
                for r in related:
                    related_lines.append(
                        f"- Section {r.section_number}: {r.title or 'Untitled'}"
                    )
                context["related_sections"] = "\n".join(related_lines)
    except Exception:
        pass

    # Try to get interpreting judgments from PostgreSQL
    try:
        from app.models.legal import Judgment
        from sqlalchemy import select

        section_ref = f'"{act.short_name}", "section": "{section.section_number}"'

        async with async_session() as session:
            stmt = (
                select(Judgment)
                .where(Judgment.sections_interpreted.contains(section_ref))
                .limit(2)
            )
            result = await session.execute(stmt)
            judgments = result.scalars().all()

            if judgments:
                j_lines = []
                for j in judgments:
                    citation = j.citation_scc or j.citation_air or ""
                    j_lines.append(
                        f"- {j.case_name} ({j.year}) [{citation}]: "
                        f"{(j.headnote or '')[:200]}"
                    )
                context["interpreting_judgments"] = "\n".join(j_lines)
    except Exception:
        pass

    return context


def build_prompt(context: dict, num_pairs: int = PAIRS_PER_SECTION) -> str:
    """Build the generation prompt from section context."""
    # Build related context block
    related_block = ""
    if context["related_sections"] or context["interpreting_judgments"]:
        related_block = RELATED_CONTEXT_TEMPLATE.format(
            related_sections=context["related_sections"] or "None available",
            interpreting_judgments=context["interpreting_judgments"] or "None available",
        )

    return GENERATION_PROMPT_TEMPLATE.format(
        num_pairs=num_pairs,
        act_name=context["act_name"],
        act_short_name=context["act_short_name"],
        act_year=context["act_year"],
        section_number=context["section_number"],
        section_title=context["section_title"],
        section_text=context["section_text"],
        domain=context["domain"],
        chapter=context["chapter"],
        related_context=related_block,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Anthropic API Client
# ═══════════════════════════════════════════════════════════════════════════════


async def call_anthropic(
    prompt: str,
    client: httpx.AsyncClient,
    api_key: str,
) -> str | None:
    """
    Call the Anthropic API with retry logic.

    Returns the text response or None on failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": MAX_TOKENS_PER_CALL,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=120.0,
            )

            if response.status_code == 200:
                data = response.json()
                # Extract text from response
                content = data.get("content", [])
                if content and content[0].get("type") == "text":
                    return content[0]["text"]
                return None

            if response.status_code == 429:
                delay = RETRY_BACKOFF * (2 ** attempt) + 10.0
                logger.warning(
                    "anthropic_rate_limited",
                    retry_in=delay,
                    attempt=attempt + 1,
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code >= 500:
                delay = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "anthropic_server_error",
                    status=response.status_code,
                    retry_in=delay,
                )
                await asyncio.sleep(delay)
                continue

            logger.error(
                "anthropic_client_error",
                status=response.status_code,
                body=response.text[:200],
            )
            return None

        except httpx.TimeoutException:
            delay = RETRY_BACKOFF * (2 ** attempt)
            logger.warning("anthropic_timeout", retry_in=delay, attempt=attempt + 1)
            await asyncio.sleep(delay)

        except Exception as e:
            logger.error("anthropic_error", error=str(e))
            return None

    logger.error("anthropic_all_retries_exhausted")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Response Parser
# ═══════════════════════════════════════════════════════════════════════════════


def parse_api_response(
    response_text: str,
    context: dict,
) -> list[SFTPair]:
    """
    Parse the Anthropic API response into SFTPair objects.

    The API should return a JSON array of {"question": ..., "answer": ...}.
    Handles common formatting issues (markdown fences, extra text).
    """
    pairs = []

    # Strip markdown code fences if present
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try parsing as JSON array
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array within the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning(
                    "api_response_parse_failed",
                    section=context.get("section_number"),
                    response_preview=text[:200],
                )
                return []
        else:
            logger.warning(
                "api_response_no_json",
                section=context.get("section_number"),
            )
            return []

    if not isinstance(items, list):
        items = [items]

    for i, item in enumerate(items):
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        if not question or not answer:
            continue

        # Generate deterministic pair ID
        pair_hash = hashlib.md5(
            f"{context['act_short_name']}/{context['section_number']}:{question}".encode()
        ).hexdigest()[:10]
        pair_id = f"syn_{pair_hash}"

        # Extract cited sections from the answer
        cited_sections = []
        section_refs = re.findall(
            r"Section\s+(\d+[A-Za-z]*)\s+of\s+([A-Za-z\s,]+?)(?:\s*\d{4})?(?=[:\.\n])",
            answer,
        )
        for sec_num, act_ref in section_refs:
            cited_sections.append(f"{act_ref.strip()}/{sec_num}")

        # Always include the source section
        source_ref = f"{context['act_short_name']}/{context['section_number']}"
        if source_ref not in cited_sections:
            cited_sections.insert(0, source_ref)

        pair = SFTPair(
            pair_id=pair_id,
            source="synthetic",
            instruction=question,
            response=answer,
            system_prompt=SFT_SYSTEM_PROMPT,
            domain=context["domain"],
            query_type=_infer_query_type(question),
            jurisdiction="central",
            status="needs_review",  # ALWAYS needs human verification
            source_section_id=context.get("section_pg_id"),
            cited_sections=cited_sections,
            cited_cases=[],
        )
        pairs.append(pair)

    return pairs


def _infer_query_type(question: str) -> str:
    """Infer query type from question phrasing."""
    q_lower = question.lower()
    if any(w in q_lower for w in ["how to", "procedure", "steps", "process", "file", "apply"]):
        return "procedure"
    if any(w in q_lower for w in ["what did the court", "held in", "precedent", "judgment"]):
        return "case_outcome"
    return "rights"


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════


class GenerationStats:
    """Track generation statistics."""

    def __init__(self):
        self.sections_processed: int = 0
        self.sections_skipped: int = 0
        self.api_calls: int = 0
        self.api_errors: int = 0
        self.pairs_generated: int = 0
        self.pairs_invalid: int = 0
        self.tokens_used: int = 0  # Rough estimate
        self.start_time: float = time.time()

    def to_dict(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            "sections_processed": self.sections_processed,
            "sections_skipped": self.sections_skipped,
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
            "pairs_generated": self.pairs_generated,
            "pairs_invalid": self.pairs_invalid,
            "tokens_used_estimate": self.tokens_used,
            "duration_seconds": round(elapsed, 2),
            "pairs_per_minute": round(
                self.pairs_generated / (elapsed / 60), 1
            ) if elapsed > 0 else 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Generation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_synthetic_pairs(
    domain_filter: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    pairs_per_section: int = PAIRS_PER_SECTION,
) -> list[SFTPair]:
    """
    Generate synthetic SFT pairs from all qualifying sections.

    Args:
        domain_filter: Only process sections from this domain.
        limit: Maximum sections to process (not pairs).
        dry_run: Show prompts without calling the API.
        pairs_per_section: How many Q&A pairs to request per section.

    Returns:
        List of SFTPair objects with status "needs_review".
    """
    from sqlalchemy import select, func
    from sqlalchemy.orm import selectinload

    # Check API key
    api_key = getattr(settings, "ANTHROPIC_API_KEY", "") or ""
    if not api_key and not dry_run:
        logger.error("no_anthropic_api_key", hint="Set ANTHROPIC_API_KEY in .env")
        print("\nERROR: Set ANTHROPIC_API_KEY in .env to use the synthetic generator.")
        print("Get a key from: https://console.anthropic.com/\n")
        return []

    stats = GenerationStats()
    all_pairs: list[SFTPair] = []

    logger.info(
        "synthetic_generation_start",
        domain_filter=domain_filter,
        limit=limit,
        dry_run=dry_run,
        pairs_per_section=pairs_per_section,
    )

    # Stream sections with act metadata
    page_size = 100
    offset = 0

    async with httpx.AsyncClient() as client:
        while True:
            async with async_session() as session:
                stmt = (
                    select(Section, Act)
                    .join(Act, Section.act_id == Act.id)
                    .where(Section.text.isnot(None))
                    .order_by(Act.domain, Section.id)
                    .offset(offset)
                    .limit(page_size)
                )
                if domain_filter:
                    stmt = stmt.where(Act.domain == domain_filter)

                result = await session.execute(stmt)
                rows = result.all()

            if not rows:
                break

            for section, act in rows:
                if limit and stats.sections_processed >= limit:
                    break

                # Skip very short sections
                text = (section.text or "").strip()
                if len(text) < 50:
                    stats.sections_skipped += 1
                    continue

                # Build context
                context = await build_section_context(section, act)
                context["section_pg_id"] = str(section.id)

                # Build prompt
                prompt = build_prompt(context, num_pairs=pairs_per_section)

                if dry_run:
                    print(f"\n{'─' * 60}")
                    print(f"  Section: {act.short_name}/{section.section_number}")
                    print(f"  Domain: {act.domain}")
                    print(f"  Prompt length: {len(prompt)} chars")
                    print(f"  Prompt preview:")
                    print(f"  {prompt[:300]}...")
                    stats.sections_processed += 1
                    continue

                # Call API
                await asyncio.sleep(API_DELAY_SECONDS)
                stats.api_calls += 1

                response_text = await call_anthropic(prompt, client, api_key)

                if not response_text:
                    stats.api_errors += 1
                    continue

                # Estimate tokens used
                stats.tokens_used += estimate_token_count(prompt) + estimate_token_count(response_text)

                # Parse response into pairs
                pairs = parse_api_response(response_text, context)

                # Validate each pair
                for pair in pairs:
                    validation = validate_response_format(pair.response)
                    if validation["valid"]:
                        all_pairs.append(pair)
                        stats.pairs_generated += 1
                    else:
                        # Still keep it but flag — annotators can fix format
                        pair.review_notes = f"Format issues: {validation['issues']}"
                        all_pairs.append(pair)
                        stats.pairs_invalid += 1
                        stats.pairs_generated += 1

                stats.sections_processed += 1

                # Progress logging
                if stats.sections_processed % 10 == 0:
                    logger.info(
                        "synthetic_generation_progress",
                        **stats.to_dict(),
                    )

            offset += page_size

            if limit and stats.sections_processed >= limit:
                break

    logger.info("synthetic_generation_complete", **stats.to_dict())
    return all_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════


def export_pairs(pairs: list[SFTPair], output_path: Path | None = None) -> Path:
    """Export pairs as JSONL."""
    path = output_path or SFT_PATHS.synthetic_raw
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    logger.info("synthetic_pairs_exported", path=str(path), count=len(pairs))
    return path


def print_summary(pairs: list[SFTPair], stats_dict: dict | None = None) -> None:
    """Print generation summary."""
    from collections import Counter

    print(f"\n{'═' * 60}")
    print(f"  Synthetic Generation — Summary")
    print(f"{'═' * 60}\n")
    print(f"  Total pairs: {len(pairs)}")

    if stats_dict:
        print(f"  Sections processed: {stats_dict.get('sections_processed', 0)}")
        print(f"  API calls: {stats_dict.get('api_calls', 0)}")
        print(f"  API errors: {stats_dict.get('api_errors', 0)}")
        print(f"  Format issues: {stats_dict.get('pairs_invalid', 0)}")
        print(f"  Est. tokens used: {stats_dict.get('tokens_used_estimate', 0):,}")
        print(f"  Duration: {stats_dict.get('duration_seconds', 0)}s")
        print(f"  Rate: {stats_dict.get('pairs_per_minute', 0)} pairs/min")

    # By domain
    domains = Counter(p.domain for p in pairs)
    print(f"\n  By domain:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        pct = count / len(pairs) * 100 if pairs else 0
        print(f"    {domain:<20} {count:>6} ({pct:.1f}%)")

    # By status
    statuses = Counter(p.status for p in pairs)
    print(f"\n  By status:")
    for status, count in sorted(statuses.items()):
        print(f"    {status:<20} {count:>6}")

    # By query type
    qtypes = Counter(p.query_type for p in pairs)
    print(f"\n  By query type:")
    for qt, count in sorted(qtypes.items(), key=lambda x: -x[1]):
        print(f"    {qt:<20} {count:>6}")

    # Sample
    if pairs:
        print(f"\n  Sample pairs (first 2):")
        for p in pairs[:2]:
            print(f"    [{p.pair_id}] ({p.domain}) [{p.status}]")
            print(f"      Q: {p.instruction[:80]}...")
            print(f"      A: {p.response[:120]}...")
            print()

    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Cost Estimation
# ═══════════════════════════════════════════════════════════════════════════════


def estimate_cost(
    num_sections: int,
    pairs_per_section: int = PAIRS_PER_SECTION,
) -> dict:
    """
    Estimate Anthropic API cost for synthetic generation.

    Based on Claude Sonnet 4 pricing:
    - Input: $3 / 1M tokens
    - Output: $15 / 1M tokens
    """
    avg_prompt_tokens = 800   # ~800 tokens per prompt
    avg_output_tokens = 1500  # ~1500 tokens per response (3 pairs)

    total_input = num_sections * avg_prompt_tokens
    total_output = num_sections * avg_output_tokens
    total_pairs = num_sections * pairs_per_section

    input_cost = (total_input / 1_000_000) * 3.0
    output_cost = (total_output / 1_000_000) * 15.0
    total_cost = input_cost + output_cost

    return {
        "sections": num_sections,
        "estimated_pairs": total_pairs,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "input_cost_usd": round(input_cost, 2),
        "output_cost_usd": round(output_cost, 2),
        "total_cost_usd": round(total_cost, 2),
        "time_minutes": round(num_sections * API_DELAY_SECONDS / 60, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra Synthetic SFT Generator (Sprint 8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m data.training.synthetic_generator --dry-run --limit 5
  python -m data.training.synthetic_generator --domain criminal --limit 20
  python -m data.training.synthetic_generator --estimate 62
  python -m data.training.synthetic_generator                      # Full run
        """,
    )
    parser.add_argument("--domain", type=str, default=None,
                        help="Only generate from this domain")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum sections to process")
    parser.add_argument("--pairs-per-section", type=int, default=PAIRS_PER_SECTION,
                        help=f"Pairs to generate per section (default: {PAIRS_PER_SECTION})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts without calling API")
    parser.add_argument("--estimate", type=int, default=None, metavar="N",
                        help="Estimate cost for N sections without running")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path")

    args = parser.parse_args()

    # Cost estimation mode
    if args.estimate:
        cost = estimate_cost(args.estimate, args.pairs_per_section)
        print(f"\n{'═' * 50}")
        print(f"  Cost Estimate for {cost['sections']} sections")
        print(f"{'═' * 50}")
        print(f"  Estimated pairs:  {cost['estimated_pairs']:,}")
        print(f"  Input tokens:     {cost['input_tokens']:,}")
        print(f"  Output tokens:    {cost['output_tokens']:,}")
        print(f"  Input cost:       ${cost['input_cost_usd']}")
        print(f"  Output cost:      ${cost['output_cost_usd']}")
        print(f"  Total cost:       ${cost['total_cost_usd']}")
        print(f"  Estimated time:   {cost['time_minutes']} minutes")
        print(f"{'═' * 50}\n")
        return

    start = time.time()

    pairs = await generate_synthetic_pairs(
        domain_filter=args.domain,
        limit=args.limit,
        dry_run=args.dry_run,
        pairs_per_section=args.pairs_per_section,
    )

    duration = round(time.time() - start, 2)

    if args.dry_run:
        print(f"\n  Dry run complete — {len(pairs)} sections previewed in {duration}s")
        return

    if pairs:
        output_path = Path(args.output) if args.output else None
        path = export_pairs(pairs, output_path)
        print_summary(pairs)
        print(f"  Exported to: {path}")
    else:
        print("\n  No pairs generated. Check API key and section data.")

    print(f"  Duration: {duration}s\n")


if __name__ == "__main__":
    asyncio.run(main())