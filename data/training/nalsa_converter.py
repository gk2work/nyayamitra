"""
NyayaMitra — NALSA / Procedure → SFT Converter (Sprint 8).

Converts procedural knowledge (37 procedures from procedure_builder.py
+ 14 NALSA FAQs) into the SFT instruction-response format. Generates
multiple question phrasings per procedure to maximise training volume.

Target: ~10,000 pairs from procedural sources.

Strategy:
    1. Load all procedures from procedure_builder + NALSA scraper
    2. For each procedure, generate 5-10 question variants:
       - Formal: "What is the procedure to file an FIR?"
       - Colloquial: "How do I file a police complaint?"
       - Scenario: "Someone stole my phone, what should I do?"
       - Hindi-English mix: "FIR kaise file karein?"
       - Partial: "Where do I go to register a complaint?"
    3. Build structured response matching the SFT template
    4. Also generate section-based pairs from cited law references
    5. Validate and export as JSONL

Usage:
    python -m data.training.nalsa_converter
    python -m data.training.nalsa_converter --domain criminal
    python -m data.training.nalsa_converter --dry-run
    python -m data.training.nalsa_converter --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from data.training.sft_config import (
    SFT_SYSTEM_PROMPT,
    SFTPair,
    SFT_PATHS,
    STANDARD_DISCLAIMER,
    format_sft_pair,
    format_applicable_law_section,
    format_procedure_section,
    estimate_token_count,
)

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Question Variant Generation
#
# Each procedure gets multiple question phrasings to teach the model
# that different wordings map to the same procedure.
# ═══════════════════════════════════════════════════════════════════════════════

# Templates per domain for generating question variants
VARIANT_TEMPLATES: dict[str, list[dict]] = {
    "criminal": [
        {"template": "What is the procedure to {action}?", "type": "procedure"},
        {"template": "How do I {action}?", "type": "procedure"},
        {"template": "I need to {action}. What should I do step by step?", "type": "procedure"},
        {"template": "What are my rights when {situation}?", "type": "rights"},
        {"template": "{situation}. What legal steps can I take?", "type": "procedure"},
        {"template": "Can you explain how to {action} in India?", "type": "procedure"},
        {"template": "What documents do I need to {action}?", "type": "procedure"},
        {"template": "Where should I go to {action}?", "type": "procedure"},
    ],
    "property": [
        {"template": "What is the process for {action}?", "type": "procedure"},
        {"template": "How to {action} in India?", "type": "procedure"},
        {"template": "What are the legal requirements for {action}?", "type": "rights"},
        {"template": "I want to {action}. What steps should I follow?", "type": "procedure"},
        {"template": "What fees and stamp duty are involved in {action}?", "type": "procedure"},
        {"template": "Which authority handles {action}?", "type": "procedure"},
    ],
    "family": [
        {"template": "How to {action} in India?", "type": "procedure"},
        {"template": "What is the legal procedure for {action}?", "type": "procedure"},
        {"template": "I need to {action}. What are my options?", "type": "procedure"},
        {"template": "What are the grounds for {action}?", "type": "rights"},
        {"template": "How long does {action} take?", "type": "procedure"},
        {"template": "Do I need a lawyer for {action}?", "type": "procedure"},
        {"template": "What court handles {action}?", "type": "procedure"},
    ],
    "consumer": [
        {"template": "How to {action}?", "type": "procedure"},
        {"template": "I bought a defective product. How to {action}?", "type": "procedure"},
        {"template": "What is the process to {action}?", "type": "procedure"},
        {"template": "Where do I file a complaint for {action}?", "type": "procedure"},
        {"template": "What compensation can I get for {action}?", "type": "rights"},
        {"template": "Can I {action} online?", "type": "procedure"},
    ],
    "labour": [
        {"template": "How to {action}?", "type": "procedure"},
        {"template": "My employer {situation}. What can I do?", "type": "rights"},
        {"template": "What are my rights regarding {action}?", "type": "rights"},
        {"template": "Where do I complain about {action}?", "type": "procedure"},
        {"template": "What is the procedure for {action}?", "type": "procedure"},
        {"template": "Is my employer legally required to {action}?", "type": "rights"},
    ],
    "constitutional": [
        {"template": "How to {action}?", "type": "procedure"},
        {"template": "What is the process to {action}?", "type": "procedure"},
        {"template": "I want to {action}. How do I do it?", "type": "procedure"},
        {"template": "What is the fee for {action}?", "type": "procedure"},
        {"template": "How long does it take to {action}?", "type": "procedure"},
        {"template": "Who is responsible for {action}?", "type": "procedure"},
    ],
    "ip": [
        {"template": "How to {action}?", "type": "procedure"},
        {"template": "What is the procedure to {action}?", "type": "procedure"},
        {"template": "I want to {action}. What steps should I follow?", "type": "procedure"},
        {"template": "What are the penalties for {action}?", "type": "rights"},
        {"template": "Where do I report {action}?", "type": "procedure"},
    ],
    "corporate": [
        {"template": "What is the legal process for {action}?", "type": "procedure"},
        {"template": "How to {action}?", "type": "procedure"},
        {"template": "What are the requirements for {action}?", "type": "rights"},
        {"template": "What is the time limit for {action}?", "type": "procedure"},
        {"template": "Can I {action} without a lawyer?", "type": "procedure"},
    ],
    "environmental": [
        {"template": "How to {action}?", "type": "procedure"},
        {"template": "Where do I complain about {action}?", "type": "procedure"},
        {"template": "What legal remedy is available for {action}?", "type": "rights"},
        {"template": "What is the procedure for {action}?", "type": "procedure"},
    ],
}

# Fallback templates for domains without specific entries
DEFAULT_TEMPLATES = [
    {"template": "How to {action}?", "type": "procedure"},
    {"template": "What is the procedure to {action}?", "type": "procedure"},
    {"template": "I need to {action}. What should I do?", "type": "procedure"},
    {"template": "What are my legal options for {action}?", "type": "rights"},
    {"template": "Where should I go for {action}?", "type": "procedure"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Action/Situation Extraction
# ═══════════════════════════════════════════════════════════════════════════════


def extract_action_phrase(question: str) -> str:
    """
    Extract the core action from a procedure question.

    "How to file an FIR (First Information Report)?" → "file an FIR"
    "What are my rights if I am arrested?" → "handle an arrest situation"
    """
    import re

    q = question.strip().rstrip("?").lower()

    # Remove common prefixes
    for prefix in [
        "how to ", "how do i ", "how can i ", "what is the procedure to ",
        "what is the process for ", "what is the process to ",
        "what are the steps to ", "what should i do to ",
    ]:
        if q.startswith(prefix):
            return q[len(prefix):].strip()

    # For "What are my rights..." questions, convert to action
    if "rights" in q:
        # "What are my rights if I am arrested?" → "being arrested"
        match = re.search(r"rights?\s+(?:if|when|regarding|about)\s+(.+)", q)
        if match:
            return match.group(1).strip()

    # Fallback: use the whole question minus "what/how"
    q = re.sub(r"^(?:what|how|where|when|can)\s+", "", q)
    return q[:80]


def extract_situation_phrase(question: str, answer: str) -> str:
    """
    Extract a situation description for scenario-based questions.

    Uses the answer's first sentence if the question is too abstract.
    """
    # Try to get situation from the answer's first sentence
    first_sentence = answer.split(".")[0].strip()
    if len(first_sentence) > 20 and len(first_sentence) < 150:
        return first_sentence.lower()

    return extract_action_phrase(question)


# ═══════════════════════════════════════════════════════════════════════════════
# Response Builder
# ═══════════════════════════════════════════════════════════════════════════════


def build_procedure_response(proc: dict) -> str:
    """
    Build a structured SFT response from a procedure dict.

    Args:
        proc: A procedure dict from procedure_builder or nalsa_scraper.
              Has: question, answer, domain, steps, relevant_law, jurisdiction.
    """
    # [APPLICABLE_LAW]
    law_entries = []
    for law in proc.get("relevant_law", []):
        law_entries.append({
            "act": law.get("act", ""),
            "section": law.get("section", ""),
            "text": "",
        })
    applicable_law = format_applicable_law_section(law_entries)

    # [PRECEDENT]
    precedent = "No specific precedent directly on point."

    # [LEGAL_POSITION]
    legal_position = proc.get("answer", "").strip()

    # [PROCEDURE]
    procedure = format_procedure_section(proc.get("steps", []))

    # [JURISDICTION_NOTE]
    jurisdiction = proc.get("jurisdiction", "central")
    if jurisdiction == "central":
        jurisdiction_note = "This is central law applicable across India. State-specific rules may add additional requirements."
    else:
        jurisdiction_note = f"This procedure may vary by state. The information provided is for {jurisdiction}."

    # [CONFIDENCE] — procedural guides are well-researched
    confidence = "High"

    return format_sft_pair(
        instruction="",
        applicable_law=applicable_law,
        precedent=precedent,
        legal_position=legal_position,
        procedure=procedure,
        jurisdiction_note=jurisdiction_note,
        confidence=confidence,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Question Variant Generator
# ═══════════════════════════════════════════════════════════════════════════════


def generate_question_variants(proc: dict) -> list[dict]:
    """
    Generate multiple question phrasings for a single procedure.

    Returns list of {"question": str, "query_type": str} dicts.
    """
    original_question = proc.get("question", "")
    domain = proc.get("domain", "general")
    answer = proc.get("answer", "")

    action = extract_action_phrase(original_question)
    situation = extract_situation_phrase(original_question, answer)

    variants = []

    # 1. Always include the original question
    variants.append({
        "question": original_question,
        "query_type": "procedure",
    })

    # 2. Apply domain-specific templates
    templates = VARIANT_TEMPLATES.get(domain, DEFAULT_TEMPLATES)

    for tmpl in templates:
        try:
            if "{action}" in tmpl["template"] and "{situation}" in tmpl["template"]:
                q = tmpl["template"].format(action=action, situation=situation)
            elif "{action}" in tmpl["template"]:
                q = tmpl["template"].format(action=action)
            elif "{situation}" in tmpl["template"]:
                q = tmpl["template"].format(situation=situation)
            else:
                continue

            # Skip if too similar to original
            if q.lower().strip("? ") == original_question.lower().strip("? "):
                continue

            # Skip if too short or too long
            if len(q) < 15 or len(q) > 300:
                continue

            variants.append({
                "question": q,
                "query_type": tmpl["type"],
            })
        except (KeyError, IndexError):
            continue

    # 3. Add a "simple Hindi-English" variant if criminal/family/consumer
    if domain in ("criminal", "family", "consumer", "labour"):
        hindi_variants = _generate_hinglish_variant(action, domain)
        variants.extend(hindi_variants)

    # Deduplicate by lowered question
    seen = set()
    unique = []
    for v in variants:
        key = v["question"].lower().strip("? ")
        if key not in seen:
            seen.add(key)
            unique.append(v)

    # Cap at 10 variants per procedure
    return unique[:10]


def _generate_hinglish_variant(action: str, domain: str) -> list[dict]:
    """Generate Hindi-English (Hinglish) question variants."""
    variants = []

    # Common Hinglish patterns citizens use
    hinglish_patterns = {
        "file an fir": [
            "FIR kaise file kare?",
            "Police complaint kaise likhaye?",
        ],
        "apply for bail": [
            "Bail ke liye kaise apply kare?",
            "Zamanat kaise milti hai?",
        ],
        "file a consumer complaint": [
            "Consumer complaint kaise kare?",
            "Upbhokta shikayat kaise darj kare?",
        ],
        "file for divorce": [
            "Divorce kaise le?",
            "Talaq ka kanooni process kya hai?",
        ],
    }

    action_lower = action.lower().strip()
    for key, questions in hinglish_patterns.items():
        if key in action_lower or action_lower in key:
            for q in questions:
                variants.append({"question": q, "query_type": "procedure"})
            break

    # Generic Hinglish if no specific match
    if not variants:
        variants.append({
            "question": f"{action.capitalize()} kaise kare?",
            "query_type": "procedure",
        })

    return variants[:2]  # Max 2 Hinglish variants


# ═══════════════════════════════════════════════════════════════════════════════
# Section-Based Pair Generation
#
# For each relevant_law entry in a procedure, generate a pair asking
# about that specific section. This adds volume and teaches the model
# about the underlying statutory provisions.
# ═══════════════════════════════════════════════════════════════════════════════


def generate_section_pairs(proc: dict) -> list[SFTPair]:
    """
    Generate pairs from the relevant_law entries of a procedure.

    For each cited section, create a question like:
    "What does Section X of Act say about {topic}?"
    """
    pairs = []
    laws = proc.get("relevant_law", [])
    answer_text = proc.get("answer", "")
    domain = proc.get("domain", "general")
    proc_id = proc.get("id", "unknown")

    for law in laws[:3]:  # Max 3 section-based pairs per procedure
        act = law.get("act", "")
        section = law.get("section", "")
        if not act or not section:
            continue

        question = f"What does Section {section} of {act} provide?"

        # Build a focused response about this section
        applicable_law = f"- Section {section} of {act}: This provision is relevant to {answer_text[:100]}..."
        response = format_sft_pair(
            instruction="",
            applicable_law=applicable_law,
            legal_position=answer_text[:400],
            confidence="Medium",
        )

        pair_hash = hashlib.md5(
            f"nalsa_sec:{act}/{section}:{proc_id}".encode()
        ).hexdigest()[:10]

        pair = SFTPair(
            pair_id=f"ns_{pair_hash}",
            source="nalsa",
            instruction=question,
            response=response,
            system_prompt=SFT_SYSTEM_PROMPT,
            domain=domain,
            query_type="section_explanation",
            jurisdiction=proc.get("jurisdiction", "central"),
            status="draft",
            source_procedure_id=proc_id,
            cited_sections=[f"{act}/{section}"],
        )
        pairs.append(pair)

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Main Conversion Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def load_all_procedures() -> list[dict]:
    """Load procedures from both procedure_builder and NALSA scraper."""
    all_procs = []

    # From procedure_builder
    try:
        from data.procedures.procedure_builder import build_all_procedures
        procs = build_all_procedures()
        all_procs.extend(procs)
        logger.info("loaded_procedure_builder", count=len(procs))
    except Exception as e:
        logger.warning("procedure_builder_load_failed", error=str(e))

    # Deduplicate by ID
    seen_ids = set()
    unique = []
    for proc in all_procs:
        pid = proc.get("id", "")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique.append(proc)

    logger.info("procedures_loaded_total", count=len(unique))
    return unique


def convert_procedures(
    domain_filter: str | None = None,
    limit: int | None = None,
) -> list[SFTPair]:
    """
    Convert all procedures into SFT pairs with multiple question variants.

    Returns list of SFTPair objects.
    """
    procedures = load_all_procedures()

    if domain_filter:
        procedures = [p for p in procedures if p.get("domain") == domain_filter]

    if limit:
        procedures = procedures[:limit]

    all_pairs: list[SFTPair] = []
    proc_count = 0

    for proc in procedures:
        proc_id = proc.get("id", f"proc_{proc_count}")
        domain = proc.get("domain", "general")

        # Build the response (same for all question variants)
        response = build_procedure_response(proc)

        # Generate question variants
        variants = generate_question_variants(proc)

        # Create SFTPair for each variant
        for i, v in enumerate(variants):
            pair_hash = hashlib.md5(
                f"nalsa:{proc_id}:{v['question']}".encode()
            ).hexdigest()[:10]

            pair = SFTPair(
                pair_id=f"nl_{pair_hash}",
                source="nalsa",
                instruction=v["question"],
                response=response,
                system_prompt=SFT_SYSTEM_PROMPT,
                domain=domain,
                query_type=v["query_type"],
                jurisdiction=proc.get("jurisdiction", "central"),
                status="draft",
                source_procedure_id=proc_id,
                cited_sections=[
                    f"{law['act']}/{law['section']}"
                    for law in proc.get("relevant_law", [])
                    if law.get("act") and law.get("section")
                ],
            )
            all_pairs.append(pair)

        # Also generate section-based pairs
        section_pairs = generate_section_pairs(proc)
        all_pairs.extend(section_pairs)

        proc_count += 1

    logger.info(
        "nalsa_conversion_complete",
        procedures=proc_count,
        total_pairs=len(all_pairs),
    )

    return all_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════


def export_pairs(pairs: list[SFTPair], output_path: Path | None = None) -> Path:
    """Export pairs as JSONL."""
    path = output_path or SFT_PATHS.nalsa_raw
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    logger.info("nalsa_pairs_exported", path=str(path), count=len(pairs))
    return path


def print_summary(pairs: list[SFTPair]) -> None:
    """Print conversion summary."""
    from collections import Counter

    print(f"\n{'═' * 60}")
    print(f"  NALSA/Procedure → SFT Conversion — Summary")
    print(f"{'═' * 60}\n")
    print(f"  Total pairs: {len(pairs)}")

    # By domain
    domains = Counter(p.domain for p in pairs)
    print(f"\n  By domain:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        pct = count / len(pairs) * 100 if pairs else 0
        print(f"    {domain:<20} {count:>6} ({pct:.1f}%)")

    # By source
    sources = Counter(p.source for p in pairs)
    print(f"\n  By source:")
    for source, count in sorted(sources.items()):
        print(f"    {source:<20} {count:>6}")

    # By query type
    qtypes = Counter(p.query_type for p in pairs)
    print(f"\n  By query type:")
    for qt, count in sorted(qtypes.items(), key=lambda x: -x[1]):
        print(f"    {qt:<20} {count:>6}")

    # Unique procedures
    unique_procs = len(set(
        p.source_procedure_id for p in pairs if p.source_procedure_id
    ))
    avg_per_proc = len(pairs) / unique_procs if unique_procs else 0
    print(f"\n  Unique source procedures: {unique_procs}")
    print(f"  Avg pairs per procedure: {avg_per_proc:.1f}")

    # Hinglish pairs
    hinglish = [p for p in pairs if any(
        w in p.instruction.lower() for w in ["kaise", "kare", "kya", "hai"]
    )]
    print(f"  Hindi-English (Hinglish) pairs: {len(hinglish)}")

    # Sample
    if pairs:
        print(f"\n  Sample pairs:")
        for p in pairs[:3]:
            print(f"    [{p.pair_id}] ({p.domain}) [{p.query_type}]")
            print(f"      Q: {p.instruction}")
            print()

    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra NALSA/Procedure → SFT Converter (Sprint 8)",
    )
    parser.add_argument("--domain", type=str, default=None,
                        help="Only convert from this domain")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum procedures to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show summary without saving")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path")

    args = parser.parse_args()

    start = time.time()

    pairs = convert_procedures(
        domain_filter=args.domain,
        limit=args.limit,
    )

    duration = round(time.time() - start, 2)

    if args.dry_run:
        print_summary(pairs)
        print(f"  (Dry run — {len(pairs)} pairs in {duration}s, not saved)")
    else:
        output_path = Path(args.output) if args.output else None
        path = export_pairs(pairs, output_path)
        print_summary(pairs)
        print(f"  Exported to: {path}")
        print(f"  Duration: {duration}s")


if __name__ == "__main__":
    main()