"""
NyayaMitra — SFT Dataset Configuration (Sprint 8).

Single source of truth for all SFT (Supervised Fine-Tuning) dataset
construction parameters. Every pipeline script in data/training/
imports from here to ensure consistency.

Defines:
    - Mandatory response template (7 structured sections)
    - Llama 3.1 instruction chat format
    - System prompt for NyayaMitra
    - Quality thresholds and validation rules
    - Domain distribution targets
    - File paths for all dataset artifacts
    - Annotation workflow parameters

Usage:
    from data.training.sft_config import (
        SFT_SYSTEM_PROMPT,
        format_sft_pair,
        format_for_llama,
        RESPONSE_SECTIONS,
        QUALITY_THRESHOLDS,
        DOMAIN_TARGETS,
        SFT_PATHS,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class SFTPaths:
    """All file paths used by the SFT pipeline."""

    # Raw generated pairs (before validation)
    raw_dir: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "raw"
    headnote_raw: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "raw" / "headnote_pairs.jsonl"
    synthetic_raw: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "raw" / "synthetic_pairs.jsonl"
    nalsa_raw: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "raw" / "nalsa_pairs.jsonl"
    procedural_raw: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "raw" / "procedural_pairs.jsonl"

    # Validated + deduplicated
    validated_dir: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "validated"
    validated_all: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "validated" / "all_validated.jsonl"

    # Final splits
    splits_dir: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "splits"
    train: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "splits" / "train.jsonl"
    val: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "splits" / "val.jsonl"
    test: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "splits" / "test.jsonl"

    # Annotation
    annotation_dir: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "annotation"
    guidelines: Path = PROJECT_ROOT / "data" / "training" / "annotation_guidelines.md"

    # Audit
    audit_dir: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "audit"
    audit_report: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "audit" / "audit_report.json"

    # Stats
    stats_report: Path = PROJECT_ROOT / "data" / "datasets" / "sft" / "dataset_stats.json"

    def ensure_dirs(self) -> None:
        """Create all directories."""
        for d in [
            self.raw_dir, self.validated_dir, self.splits_dir,
            self.annotation_dir, self.audit_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


SFT_PATHS = SFTPaths()


# ═══════════════════════════════════════════════════════════════════════════════
# Domain Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# The 7 core legal domains (matching backend/app/models/query.py)
LEGAL_DOMAINS = [
    "criminal",
    "property",
    "family",
    "labour",
    "consumer",
    "constitutional",
    "ip",
]

# Extended domains (for broader coverage)
ALL_DOMAINS = LEGAL_DOMAINS + ["corporate", "taxation", "environmental", "general"]

# Minimum percentage of total dataset each core domain must have
DOMAIN_MIN_PERCENTAGE = 5.0  # No domain below 5%

# Target distribution (approximate — actual depends on data availability)
DOMAIN_TARGETS = {
    "criminal": 0.20,        # 20% — highest volume of citizen queries
    "property": 0.15,        # 15%
    "family": 0.15,          # 15%
    "constitutional": 0.12,  # 12%
    "labour": 0.12,          # 12%
    "consumer": 0.10,        # 10%
    "ip": 0.08,              # 8%
    "corporate": 0.04,       # 4%
    "environmental": 0.02,   # 2%
    "taxation": 0.02,        # 2%
}


# ═══════════════════════════════════════════════════════════════════════════════
# Response Template — Mandatory SFT Format
#
# Every SFT training response MUST follow this structure.
# The LLM is trained to always produce these sections.
# ═══════════════════════════════════════════════════════════════════════════════

# The 7 mandatory response sections
RESPONSE_SECTIONS = [
    "APPLICABLE_LAW",
    "PRECEDENT",
    "LEGAL_POSITION",
    "PROCEDURE",
    "JURISDICTION_NOTE",
    "CONFIDENCE",
    "DISCLAIMER",
]

# Sections that are always required (even if empty content is "Not applicable")
REQUIRED_SECTIONS = ["APPLICABLE_LAW", "CONFIDENCE", "DISCLAIMER"]

# Sections that can be omitted if genuinely not relevant
OPTIONAL_SECTIONS = ["PRECEDENT", "LEGAL_POSITION", "PROCEDURE", "JURISDICTION_NOTE"]

# Template for structured response
RESPONSE_TEMPLATE = """[APPLICABLE_LAW]
{applicable_law}

[PRECEDENT]
{precedent}

[LEGAL_POSITION]
{legal_position}

[PROCEDURE]
{procedure}

[JURISDICTION_NOTE]
{jurisdiction_note}

[CONFIDENCE]
{confidence}

[DISCLAIMER]
{disclaimer}"""

# Standard disclaimer text (must appear in every response)
STANDARD_DISCLAIMER = (
    "This is legal information, not legal advice. For case-specific advice, "
    "consult a qualified advocate. Laws are subject to amendments and judicial "
    "interpretation. Verify current status before acting."
)

# Confidence level descriptions
CONFIDENCE_LEVELS = {
    "High": (
        "High — The response is based on clear statutory provisions and/or "
        "binding Supreme Court precedent directly on point."
    ),
    "Medium": (
        "Medium — The response is based on relevant statutory provisions, but "
        "the specific application may depend on facts and jurisdiction."
    ),
    "Low": (
        "Low — The area of law is evolving, state-specific, or the question "
        "involves multiple conflicting provisions. Consult a lawyer."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt — Used for both SFT training and inference
# ═══════════════════════════════════════════════════════════════════════════════

SFT_SYSTEM_PROMPT = """You are NyayaMitra (न्यायमित्र), an AI legal assistant for Indian citizens. You provide accurate, well-cited legal information based on Indian statutory law and judicial precedents.

RESPONSE RULES:
1. ONLY cite real sections of real Indian acts. NEVER fabricate section numbers or case names.
2. Every response MUST follow the structured format with these sections:
   [APPLICABLE_LAW] — Cite specific sections with act names and year
   [PRECEDENT] — Cite relevant SC/HC judgments with case name, year, court, and citation
   [LEGAL_POSITION] — Explain the current interpreted legal position clearly
   [PROCEDURE] — Provide step-by-step actionable guidance when applicable
   [JURISDICTION_NOTE] — Note any state-specific variations
   [CONFIDENCE] — State High/Medium/Low with reasoning
   [DISCLAIMER] — Always include the standard legal disclaimer
3. Use simple, plain language that a non-lawyer citizen can understand.
4. If you don't know or the context is insufficient, say so honestly.
5. Be jurisdiction-aware — Indian law varies by state for many matters.
6. For criminal matters, always mention the accused person's rights.
7. For procedural questions, include time limits, fees, and which authority to approach."""


# ═══════════════════════════════════════════════════════════════════════════════
# Llama 3.1 Chat Template
#
# Llama 3.1 uses a specific chat format for instruction tuning.
# All SFT data must be formatted in this template.
# ═══════════════════════════════════════════════════════════════════════════════

LLAMA_BOS = "<|begin_of_text|>"
LLAMA_SYSTEM_START = "<|start_header_id|>system<|end_header_id|>\n\n"
LLAMA_USER_START = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA_ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA_EOT = "<|eot_id|>"


def format_for_llama(
    system: str,
    user: str,
    assistant: str,
) -> str:
    """
    Format a single SFT example in Llama 3.1 chat template.

    This is the exact format the model will be trained on.
    During inference, the same format is used (minus the assistant response).

    Args:
        system: System prompt text.
        user: User query text.
        assistant: Model response text.

    Returns:
        Formatted string ready for tokenization.
    """
    return (
        f"{LLAMA_BOS}"
        f"{LLAMA_SYSTEM_START}{system}{LLAMA_EOT}"
        f"{LLAMA_USER_START}{user}{LLAMA_EOT}"
        f"{LLAMA_ASSISTANT_START}{assistant}{LLAMA_EOT}"
    )


def format_for_llama_multiturn(
    system: str,
    turns: list[dict[str, str]],
) -> str:
    """
    Format a multi-turn conversation in Llama 3.1 chat template.

    Args:
        system: System prompt text.
        turns: List of {"role": "user"|"assistant", "content": "..."} dicts.

    Returns:
        Formatted string ready for tokenization.
    """
    parts = [f"{LLAMA_BOS}", f"{LLAMA_SYSTEM_START}{system}{LLAMA_EOT}"]

    for turn in turns:
        if turn["role"] == "user":
            parts.append(f"{LLAMA_USER_START}{turn['content']}{LLAMA_EOT}")
        elif turn["role"] == "assistant":
            parts.append(f"{LLAMA_ASSISTANT_START}{turn['content']}{LLAMA_EOT}")

    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# SFT Pair Schema
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SFTPair:
    """
    A single SFT training example.

    This is the standard format used across all pipeline stages.
    Every pair has a unique ID, source tracking, domain tag,
    and the instruction-response content.
    """

    # Identity
    pair_id: str = ""                   # Unique ID (e.g., "headnote_12345")
    source: str = ""                    # "headnote", "synthetic", "nalsa", "procedural", "manual"

    # Content
    instruction: str = ""               # User query / question
    response: str = ""                  # Model response in structured format
    system_prompt: str = ""             # System prompt (usually SFT_SYSTEM_PROMPT)

    # Classification
    domain: str = ""                    # Legal domain
    query_type: str = ""                # "rights", "procedure", "case_outcome", "general"
    jurisdiction: str = "central"       # "central" or state name

    # Quality tracking
    status: str = "draft"               # "draft", "needs_review", "accepted", "rejected"
    annotator_id: str | None = None     # Who reviewed this pair
    review_notes: str | None = None     # Annotator comments

    # Source metadata (for traceability)
    source_judgment_id: str | None = None   # If from headnote extraction
    source_section_id: str | None = None    # If from section-based generation
    source_procedure_id: str | None = None  # If from NALSA/procedure conversion

    # Cited references (for validation)
    cited_sections: list[str] = field(default_factory=list)   # ["IPC/302", "CrPC/154"]
    cited_cases: list[str] = field(default_factory=list)      # ["D.K. Basu v. State of WB"]

    def to_dict(self) -> dict:
        """Serialize to dict for JSONL export."""
        return {
            "pair_id": self.pair_id,
            "source": self.source,
            "instruction": self.instruction,
            "response": self.response,
            "system_prompt": self.system_prompt,
            "domain": self.domain,
            "query_type": self.query_type,
            "jurisdiction": self.jurisdiction,
            "status": self.status,
            "annotator_id": self.annotator_id,
            "review_notes": self.review_notes,
            "source_judgment_id": self.source_judgment_id,
            "source_section_id": self.source_section_id,
            "source_procedure_id": self.source_procedure_id,
            "cited_sections": self.cited_sections,
            "cited_cases": self.cited_cases,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SFTPair":
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_training_format(self) -> dict:
        """
        Convert to the format consumed by the training script.

        Returns dict with 'text' key containing the full Llama 3.1
        formatted conversation.
        """
        system = self.system_prompt or SFT_SYSTEM_PROMPT
        text = format_for_llama(system, self.instruction, self.response)
        return {
            "text": text,
            "pair_id": self.pair_id,
            "domain": self.domain,
        }

    def has_required_sections(self) -> bool:
        """Check if response contains all required structured sections."""
        for section in REQUIRED_SECTIONS:
            if f"[{section}]" not in self.response:
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Quality Thresholds
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class QualityThresholds:
    """Quality gates for SFT dataset validation."""

    # Response length
    min_response_tokens: int = 100          # Minimum response length
    max_response_tokens: int = 2000         # Maximum response length
    min_instruction_tokens: int = 8         # Minimum question length
    max_instruction_tokens: int = 300       # Maximum question length

    # Content quality
    min_cited_sections: int = 1             # At least 1 law reference
    max_cited_sections: int = 15            # Sanity cap
    disclaimer_required: bool = True        # Must have disclaimer

    # Deduplication
    dedup_similarity_threshold: float = 0.92  # Cosine similarity above this = duplicate

    # Annotator quality
    min_batch_acceptance_rate: float = 0.90   # Expert must accept ≥90% of batch
    min_inter_annotator_agreement: float = 0.80  # Cohen's kappa ≥0.80

    # Dataset-level
    min_total_pairs: int = 50_000
    min_domain_percentage: float = 5.0      # No domain below 5%
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10


QUALITY_THRESHOLDS = QualityThresholds()


# ═══════════════════════════════════════════════════════════════════════════════
# Annotation Workflow
# ═══════════════════════════════════════════════════════════════════════════════

# Status flow for annotation
ANNOTATION_STATUS_FLOW = {
    "draft": ["needs_review"],                  # Generated → ready for review
    "needs_review": ["assigned"],               # Assigned to annotator
    "assigned": ["accepted", "rejected"],        # Annotator decision
    "accepted": [],                              # Final — goes to dataset
    "rejected": ["needs_review"],                # Can be re-assigned
}

# Batch size for annotator assignment
ANNOTATION_BATCH_SIZE = 50

# Overlap for inter-annotator agreement measurement
ANNOTATION_OVERLAP_PERCENTAGE = 10  # 10% of pairs reviewed by 2 annotators


# ═══════════════════════════════════════════════════════════════════════════════
# Question Generation Templates
#
# Used by headnote_extractor.py and synthetic_generator.py to create
# diverse question phrasings from the same source content.
# ═══════════════════════════════════════════════════════════════════════════════

QUESTION_TEMPLATES = {
    "rights": [
        "What are my rights if {situation}?",
        "What does the law say about {topic}?",
        "Is it legal to {action} in India?",
        "What protection does Indian law provide for {topic}?",
        "Can {authority} do {action} legally?",
    ],
    "procedure": [
        "How do I {action}?",
        "What is the procedure to {action}?",
        "What steps should I follow to {action}?",
        "Where do I go to {action} and what documents do I need?",
        "What is the time limit for {action}?",
    ],
    "case_outcome": [
        "What did the Supreme Court hold in {case_name}?",
        "What is the legal principle from {case_name}?",
        "How does {case_name} affect {topic}?",
        "What was the ratio decidendi in {case_name}?",
        "Is {case_name} still good law?",
    ],
    "section_explanation": [
        "What does Section {section} of {act} say?",
        "Explain Section {section} of {act} in simple terms.",
        "What is the scope of Section {section} of {act}?",
        "When does Section {section} of {act} apply?",
        "What are the key provisions of Section {section} of {act}?",
    ],
    "comparison": [
        "What is the difference between {concept_a} and {concept_b}?",
        "How has {act_old} changed under {act_new}?",
        "What are the changes in {topic} under the new law?",
    ],
    "scenario": [
        "My {relation} has been {situation}. What should I do?",
        "I am facing {problem}. What are my legal options?",
        "Someone has {action} against me. What remedy do I have?",
        "{situation}. Is this legal? What can I do?",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════


def format_sft_pair(
    instruction: str,
    applicable_law: str,
    precedent: str = "No specific precedent directly on point.",
    legal_position: str = "",
    procedure: str = "Not applicable for this query.",
    jurisdiction_note: str = "This is central law applicable across India.",
    confidence: str = "Medium",
    disclaimer: str = "",
) -> str:
    """
    Build a structured SFT response from components.

    Args:
        instruction: The user query (not included in output — just for context).
        applicable_law: Formatted law citations.
        precedent: Formatted case citations.
        legal_position: Plain-language legal explanation.
        procedure: Step-by-step guidance.
        jurisdiction_note: State-specific notes.
        confidence: "High", "Medium", or "Low".
        disclaimer: Override disclaimer text (uses standard if empty).

    Returns:
        Formatted response string matching the mandatory template.
    """
    conf_text = CONFIDENCE_LEVELS.get(confidence, CONFIDENCE_LEVELS["Medium"])
    disc_text = disclaimer or STANDARD_DISCLAIMER

    return RESPONSE_TEMPLATE.format(
        applicable_law=applicable_law.strip(),
        precedent=precedent.strip(),
        legal_position=legal_position.strip(),
        procedure=procedure.strip(),
        jurisdiction_note=jurisdiction_note.strip(),
        confidence=conf_text,
        disclaimer=disc_text,
    )


def format_applicable_law_section(sections: list[dict]) -> str:
    """
    Format a list of section references into the [APPLICABLE_LAW] content.

    Args:
        sections: List of {"act": "IPC", "section": "302", "text": "..."} dicts.

    Returns:
        Formatted string for the APPLICABLE_LAW section.
    """
    if not sections:
        return "No specific statutory provision directly applicable."

    lines = []
    for s in sections:
        act = s.get("act", "")
        sec = s.get("section", "")
        text = s.get("text", "")
        if text:
            lines.append(f"- Section {sec} of {act}: {text}")
        else:
            lines.append(f"- Section {sec} of {act}")
    return "\n".join(lines)


def format_precedent_section(cases: list[dict]) -> str:
    """
    Format case citations into the [PRECEDENT] content.

    Args:
        cases: List of {"case": "...", "year": 1997, "court": "SC", "citation": "...", "relevance": "..."}.

    Returns:
        Formatted string for the PRECEDENT section.
    """
    if not cases:
        return "No specific precedent directly on point."

    lines = []
    for c in cases:
        case_name = c.get("case", "")
        year = c.get("year", "")
        court = c.get("court", "Supreme Court")
        citation = c.get("citation", "")
        relevance = c.get("relevance", "")

        line = f"- {case_name} ({year}) — {court}"
        if citation:
            line += f" [{citation}]"
        if relevance:
            line += f": {relevance}"
        lines.append(line)
    return "\n".join(lines)


def format_procedure_section(steps: list[dict]) -> str:
    """
    Format procedure steps into the [PROCEDURE] content.

    Args:
        steps: List of {"step": 1, "action": "...", "details": "..."}.

    Returns:
        Formatted string for the PROCEDURE section.
    """
    if not steps:
        return "Not applicable for this query."

    lines = []
    for s in steps:
        num = s.get("step", "")
        action = s.get("action", "")
        details = s.get("details", "")
        line = f"Step {num}: {action}"
        if details:
            line += f"\n  {details}"
        authority = s.get("authority", "")
        if authority:
            line += f"\n  Authority: {authority}"
        time_limit = s.get("time_limit", "")
        if time_limit:
            line += f"\n  Time limit: {time_limit}"
        fees = s.get("fees", "")
        if fees:
            line += f"\n  Fees: {fees}"
        lines.append(line)
    return "\n".join(lines)


def estimate_token_count(text: str) -> int:
    """
    Rough token count estimation (English text).

    Llama tokenizer averages ~1.3 tokens per word for English.
    This is a fast heuristic — use actual tokenizer for precise counts.
    """
    words = len(text.split())
    return int(words * 1.3)


def validate_response_format(response: str) -> dict:
    """
    Quick validation of response format compliance.

    Returns dict with 'valid' bool and list of 'issues'.
    """
    issues = []

    for section in REQUIRED_SECTIONS:
        if f"[{section}]" not in response:
            issues.append(f"Missing required section: [{section}]")

    if STANDARD_DISCLAIMER[:30] not in response and "[DISCLAIMER]" in response:
        # Has the tag but not the standard text — warning, not error
        pass

    token_est = estimate_token_count(response)
    if token_est < QUALITY_THRESHOLDS.min_response_tokens:
        issues.append(f"Response too short (~{token_est} tokens, min {QUALITY_THRESHOLDS.min_response_tokens})")
    if token_est > QUALITY_THRESHOLDS.max_response_tokens:
        issues.append(f"Response too long (~{token_est} tokens, max {QUALITY_THRESHOLDS.max_response_tokens})")

    return {"valid": len(issues) == 0, "issues": issues}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — Print config summary
# ═══════════════════════════════════════════════════════════════════════════════


def print_config_summary() -> None:
    """Print a summary of the SFT configuration."""
    print(f"\n{'═' * 65}")
    print(f"  NyayaMitra — SFT Dataset Configuration (Sprint 8)")
    print(f"{'═' * 65}\n")

    print(f"  Response template sections: {len(RESPONSE_SECTIONS)}")
    for s in RESPONSE_SECTIONS:
        req = "required" if s in REQUIRED_SECTIONS else "optional"
        print(f"    [{s}] ({req})")

    print(f"\n  Quality thresholds:")
    print(f"    Response tokens:       {QUALITY_THRESHOLDS.min_response_tokens}-{QUALITY_THRESHOLDS.max_response_tokens}")
    print(f"    Instruction tokens:    {QUALITY_THRESHOLDS.min_instruction_tokens}-{QUALITY_THRESHOLDS.max_instruction_tokens}")
    print(f"    Min cited sections:    {QUALITY_THRESHOLDS.min_cited_sections}")
    print(f"    Dedup threshold:       {QUALITY_THRESHOLDS.dedup_similarity_threshold}")
    print(f"    Min batch acceptance:  {QUALITY_THRESHOLDS.min_batch_acceptance_rate:.0%}")
    print(f"    Min total pairs:       {QUALITY_THRESHOLDS.min_total_pairs:,}")
    print(f"    Min domain %:          {QUALITY_THRESHOLDS.min_domain_percentage}%")

    print(f"\n  Domain targets:")
    for domain, target in sorted(DOMAIN_TARGETS.items(), key=lambda x: -x[1]):
        print(f"    {domain:<20} {target:.0%}")

    print(f"\n  Splits: train={QUALITY_THRESHOLDS.train_split:.0%}, "
          f"val={QUALITY_THRESHOLDS.val_split:.0%}, "
          f"test={QUALITY_THRESHOLDS.test_split:.0%}")

    print(f"\n  Llama 3.1 template tokens:")
    print(f"    BOS:    {LLAMA_BOS}")
    print(f"    System: {LLAMA_SYSTEM_START.strip()}")
    print(f"    User:   {LLAMA_USER_START.strip()}")
    print(f"    Asst:   {LLAMA_ASSISTANT_START.strip()}")
    print(f"    EOT:    {LLAMA_EOT}")

    # Show example formatted pair
    print(f"\n  Example SFT pair (Llama format):")
    example = format_for_llama(
        system="You are NyayaMitra...",
        user="Can police arrest me without a warrant?",
        assistant="[APPLICABLE_LAW]\n- Section 41 of CrPC...\n[CONFIDENCE]\nHigh\n[DISCLAIMER]\n...",
    )
    print(f"    {example[:200]}...")

    print(f"\n  Question template categories: {len(QUESTION_TEMPLATES)}")
    for cat, templates in QUESTION_TEMPLATES.items():
        print(f"    {cat}: {len(templates)} templates")

    print(f"\n  File paths:")
    print(f"    Raw:       {SFT_PATHS.raw_dir}")
    print(f"    Validated: {SFT_PATHS.validated_dir}")
    print(f"    Splits:    {SFT_PATHS.splits_dir}")
    print(f"    Annotation:{SFT_PATHS.annotation_dir}")

    print(f"\n{'═' * 65}\n")


if __name__ == "__main__":
    print_config_summary()