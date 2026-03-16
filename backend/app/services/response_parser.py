"""
NyayaMitra — LLM Response Parser.

Parses the LLM's markdown/structured output back into the Pydantic
QueryResponse fields. The LLM is prompted to produce a response with
markdown headers (## Answer, ## Applicable Law, etc.), and this module
extracts the structured data from that free text.

Extraction strategies:
    1. Section-header splitting — split response by ## headers
    2. Regex citation extraction — find "Section X of Act Y" patterns
    3. Case citation extraction — find "Case Name (Year) — Court" patterns
    4. Procedure extraction — find numbered steps

The parser is best-effort: if the LLM deviates from the expected format,
the raw answer text is preserved and structured fields may be empty.
The retrieval-based structured data (from query.py _generate_response)
serves as the primary source of ApplicableLaw and Precedent objects;
this parser supplements by extracting any additional citations the LLM
mentioned in its answer text.

Usage:
    from app.services.response_parser import parse_llm_response

    parsed = parse_llm_response(llm_text)
    # parsed["answer"], parsed["applicable_law"], parsed["precedents"],
    # parsed["procedure"], parsed["jurisdiction_notes"]
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Section Header Splitter
# ═══════════════════════════════════════════════════════════════════════════════

# Matches ## Header or **Header:** style sections
HEADER_PATTERN = re.compile(
    r"^(?:#{1,3}\s+|(?:\*\*))?"
    r"(Answer|Applicable Law|Key Precedents?|What You Should Do|"
    r"Procedure|Important Notes?|Jurisdiction)"
    r"(?:\*\*)?:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_sections(text: str) -> dict[str, str]:
    """
    Split LLM response text by markdown section headers.

    Returns a dict mapping normalized section names to their content.
    Unrecognized content before the first header goes into "preamble".
    """
    sections: dict[str, str] = {}
    current_key = "preamble"
    current_lines: list[str] = []

    for line in text.split("\n"):
        match = HEADER_PATTERN.match(line.strip())
        if match:
            # Save previous section
            if current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
            # Start new section
            header = match.group(1).lower().strip()
            current_key = _normalize_header(header)
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def _normalize_header(header: str) -> str:
    """Normalize section header names to canonical keys."""
    header = header.lower().strip()
    if "answer" in header:
        return "answer"
    if "applicable" in header or "law" in header:
        return "applicable_law"
    if "precedent" in header:
        return "precedents"
    if "should do" in header or "procedure" in header:
        return "procedure"
    if "note" in header or "jurisdiction" in header:
        return "notes"
    return header


# ═══════════════════════════════════════════════════════════════════════════════
# Citation Extractors
# ═══════════════════════════════════════════════════════════════════════════════

# Matches: "Section 41 of the Code of Criminal Procedure, 1973"
#          "Section 302 of IPC"
#          "Section 498A of the Indian Penal Code, 1860"
#          "Article 21 of the Constitution of India"
SECTION_PATTERN = re.compile(
    r"(?:Section|Article|Rule)\s+"
    r"(\d+[A-Za-z]?(?:\(\d+\))?(?:\([a-z]\))?)"
    r"\s+of\s+(?:the\s+)?"
    r"([A-Z][A-Za-z\s,()]+?\d{4}|[A-Z][A-Za-z\s,()]+?Act|"
    r"IPC|CrPC|CPC|Constitution of India|Constitution|"
    r"TPA|HMA|SMA|CPA|RERA|DV Act|POSH|IT Act|RTI|"
    r"Copyright Act|ID Act|Contract Act)",
    re.IGNORECASE,
)

# Matches: "D.K. Basu v. State of West Bengal (1997)"
#          "Arnesh Kumar v. State of Bihar (2014) — Supreme Court"
#          "Lalita Kumari v. Government of Uttar Pradesh (2013)"
CASE_PATTERN = re.compile(
    r"([A-Z][A-Za-z\.\s]+?)\s+v\.?\s+"
    r"([A-Z][A-Za-z\.\s]+?)"
    r"\s*\((\d{4})\)"
    r"(?:\s*[—–-]\s*(Supreme Court|High Court|[A-Za-z\s]+Court))?"
)

# Matches SCC/AIR citations: "(2014) 8 SCC 273" or "AIR 2014 SC 2756"
SCC_PATTERN = re.compile(r"\((\d{4})\)\s+\d+\s+SCC\s+\d+")
AIR_PATTERN = re.compile(r"AIR\s+\d{4}\s+SC\s+\d+")


def extract_section_citations(text: str) -> list[dict]:
    """
    Extract statutory section citations from text.

    Returns list of dicts with: section, act, text (surrounding context).
    """
    citations = []
    seen = set()

    for match in SECTION_PATTERN.finditer(text):
        section_num = match.group(1).strip()
        act_name = match.group(2).strip()

        # Normalize trailing commas/spaces
        act_name = act_name.rstrip(" ,")

        key = f"{act_name}/{section_num}".lower()
        if key in seen:
            continue
        seen.add(key)

        # Extract surrounding context (the sentence containing this citation)
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 150)
        context = text[start:end].strip()
        # Trim to sentence boundaries
        if "." in context[50:]:
            context = context[:50 + context[50:].index(".") + 1]

        citations.append({
            "section": section_num,
            "act": act_name,
            "text": context,
        })

    return citations


def extract_case_citations(text: str) -> list[dict]:
    """
    Extract case citations from text.

    Returns list of dicts with: case, year, court, citation, relevance.
    """
    citations = []
    seen = set()

    for match in CASE_PATTERN.finditer(text):
        petitioner = match.group(1).strip()
        respondent = match.group(2).strip()
        year = int(match.group(3))
        court = match.group(4) or "Supreme Court"
        court = court.strip()

        case_name = f"{petitioner} v. {respondent}"

        # Deduplicate
        key = case_name.lower()
        if key in seen:
            continue
        seen.add(key)

        # Extract surrounding context for relevance
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 200)
        context = text[start:end].strip()
        if "." in context[50:]:
            context = context[:50 + context[50:].index(".") + 1]

        # Try to find a formal citation nearby
        nearby_text = text[max(0, match.start() - 30):min(len(text), match.end() + 80)]
        scc_match = SCC_PATTERN.search(nearby_text)
        air_match = AIR_PATTERN.search(nearby_text)
        citation = ""
        if scc_match:
            citation = scc_match.group(0)
        elif air_match:
            citation = air_match.group(0)

        citations.append({
            "case": case_name,
            "year": year,
            "court": court,
            "citation": citation,
            "relevance": context,
        })

    return citations


def extract_procedure_steps(text: str) -> list[dict]:
    """
    Extract numbered procedure steps from text.

    Looks for patterns like:
        1. File an FIR at the nearest police station.
        2. Obtain a copy of the FIR.
    """
    steps = []
    step_pattern = re.compile(
        r"^\s*(\d+)\.\s+\*{0,2}(.+?)(?:\*{0,2})\s*$",
        re.MULTILINE,
    )

    for match in step_pattern.finditer(text):
        step_num = int(match.group(1))
        action_text = match.group(2).strip()

        # Remove markdown bold markers
        action_text = action_text.replace("**", "").strip()

        # Split into action (first sentence) and details (rest)
        sentences = action_text.split(". ", 1)
        action = sentences[0].strip().rstrip(".")
        details = sentences[1].strip() if len(sentences) > 1 else ""

        steps.append({
            "step": step_num,
            "action": action,
            "details": details,
        })

    return steps


# ═══════════════════════════════════════════════════════════════════════════════
# Main Parser
# ═══════════════════════════════════════════════════════════════════════════════


def parse_llm_response(text: str) -> dict:
    """
    Parse an LLM response into structured fields.

    Attempts to split by section headers first. If no headers are found,
    treats the entire text as the answer and extracts citations from it.

    Args:
        text: The raw LLM response text.

    Returns:
        dict with keys:
            answer: str — the main explanation
            applicable_law: list[dict] — extracted section citations
            precedents: list[dict] — extracted case citations
            procedure: list[dict] — extracted procedure steps
            jurisdiction_notes: str — any jurisdiction-specific notes
    """
    if not text:
        return {
            "answer": "",
            "applicable_law": [],
            "precedents": [],
            "procedure": [],
            "jurisdiction_notes": "",
        }

    # Try to split by section headers
    sections = split_sections(text)

    # Build the answer from header-split sections or fall back to full text
    if "answer" in sections:
        answer = sections["answer"]
    elif "preamble" in sections and len(sections) > 1:
        answer = sections["preamble"]
    else:
        # No headers found — use the full text as the answer
        answer = text

    # Extract citations from the applicable_law section if it exists,
    # otherwise extract from the full text
    law_text = sections.get("applicable_law", text)
    applicable_law = extract_section_citations(law_text)

    # Extract case citations from precedents section or full text
    prec_text = sections.get("precedents", text)
    precedents = extract_case_citations(prec_text)

    # Extract procedure steps
    proc_text = sections.get("procedure", "")
    procedure = extract_procedure_steps(proc_text)

    # Jurisdiction notes
    jurisdiction_notes = sections.get("notes", "")

    logger.info(
        "response_parsed",
        sections_found=len(sections),
        laws_extracted=len(applicable_law),
        cases_extracted=len(precedents),
        steps_extracted=len(procedure),
    )

    return {
        "answer": answer,
        "applicable_law": applicable_law,
        "precedents": precedents,
        "procedure": procedure,
        "jurisdiction_notes": jurisdiction_notes,
    }


def merge_parsed_with_retrieval(
    parsed: dict,
    retrieval_laws: list,
    retrieval_precedents: list,
) -> dict:
    """
    Merge LLM-parsed citations with retrieval-based citations.

    Retrieval-based citations (from _generate_response in query.py) are
    the primary source — they come directly from the database with verified
    metadata. LLM-parsed citations supplement by capturing any additional
    citations the LLM mentioned that weren't in the retrieval results.

    Args:
        parsed: Output from parse_llm_response().
        retrieval_laws: ApplicableLaw objects from retrieval.
        retrieval_precedents: Precedent objects from retrieval.

    Returns:
        dict with merged applicable_law and precedents lists.
    """
    # Build sets of existing citations for dedup
    existing_sections = set()
    for law in retrieval_laws:
        key = f"{getattr(law, 'act', '')}/{getattr(law, 'section', '')}".lower()
        existing_sections.add(key)

    existing_cases = set()
    for prec in retrieval_precedents:
        existing_cases.add(getattr(prec, "case", "").lower())

    # Add LLM-extracted citations that aren't already in retrieval results
    new_laws = []
    for law_dict in parsed.get("applicable_law", []):
        key = f"{law_dict['act']}/{law_dict['section']}".lower()
        if key not in existing_sections:
            new_laws.append(law_dict)
            existing_sections.add(key)

    new_precedents = []
    for prec_dict in parsed.get("precedents", []):
        if prec_dict["case"].lower() not in existing_cases:
            new_precedents.append(prec_dict)
            existing_cases.add(prec_dict["case"].lower())

    logger.info(
        "citations_merged",
        retrieval_laws=len(retrieval_laws),
        retrieval_precedents=len(retrieval_precedents),
        llm_new_laws=len(new_laws),
        llm_new_precedents=len(new_precedents),
    )

    return {
        "new_laws": new_laws,
        "new_precedents": new_precedents,
    }