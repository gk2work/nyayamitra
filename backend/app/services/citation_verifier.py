"""
NyayaMitra — Citation Verification Service.

Post-generation checks that ensure every citation in the LLM response
actually exists in the database. This is a CRITICAL safety layer —
the system design mandates that fabricated citations never reach users.

Verification checks (per citation):
    1. Section existence:  Does Section X of Act Y exist in PostgreSQL?
    2. Case existence:     Does this case_name exist in the judgments table?
    3. Citation format:    Does the SCC/AIR citation match standard patterns?
    4. Overruling check:   Is the cited judgment overruled? If so, flag it.

Aggregate checks (per response):
    5. Failure threshold:  If >30% citations fail, trigger LLM regeneration
                           with a stricter grounding prompt.

Usage:
    from app.services.citation_verifier import get_citation_verifier

    verifier = await get_citation_verifier()
    report = await verifier.verify_response(query_response)
    # report.all_verified, report.accuracy, report.section_results, ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from app.config import settings

logger = structlog.get_logger()

# Citation format patterns
SCC_PATTERN = re.compile(r"^\(\d{4}\)\s+\d+\s+SCC\s+\d+$")
AIR_PATTERN = re.compile(r"^AIR\s+\d{4}\s+SC\s+\d+$")

# Abbreviation → canonical act name (for flexible matching)
ACT_ABBREVIATIONS = {
    "IPC": "Indian Penal Code, 1860",
    "Indian Penal Code": "Indian Penal Code, 1860",
    "CrPC": "Code of Criminal Procedure, 1973",
    "Cr.P.C.": "Code of Criminal Procedure, 1973",
    "CPC": "Code of Civil Procedure, 1908",
    "Constitution": "Constitution of India",
    "Constitution of India": "Constitution of India",
    "TPA": "Transfer of Property Act, 1882",
    "HMA": "Hindu Marriage Act, 1955",
    "SMA": "Special Marriage Act, 1954",
    "DV Act": "Protection of Women from Domestic Violence Act, 2005",
    "CPA": "Consumer Protection Act, 2019",
    "RERA": "Real Estate (Regulation and Development) Act, 2016",
    "RTI": "Right to Information Act, 2005",
    "IT Act": "Information Technology Act, 2000",
    "Copyright Act": "Copyright Act, 1957",
    "ID Act": "Industrial Disputes Act, 1947",
    "POSH": "Sexual Harassment of Women at Workplace Act, 2013",
    "Contract Act": "Indian Contract Act, 1872",
    "Evidence Act": "Indian Evidence Act, 1872",
    "Dowry Prohibition Act": "Dowry Prohibition Act, 1961",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Verification Result Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SectionVerification:
    """Result of verifying a single section citation."""

    act: str
    section: str
    exists: bool = False
    status: str = ""  # active, repealed, amended
    matched_act_name: str = ""  # canonical act name found in DB
    error: str = ""


@dataclass
class CaseVerification:
    """Result of verifying a single case citation."""

    case_name: str
    exists: bool = False
    is_overruled: bool = False
    overruled_by: str = ""
    citation_format_valid: bool = True  # SCC/AIR format check
    matched_case_name: str = ""  # exact name found in DB
    error: str = ""


@dataclass
class VerificationReport:
    """Aggregate verification report for an entire response."""

    total_sections: int = 0
    verified_sections: int = 0
    failed_sections: int = 0
    section_results: list[SectionVerification] = field(default_factory=list)

    total_cases: int = 0
    verified_cases: int = 0
    failed_cases: int = 0
    overruled_cases: int = 0
    case_results: list[CaseVerification] = field(default_factory=list)

    total_citations: int = 0
    verified_citations: int = 0
    accuracy: float = 0.0
    all_verified: bool = False
    regeneration_triggered: bool = False

    def compute_summary(self):
        """Compute aggregate metrics from individual results."""
        self.total_sections = len(self.section_results)
        self.verified_sections = sum(1 for s in self.section_results if s.exists)
        self.failed_sections = self.total_sections - self.verified_sections

        self.total_cases = len(self.case_results)
        self.verified_cases = sum(1 for c in self.case_results if c.exists)
        self.failed_cases = self.total_cases - self.verified_cases
        self.overruled_cases = sum(1 for c in self.case_results if c.is_overruled)

        self.total_citations = self.total_sections + self.total_cases
        self.verified_citations = self.verified_sections + self.verified_cases
        self.accuracy = (
            self.verified_citations / self.total_citations
            if self.total_citations > 0
            else 1.0
        )
        self.all_verified = (
            self.failed_sections == 0 and self.failed_cases == 0
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Citation Verifier
# ═══════════════════════════════════════════════════════════════════════════════


class CitationVerifier:
    """
    Verifies citations in LLM responses against the database.

    Uses PostgreSQL for section/case existence checks and
    Neo4j for overruling relationship checks.
    """

    def __init__(self):
        self._act_cache: dict[str, set[str]] = {}  # act_name → {section_numbers}
        self._case_cache: dict[str, dict] = {}  # case_name_lower → {case_name, is_overruled, ...}
        self._cache_loaded = False

    async def load_cache(self) -> None:
        """
        Pre-load all acts/sections/judgments into memory for fast lookup.

        This is efficient because our current dataset is small (~60 sections,
        ~20 judgments). For larger datasets, switch to per-query DB lookups.
        """
        if self._cache_loaded:
            return

        try:
            from sqlalchemy import select
            from app.database import async_session
            from app.models.legal import Act, Section, Judgment

            async with async_session() as session:
                # Load all acts and their sections
                result = await session.execute(
                    select(Act)
                )
                acts = result.scalars().all()

                for act in acts:
                    # Index by both full name and short name
                    sec_result = await session.execute(
                        select(Section.section_number).where(Section.act_id == act.id)
                    )
                    section_numbers = {row[0] for row in sec_result.all()}

                    self._act_cache[act.name.lower()] = section_numbers
                    if act.short_name:
                        self._act_cache[act.short_name.lower()] = section_numbers

                # Load all judgments
                result = await session.execute(select(Judgment))
                judgments = result.scalars().all()

                for j in judgments:
                    self._case_cache[j.case_name.lower()] = {
                        "case_name": j.case_name,
                        "year": j.year,
                        "court": j.court,
                        "is_overruled": j.is_overruled,
                        "overruled_by": j.overruled_by or "",
                        "citation_scc": j.citation_scc or "",
                        "citation_air": j.citation_air or "",
                    }

            self._cache_loaded = True
            logger.info(
                "citation_cache_loaded",
                acts=len(self._act_cache),
                cases=len(self._case_cache),
            )

        except Exception as e:
            logger.warning("citation_cache_failed", error=str(e))

    # ─── Section Verification ────────────────────────────────────────────

    async def verify_section(self, act: str, section: str) -> SectionVerification:
        """
        Verify that a section citation exists in the database.

        Tries multiple matching strategies:
            1. Exact act name match
            2. Abbreviation expansion (IPC → Indian Penal Code, 1860)
            3. Fuzzy act name match (contains search)
        """
        await self.load_cache()

        result = SectionVerification(act=act, section=section)
        act_lower = act.lower().strip()
        section_clean = section.strip()

        # Strategy 1: Direct match
        if act_lower in self._act_cache:
            sections = self._act_cache[act_lower]
            if section_clean in sections:
                result.exists = True
                result.matched_act_name = act
                result.status = "active"
                return result

        # Strategy 2: Abbreviation expansion
        for abbr, full_name in ACT_ABBREVIATIONS.items():
            if abbr.lower() == act_lower or full_name.lower() == act_lower:
                full_lower = full_name.lower()
                if full_lower in self._act_cache:
                    sections = self._act_cache[full_lower]
                    if section_clean in sections:
                        result.exists = True
                        result.matched_act_name = full_name
                        result.status = "active"
                        return result
                # Also check short name
                abbr_lower = abbr.lower()
                if abbr_lower in self._act_cache:
                    sections = self._act_cache[abbr_lower]
                    if section_clean in sections:
                        result.exists = True
                        result.matched_act_name = abbr
                        result.status = "active"
                        return result

        # Strategy 3: Fuzzy match (act name contains search)
        for cached_name, sections in self._act_cache.items():
            if act_lower in cached_name or cached_name in act_lower:
                if section_clean in sections:
                    result.exists = True
                    result.matched_act_name = cached_name
                    result.status = "active"
                    return result

        result.error = f"Section {section} of {act} not found in database"
        return result

    # ─── Case Verification ───────────────────────────────────────────────

    async def verify_case(self, case_name: str, citation: str = "") -> CaseVerification:
        """
        Verify that a case citation exists in the database.

        Tries multiple matching strategies:
            1. Exact case name match (case-insensitive)
            2. Partial match (petitioner name)
            3. "v." normalization (v. / vs. / v / versus)
        """
        await self.load_cache()

        result = CaseVerification(case_name=case_name)
        name_lower = case_name.lower().strip()

        # Normalize "v." variations
        name_normalized = re.sub(r"\s+v\.?\s+|\s+vs\.?\s+|\s+versus\s+", " v. ", name_lower)

        # Strategy 1: Exact match
        if name_lower in self._case_cache:
            match = self._case_cache[name_lower]
            result.exists = True
            result.matched_case_name = match["case_name"]
            result.is_overruled = match["is_overruled"]
            result.overruled_by = match["overruled_by"]
        else:
            # Strategy 2: Normalized match
            for cached_lower, match in self._case_cache.items():
                cached_normalized = re.sub(
                    r"\s+v\.?\s+|\s+vs\.?\s+|\s+versus\s+", " v. ", cached_lower
                )
                if name_normalized == cached_normalized:
                    result.exists = True
                    result.matched_case_name = match["case_name"]
                    result.is_overruled = match["is_overruled"]
                    result.overruled_by = match["overruled_by"]
                    break

            # Strategy 3: Partial match (petitioner name before "v.")
            if not result.exists:
                petitioner = name_normalized.split(" v. ")[0].strip()
                if len(petitioner) > 3:
                    for cached_lower, match in self._case_cache.items():
                        if petitioner in cached_lower:
                            result.exists = True
                            result.matched_case_name = match["case_name"]
                            result.is_overruled = match["is_overruled"]
                            result.overruled_by = match["overruled_by"]
                            break

        # Citation format check
        if citation:
            citation_clean = citation.strip()
            if citation_clean:
                is_scc = bool(SCC_PATTERN.match(citation_clean))
                is_air = bool(AIR_PATTERN.match(citation_clean))
                result.citation_format_valid = is_scc or is_air
            else:
                result.citation_format_valid = True  # empty citation is OK

        if not result.exists:
            result.error = f"Case '{case_name}' not found in database"

        return result

    # ─── Full Response Verification ──────────────────────────────────────

    async def verify_response(
        self,
        applicable_law: list,
        precedents: list,
    ) -> VerificationReport:
        """
        Verify all citations in a QueryResponse.

        Args:
            applicable_law: List of ApplicableLaw objects.
            precedents: List of Precedent objects.

        Returns:
            VerificationReport with per-citation results and aggregate metrics.
        """
        await self.load_cache()

        report = VerificationReport()

        # Verify sections
        for law in applicable_law:
            act_name = getattr(law, "act", "")
            section_num = getattr(law, "section", "")
            if act_name and section_num:
                sv = await self.verify_section(act_name, section_num)
                report.section_results.append(sv)

        # Verify cases
        for prec in precedents:
            case_name = getattr(prec, "case", "")
            citation = getattr(prec, "citation", "")
            if case_name:
                cv = await self.verify_case(case_name, citation)
                report.case_results.append(cv)

        report.compute_summary()

        # Check regeneration threshold
        failure_threshold = getattr(
            settings, "CITATION_FAILURE_THRESHOLD", 0.3
        )
        if report.total_citations > 0 and report.accuracy < (1.0 - failure_threshold):
            report.regeneration_triggered = True

        logger.info(
            "citations_verified",
            total=report.total_citations,
            verified=report.verified_citations,
            accuracy=round(report.accuracy, 3),
            failed_sections=report.failed_sections,
            failed_cases=report.failed_cases,
            overruled=report.overruled_cases,
            regeneration=report.regeneration_triggered,
        )

        return report

    def build_strict_grounding_prompt(
        self,
        query: str,
        verified_sections: list[SectionVerification],
        verified_cases: list[CaseVerification],
    ) -> str:
        """
        Build a stricter LLM prompt that lists only verified citations.

        Used when regeneration is triggered (>30% citations failed).
        The prompt explicitly tells the LLM to ONLY cite the listed
        sections and cases, preventing fabrication.
        """
        parts = [
            f"USER QUESTION: {query}",
            "",
            "IMPORTANT: Only cite the following VERIFIED legal provisions and cases.",
            "Do NOT cite any section or case that is not in this list.",
            "",
            "VERIFIED SECTIONS:",
        ]

        for sv in verified_sections:
            if sv.exists:
                parts.append(f"  - Section {sv.section} of {sv.matched_act_name}")

        parts.append("")
        parts.append("VERIFIED CASES:")

        for cv in verified_cases:
            if cv.exists:
                overruled_tag = " [OVERRULED]" if cv.is_overruled else ""
                parts.append(f"  - {cv.matched_case_name}{overruled_tag}")

        parts.append("")
        parts.append(
            "If the above provisions do not adequately answer the question, "
            "say so honestly rather than citing unverified sources."
        )

        return "\n".join(parts)

    def annotate_overruled_precedents(self, precedents: list, report: VerificationReport) -> list:
        """
        Annotate precedents with overruling warnings.

        For any cited case that has been overruled, append
        "[OVERRULED by X]" to the relevance field so the user
        and LLM both see the warning.
        """
        case_results_map = {
            cv.case_name.lower(): cv for cv in report.case_results
        }

        for prec in precedents:
            case_name = getattr(prec, "case", "")
            cv = case_results_map.get(case_name.lower())
            if cv and cv.is_overruled and cv.overruled_by:
                current_relevance = getattr(prec, "relevance", "")
                warning = f" ⚠️ [OVERRULED by {cv.overruled_by}]"
                if warning not in current_relevance:
                    prec.relevance = current_relevance + warning

        return precedents

    def filter_unverified_citations(
        self,
        applicable_law: list,
        precedents: list,
        report: VerificationReport,
    ) -> tuple[list, list]:
        """
        Remove citations that failed verification.

        Returns filtered (applicable_law, precedents) with only
        verified citations. Unverified citations are logged but
        not shown to the user.
        """
        # Build lookup sets
        verified_sections = {
            (sv.act.lower(), sv.section)
            for sv in report.section_results
            if sv.exists
        }
        verified_cases = {
            cv.case_name.lower()
            for cv in report.case_results
            if cv.exists
        }

        filtered_laws = []
        removed_laws = []
        for law in applicable_law:
            key = (getattr(law, "act", "").lower(), getattr(law, "section", ""))
            if key in verified_sections:
                filtered_laws.append(law)
            else:
                removed_laws.append(f"{law.act} S.{law.section}")

        filtered_precs = []
        removed_precs = []
        for prec in precedents:
            if getattr(prec, "case", "").lower() in verified_cases:
                filtered_precs.append(prec)
            else:
                removed_precs.append(prec.case)

        if removed_laws or removed_precs:
            logger.warning(
                "unverified_citations_removed",
                removed_sections=removed_laws,
                removed_cases=removed_precs,
            )

        return filtered_laws, filtered_precs


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_citation_verifier: CitationVerifier | None = None


async def get_citation_verifier() -> CitationVerifier:
    """Get or create the singleton citation verifier."""
    global _citation_verifier
    if _citation_verifier is None:
        _citation_verifier = CitationVerifier()
        await _citation_verifier.load_cache()
    return _citation_verifier