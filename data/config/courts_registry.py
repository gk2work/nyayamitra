"""
NyayaMitra — Courts Registry.

Master registry of Indian courts for judgment ingestion. Each entry
contains the metadata needed for scraping Indian Kanoon, filtering
by jurisdiction, and classifying judgments.

Court hierarchy:
    Supreme Court of India (SC)
    └── 25 High Courts (HC)
        └── District / Sessions Courts (not ingested in Sprint 7)

Additionally includes key tribunals whose orders are frequently cited.

Ingestion priority is based on caseload and population served:
    P0 — SC + top 10 HCs by state population (covers ~75% of India)
    P1 — Remaining 15 HCs
    P2 — Tribunals

The registry drives:
    1. data/scrapers/indian_kanoon.py — which courts to fetch judgments from
    2. data/embeddings/bulk_indexer.py — court metadata tagging
    3. backend/app/services/query_router.py — jurisdiction → court mapping
    4. data/scripts/coverage_report.py — per-court ingestion stats

Usage:
    from data.config.courts_registry import (
        COURTS_REGISTRY,
        get_courts_for_ingestion,
        get_hc_for_state,
        SUPREME_COURT,
    )

    # All P0 courts
    p0 = get_courts_for_ingestion("P0")

    # Which HC covers Maharashtra?
    hc = get_hc_for_state("Maharashtra")  # → Bombay High Court
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════════


class CourtType(str, Enum):
    """Type of court."""

    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    TRIBUNAL = "tribunal"


class Priority(str, Enum):
    """Ingestion priority tier."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


@dataclass(frozen=True)
class CourtEntry:
    """A single court in the registry."""

    # Identity
    name: str                           # Full official name
    short_name: str                     # Abbreviation (e.g., "BOM HC", "DEL HC")
    court_type: CourtType               # SC, HC, or Tribunal

    # Jurisdiction
    states: list[str] = field(default_factory=list)  # States/UTs under jurisdiction
    principal_seat: str = ""            # City of the principal bench
    benches: list[str] = field(default_factory=list)  # Additional bench locations

    # Scraping — Indian Kanoon
    indian_kanoon_id: str = ""          # Court identifier used in IK API/URLs
    indian_kanoon_doc_type: str = ""    # e.g., "supremecourt", "highcourt"

    # Ingestion
    priority: Priority = Priority.P1
    target_judgments: int = 5000        # Target number of judgments to ingest
    year_range_start: int = 2019        # Fetch judgments from this year onwards
    year_range_end: int = 2025          # Up to this year

    # Metadata
    established: int = 0                # Year established
    notes: str = ""

    @property
    def jurisdiction_key(self) -> str:
        """Primary state key for jurisdiction matching."""
        return self.states[0] if self.states else "India"


# ═══════════════════════════════════════════════════════════════════════════════
# Supreme Court
# ═══════════════════════════════════════════════════════════════════════════════

SUPREME_COURT = CourtEntry(
    name="Supreme Court of India",
    short_name="SC",
    court_type=CourtType.SUPREME_COURT,
    states=["India"],
    principal_seat="New Delhi",
    indian_kanoon_id="supremecourt",
    indian_kanoon_doc_type="supremecourt",
    priority=Priority.P0,
    target_judgments=15000,
    year_range_start=2015,
    year_range_end=2025,
    established=1950,
    notes="All SC judgments are binding across India. Highest priority.",
)


# ═══════════════════════════════════════════════════════════════════════════════
# High Courts — P0 (Top 10 by state population)
# ═══════════════════════════════════════════════════════════════════════════════

HIGH_COURTS_P0: list[CourtEntry] = [
    CourtEntry(
        name="Allahabad High Court",
        short_name="ALL HC",
        court_type=CourtType.HIGH_COURT,
        states=["Uttar Pradesh"],
        principal_seat="Prayagraj",
        benches=["Lucknow"],
        indian_kanoon_id="allahabadhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P0,
        target_judgments=8000,
        year_range_start=2019,
        established=1866,
        notes="Largest HC by caseload. UP is most populous state (~240M).",
    ),
    CourtEntry(
        name="Bombay High Court",
        short_name="BOM HC",
        court_type=CourtType.HIGH_COURT,
        states=["Maharashtra", "Goa"],
        principal_seat="Mumbai",
        benches=["Nagpur", "Aurangabad", "Panaji"],
        indian_kanoon_id="bombayhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P0,
        target_judgments=8000,
        year_range_start=2019,
        established=1862,
        notes="Covers Mumbai — major commercial jurisdiction.",
    ),
    CourtEntry(
        name="Madras High Court",
        short_name="MAD HC",
        court_type=CourtType.HIGH_COURT,
        states=["Tamil Nadu", "Puducherry"],
        principal_seat="Chennai",
        benches=["Madurai"],
        indian_kanoon_id="madrashighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P0,
        target_judgments=6000,
        year_range_start=2019,
        established=1862,
    ),
    CourtEntry(
        name="Calcutta High Court",
        short_name="CAL HC",
        court_type=CourtType.HIGH_COURT,
        states=["West Bengal", "Andaman and Nicobar Islands"],
        principal_seat="Kolkata",
        benches=["Port Blair"],
        indian_kanoon_id="calcuttahighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P0,
        target_judgments=6000,
        year_range_start=2019,
        established=1862,
    ),
    CourtEntry(
        name="Delhi High Court",
        short_name="DEL HC",
        court_type=CourtType.HIGH_COURT,
        states=["Delhi"],
        principal_seat="New Delhi",
        indian_kanoon_id="delhihighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P0,
        target_judgments=8000,
        year_range_start=2019,
        established=1966,
        notes="Major IP, commercial, and constitutional law jurisdiction.",
    ),
    CourtEntry(
        name="Karnataka High Court",
        short_name="KAR HC",
        court_type=CourtType.HIGH_COURT,
        states=["Karnataka"],
        principal_seat="Bengaluru",
        benches=["Dharwad", "Kalaburagi"],
        indian_kanoon_id="karnatakahighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P0,
        target_judgments=6000,
        year_range_start=2019,
        established=1884,
        notes="Bengaluru — major IT/tech industry jurisdiction.",
    ),
    CourtEntry(
        name="Gujarat High Court",
        short_name="GUJ HC",
        court_type=CourtType.HIGH_COURT,
        states=["Gujarat"],
        principal_seat="Ahmedabad",
        indian_kanoon_id="gujarathighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P0,
        target_judgments=5000,
        year_range_start=2019,
        established=1960,
    ),
    CourtEntry(
        name="Rajasthan High Court",
        short_name="RAJ HC",
        court_type=CourtType.HIGH_COURT,
        states=["Rajasthan"],
        principal_seat="Jodhpur",
        benches=["Jaipur"],
        indian_kanoon_id="rajasthanhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P0,
        target_judgments=5000,
        year_range_start=2019,
        established=1949,
    ),
    CourtEntry(
        name="Madhya Pradesh High Court",
        short_name="MP HC",
        court_type=CourtType.HIGH_COURT,
        states=["Madhya Pradesh"],
        principal_seat="Jabalpur",
        benches=["Indore", "Gwalior"],
        indian_kanoon_id="madhyapradeshhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P0,
        target_judgments=5000,
        year_range_start=2019,
        established=1956,
    ),
    CourtEntry(
        name="Patna High Court",
        short_name="PAT HC",
        court_type=CourtType.HIGH_COURT,
        states=["Bihar"],
        principal_seat="Patna",
        indian_kanoon_id="patnahighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P0,
        target_judgments=5000,
        year_range_start=2019,
        established=1916,
        notes="Bihar — 3rd most populous state (~130M).",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# High Courts — P1 (Remaining 15 HCs)
# ═══════════════════════════════════════════════════════════════════════════════

HIGH_COURTS_P1: list[CourtEntry] = [
    CourtEntry(
        name="Andhra Pradesh High Court",
        short_name="AP HC",
        court_type=CourtType.HIGH_COURT,
        states=["Andhra Pradesh"],
        principal_seat="Amaravati",
        indian_kanoon_id="andhrapradeshhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=3000,
        year_range_start=2019,
        established=2019,
        notes="Bifurcated from combined AP HC in 2019.",
    ),
    CourtEntry(
        name="Telangana High Court",
        short_name="TEL HC",
        court_type=CourtType.HIGH_COURT,
        states=["Telangana"],
        principal_seat="Hyderabad",
        indian_kanoon_id="telanganahighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=3000,
        year_range_start=2019,
        established=2019,
        notes="Hyderabad — major tech hub jurisdiction.",
    ),
    CourtEntry(
        name="Kerala High Court",
        short_name="KER HC",
        court_type=CourtType.HIGH_COURT,
        states=["Kerala", "Lakshadweep"],
        principal_seat="Kochi",
        indian_kanoon_id="keralahighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=3000,
        year_range_start=2019,
        established=1956,
    ),
    CourtEntry(
        name="Punjab and Haryana High Court",
        short_name="P&H HC",
        court_type=CourtType.HIGH_COURT,
        states=["Punjab", "Haryana", "Chandigarh"],
        principal_seat="Chandigarh",
        indian_kanoon_id="punjabhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=4000,
        year_range_start=2019,
        established=1947,
        notes="Joint HC for Punjab, Haryana, and Chandigarh UT.",
    ),
    CourtEntry(
        name="Orissa High Court",
        short_name="ORI HC",
        court_type=CourtType.HIGH_COURT,
        states=["Odisha"],
        principal_seat="Cuttack",
        indian_kanoon_id="orissahighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=2000,
        year_range_start=2019,
        established=1948,
    ),
    CourtEntry(
        name="Jharkhand High Court",
        short_name="JHR HC",
        court_type=CourtType.HIGH_COURT,
        states=["Jharkhand"],
        principal_seat="Ranchi",
        indian_kanoon_id="jharkhandhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=2000,
        year_range_start=2019,
        established=2000,
    ),
    CourtEntry(
        name="Chhattisgarh High Court",
        short_name="CG HC",
        court_type=CourtType.HIGH_COURT,
        states=["Chhattisgarh"],
        principal_seat="Bilaspur",
        indian_kanoon_id="chhattisgarthhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=2000,
        year_range_start=2019,
        established=2000,
    ),
    CourtEntry(
        name="Uttarakhand High Court",
        short_name="UTT HC",
        court_type=CourtType.HIGH_COURT,
        states=["Uttarakhand"],
        principal_seat="Nainital",
        indian_kanoon_id="uttarakhandhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=2000,
        year_range_start=2019,
        established=2000,
    ),
    CourtEntry(
        name="Himachal Pradesh High Court",
        short_name="HP HC",
        court_type=CourtType.HIGH_COURT,
        states=["Himachal Pradesh"],
        principal_seat="Shimla",
        indian_kanoon_id="himachalpradeshhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=1500,
        year_range_start=2019,
        established=1971,
    ),
    CourtEntry(
        name="Jammu and Kashmir and Ladakh High Court",
        short_name="J&K HC",
        court_type=CourtType.HIGH_COURT,
        states=["Jammu and Kashmir", "Ladakh"],
        principal_seat="Srinagar",
        benches=["Jammu"],
        indian_kanoon_id="jammukashmirhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=1500,
        year_range_start=2019,
        established=1928,
        notes="Renamed after bifurcation of J&K into two UTs in 2019.",
    ),
    CourtEntry(
        name="Gauhati High Court",
        short_name="GAU HC",
        court_type=CourtType.HIGH_COURT,
        states=["Assam", "Nagaland", "Mizoram", "Arunachal Pradesh"],
        principal_seat="Guwahati",
        benches=["Kohima", "Aizawl", "Itanagar"],
        indian_kanoon_id="gauhatihighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=2000,
        year_range_start=2019,
        established=1948,
        notes="Covers 4 NE states.",
    ),
    CourtEntry(
        name="Tripura High Court",
        short_name="TRI HC",
        court_type=CourtType.HIGH_COURT,
        states=["Tripura"],
        principal_seat="Agartala",
        indian_kanoon_id="tripurahighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=1000,
        year_range_start=2019,
        established=2013,
    ),
    CourtEntry(
        name="Meghalaya High Court",
        short_name="MEG HC",
        court_type=CourtType.HIGH_COURT,
        states=["Meghalaya"],
        principal_seat="Shillong",
        indian_kanoon_id="meghalayahighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=1000,
        year_range_start=2019,
        established=2013,
    ),
    CourtEntry(
        name="Manipur High Court",
        short_name="MAN HC",
        court_type=CourtType.HIGH_COURT,
        states=["Manipur"],
        principal_seat="Imphal",
        indian_kanoon_id="manipurhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=1000,
        year_range_start=2019,
        established=2013,
    ),
    CourtEntry(
        name="Sikkim High Court",
        short_name="SIK HC",
        court_type=CourtType.HIGH_COURT,
        states=["Sikkim"],
        principal_seat="Gangtok",
        indian_kanoon_id="sikkimhighcourt",
        indian_kanoon_doc_type="highcourt",
        priority=Priority.P1,
        target_judgments=500,
        year_range_start=2019,
        established=1975,
        notes="Smallest HC by caseload.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Tribunals — P2
# ═══════════════════════════════════════════════════════════════════════════════

TRIBUNALS: list[CourtEntry] = [
    CourtEntry(
        name="National Consumer Disputes Redressal Commission",
        short_name="NCDRC",
        court_type=CourtType.TRIBUNAL,
        states=["India"],
        principal_seat="New Delhi",
        indian_kanoon_id="ncdrc",
        indian_kanoon_doc_type="tribunal",
        priority=Priority.P2,
        target_judgments=3000,
        year_range_start=2019,
        established=1988,
        notes="Apex consumer forum. Orders frequently cited in CPA cases.",
    ),
    CourtEntry(
        name="National Company Law Tribunal",
        short_name="NCLT",
        court_type=CourtType.TRIBUNAL,
        states=["India"],
        principal_seat="New Delhi",
        benches=[
            "Mumbai", "Chennai", "Kolkata", "Bengaluru", "Ahmedabad",
            "Hyderabad", "Chandigarh", "Jaipur", "Kochi", "Cuttack",
            "Guwahati", "Amaravati", "Indore",
        ],
        indian_kanoon_id="nclt",
        indian_kanoon_doc_type="tribunal",
        priority=Priority.P2,
        target_judgments=3000,
        year_range_start=2019,
        established=2016,
        notes="IBC cases, company law matters, winding up.",
    ),
    CourtEntry(
        name="National Company Law Appellate Tribunal",
        short_name="NCLAT",
        court_type=CourtType.TRIBUNAL,
        states=["India"],
        principal_seat="New Delhi",
        benches=["Chennai"],
        indian_kanoon_id="nclat",
        indian_kanoon_doc_type="tribunal",
        priority=Priority.P2,
        target_judgments=2000,
        year_range_start=2019,
        established=2016,
    ),
    CourtEntry(
        name="National Green Tribunal",
        short_name="NGT",
        court_type=CourtType.TRIBUNAL,
        states=["India"],
        principal_seat="New Delhi",
        benches=["Bhopal", "Pune", "Kolkata", "Chennai"],
        indian_kanoon_id="ngt",
        indian_kanoon_doc_type="tribunal",
        priority=Priority.P2,
        target_judgments=2000,
        year_range_start=2019,
        established=2010,
    ),
    CourtEntry(
        name="Debt Recovery Appellate Tribunal",
        short_name="DRAT",
        court_type=CourtType.TRIBUNAL,
        states=["India"],
        principal_seat="New Delhi",
        indian_kanoon_id="drat",
        indian_kanoon_doc_type="tribunal",
        priority=Priority.P2,
        target_judgments=1000,
        year_range_start=2020,
        established=1993,
    ),
    CourtEntry(
        name="Securities Appellate Tribunal",
        short_name="SAT",
        court_type=CourtType.TRIBUNAL,
        states=["India"],
        principal_seat="Mumbai",
        indian_kanoon_id="sat",
        indian_kanoon_doc_type="tribunal",
        priority=Priority.P2,
        target_judgments=1000,
        year_range_start=2020,
        established=1992,
        notes="Appeals against SEBI orders.",
    ),
    CourtEntry(
        name="Income Tax Appellate Tribunal",
        short_name="ITAT",
        court_type=CourtType.TRIBUNAL,
        states=["India"],
        principal_seat="Mumbai",
        indian_kanoon_id="itat",
        indian_kanoon_doc_type="tribunal",
        priority=Priority.P2,
        target_judgments=2000,
        year_range_start=2020,
        established=1941,
        notes="Largest tribunal by volume. Tax law matters.",
    ),
    CourtEntry(
        name="Central Administrative Tribunal",
        short_name="CAT",
        court_type=CourtType.TRIBUNAL,
        states=["India"],
        principal_seat="New Delhi",
        indian_kanoon_id="cat",
        indian_kanoon_doc_type="tribunal",
        priority=Priority.P2,
        target_judgments=1500,
        year_range_start=2020,
        established=1985,
        notes="Service matters of central government employees.",
    ),
    CourtEntry(
        name="Armed Forces Tribunal",
        short_name="AFT",
        court_type=CourtType.TRIBUNAL,
        states=["India"],
        principal_seat="New Delhi",
        benches=["Chandigarh", "Lucknow", "Kolkata", "Mumbai",
                 "Chennai", "Kochi", "Jaipur", "Guwahati", "Srinagar"],
        indian_kanoon_id="aft",
        indian_kanoon_doc_type="tribunal",
        priority=Priority.P2,
        target_judgments=1000,
        year_range_start=2020,
        established=2009,
    ),
    CourtEntry(
        name="Real Estate Appellate Tribunal",
        short_name="REAT",
        court_type=CourtType.TRIBUNAL,
        states=["India"],
        principal_seat="New Delhi",
        indian_kanoon_id="reat",
        indian_kanoon_doc_type="tribunal",
        priority=Priority.P2,
        target_judgments=500,
        year_range_start=2020,
        established=2017,
        notes="Appeals against RERA authority orders.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# State → HC Mapping
#
# Maps every Indian state and UT to its jurisdictional High Court.
# Used by the query router to scope retrieval to relevant HC judgments.
# ═══════════════════════════════════════════════════════════════════════════════

STATE_TO_HC: dict[str, str] = {
    # P0 High Courts
    "Uttar Pradesh": "ALL HC",
    "Maharashtra": "BOM HC",
    "Goa": "BOM HC",
    "Tamil Nadu": "MAD HC",
    "Puducherry": "MAD HC",
    "West Bengal": "CAL HC",
    "Andaman and Nicobar Islands": "CAL HC",
    "Delhi": "DEL HC",
    "Karnataka": "KAR HC",
    "Gujarat": "GUJ HC",
    "Rajasthan": "RAJ HC",
    "Madhya Pradesh": "MP HC",
    "Bihar": "PAT HC",
    # P1 High Courts
    "Andhra Pradesh": "AP HC",
    "Telangana": "TEL HC",
    "Kerala": "KER HC",
    "Lakshadweep": "KER HC",
    "Punjab": "P&H HC",
    "Haryana": "P&H HC",
    "Chandigarh": "P&H HC",
    "Odisha": "ORI HC",
    "Jharkhand": "JHR HC",
    "Chhattisgarh": "CG HC",
    "Uttarakhand": "UTT HC",
    "Himachal Pradesh": "HP HC",
    "Jammu and Kashmir": "J&K HC",
    "Ladakh": "J&K HC",
    "Assam": "GAU HC",
    "Nagaland": "GAU HC",
    "Mizoram": "GAU HC",
    "Arunachal Pradesh": "GAU HC",
    "Tripura": "TRI HC",
    "Meghalaya": "MEG HC",
    "Manipur": "MAN HC",
    "Sikkim": "SIK HC",
    # UTs without a dedicated entry (mapped to nearest)
    "Dadra and Nagar Haveli and Daman and Diu": "BOM HC",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregated Registry
# ═══════════════════════════════════════════════════════════════════════════════

ALL_HIGH_COURTS: list[CourtEntry] = HIGH_COURTS_P0 + HIGH_COURTS_P1

COURTS_REGISTRY: dict[str, list[CourtEntry]] = {
    "supreme_court": [SUPREME_COURT],
    "high_courts_p0": HIGH_COURTS_P0,
    "high_courts_p1": HIGH_COURTS_P1,
    "tribunals": TRIBUNALS,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Query Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def get_all_courts() -> list[CourtEntry]:
    """Return every court as a flat list."""
    return [SUPREME_COURT] + ALL_HIGH_COURTS + TRIBUNALS


def get_courts_for_ingestion(max_priority: str = "P1") -> list[CourtEntry]:
    """
    Return courts ordered for ingestion, filtered by priority.

    Default is P0 + P1 (SC + all 25 HCs), excluding tribunals.
    """
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    max_tier = priority_order.get(max_priority.upper(), 1)

    courts = [
        c for c in get_all_courts()
        if priority_order.get(c.priority.value, 99) <= max_tier
    ]

    return sorted(courts, key=lambda c: (
        priority_order[c.priority.value],
        0 if c.court_type == CourtType.SUPREME_COURT else 1,
        c.name,
    ))


def get_hc_for_state(state: str) -> CourtEntry | None:
    """
    Get the High Court entry for a given state or UT.

    Args:
        state: State or Union Territory name (case-insensitive).

    Returns:
        Matching CourtEntry or None if not found.
    """
    # Normalize input
    state_lower = state.strip().lower()

    # Search in STATE_TO_HC mapping
    for mapped_state, hc_short in STATE_TO_HC.items():
        if mapped_state.lower() == state_lower:
            # Find the matching court entry
            for court in ALL_HIGH_COURTS:
                if court.short_name == hc_short:
                    return court
            break

    # Fallback: search court states directly
    for court in ALL_HIGH_COURTS:
        for court_state in court.states:
            if court_state.lower() == state_lower:
                return court

    return None


def get_court_by_short_name(short_name: str) -> CourtEntry | None:
    """Look up a court by its short name (e.g., 'DEL HC', 'SC')."""
    target = short_name.strip().upper()
    for court in get_all_courts():
        if court.short_name.upper() == target:
            return court
    return None


def get_total_target_judgments(max_priority: str = "P1") -> int:
    """Get the total target number of judgments across all courts at given priority."""
    return sum(c.target_judgments for c in get_courts_for_ingestion(max_priority))


def print_registry_summary() -> None:
    """Print a summary of the courts registry."""
    all_courts = get_all_courts()
    hcs = ALL_HIGH_COURTS

    print("\n" + "=" * 70)
    print("  NyayaMitra — Courts Registry Summary")
    print("=" * 70)

    print(f"\n  Total courts registered: {len(all_courts)}")
    print(f"    Supreme Court:  1")
    print(f"    High Courts:    {len(hcs)} (P0: {len(HIGH_COURTS_P0)}, P1: {len(HIGH_COURTS_P1)})")
    print(f"    Tribunals:      {len(TRIBUNALS)}")

    # Target judgments
    sc_target = SUPREME_COURT.target_judgments
    p0_hc_target = sum(c.target_judgments for c in HIGH_COURTS_P0)
    p1_hc_target = sum(c.target_judgments for c in HIGH_COURTS_P1)
    trib_target = sum(c.target_judgments for c in TRIBUNALS)

    print(f"\n  Target judgment counts:")
    print(f"    SC:             {sc_target:>8,}")
    print(f"    P0 HCs (10):    {p0_hc_target:>8,}")
    print(f"    P1 HCs (15):    {p1_hc_target:>8,}")
    print(f"    Tribunals:      {trib_target:>8,}")
    print(f"    ─────────────────────────")
    total = sc_target + p0_hc_target + p1_hc_target + trib_target
    print(f"    TOTAL:          {total:>8,}")

    # Sprint 7 target (SC + P0 HCs)
    sprint7 = sc_target + p0_hc_target
    print(f"\n  Sprint 7 scope (SC + P0 HCs): {sprint7:,} judgments")

    # P0 court details
    print(f"\n  P0 Courts (SC + top 10 HCs):")
    print(f"  {'Court':<35} {'Seat':<16} {'States':<24} {'Target':>8}")
    print("  " + "─" * 85)
    print(f"  {'Supreme Court of India':<35} {'New Delhi':<16} {'All India':<24} {sc_target:>8,}")
    for c in HIGH_COURTS_P0:
        states_str = ", ".join(c.states[:2])
        if len(c.states) > 2:
            states_str += f" +{len(c.states) - 2}"
        print(f"  {c.name:<35} {c.principal_seat:<16} {states_str:<24} {c.target_judgments:>8,}")

    # State coverage
    all_states = set(STATE_TO_HC.keys())
    p0_states = set()
    for c in HIGH_COURTS_P0:
        p0_states.update(c.states)
    p0_states.update(
        s for s, hc in STATE_TO_HC.items()
        if any(c.short_name == hc for c in HIGH_COURTS_P0)
    )

    print(f"\n  Jurisdiction coverage:")
    print(f"    Total states/UTs mapped: {len(all_states)}")
    print(f"    Covered by P0 HCs:       {len(p0_states)}")
    print(f"    P1 HCs needed for:       {len(all_states) - len(p0_states)} remaining states")

    print("\n" + "=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_registry_summary()

    # Quick validation: every state maps to a valid court
    print("  State → HC mapping validation:")
    errors = []
    for state, hc_short in STATE_TO_HC.items():
        court = get_court_by_short_name(hc_short)
        if court is None:
            errors.append(f"    ✗ {state} → {hc_short} (COURT NOT FOUND)")
    if errors:
        for e in errors:
            print(e)
    else:
        print(f"    ✓ All {len(STATE_TO_HC)} states/UTs map to valid courts.")

    # Test get_hc_for_state
    print("\n  Sample lookups:")
    for test in ["Maharashtra", "Delhi", "Tamil Nadu", "Assam", "Sikkim"]:
        hc = get_hc_for_state(test)
        print(f"    {test:<20} → {hc.short_name if hc else 'NOT FOUND'}")
    print()