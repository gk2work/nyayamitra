"""
NyayaMitra — Acts Registry.

Master registry of Indian legislation to ingest. Each entry contains the
metadata needed for scraping, parsing, indexing, and classification.

Organized by legal domain (criminal, property, family, labour, consumer,
constitutional, IP). Priority tiers:

    P0 — Core acts that cover 80%+ of citizen queries. Must be ingested first.
    P1 — Important acts frequently cited in courts. Second batch.
    P2 — Supplementary / niche acts. Third batch.

The registry drives:
    1. data/scrapers/india_code.py  — what to scrape from legislative.gov.in
    2. data/embeddings/bulk_indexer.py — domain tagging during indexing
    3. backend/app/services/query_router.py — domain → relevant acts mapping
    4. evaluation — coverage reporting

Usage:
    from data.config.acts_registry import ACTS_REGISTRY, get_acts_by_domain, get_acts_by_priority

    # All P0 acts
    p0 = get_acts_by_priority("P0")

    # Criminal domain only
    criminal = get_acts_by_domain("criminal")

    # All acts as flat list
    all_acts = get_all_acts()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════════


class LegalDomain(str, Enum):
    """Legal domain classification matching backend/app/models/query.py."""

    CRIMINAL = "criminal"
    PROPERTY = "property"
    FAMILY = "family"
    LABOUR = "labour"
    CONSUMER = "consumer"
    CONSTITUTIONAL = "constitutional"
    IP = "ip"
    CORPORATE = "corporate"
    TAXATION = "taxation"
    ENVIRONMENTAL = "environmental"


class Priority(str, Enum):
    """Ingestion priority tier."""

    P0 = "P0"  # Core — must have
    P1 = "P1"  # Important — should have
    P2 = "P2"  # Supplementary — nice to have


class ActStatus(str, Enum):
    """Current legislative status."""

    ACTIVE = "active"
    REPEALED = "repealed"
    PARTIALLY_REPEALED = "partially_repealed"
    REPLACED = "replaced"  # Superseded by new legislation


@dataclass(frozen=True)
class ActEntry:
    """A single act in the registry."""

    # Identity
    name: str                       # Full official name
    short_name: str                 # Citation short form (e.g., "IPC", "CrPC")
    year: int                       # Year of enactment
    act_number: str                 # Act number (e.g., "45 of 1860")

    # Classification
    domain: LegalDomain             # Primary legal domain
    priority: Priority              # Ingestion priority
    status: ActStatus = ActStatus.ACTIVE

    # Scraping
    india_code_id: str = ""         # India Code identifier / URL slug
    alt_source_url: str = ""        # Fallback source if India Code fails

    # Metadata
    replaced_by: str = ""           # If repealed, what replaced it
    secondary_domains: list[LegalDomain] = field(default_factory=list)
    notes: str = ""                 # Special parsing notes

    @property
    def citation_key(self) -> str:
        """Standard citation form: 'Short Name, Year'."""
        return f"{self.short_name}, {self.year}"

    @property
    def all_domains(self) -> list[LegalDomain]:
        """Primary + secondary domains."""
        return [self.domain] + list(self.secondary_domains)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — Criminal Law
# ═══════════════════════════════════════════════════════════════════════════════

CRIMINAL_ACTS: list[ActEntry] = [
    # --- P0: Core Criminal ---
    ActEntry(
        name="Indian Penal Code",
        short_name="IPC",
        year=1860,
        act_number="45 of 1860",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P0,
        status=ActStatus.REPLACED,
        india_code_id="indian-penal-code-1860",
        replaced_by="Bharatiya Nyaya Sanhita, 2023",
        notes="Replaced by BNS w.e.f. 01-07-2024 but still relevant for pre-2024 cases",
    ),
    ActEntry(
        name="Bharatiya Nyaya Sanhita",
        short_name="BNS",
        year=2023,
        act_number="45 of 2023",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P0,
        india_code_id="bharatiya-nyaya-sanhita-2023",
        notes="Replaces IPC. Effective 01-07-2024",
    ),
    ActEntry(
        name="Code of Criminal Procedure",
        short_name="CrPC",
        year=1973,
        act_number="2 of 1974",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P0,
        status=ActStatus.REPLACED,
        india_code_id="code-of-criminal-procedure-1973",
        replaced_by="Bharatiya Nagarik Suraksha Sanhita, 2023",
        notes="Replaced by BNSS w.e.f. 01-07-2024 but still relevant for pending cases",
    ),
    ActEntry(
        name="Bharatiya Nagarik Suraksha Sanhita",
        short_name="BNSS",
        year=2023,
        act_number="46 of 2023",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P0,
        india_code_id="bharatiya-nagarik-suraksha-sanhita-2023",
        notes="Replaces CrPC. Effective 01-07-2024",
    ),
    ActEntry(
        name="Indian Evidence Act",
        short_name="IEA",
        year=1872,
        act_number="1 of 1872",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P0,
        status=ActStatus.REPLACED,
        india_code_id="indian-evidence-act-1872",
        replaced_by="Bharatiya Sakshya Adhiniyam, 2023",
        secondary_domains=[LegalDomain.PROPERTY, LegalDomain.FAMILY],
    ),
    ActEntry(
        name="Bharatiya Sakshya Adhiniyam",
        short_name="BSA",
        year=2023,
        act_number="47 of 2023",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P0,
        india_code_id="bharatiya-sakshya-adhiniyam-2023",
        secondary_domains=[LegalDomain.PROPERTY, LegalDomain.FAMILY],
        notes="Replaces Indian Evidence Act. Effective 01-07-2024",
    ),

    # --- P1: Important Criminal ---
    ActEntry(
        name="Narcotic Drugs and Psychotropic Substances Act",
        short_name="NDPS Act",
        year=1985,
        act_number="61 of 1985",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P1,
        india_code_id="narcotic-drugs-and-psychotropic-substances-act-1985",
    ),
    ActEntry(
        name="Prevention of Corruption Act",
        short_name="PCA",
        year=1988,
        act_number="49 of 1988",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P1,
        india_code_id="prevention-of-corruption-act-1988",
    ),
    ActEntry(
        name="Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act",
        short_name="SC/ST Act",
        year=1989,
        act_number="33 of 1989",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P1,
        india_code_id="sc-st-prevention-of-atrocities-act-1989",
    ),
    ActEntry(
        name="Protection of Children from Sexual Offences Act",
        short_name="POCSO",
        year=2012,
        act_number="32 of 2012",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P1,
        india_code_id="protection-of-children-from-sexual-offences-act-2012",
    ),
    ActEntry(
        name="Juvenile Justice (Care and Protection of Children) Act",
        short_name="JJ Act",
        year=2015,
        act_number="2 of 2016",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P1,
        india_code_id="juvenile-justice-act-2015",
    ),
    ActEntry(
        name="Unlawful Activities (Prevention) Act",
        short_name="UAPA",
        year=1967,
        act_number="37 of 1967",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P1,
        india_code_id="unlawful-activities-prevention-act-1967",
    ),
    ActEntry(
        name="Arms Act",
        short_name="Arms Act",
        year=1959,
        act_number="54 of 1959",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P2,
        india_code_id="arms-act-1959",
    ),
    ActEntry(
        name="Dowry Prohibition Act",
        short_name="Dowry Prohibition Act",
        year=1961,
        act_number="28 of 1961",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P1,
        india_code_id="dowry-prohibition-act-1961",
        secondary_domains=[LegalDomain.FAMILY],
    ),
    ActEntry(
        name="Prevention of Money Laundering Act",
        short_name="PMLA",
        year=2002,
        act_number="15 of 2003",
        domain=LegalDomain.CRIMINAL,
        priority=Priority.P1,
        india_code_id="prevention-of-money-laundering-act-2002",
        secondary_domains=[LegalDomain.CORPORATE],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — Property & Real Estate
# ═══════════════════════════════════════════════════════════════════════════════

PROPERTY_ACTS: list[ActEntry] = [
    # --- P0: Core Property ---
    ActEntry(
        name="Transfer of Property Act",
        short_name="TPA",
        year=1882,
        act_number="4 of 1882",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P0,
        india_code_id="transfer-of-property-act-1882",
    ),
    ActEntry(
        name="Registration Act",
        short_name="Registration Act",
        year=1908,
        act_number="16 of 1908",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P0,
        india_code_id="registration-act-1908",
    ),
    ActEntry(
        name="Indian Stamp Act",
        short_name="Stamp Act",
        year=1899,
        act_number="2 of 1899",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P0,
        india_code_id="indian-stamp-act-1899",
    ),
    ActEntry(
        name="Real Estate (Regulation and Development) Act",
        short_name="RERA",
        year=2016,
        act_number="16 of 2016",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P0,
        india_code_id="real-estate-regulation-and-development-act-2016",
        secondary_domains=[LegalDomain.CONSUMER],
    ),

    # --- P1: Important Property ---
    ActEntry(
        name="Indian Easements Act",
        short_name="Easements Act",
        year=1882,
        act_number="5 of 1882",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P1,
        india_code_id="indian-easements-act-1882",
    ),
    ActEntry(
        name="Specific Relief Act",
        short_name="Specific Relief Act",
        year=1963,
        act_number="47 of 1963",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P1,
        india_code_id="specific-relief-act-1963",
    ),
    ActEntry(
        name="Benami Transactions (Prohibition) Act",
        short_name="Benami Act",
        year=1988,
        act_number="45 of 1988",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P1,
        india_code_id="benami-transactions-prohibition-act-1988",
    ),
    ActEntry(
        name="Land Acquisition, Rehabilitation and Resettlement Act",
        short_name="LARR Act",
        year=2013,
        act_number="30 of 2013",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P1,
        india_code_id="land-acquisition-act-2013",
    ),
    ActEntry(
        name="Indian Contract Act",
        short_name="Contract Act",
        year=1872,
        act_number="9 of 1872",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P0,
        india_code_id="indian-contract-act-1872",
        secondary_domains=[LegalDomain.CORPORATE, LegalDomain.CONSUMER],
        notes="General law of contracts — cross-domain relevance",
    ),
    ActEntry(
        name="Sale of Goods Act",
        short_name="Sale of Goods Act",
        year=1930,
        act_number="3 of 1930",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P1,
        india_code_id="sale-of-goods-act-1930",
        secondary_domains=[LegalDomain.CONSUMER],
    ),
    ActEntry(
        name="Partition Act",
        short_name="Partition Act",
        year=1893,
        act_number="4 of 1893",
        domain=LegalDomain.PROPERTY,
        priority=Priority.P2,
        india_code_id="partition-act-1893",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — Family & Personal Law
# ═══════════════════════════════════════════════════════════════════════════════

FAMILY_ACTS: list[ActEntry] = [
    # --- P0: Core Family ---
    ActEntry(
        name="Hindu Marriage Act",
        short_name="HMA",
        year=1955,
        act_number="25 of 1955",
        domain=LegalDomain.FAMILY,
        priority=Priority.P0,
        india_code_id="hindu-marriage-act-1955",
    ),
    ActEntry(
        name="Hindu Succession Act",
        short_name="HSA",
        year=1956,
        act_number="30 of 1956",
        domain=LegalDomain.FAMILY,
        priority=Priority.P0,
        india_code_id="hindu-succession-act-1956",
        secondary_domains=[LegalDomain.PROPERTY],
    ),
    ActEntry(
        name="Special Marriage Act",
        short_name="SMA",
        year=1954,
        act_number="43 of 1954",
        domain=LegalDomain.FAMILY,
        priority=Priority.P0,
        india_code_id="special-marriage-act-1954",
    ),
    ActEntry(
        name="Protection of Women from Domestic Violence Act",
        short_name="DV Act",
        year=2005,
        act_number="43 of 2005",
        domain=LegalDomain.FAMILY,
        priority=Priority.P0,
        india_code_id="protection-of-women-from-domestic-violence-act-2005",
        secondary_domains=[LegalDomain.CRIMINAL],
    ),

    # --- P1: Important Family ---
    ActEntry(
        name="Hindu Minority and Guardianship Act",
        short_name="HMGA",
        year=1956,
        act_number="32 of 1956",
        domain=LegalDomain.FAMILY,
        priority=Priority.P1,
        india_code_id="hindu-minority-and-guardianship-act-1956",
    ),
    ActEntry(
        name="Hindu Adoptions and Maintenance Act",
        short_name="HAMA",
        year=1956,
        act_number="78 of 1956",
        domain=LegalDomain.FAMILY,
        priority=Priority.P1,
        india_code_id="hindu-adoptions-and-maintenance-act-1956",
    ),
    ActEntry(
        name="Guardians and Wards Act",
        short_name="GWA",
        year=1890,
        act_number="8 of 1890",
        domain=LegalDomain.FAMILY,
        priority=Priority.P1,
        india_code_id="guardians-and-wards-act-1890",
    ),
    ActEntry(
        name="Indian Divorce Act",
        short_name="Divorce Act",
        year=1869,
        act_number="4 of 1869",
        domain=LegalDomain.FAMILY,
        priority=Priority.P1,
        india_code_id="indian-divorce-act-1869",
        notes="Applicable to Christians",
    ),
    ActEntry(
        name="Dissolution of Muslim Marriages Act",
        short_name="DMMA",
        year=1939,
        act_number="8 of 1939",
        domain=LegalDomain.FAMILY,
        priority=Priority.P1,
        india_code_id="dissolution-of-muslim-marriages-act-1939",
    ),
    ActEntry(
        name="Muslim Women (Protection of Rights on Divorce) Act",
        short_name="Muslim Women Act",
        year=1986,
        act_number="25 of 1986",
        domain=LegalDomain.FAMILY,
        priority=Priority.P1,
        india_code_id="muslim-women-protection-of-rights-on-divorce-act-1986",
    ),
    ActEntry(
        name="Prohibition of Child Marriage Act",
        short_name="PCMA",
        year=2006,
        act_number="6 of 2007",
        domain=LegalDomain.FAMILY,
        priority=Priority.P1,
        india_code_id="prohibition-of-child-marriage-act-2006",
    ),
    ActEntry(
        name="Indian Succession Act",
        short_name="ISA",
        year=1925,
        act_number="39 of 1925",
        domain=LegalDomain.FAMILY,
        priority=Priority.P1,
        india_code_id="indian-succession-act-1925",
        secondary_domains=[LegalDomain.PROPERTY],
        notes="Applies to Christians, Parsis, and intestate succession",
    ),
    ActEntry(
        name="Maintenance and Welfare of Parents and Senior Citizens Act",
        short_name="Senior Citizens Act",
        year=2007,
        act_number="56 of 2007",
        domain=LegalDomain.FAMILY,
        priority=Priority.P1,
        india_code_id="maintenance-and-welfare-of-parents-and-senior-citizens-act-2007",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — Labour & Employment
# ═══════════════════════════════════════════════════════════════════════════════

LABOUR_ACTS: list[ActEntry] = [
    # --- P0: Core Labour (New Codes) ---
    ActEntry(
        name="Code on Wages",
        short_name="Wages Code",
        year=2019,
        act_number="29 of 2019",
        domain=LegalDomain.LABOUR,
        priority=Priority.P0,
        india_code_id="code-on-wages-2019",
        notes="Consolidates Minimum Wages Act, Payment of Wages Act, etc.",
    ),
    ActEntry(
        name="Industrial Relations Code",
        short_name="IR Code",
        year=2020,
        act_number="35 of 2020",
        domain=LegalDomain.LABOUR,
        priority=Priority.P0,
        india_code_id="industrial-relations-code-2020",
        notes="Consolidates ID Act, Trade Unions Act, etc.",
    ),
    ActEntry(
        name="Code on Social Security",
        short_name="SS Code",
        year=2020,
        act_number="36 of 2020",
        domain=LegalDomain.LABOUR,
        priority=Priority.P0,
        india_code_id="code-on-social-security-2020",
        notes="Consolidates EPF, ESI, Gratuity, Maternity Benefit, etc.",
    ),
    ActEntry(
        name="Occupational Safety, Health and Working Conditions Code",
        short_name="OSH Code",
        year=2020,
        act_number="37 of 2020",
        domain=LegalDomain.LABOUR,
        priority=Priority.P0,
        india_code_id="osh-code-2020",
        notes="Consolidates Factories Act, Mines Act, etc.",
    ),

    # --- P1: Legacy labour laws (still operational / heavily cited) ---
    ActEntry(
        name="Industrial Disputes Act",
        short_name="ID Act",
        year=1947,
        act_number="14 of 1947",
        domain=LegalDomain.LABOUR,
        priority=Priority.P1,
        india_code_id="industrial-disputes-act-1947",
        notes="Being replaced by IR Code but still operational in most states",
    ),
    ActEntry(
        name="Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act",
        short_name="POSH Act",
        year=2013,
        act_number="14 of 2013",
        domain=LegalDomain.LABOUR,
        priority=Priority.P0,
        india_code_id="sexual-harassment-of-women-at-workplace-act-2013",
        secondary_domains=[LegalDomain.CRIMINAL],
    ),
    ActEntry(
        name="Payment of Gratuity Act",
        short_name="Gratuity Act",
        year=1972,
        act_number="39 of 1972",
        domain=LegalDomain.LABOUR,
        priority=Priority.P1,
        india_code_id="payment-of-gratuity-act-1972",
    ),
    ActEntry(
        name="Employees' Provident Funds and Miscellaneous Provisions Act",
        short_name="EPF Act",
        year=1952,
        act_number="19 of 1952",
        domain=LegalDomain.LABOUR,
        priority=Priority.P1,
        india_code_id="employees-provident-funds-act-1952",
    ),
    ActEntry(
        name="Employees' State Insurance Act",
        short_name="ESI Act",
        year=1948,
        act_number="34 of 1948",
        domain=LegalDomain.LABOUR,
        priority=Priority.P1,
        india_code_id="employees-state-insurance-act-1948",
    ),
    ActEntry(
        name="Minimum Wages Act",
        short_name="Minimum Wages Act",
        year=1948,
        act_number="11 of 1948",
        domain=LegalDomain.LABOUR,
        priority=Priority.P1,
        india_code_id="minimum-wages-act-1948",
    ),
    ActEntry(
        name="Maternity Benefit Act",
        short_name="Maternity Benefit Act",
        year=1961,
        act_number="53 of 1961",
        domain=LegalDomain.LABOUR,
        priority=Priority.P1,
        india_code_id="maternity-benefit-act-1961",
    ),
    ActEntry(
        name="Equal Remuneration Act",
        short_name="ERA",
        year=1976,
        act_number="25 of 1976",
        domain=LegalDomain.LABOUR,
        priority=Priority.P2,
        india_code_id="equal-remuneration-act-1976",
    ),
    ActEntry(
        name="Contract Labour (Regulation and Abolition) Act",
        short_name="CLRA Act",
        year=1970,
        act_number="37 of 1970",
        domain=LegalDomain.LABOUR,
        priority=Priority.P2,
        india_code_id="contract-labour-act-1970",
    ),
    ActEntry(
        name="Bonded Labour System (Abolition) Act",
        short_name="Bonded Labour Act",
        year=1976,
        act_number="19 of 1976",
        domain=LegalDomain.LABOUR,
        priority=Priority.P2,
        india_code_id="bonded-labour-system-abolition-act-1976",
    ),
    ActEntry(
        name="Child and Adolescent Labour (Prohibition and Regulation) Act",
        short_name="Child Labour Act",
        year=1986,
        act_number="61 of 1986",
        domain=LegalDomain.LABOUR,
        priority=Priority.P1,
        india_code_id="child-labour-prohibition-and-regulation-act-1986",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — Consumer Protection
# ═══════════════════════════════════════════════════════════════════════════════

CONSUMER_ACTS: list[ActEntry] = [
    ActEntry(
        name="Consumer Protection Act",
        short_name="CPA",
        year=2019,
        act_number="35 of 2019",
        domain=LegalDomain.CONSUMER,
        priority=Priority.P0,
        india_code_id="consumer-protection-act-2019",
        notes="Replaces CPA 1986. Includes e-commerce, product liability, mediation.",
    ),
    ActEntry(
        name="Consumer Protection Act",
        short_name="CPA 1986",
        year=1986,
        act_number="68 of 1986",
        domain=LegalDomain.CONSUMER,
        priority=Priority.P1,
        status=ActStatus.REPEALED,
        india_code_id="consumer-protection-act-1986",
        replaced_by="Consumer Protection Act, 2019",
        notes="Repealed but many pending cases still under this act",
    ),
    ActEntry(
        name="Food Safety and Standards Act",
        short_name="FSSA",
        year=2006,
        act_number="34 of 2006",
        domain=LegalDomain.CONSUMER,
        priority=Priority.P1,
        india_code_id="food-safety-and-standards-act-2006",
    ),
    ActEntry(
        name="Legal Metrology Act",
        short_name="Legal Metrology Act",
        year=2009,
        act_number="1 of 2010",
        domain=LegalDomain.CONSUMER,
        priority=Priority.P2,
        india_code_id="legal-metrology-act-2009",
    ),
    ActEntry(
        name="Competition Act",
        short_name="Competition Act",
        year=2002,
        act_number="12 of 2003",
        domain=LegalDomain.CONSUMER,
        priority=Priority.P1,
        india_code_id="competition-act-2002",
        secondary_domains=[LegalDomain.CORPORATE],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — Constitutional & Administrative
# ═══════════════════════════════════════════════════════════════════════════════

CONSTITUTIONAL_ACTS: list[ActEntry] = [
    ActEntry(
        name="Constitution of India",
        short_name="Constitution",
        year=1950,
        act_number="N/A",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P0,
        india_code_id="constitution-of-india",
        notes="Fundamental rights, DPSP, fundamental duties, schedules. Parse article-wise.",
    ),
    ActEntry(
        name="Right to Information Act",
        short_name="RTI Act",
        year=2005,
        act_number="22 of 2005",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P0,
        india_code_id="right-to-information-act-2005",
    ),
    ActEntry(
        name="Code of Civil Procedure",
        short_name="CPC",
        year=1908,
        act_number="5 of 1908",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P0,
        india_code_id="code-of-civil-procedure-1908",
        secondary_domains=[LegalDomain.PROPERTY, LegalDomain.FAMILY],
        notes="General procedural law for civil courts — cross-domain",
    ),
    ActEntry(
        name="Limitation Act",
        short_name="Limitation Act",
        year=1963,
        act_number="36 of 1963",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P0,
        india_code_id="limitation-act-1963",
        secondary_domains=[LegalDomain.PROPERTY, LegalDomain.FAMILY, LegalDomain.CRIMINAL],
        notes="Time limits for filing suits — universally applicable",
    ),
    ActEntry(
        name="Indian Succession Act",
        short_name="ISA",
        year=1925,
        act_number="39 of 1925",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P1,
        india_code_id="indian-succession-act-1925",
        secondary_domains=[LegalDomain.FAMILY, LegalDomain.PROPERTY],
    ),
    ActEntry(
        name="Arbitration and Conciliation Act",
        short_name="Arbitration Act",
        year=1996,
        act_number="26 of 1996",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P1,
        india_code_id="arbitration-and-conciliation-act-1996",
        secondary_domains=[LegalDomain.CORPORATE],
    ),
    ActEntry(
        name="Legal Services Authorities Act",
        short_name="LSA Act",
        year=1987,
        act_number="39 of 1987",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P1,
        india_code_id="legal-services-authorities-act-1987",
        notes="NALSA, free legal aid, Lok Adalats",
    ),
    ActEntry(
        name="Representation of the People Act",
        short_name="RPA",
        year=1951,
        act_number="43 of 1951",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P1,
        india_code_id="representation-of-the-people-act-1951",
    ),
    ActEntry(
        name="National Security Act",
        short_name="NSA",
        year=1980,
        act_number="65 of 1980",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P2,
        india_code_id="national-security-act-1980",
        secondary_domains=[LegalDomain.CRIMINAL],
    ),
    ActEntry(
        name="Administrative Tribunals Act",
        short_name="AT Act",
        year=1985,
        act_number="13 of 1985",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P2,
        india_code_id="administrative-tribunals-act-1985",
    ),
    ActEntry(
        name="Contempt of Courts Act",
        short_name="Contempt Act",
        year=1971,
        act_number="70 of 1971",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P2,
        india_code_id="contempt-of-courts-act-1971",
    ),
    ActEntry(
        name="Lok Pal and Lokayuktas Act",
        short_name="Lokpal Act",
        year=2013,
        act_number="1 of 2014",
        domain=LegalDomain.CONSTITUTIONAL,
        priority=Priority.P2,
        india_code_id="lokpal-and-lokayuktas-act-2013",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — Intellectual Property
# ═══════════════════════════════════════════════════════════════════════════════

IP_ACTS: list[ActEntry] = [
    ActEntry(
        name="Information Technology Act",
        short_name="IT Act",
        year=2000,
        act_number="21 of 2000",
        domain=LegalDomain.IP,
        priority=Priority.P0,
        india_code_id="information-technology-act-2000",
        secondary_domains=[LegalDomain.CRIMINAL],
        notes="Cybercrime, digital signatures, intermediary liability",
    ),
    ActEntry(
        name="Copyright Act",
        short_name="Copyright Act",
        year=1957,
        act_number="14 of 1957",
        domain=LegalDomain.IP,
        priority=Priority.P0,
        india_code_id="copyright-act-1957",
    ),
    ActEntry(
        name="Patents Act",
        short_name="Patents Act",
        year=1970,
        act_number="39 of 1970",
        domain=LegalDomain.IP,
        priority=Priority.P0,
        india_code_id="patents-act-1970",
    ),
    ActEntry(
        name="Trade Marks Act",
        short_name="Trade Marks Act",
        year=1999,
        act_number="47 of 1999",
        domain=LegalDomain.IP,
        priority=Priority.P0,
        india_code_id="trade-marks-act-1999",
    ),
    ActEntry(
        name="Designs Act",
        short_name="Designs Act",
        year=2000,
        act_number="16 of 2000",
        domain=LegalDomain.IP,
        priority=Priority.P1,
        india_code_id="designs-act-2000",
    ),
    ActEntry(
        name="Geographical Indications of Goods (Registration and Protection) Act",
        short_name="GI Act",
        year=1999,
        act_number="48 of 1999",
        domain=LegalDomain.IP,
        priority=Priority.P2,
        india_code_id="geographical-indications-act-1999",
    ),
    ActEntry(
        name="Semiconductor Integrated Circuits Layout-Design Act",
        short_name="SICLD Act",
        year=2000,
        act_number="37 of 2000",
        domain=LegalDomain.IP,
        priority=Priority.P2,
        india_code_id="semiconductor-integrated-circuits-layout-design-act-2000",
    ),
    ActEntry(
        name="Digital Personal Data Protection Act",
        short_name="DPDP Act",
        year=2023,
        act_number="22 of 2023",
        domain=LegalDomain.IP,
        priority=Priority.P0,
        india_code_id="digital-personal-data-protection-act-2023",
        secondary_domains=[LegalDomain.CONSTITUTIONAL],
        notes="India's data protection law. Effective 2024-25.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — Corporate & Commercial
# ═══════════════════════════════════════════════════════════════════════════════

CORPORATE_ACTS: list[ActEntry] = [
    ActEntry(
        name="Companies Act",
        short_name="Companies Act",
        year=2013,
        act_number="18 of 2013",
        domain=LegalDomain.CORPORATE,
        priority=Priority.P0,
        india_code_id="companies-act-2013",
    ),
    ActEntry(
        name="Limited Liability Partnership Act",
        short_name="LLP Act",
        year=2008,
        act_number="6 of 2009",
        domain=LegalDomain.CORPORATE,
        priority=Priority.P1,
        india_code_id="limited-liability-partnership-act-2008",
    ),
    ActEntry(
        name="Insolvency and Bankruptcy Code",
        short_name="IBC",
        year=2016,
        act_number="31 of 2016",
        domain=LegalDomain.CORPORATE,
        priority=Priority.P0,
        india_code_id="insolvency-and-bankruptcy-code-2016",
    ),
    ActEntry(
        name="Negotiable Instruments Act",
        short_name="NI Act",
        year=1881,
        act_number="26 of 1881",
        domain=LegalDomain.CORPORATE,
        priority=Priority.P0,
        india_code_id="negotiable-instruments-act-1881",
        secondary_domains=[LegalDomain.CRIMINAL],
        notes="Section 138 (cheque bounce) is one of the most litigated provisions in India",
    ),
    ActEntry(
        name="Partnership Act",
        short_name="Partnership Act",
        year=1932,
        act_number="9 of 1932",
        domain=LegalDomain.CORPORATE,
        priority=Priority.P1,
        india_code_id="indian-partnership-act-1932",
    ),
    ActEntry(
        name="Securities and Exchange Board of India Act",
        short_name="SEBI Act",
        year=1992,
        act_number="15 of 1992",
        domain=LegalDomain.CORPORATE,
        priority=Priority.P1,
        india_code_id="sebi-act-1992",
    ),
    ActEntry(
        name="Foreign Exchange Management Act",
        short_name="FEMA",
        year=1999,
        act_number="42 of 1999",
        domain=LegalDomain.CORPORATE,
        priority=Priority.P1,
        india_code_id="foreign-exchange-management-act-1999",
    ),
    ActEntry(
        name="Reserve Bank of India Act",
        short_name="RBI Act",
        year=1934,
        act_number="2 of 1934",
        domain=LegalDomain.CORPORATE,
        priority=Priority.P2,
        india_code_id="reserve-bank-of-india-act-1934",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — Taxation
# ═══════════════════════════════════════════════════════════════════════════════

TAXATION_ACTS: list[ActEntry] = [
    ActEntry(
        name="Income Tax Act",
        short_name="IT Act 1961",
        year=1961,
        act_number="43 of 1961",
        domain=LegalDomain.TAXATION,
        priority=Priority.P1,
        india_code_id="income-tax-act-1961",
        notes="Massive act — 298 sections + schedules. Parse carefully.",
    ),
    ActEntry(
        name="Central Goods and Services Tax Act",
        short_name="CGST Act",
        year=2017,
        act_number="12 of 2017",
        domain=LegalDomain.TAXATION,
        priority=Priority.P1,
        india_code_id="central-goods-and-services-tax-act-2017",
    ),
    ActEntry(
        name="Integrated Goods and Services Tax Act",
        short_name="IGST Act",
        year=2017,
        act_number="13 of 2017",
        domain=LegalDomain.TAXATION,
        priority=Priority.P2,
        india_code_id="integrated-goods-and-services-tax-act-2017",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — Environmental
# ═══════════════════════════════════════════════════════════════════════════════

ENVIRONMENTAL_ACTS: list[ActEntry] = [
    ActEntry(
        name="Environment (Protection) Act",
        short_name="EPA",
        year=1986,
        act_number="29 of 1986",
        domain=LegalDomain.ENVIRONMENTAL,
        priority=Priority.P1,
        india_code_id="environment-protection-act-1986",
    ),
    ActEntry(
        name="National Green Tribunal Act",
        short_name="NGT Act",
        year=2010,
        act_number="19 of 2010",
        domain=LegalDomain.ENVIRONMENTAL,
        priority=Priority.P1,
        india_code_id="national-green-tribunal-act-2010",
    ),
    ActEntry(
        name="Water (Prevention and Control of Pollution) Act",
        short_name="Water Act",
        year=1974,
        act_number="6 of 1974",
        domain=LegalDomain.ENVIRONMENTAL,
        priority=Priority.P2,
        india_code_id="water-prevention-and-control-of-pollution-act-1974",
    ),
    ActEntry(
        name="Air (Prevention and Control of Pollution) Act",
        short_name="Air Act",
        year=1981,
        act_number="14 of 1981",
        domain=LegalDomain.ENVIRONMENTAL,
        priority=Priority.P2,
        india_code_id="air-prevention-and-control-of-pollution-act-1981",
    ),
    ActEntry(
        name="Wildlife Protection Act",
        short_name="Wildlife Act",
        year=1972,
        act_number="53 of 1972",
        domain=LegalDomain.ENVIRONMENTAL,
        priority=Priority.P2,
        india_code_id="wildlife-protection-act-1972",
    ),
    ActEntry(
        name="Forest (Conservation) Act",
        short_name="Forest Act",
        year=1980,
        act_number="69 of 1980",
        domain=LegalDomain.ENVIRONMENTAL,
        priority=Priority.P2,
        india_code_id="forest-conservation-act-1980",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregated Registry
# ═══════════════════════════════════════════════════════════════════════════════

ACTS_REGISTRY: dict[str, list[ActEntry]] = {
    "criminal": CRIMINAL_ACTS,
    "property": PROPERTY_ACTS,
    "family": FAMILY_ACTS,
    "labour": LABOUR_ACTS,
    "consumer": CONSUMER_ACTS,
    "constitutional": CONSTITUTIONAL_ACTS,
    "ip": IP_ACTS,
    "corporate": CORPORATE_ACTS,
    "taxation": TAXATION_ACTS,
    "environmental": ENVIRONMENTAL_ACTS,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Query Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def get_all_acts() -> list[ActEntry]:
    """Return all acts as a flat list."""
    return [act for acts in ACTS_REGISTRY.values() for act in acts]


def get_acts_by_domain(domain: str) -> list[ActEntry]:
    """Return all acts for a given domain."""
    return ACTS_REGISTRY.get(domain.lower(), [])


def get_acts_by_priority(priority: str) -> list[ActEntry]:
    """Return all acts at a given priority level across all domains."""
    target = Priority(priority.upper())
    return [act for act in get_all_acts() if act.priority == target]


def get_active_acts() -> list[ActEntry]:
    """Return only currently active (not repealed/replaced) acts."""
    return [act for act in get_all_acts() if act.status == ActStatus.ACTIVE]


def get_acts_for_ingestion(max_priority: str = "P2") -> list[ActEntry]:
    """
    Return acts ordered for ingestion: P0 first, then P1, then P2.

    Args:
        max_priority: Maximum priority tier to include. "P0" for core only,
                      "P1" for core + important, "P2" for everything.
    """
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    max_tier = priority_order.get(max_priority.upper(), 2)

    acts = [
        act
        for act in get_all_acts()
        if priority_order.get(act.priority.value, 99) <= max_tier
    ]

    return sorted(acts, key=lambda a: (priority_order[a.priority.value], a.domain.value, a.year))


def get_short_name_index() -> dict[str, ActEntry]:
    """
    Build a lookup index from short_name → ActEntry.

    Useful for citation verification (e.g., "IPC" → full ActEntry).
    """
    index: dict[str, ActEntry] = {}
    for act in get_all_acts():
        index[act.short_name] = act
        # Also index by citation_key
        index[act.citation_key] = act
    return index


def print_registry_summary() -> None:
    """Print a summary of the registry for debugging."""
    all_acts = get_all_acts()

    print("\n" + "=" * 65)
    print("  NyayaMitra — Acts Registry Summary")
    print("=" * 65)
    print(f"\n  Total acts registered: {len(all_acts)}")

    # By priority
    for p in Priority:
        count = len([a for a in all_acts if a.priority == p])
        print(f"    {p.value}: {count} acts")

    # By domain
    print()
    for domain_key, acts in ACTS_REGISTRY.items():
        active = len([a for a in acts if a.status == ActStatus.ACTIVE])
        replaced = len([a for a in acts if a.status in (ActStatus.REPEALED, ActStatus.REPLACED)])
        print(f"  {domain_key:<16} {len(acts):>3} acts ({active} active, {replaced} repealed/replaced)")

    # By status
    print()
    for s in ActStatus:
        count = len([a for a in all_acts if a.status == s])
        if count:
            print(f"    {s.value}: {count}")

    print("\n" + "=" * 65 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_registry_summary()

    # Show ingestion plan
    print("  Ingestion Order (P0 first):")
    print("  " + "-" * 60)
    for i, act in enumerate(get_acts_for_ingestion("P0"), 1):
        status = f" [{act.status.value}]" if act.status != ActStatus.ACTIVE else ""
        print(f"    {i:>3}. {act.citation_key:<35} ({act.domain.value}){status}")

    print(f"\n  P0 total: {len(get_acts_by_priority('P0'))} acts")
    print(f"  P0+P1 total: {len(get_acts_by_priority('P0')) + len(get_acts_by_priority('P1'))} acts")
    print(f"  P0+P1+P2 total: {len(get_all_acts())} acts\n")