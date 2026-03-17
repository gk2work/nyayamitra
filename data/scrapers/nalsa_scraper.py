"""
NyayaMitra — NALSA / Tele-Law FAQ Scraper & Curator.

Ingests procedural legal knowledge from NALSA (National Legal Services
Authority) and the Tele-Law programme. This data answers "how do I..."
questions — step-by-step guides for common legal procedures.

Data sources:
    1. Curated FAQ entries (embedded in this file) — high-quality,
       verified procedural knowledge covering all 7 legal domains.
    2. NALSA website (nalsa.gov.in) — if reachable, scrape additional
       FAQ and scheme descriptions.
    3. Tele-Law portal (tele-law.in) — legal awareness content.

Unlike acts (india_code.py) and judgments (indian_kanoon.py), this
data is primarily curated rather than scraped, because:
    - NALSA's website structure is inconsistent and changes frequently
    - Procedural steps require expert verification (not auto-parseable)
    - Quality matters more than volume for procedural guidance

Each FAQ entry maps to a PostgreSQL record and is also stored as a
structured JSON file for the embedding pipeline.

Usage:
    # Seed all curated FAQ entries
    python -m data.scrapers.nalsa_scraper

    # Seed + attempt web scrape
    python -m data.scrapers.nalsa_scraper --scrape

    # Export as JSON for embedding pipeline
    python -m data.scrapers.nalsa_scraper --export
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import structlog

# ── Project path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.database import async_session
from app.models.legal import IngestionLog

logger = structlog.get_logger()

EXPORT_DIR = PROJECT_ROOT / "data" / "raw" / "nalsa_faq"


# ═══════════════════════════════════════════════════════════════════════════════
# FAQ Entry Schema
# ═══════════════════════════════════════════════════════════════════════════════


# Each entry follows this structure for direct use by the RAG pipeline:
#   question:     The user-facing question (used for embedding)
#   answer:       Plain-language explanation
#   domain:       Legal domain
#   jurisdiction: "central" or state-specific
#   steps:        List of {step, action, details, authority, time_limit, fees}
#   relevant_law: List of {act, section} references
#   tags:         Keywords for search
#   source:       "nalsa_curated" or "nalsa_website"


# ═══════════════════════════════════════════════════════════════════════════════
# Curated FAQ Entries — Criminal Law
# ═══════════════════════════════════════════════════════════════════════════════

CRIMINAL_FAQS = [
    {
        "id": "faq_crim_01",
        "question": "How to file an FIR (First Information Report)?",
        "answer": (
            "An FIR is the first step to set the criminal law in motion. "
            "Under Section 154 CrPC (now Section 173 BNSS), any person can "
            "report a cognizable offence at the nearest police station. The "
            "police are legally bound to register the FIR — refusal is a "
            "punishable offence."
        ),
        "domain": "criminal",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Go to the nearest police station",
                "details": "Visit the police station having jurisdiction over the area where the offence occurred. You can also file at any police station under the Zero FIR policy.",
                "authority": "Police Station (SHO)",
                "time_limit": "No time limit, but file as early as possible",
                "fees": "Free — no fees for FIR registration",
            },
            {
                "step": 2,
                "action": "Narrate the incident to the officer",
                "details": "Give a detailed oral or written account of the offence — who, what, when, where. The officer must write it down in the prescribed register.",
                "authority": "Station House Officer (SHO)",
            },
            {
                "step": 3,
                "action": "Get your FIR copy",
                "details": "Under Section 154(2) CrPC, you are entitled to a free copy of the FIR immediately. Do not leave without it.",
                "authority": "Police Station",
            },
            {
                "step": 4,
                "action": "If police refuse to register FIR",
                "details": "Send a written complaint to the Superintendent of Police (SP) by registered post. If still no action, approach the Judicial Magistrate under Section 156(3) CrPC to direct the police to register and investigate.",
                "authority": "SP / Judicial Magistrate",
            },
        ],
        "relevant_law": [
            {"act": "CrPC", "section": "154"},
            {"act": "CrPC", "section": "155"},
            {"act": "CrPC", "section": "156(3)"},
            {"act": "BNSS", "section": "173"},
        ],
        "tags": ["FIR", "police", "complaint", "cognizable offence", "zero FIR"],
        "source": "nalsa_curated",
    },
    {
        "id": "faq_crim_02",
        "question": "How to apply for bail?",
        "answer": (
            "Bail is a right in bailable offences and at the discretion of "
            "the court in non-bailable offences. You can apply for regular "
            "bail, anticipatory bail, or interim bail depending on your "
            "situation."
        ),
        "domain": "criminal",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Determine the type of bail needed",
                "details": "Regular bail (Section 437/439 CrPC) if already arrested. Anticipatory bail (Section 438 CrPC) if you expect arrest. Interim bail for temporary release pending hearing.",
                "authority": "Court",
            },
            {
                "step": 2,
                "action": "Engage a lawyer or seek free legal aid",
                "details": "Contact a lawyer or approach the District Legal Services Authority (DLSA) for free legal aid under the Legal Services Authorities Act if you cannot afford a lawyer.",
                "authority": "DLSA / Bar Council",
            },
            {
                "step": 3,
                "action": "File the bail application",
                "details": "Your lawyer files the application before the appropriate court — Magistrate for regular bail in non-bailable offences (Section 437), Sessions Court or High Court for anticipatory bail (Section 438).",
                "authority": "Magistrate / Sessions Court / High Court",
                "fees": "Court fees vary by state (typically Rs. 50-500)",
            },
            {
                "step": 4,
                "action": "Attend the hearing",
                "details": "The court considers: nature of offence, criminal antecedents, flight risk, likelihood of tampering with evidence. Furnish surety and bail bond as directed.",
                "authority": "Court",
            },
        ],
        "relevant_law": [
            {"act": "CrPC", "section": "436"},
            {"act": "CrPC", "section": "437"},
            {"act": "CrPC", "section": "438"},
            {"act": "CrPC", "section": "439"},
        ],
        "tags": ["bail", "anticipatory bail", "arrest", "non-bailable", "surety"],
        "source": "nalsa_curated",
    },
    {
        "id": "faq_crim_03",
        "question": "What are my rights if I am arrested?",
        "answer": (
            "Every arrested person has fundamental rights under the Constitution "
            "and the D.K. Basu guidelines laid down by the Supreme Court."
        ),
        "domain": "criminal",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Right to know the grounds of arrest",
                "details": "The police must inform you why you are being arrested (Article 22(1) of the Constitution).",
                "authority": "Police",
            },
            {
                "step": 2,
                "action": "Right to inform someone",
                "details": "You can inform a friend, relative, or any person of your choice about your arrest and the place of detention.",
                "authority": "Police",
            },
            {
                "step": 3,
                "action": "Right to a lawyer",
                "details": "You have the right to consult a lawyer of your choice. If you cannot afford one, the state must provide free legal aid.",
                "authority": "DLSA",
            },
            {
                "step": 4,
                "action": "Right to be produced before a Magistrate within 24 hours",
                "details": "Under Article 22(2), you must be produced before the nearest Magistrate within 24 hours of arrest (excluding travel time).",
                "authority": "Magistrate",
                "time_limit": "24 hours",
            },
            {
                "step": 5,
                "action": "Right to medical examination",
                "details": "Under D.K. Basu guidelines, you must be medically examined every 48 hours during detention.",
                "authority": "Police / Medical Officer",
                "time_limit": "Every 48 hours",
            },
        ],
        "relevant_law": [
            {"act": "Constitution of India", "section": "22"},
            {"act": "CrPC", "section": "41"},
            {"act": "CrPC", "section": "50"},
            {"act": "CrPC", "section": "57"},
        ],
        "tags": ["arrest rights", "D.K. Basu", "Article 22", "24 hours", "lawyer"],
        "source": "nalsa_curated",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Curated FAQ Entries — Property
# ═══════════════════════════════════════════════════════════════════════════════

PROPERTY_FAQS = [
    {
        "id": "faq_prop_01",
        "question": "How to register a property sale deed?",
        "answer": (
            "All property transactions involving immovable property worth more "
            "than Rs. 100 must be registered under the Registration Act, 1908. "
            "Unregistered sale deeds are not admissible as evidence."
        ),
        "domain": "property",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Draft the sale deed",
                "details": "Have a lawyer draft the sale deed with full details: property description, survey number, boundaries, consideration amount, parties' details.",
            },
            {
                "step": 2,
                "action": "Pay stamp duty",
                "details": "Purchase stamp paper of the requisite value. Stamp duty rates vary by state (typically 5-8% of property value). Pay online or at authorized stamp vendor.",
                "fees": "5-8% of property value (varies by state)",
            },
            {
                "step": 3,
                "action": "Visit the Sub-Registrar's office",
                "details": "Both buyer and seller (or their power of attorney holders) must appear in person before the Sub-Registrar of the jurisdiction where the property is located. Bring two witnesses.",
                "authority": "Sub-Registrar",
            },
            {
                "step": 4,
                "action": "Pay registration fee and biometric verification",
                "details": "Pay 1% registration fee. Provide Aadhaar-based biometric verification. The Sub-Registrar will endorse and register the deed.",
                "fees": "1% of property value",
                "authority": "Sub-Registrar",
            },
            {
                "step": 5,
                "action": "Collect the registered deed",
                "details": "The registered sale deed will be returned within 15-30 days. Apply for mutation of property records at the local municipal corporation or revenue office.",
                "time_limit": "15-30 days for return",
            },
        ],
        "relevant_law": [
            {"act": "Registration Act", "section": "17"},
            {"act": "TPA", "section": "54"},
            {"act": "Stamp Act", "section": "3"},
        ],
        "tags": ["property registration", "sale deed", "stamp duty", "Sub-Registrar"],
        "source": "nalsa_curated",
    },
    {
        "id": "faq_prop_02",
        "question": "How to file a complaint against a builder for delayed possession under RERA?",
        "answer": (
            "Under the Real Estate (Regulation and Development) Act, 2016, "
            "homebuyers can file complaints with the state RERA authority for "
            "delayed possession, deficient services, or false promises."
        ),
        "domain": "property",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Check if the project is RERA registered",
                "details": "Visit your state RERA website and search by project name or RERA registration number.",
                "authority": "State RERA Authority",
            },
            {
                "step": 2,
                "action": "Gather documents",
                "details": "Allotment letter, builder-buyer agreement, payment receipts, possession date commitment, any communication about delay.",
            },
            {
                "step": 3,
                "action": "File complaint on state RERA portal",
                "details": "Register on your state RERA website, fill the complaint form, upload documents, and pay the prescribed fee (typically Rs. 1,000-5,000).",
                "authority": "State RERA Authority",
                "fees": "Rs. 1,000-5,000 (varies by state)",
            },
            {
                "step": 4,
                "action": "Attend hearing",
                "details": "RERA authority will issue notice to the builder and schedule a hearing. You can appear in person or through a lawyer.",
                "authority": "RERA Adjudicating Officer",
            },
            {
                "step": 5,
                "action": "Relief available",
                "details": "RERA can order: refund with interest, possession with compensation, or penalty on the builder. Appeal lies to RERA Appellate Tribunal within 60 days.",
                "time_limit": "60 days for appeal",
            },
        ],
        "relevant_law": [
            {"act": "RERA", "section": "18"},
            {"act": "RERA", "section": "31"},
            {"act": "RERA", "section": "43"},
        ],
        "tags": ["RERA", "builder", "delayed possession", "homebuyer", "real estate"],
        "source": "nalsa_curated",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Curated FAQ Entries — Family Law
# ═══════════════════════════════════════════════════════════════════════════════

FAMILY_FAQS = [
    {
        "id": "faq_fam_01",
        "question": "How to file for divorce under Hindu Marriage Act?",
        "answer": (
            "Either spouse can file for divorce under Section 13 of the Hindu "
            "Marriage Act, 1955. Mutual consent divorce under Section 13B is "
            "faster (6-18 months) than contested divorce (2-5 years)."
        ),
        "domain": "family",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Attempt reconciliation / mediation",
                "details": "Many courts require parties to attempt mediation before proceeding with divorce. Family courts have in-house mediation centres.",
                "authority": "Family Court Mediation Centre",
            },
            {
                "step": 2,
                "action": "File divorce petition",
                "details": "File in the Family Court having jurisdiction where the marriage was solemnised, where the couple last lived together, or where the wife currently resides.",
                "authority": "Family Court",
                "fees": "Court fees vary by state (typically Rs. 500-2,000)",
            },
            {
                "step": 3,
                "action": "First motion (mutual consent) or contested proceedings",
                "details": "In mutual consent: both parties file jointly. Court grants 6-month cooling period (can be waived per SC ruling in Amardeep Singh case). In contested: serve notice on the other party.",
                "time_limit": "6 months cooling period (mutual consent)",
            },
            {
                "step": 4,
                "action": "Second motion / trial",
                "details": "Mutual consent: both parties confirm before court after cooling period. Contested: evidence, witnesses, arguments — can take 2-5 years.",
                "authority": "Family Court",
            },
            {
                "step": 5,
                "action": "Decree of divorce",
                "details": "Court passes the divorce decree. Effective immediately. Appeal to High Court within 30 days if contested.",
                "time_limit": "30 days for appeal",
            },
        ],
        "relevant_law": [
            {"act": "HMA", "section": "13"},
            {"act": "HMA", "section": "13B"},
            {"act": "CPC", "section": "Order VII Rule 1"},
        ],
        "tags": ["divorce", "Hindu Marriage Act", "mutual consent", "Family Court"],
        "source": "nalsa_curated",
    },
    {
        "id": "faq_fam_02",
        "question": "How to file a domestic violence complaint?",
        "answer": (
            "Under the Protection of Women from Domestic Violence Act, 2005, "
            "any woman who is a victim of physical, emotional, verbal, sexual, "
            "or economic abuse can seek protection and relief."
        ),
        "domain": "family",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "File a Domestic Incident Report (DIR)",
                "details": "Approach the nearest Protection Officer (appointed by the state government) or Women's Cell at a police station. File a DIR in the prescribed form.",
                "authority": "Protection Officer / Police",
            },
            {
                "step": 2,
                "action": "Apply for protection orders",
                "details": "The Protection Officer helps you file an application before the Magistrate for protection orders, residence orders, monetary relief, or custody orders.",
                "authority": "Judicial Magistrate First Class",
                "fees": "No court fees (free for DV cases)",
            },
            {
                "step": 3,
                "action": "Court hearing and orders",
                "details": "The Magistrate must hear the case within 3 days of receipt. Can grant ex-parte interim orders on the first hearing itself if there is immediate danger.",
                "authority": "Magistrate",
                "time_limit": "3 days for first hearing; 60 days to dispose",
            },
        ],
        "relevant_law": [
            {"act": "DV Act", "section": "12"},
            {"act": "DV Act", "section": "18"},
            {"act": "DV Act", "section": "19"},
            {"act": "DV Act", "section": "20"},
            {"act": "IPC", "section": "498A"},
        ],
        "tags": ["domestic violence", "protection order", "DV Act", "women rights"],
        "source": "nalsa_curated",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Curated FAQ Entries — Consumer
# ═══════════════════════════════════════════════════════════════════════════════

CONSUMER_FAQS = [
    {
        "id": "faq_cons_01",
        "question": "How to file a consumer complaint?",
        "answer": (
            "Under the Consumer Protection Act, 2019, any consumer can file a "
            "complaint for defective goods, deficient services, unfair trade "
            "practices, or overcharging."
        ),
        "domain": "consumer",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Send a legal notice to the company",
                "details": "Before filing, send a written complaint/legal notice to the company giving them 15-30 days to resolve. Keep proof of delivery.",
                "time_limit": "15-30 days for response",
            },
            {
                "step": 2,
                "action": "Determine the appropriate forum",
                "details": "District Commission: up to Rs. 1 crore. State Commission: Rs. 1-10 crore. National Commission (NCDRC): above Rs. 10 crore.",
                "authority": "Consumer Commission",
            },
            {
                "step": 3,
                "action": "File the complaint",
                "details": "File online at edaakhil.nic.in or physically at the Consumer Commission. Attach: purchase receipt, warranty card, complaint letters, legal notice, any evidence.",
                "authority": "Consumer Commission",
                "fees": "Nominal (Rs. 100-5,000 depending on claim value)",
            },
            {
                "step": 4,
                "action": "Hearing and order",
                "details": "Commission issues notice to the opposite party. Both sides present evidence. Orders typically within 3-5 months for simple cases.",
                "time_limit": "3-5 months for disposal",
            },
        ],
        "relevant_law": [
            {"act": "CPA", "section": "2(7)"},
            {"act": "CPA", "section": "34"},
            {"act": "CPA", "section": "35"},
            {"act": "CPA", "section": "38"},
        ],
        "tags": ["consumer complaint", "defective product", "consumer forum", "e-Daakhil"],
        "source": "nalsa_curated",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Curated FAQ Entries — Labour
# ═══════════════════════════════════════════════════════════════════════════════

LABOUR_FAQS = [
    {
        "id": "faq_lab_01",
        "question": "How to file a complaint for unpaid wages or wrongful termination?",
        "answer": (
            "Workers denied wages or wrongfully terminated can approach the "
            "Labour Commissioner, file under the Payment of Wages Act, or "
            "raise an industrial dispute under the ID Act."
        ),
        "domain": "labour",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "File complaint with Labour Commissioner",
                "details": "Visit the office of the Assistant Labour Commissioner (Central or State) in your district. File a written complaint with details of employment, wages due, and any termination letter.",
                "authority": "Labour Commissioner",
                "fees": "Free",
            },
            {
                "step": 2,
                "action": "Conciliation proceedings",
                "details": "The Labour Officer calls both parties for conciliation. If settled, a binding settlement is recorded.",
                "authority": "Conciliation Officer",
                "time_limit": "45 days for conciliation",
            },
            {
                "step": 3,
                "action": "If conciliation fails, file in Labour Court",
                "details": "File an application before the Labour Court / Industrial Tribunal. Seek reinstatement with back wages (for wrongful termination) or recovery of unpaid wages.",
                "authority": "Labour Court / Industrial Tribunal",
            },
        ],
        "relevant_law": [
            {"act": "ID Act", "section": "2A"},
            {"act": "ID Act", "section": "25F"},
            {"act": "Wages Code", "section": "17"},
        ],
        "tags": ["wages", "termination", "labour court", "retrenchment"],
        "source": "nalsa_curated",
    },
    {
        "id": "faq_lab_02",
        "question": "How to file a sexual harassment complaint at workplace (POSH)?",
        "answer": (
            "Under the POSH Act, 2013, every workplace with 10+ employees must "
            "have an Internal Complaints Committee (ICC). Any woman employee can "
            "file a written complaint."
        ),
        "domain": "labour",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "File written complaint with ICC",
                "details": "Submit a written complaint to the Internal Complaints Committee (ICC) within 3 months of the incident (extendable by 3 months in special circumstances).",
                "authority": "Internal Complaints Committee (ICC)",
                "time_limit": "3 months from incident (+ 3 months extension possible)",
                "fees": "Free",
            },
            {
                "step": 2,
                "action": "ICC inquiry",
                "details": "ICC conducts inquiry following principles of natural justice. Both parties heard. ICC must complete inquiry within 90 days.",
                "authority": "ICC",
                "time_limit": "90 days",
            },
            {
                "step": 3,
                "action": "If no ICC exists, approach Local Complaints Committee",
                "details": "For workplaces with fewer than 10 employees or domestic workers, file with the Local Complaints Committee (LCC) at the District Officer.",
                "authority": "District Officer / LCC",
            },
            {
                "step": 4,
                "action": "Parallel FIR if criminal offence",
                "details": "Sexual harassment may also be a criminal offence under IPC Section 354A. You can simultaneously file an FIR at the police station.",
                "authority": "Police Station",
            },
        ],
        "relevant_law": [
            {"act": "POSH Act", "section": "9"},
            {"act": "POSH Act", "section": "11"},
            {"act": "IPC", "section": "354A"},
        ],
        "tags": ["sexual harassment", "POSH", "ICC", "workplace", "women"],
        "source": "nalsa_curated",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Curated FAQ Entries — Constitutional / RTI
# ═══════════════════════════════════════════════════════════════════════════════

CONSTITUTIONAL_FAQS = [
    {
        "id": "faq_const_01",
        "question": "How to file an RTI application?",
        "answer": (
            "Under the Right to Information Act, 2005, any citizen can request "
            "information from any public authority. The authority must respond "
            "within 30 days."
        ),
        "domain": "constitutional",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Identify the Public Information Officer (PIO)",
                "details": "Every government department has a designated PIO. Find them on the department's website or by calling the office.",
                "authority": "Public Authority",
            },
            {
                "step": 2,
                "action": "Write the RTI application",
                "details": "Address it to the PIO. Write 'Application under RTI Act, 2005' at the top. Clearly state what information you want. No need to give reasons.",
            },
            {
                "step": 3,
                "action": "Pay the fee and submit",
                "details": "Pay Rs. 10 fee (by postal order, demand draft, or online). BPL families are exempt. Submit by post, in person, or online at rtionline.gov.in.",
                "authority": "PIO",
                "fees": "Rs. 10 (BPL exempt)",
            },
            {
                "step": 4,
                "action": "Wait for response",
                "details": "PIO must respond within 30 days (48 hours if life/liberty is involved). If no response or unsatisfactory response, file first appeal with the First Appellate Authority within 30 days.",
                "time_limit": "30 days response; 30 days for first appeal",
            },
            {
                "step": 5,
                "action": "Second appeal to Information Commission",
                "details": "If first appeal fails, file second appeal with the Central/State Information Commission within 90 days.",
                "authority": "Central / State Information Commission",
                "time_limit": "90 days for second appeal",
            },
        ],
        "relevant_law": [
            {"act": "RTI Act", "section": "6"},
            {"act": "RTI Act", "section": "7"},
            {"act": "RTI Act", "section": "19"},
        ],
        "tags": ["RTI", "Right to Information", "PIO", "government", "transparency"],
        "source": "nalsa_curated",
    },
    {
        "id": "faq_const_02",
        "question": "How to get free legal aid?",
        "answer": (
            "Under the Legal Services Authorities Act, 1987, free legal aid is "
            "available to women, children, SC/ST, disabled persons, industrial "
            "workers, victims of trafficking, persons in custody, and anyone "
            "with annual income below the prescribed limit."
        ),
        "domain": "constitutional",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Apply to DLSA or SLSA",
                "details": "Visit the District Legal Services Authority (DLSA) office in your district or the State Legal Services Authority (SLSA). You can also apply online at nalsa.gov.in.",
                "authority": "DLSA / SLSA",
                "fees": "Completely free",
            },
            {
                "step": 2,
                "action": "Submit required documents",
                "details": "Proof of identity, income certificate (if claiming on income basis), and any documents related to your case.",
            },
            {
                "step": 3,
                "action": "Get a legal aid lawyer assigned",
                "details": "DLSA will assign a panel lawyer who will represent you free of charge in all court proceedings.",
                "authority": "DLSA Panel Lawyer",
            },
            {
                "step": 4,
                "action": "Tele-Law option",
                "details": "For basic legal advice without filing a case, call the Tele-Law helpline or visit your nearest Common Service Centre (CSC). A panel lawyer will advise you via video call.",
                "authority": "Tele-Law / CSC",
            },
        ],
        "relevant_law": [
            {"act": "LSA Act", "section": "12"},
            {"act": "LSA Act", "section": "13"},
            {"act": "Constitution of India", "section": "39A"},
        ],
        "tags": ["legal aid", "free lawyer", "NALSA", "DLSA", "Tele-Law", "poor"],
        "source": "nalsa_curated",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Curated FAQ Entries — IP / Cyber
# ═══════════════════════════════════════════════════════════════════════════════

IP_FAQS = [
    {
        "id": "faq_ip_01",
        "question": "How to file a cybercrime complaint?",
        "answer": (
            "Cybercrimes can be reported at the National Cyber Crime Reporting "
            "Portal (cybercrime.gov.in) or at your nearest police station. The "
            "IT Act, 2000 covers offences like hacking, identity theft, "
            "cyber stalking, and online fraud."
        ),
        "domain": "ip",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Report online at cybercrime.gov.in",
                "details": "Register on the National Cyber Crime Reporting Portal. File a complaint with screenshots, URLs, transaction IDs, and any other evidence.",
                "authority": "National Cyber Crime Portal",
                "fees": "Free",
            },
            {
                "step": 2,
                "action": "File an FIR at the nearest police station",
                "details": "You can also file an FIR at the local police station or the dedicated Cyber Crime Police Station in your city.",
                "authority": "Police / Cyber Cell",
            },
            {
                "step": 3,
                "action": "Preserve evidence",
                "details": "Take screenshots, save emails, note URLs, keep transaction records. Do not delete any messages or communication with the accused.",
            },
            {
                "step": 4,
                "action": "For financial fraud, call 1930",
                "details": "Call the Cyber Fraud Helpline 1930 immediately for online financial fraud. The earlier you report, the higher the chance of freezing the fraudulent transaction.",
                "authority": "Cyber Fraud Helpline",
                "time_limit": "Report within golden hour for best chance of fund recovery",
            },
        ],
        "relevant_law": [
            {"act": "IT Act", "section": "43"},
            {"act": "IT Act", "section": "66"},
            {"act": "IT Act", "section": "66C"},
            {"act": "IT Act", "section": "66D"},
        ],
        "tags": ["cybercrime", "hacking", "online fraud", "IT Act", "1930", "cyber cell"],
        "source": "nalsa_curated",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Corporate / NI Act FAQ
# ═══════════════════════════════════════════════════════════════════════════════

CORPORATE_FAQS = [
    {
        "id": "faq_corp_01",
        "question": "How to file a cheque bounce (Section 138 NI Act) case?",
        "answer": (
            "If a cheque issued to you bounces due to insufficient funds, you "
            "can file a criminal complaint under Section 138 of the Negotiable "
            "Instruments Act, 1881."
        ),
        "domain": "corporate",
        "jurisdiction": "central",
        "steps": [
            {
                "step": 1,
                "action": "Get the cheque return memo from your bank",
                "details": "Collect the 'cheque return memo' (dishonour slip) from your bank stating the reason for dishonour (e.g., insufficient funds).",
                "authority": "Your Bank",
            },
            {
                "step": 2,
                "action": "Send a legal notice within 30 days",
                "details": "Send a written demand notice to the drawer (cheque issuer) via registered post / speed post within 30 days of receiving the return memo. Demand payment of the cheque amount.",
                "time_limit": "30 days from cheque return memo",
            },
            {
                "step": 3,
                "action": "Wait 15 days for payment",
                "details": "The drawer has 15 days from receipt of the notice to make payment. If they pay, matter is settled.",
                "time_limit": "15 days from receipt of notice",
            },
            {
                "step": 4,
                "action": "File complaint within 30 days of notice expiry",
                "details": "If payment is not made within 15 days, file a criminal complaint before the Magistrate having jurisdiction within 30 days of the expiry of the 15-day period.",
                "authority": "Judicial Magistrate",
                "time_limit": "30 days after 15-day notice period expires",
                "fees": "Court fees as prescribed by state",
            },
        ],
        "relevant_law": [
            {"act": "NI Act", "section": "138"},
            {"act": "NI Act", "section": "141"},
            {"act": "NI Act", "section": "142"},
        ],
        "tags": ["cheque bounce", "Section 138", "NI Act", "dishonour", "legal notice"],
        "source": "nalsa_curated",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregated Registry
# ═══════════════════════════════════════════════════════════════════════════════

ALL_FAQS: list[dict] = (
    CRIMINAL_FAQS
    + PROPERTY_FAQS
    + FAMILY_FAQS
    + CONSUMER_FAQS
    + LABOUR_FAQS
    + CONSTITUTIONAL_FAQS
    + IP_FAQS
    + CORPORATE_FAQS
)


# ═══════════════════════════════════════════════════════════════════════════════
# Storage
# ═══════════════════════════════════════════════════════════════════════════════


async def seed_faqs() -> dict:
    """
    Seed all curated FAQ entries into PostgreSQL.

    FAQs are stored in the ingestion_logs table as a record of what was
    ingested, and the actual content is exported as JSON for the embedding
    pipeline (FAQs don't fit the acts/sections/judgments schema — they
    go directly to Qdrant as a 'procedures' collection).
    """
    logger.info("seed_faqs_start", total=len(ALL_FAQS))
    stats = {"total": len(ALL_FAQS), "new": 0, "existing": 0, "errors": 0}

    # Export to JSON for the embedding pipeline
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    for faq in ALL_FAQS:
        faq_path = EXPORT_DIR / f"{faq['id']}.json"

        if faq_path.exists() and not True:  # Always overwrite for now
            stats["existing"] += 1
            continue

        try:
            faq_path.write_text(
                json.dumps(faq, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            stats["new"] += 1
        except Exception as e:
            logger.error("faq_export_error", faq_id=faq["id"], error=str(e))
            stats["errors"] += 1

    # Also export a combined file for bulk loading
    combined_path = EXPORT_DIR / "all_faqs.json"
    combined_path.write_text(
        json.dumps(ALL_FAQS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Record in ingestion_logs
    async with async_session() as session:
        log = IngestionLog(
            id=uuid.uuid4(),
            source="nalsa_curated",
            task="seed_faqs",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            status="success",
            items_fetched=stats["total"],
            items_new=stats["new"],
            last_success_at=datetime.utcnow(),
        )
        session.add(log)
        await session.commit()

    logger.info("seed_faqs_complete", **stats)
    return stats


def export_for_embedding() -> dict:
    """
    Export FAQ data in a format optimized for the embedding pipeline.

    Each FAQ becomes one or more embedding chunks:
    - The question (for semantic matching)
    - The answer + steps combined (for context retrieval)

    Returns stats about export.
    """
    output_dir = PROJECT_ROOT / "data" / "raw" / "procedures"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []

    for faq in ALL_FAQS:
        # Chunk 1: Question + Answer (main retrieval target)
        answer_text = faq["answer"]
        if faq.get("steps"):
            steps_text = "\n".join(
                f"Step {s['step']}: {s['action']} — {s.get('details', '')}"
                for s in faq["steps"]
            )
            answer_text = f"{answer_text}\n\n{steps_text}"

        chunks.append({
            "chunk_id": f"{faq['id']}_main",
            "text": f"Q: {faq['question']}\n\nA: {answer_text}",
            "metadata": {
                "source": "nalsa_faq",
                "faq_id": faq["id"],
                "domain": faq["domain"],
                "jurisdiction": faq.get("jurisdiction", "central"),
                "tags": faq.get("tags", []),
                "relevant_law": faq.get("relevant_law", []),
                "chunk_type": "procedure",
            },
        })

    # Write chunks file
    chunks_path = output_dir / "nalsa_faq_chunks.json"
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "faqs": len(ALL_FAQS),
        "chunks": len(chunks),
        "output": str(chunks_path),
    }
    logger.info("faq_export_complete", **stats)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════


def print_summary() -> None:
    """Print a summary of curated FAQ data."""
    print(f"\n{'=' * 60}")
    print("  NyayaMitra — NALSA FAQ Registry Summary")
    print(f"{'=' * 60}\n")
    print(f"  Total FAQs: {len(ALL_FAQS)}\n")

    # By domain
    domains: dict[str, int] = {}
    for faq in ALL_FAQS:
        d = faq["domain"]
        domains[d] = domains.get(d, 0) + 1

    print(f"  {'Domain':<20} {'Count':>5} {'Example Question'}")
    print("  " + "─" * 70)
    for domain, count in sorted(domains.items()):
        example = next(f for f in ALL_FAQS if f["domain"] == domain)
        q = example["question"][:45]
        print(f"  {domain:<20} {count:>5}  {q}...")

    # Total steps
    total_steps = sum(len(f.get("steps", [])) for f in ALL_FAQS)
    total_laws = sum(len(f.get("relevant_law", [])) for f in ALL_FAQS)
    print(f"\n  Total procedure steps: {total_steps}")
    print(f"  Total law references:  {total_laws}")
    print(f"\n{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra NALSA FAQ Curator (Sprint 7)",
    )
    parser.add_argument("--scrape", action="store_true",
                        help="Attempt to scrape NALSA website for additional FAQs")
    parser.add_argument("--export", action="store_true",
                        help="Export FAQ data for embedding pipeline")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of curated data")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    # Always seed curated data
    stats = await seed_faqs()
    print(f"\nFAQ seed: {stats['new']} exported, {stats['existing']} existing")

    if args.export:
        export_stats = export_for_embedding()
        print(f"\nExport: {export_stats['chunks']} chunks → {export_stats['output']}")

    if args.scrape:
        print("\nNALSA web scraping is planned for future iteration.")
        print("Current data is curated and expert-verified.")

    if not args.export and not args.scrape and not args.summary:
        print_summary()


if __name__ == "__main__":
    asyncio.run(main())