"""
NyayaMitra — Procedural Knowledge Base Builder (Sprint 7).

Builds a comprehensive set of 50 step-by-step legal procedure guides
covering all 7+ legal domains. These guides answer "how do I..." questions
that citizens commonly ask.

Data sources:
    1. data/scrapers/nalsa_scraper.py — 14 curated FAQ entries (Sprint 7)
    2. data/procedures/procedures_data.json — 36 additional procedures (this file generates them)

Combined output: 50 procedures → chunked for Qdrant 'procedures' collection.

Each procedure has:
    - question: The user-facing question
    - answer: Plain-language explanation
    - domain: Legal domain classification
    - jurisdiction: central or state-specific
    - steps: [{step, action, details, authority, time_limit, fees}]
    - relevant_law: [{act, section}]
    - tags: Search keywords

Usage:
    # Build all procedures and export for embedding
    python -m data.procedures.procedure_builder

    # Export only (don't rebuild)
    python -m data.procedures.procedure_builder --export-only

    # Print summary
    python -m data.procedures.procedure_builder --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCEDURES_JSON = Path(__file__).parent / "procedures_data.json"
CHUNKS_OUTPUT = PROJECT_ROOT / "data" / "raw" / "procedures" / "all_procedures_chunks.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Additional Procedures (beyond the 14 in nalsa_scraper.py)
#
# These 36 entries bring the total to 50 procedures across all domains.
# ═══════════════════════════════════════════════════════════════════════════════

ADDITIONAL_PROCEDURES: list[dict] = [
    # ── Criminal (3 more → total 6 with NALSA's 3) ───────────────────────
    {
        "id": "proc_crim_04",
        "question": "How to file a private criminal complaint before a Magistrate?",
        "answer": (
            "If the police refuse to act or the offence is non-cognizable, you can "
            "file a private complaint directly before the Judicial Magistrate under "
            "Section 200 CrPC (Section 223 BNSS)."
        ),
        "domain": "criminal",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Draft the complaint", "details": "Write a complaint stating facts of the offence, name and address of the accused, witnesses, and evidence. Include your name, address, and sign it."},
            {"step": 2, "action": "File before the Magistrate", "details": "Submit the complaint before the Judicial Magistrate First Class (JMFC) having jurisdiction. The Magistrate examines you on oath under Section 200 CrPC.", "authority": "JMFC", "fees": "Court fees as per state rules (Rs. 50-500)"},
            {"step": 3, "action": "Magistrate orders inquiry or investigation", "details": "The Magistrate may order inquiry under Section 202, direct police investigation under Section 156(3), or take cognizance and issue process (summons/warrant) to the accused.", "authority": "Magistrate"},
            {"step": 4, "action": "Trial proceedings", "details": "If cognizance is taken, the accused is summoned. Trial proceeds with examination of complainant's witnesses, cross-examination, and defence evidence.", "authority": "Magistrate"},
        ],
        "relevant_law": [
            {"act": "CrPC", "section": "200"},
            {"act": "CrPC", "section": "202"},
            {"act": "CrPC", "section": "156(3)"},
        ],
        "tags": ["private complaint", "Magistrate", "Section 200", "non-cognizable"],
        "source": "curated",
    },
    {
        "id": "proc_crim_05",
        "question": "How to get anticipatory bail?",
        "answer": (
            "Anticipatory bail under Section 438 CrPC allows a person to seek bail "
            "in anticipation of arrest. It must be filed before the Sessions Court "
            "or High Court."
        ),
        "domain": "criminal",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Engage a lawyer immediately", "details": "Anticipatory bail requires legal representation. Contact a lawyer or apply for free legal aid through DLSA.", "authority": "DLSA / Advocate"},
            {"step": 2, "action": "File application before Sessions Court or High Court", "details": "Application under Section 438 CrPC with an affidavit explaining the facts, fear of arrest, and grounds for bail.", "authority": "Sessions Court / High Court", "fees": "Court fees (Rs. 500-2,000 depending on state)"},
            {"step": 3, "action": "Court hearing", "details": "Court may grant interim protection immediately and issue notice to the prosecution. Full hearing follows where court considers nature of offence, applicant's antecedents, and likelihood of fleeing.", "authority": "Court"},
            {"step": 4, "action": "Conditions and compliance", "details": "If granted, court imposes conditions — surrender passport, mark attendance at police station, not leave jurisdiction without permission. Breach leads to cancellation.", "authority": "Court / Police"},
        ],
        "relevant_law": [
            {"act": "CrPC", "section": "438"},
            {"act": "BNSS", "section": "482"},
        ],
        "tags": ["anticipatory bail", "Section 438", "pre-arrest bail", "Sessions Court"],
        "source": "curated",
    },
    {
        "id": "proc_crim_06",
        "question": "How to file a complaint for online harassment or cyber stalking?",
        "answer": (
            "Online harassment and cyber stalking are punishable under the IT Act "
            "and IPC/BNS. You can report at cybercrime.gov.in and also file an FIR."
        ),
        "domain": "criminal",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Preserve all evidence", "details": "Take screenshots of harassing messages, emails, social media posts. Note usernames, URLs, timestamps. Do not delete anything."},
            {"step": 2, "action": "Report on cybercrime.gov.in", "details": "File a complaint on the National Cyber Crime Reporting Portal with all evidence attached.", "authority": "Cyber Crime Portal", "fees": "Free"},
            {"step": 3, "action": "File FIR at police station", "details": "Visit the nearest police station or Cyber Crime Police Station. File FIR under Section 354D IPC (stalking), Section 509 IPC (word/gesture to insult modesty), and Section 67 IT Act.", "authority": "Police / Cyber Cell"},
            {"step": 4, "action": "Apply for protection", "details": "If the harassment is severe, apply to the Magistrate for a protection order. You can also seek removal of content by reporting to the social media platform.", "authority": "Magistrate / Platform"},
        ],
        "relevant_law": [
            {"act": "IT Act", "section": "66A"},
            {"act": "IT Act", "section": "67"},
            {"act": "IPC", "section": "354D"},
            {"act": "IPC", "section": "509"},
        ],
        "tags": ["cyber stalking", "online harassment", "IT Act", "cyber crime"],
        "source": "curated",
    },

    # ── Property (3 more → total 5 with NALSA's 2) ───────────────────────
    {
        "id": "proc_prop_03",
        "question": "How to evict a tenant who refuses to vacate?",
        "answer": (
            "Eviction proceedings depend on whether the tenancy is governed by state "
            "Rent Control Act or the Transfer of Property Act. The landlord must "
            "follow due process — self-help eviction is illegal."
        ),
        "domain": "property",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Serve a legal notice to vacate", "details": "Send a notice under Section 106 TPA giving the tenant 15 days (monthly tenancy) or 6 months (yearly tenancy) to vacate. Send by registered post with AD.", "time_limit": "15 days to 6 months depending on tenancy type"},
            {"step": 2, "action": "File eviction suit", "details": "If tenant doesn't vacate after notice period, file an eviction petition before the Rent Controller (if Rent Control Act applies) or a civil suit before the Civil Court.", "authority": "Rent Controller / Civil Court", "fees": "Court fees based on annual rent"},
            {"step": 3, "action": "Court proceedings", "details": "Court issues summons to tenant. Both parties present evidence. Grounds for eviction: non-payment of rent, subletting, damage to property, personal need of landlord, etc.", "authority": "Court"},
            {"step": 4, "action": "Execution of eviction order", "details": "If court orders eviction, and tenant still refuses, apply for execution of decree. Court bailiff will enforce the eviction order.", "authority": "Court Bailiff"},
        ],
        "relevant_law": [
            {"act": "TPA", "section": "106"},
            {"act": "TPA", "section": "111"},
            {"act": "CPC", "section": "Order XII Rule 6"},
        ],
        "tags": ["eviction", "tenant", "landlord", "rent control", "vacate"],
        "source": "curated",
    },
    {
        "id": "proc_prop_04",
        "question": "How to file a mutation application for inherited property?",
        "answer": (
            "After inheriting property (through will or succession), you need to get "
            "the revenue records updated in your name through a mutation application "
            "at the local tehsil/municipal office."
        ),
        "domain": "property",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Gather required documents", "details": "Death certificate of the deceased, succession certificate / probate / legal heir certificate, original property documents, identity proof of all legal heirs, no objection from other heirs (if applicable)."},
            {"step": 2, "action": "Apply at tehsil / municipal office", "details": "Submit mutation application at the Tehsildar's office (rural) or Municipal Corporation (urban) along with documents and prescribed fee.", "authority": "Tehsildar / Municipal Corporation", "fees": "Rs. 100-1,000 (varies by state)"},
            {"step": 3, "action": "Verification and hearing", "details": "Revenue officer verifies documents, publishes notice for objections (15-30 days), conducts spot inspection if needed, and holds hearing.", "authority": "Revenue Officer", "time_limit": "30-90 days"},
            {"step": 4, "action": "Mutation order", "details": "If no valid objection, revenue records are updated in your name. Collect the mutated extract (khata/patta/7/12 extract depending on state).", "authority": "Tehsildar"},
        ],
        "relevant_law": [
            {"act": "HSA", "section": "8"},
            {"act": "ISA", "section": "57"},
        ],
        "tags": ["mutation", "inheritance", "property transfer", "succession", "tehsil", "khata"],
        "source": "curated",
    },
    {
        "id": "proc_prop_05",
        "question": "How to apply for a succession certificate?",
        "answer": (
            "A succession certificate under the Indian Succession Act, 1925 is "
            "required to establish the legal right of an heir to the debts and "
            "securities of a deceased person."
        ),
        "domain": "property",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "File petition in District Court", "details": "File a petition under Section 372 of the Indian Succession Act before the District Court having jurisdiction where the deceased ordinarily resided.", "authority": "District Court", "fees": "Court fees based on value of estate (varies by state)"},
            {"step": 2, "action": "Court publishes notice", "details": "Court publishes notice in newspaper giving 45 days for objections from any person.", "time_limit": "45 days for objections"},
            {"step": 3, "action": "Hearing and order", "details": "If no objection, court grants succession certificate specifying debts and securities the heir is entitled to collect.", "authority": "District Court"},
            {"step": 4, "action": "Use certificate for asset transfer", "details": "Present the succession certificate to banks, mutual fund houses, insurance companies, etc. to claim the deceased's assets.", "authority": "Banks / Financial institutions"},
        ],
        "relevant_law": [
            {"act": "ISA", "section": "372"},
            {"act": "ISA", "section": "373"},
            {"act": "ISA", "section": "381"},
        ],
        "tags": ["succession certificate", "inheritance", "death", "legal heir", "bank"],
        "source": "curated",
    },

    # ── Family (3 more → total 5 with NALSA's 2) ─────────────────────────
    {
        "id": "proc_fam_03",
        "question": "How to claim maintenance under Section 125 CrPC?",
        "answer": (
            "Section 125 CrPC provides for maintenance to wives, children, and "
            "parents who are unable to maintain themselves. It is a quick remedy "
            "available in the Magistrate's court."
        ),
        "domain": "family",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "File application under Section 125 CrPC", "details": "File before the Magistrate of the area where the applicant resides or where the respondent resides or works. No court fee in most states.", "authority": "Judicial Magistrate First Class", "fees": "Free or minimal (Rs. 50-100)"},
            {"step": 2, "action": "Court issues notice", "details": "Magistrate issues notice to the respondent (husband/father/son) to appear and show cause why maintenance should not be ordered.", "authority": "Magistrate"},
            {"step": 3, "action": "Interim maintenance", "details": "Court can order interim maintenance pending final hearing if the applicant's need is established prima facie.", "authority": "Magistrate"},
            {"step": 4, "action": "Final order", "details": "After hearing both sides, Magistrate fixes monthly maintenance amount based on respondent's income, applicant's needs, and standard of living. Non-payment can lead to arrest.", "authority": "Magistrate", "time_limit": "Should be decided within 60 days (as per amendment)"},
        ],
        "relevant_law": [
            {"act": "CrPC", "section": "125"},
            {"act": "CrPC", "section": "128"},
            {"act": "BNSS", "section": "144"},
        ],
        "tags": ["maintenance", "Section 125", "wife", "children", "alimony"],
        "source": "curated",
    },
    {
        "id": "proc_fam_04",
        "question": "How to get a child custody order?",
        "answer": (
            "Child custody disputes are decided by the Family Court based on the "
            "welfare of the child as the paramount consideration, not the rights "
            "of the parents."
        ),
        "domain": "family",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "File custody petition in Family Court", "details": "File under Section 26 of HMA (for Hindus) or the Guardians and Wards Act, 1890 (for all). The court where the child ordinarily resides has jurisdiction.", "authority": "Family Court", "fees": "Court fees (Rs. 500-2,000)"},
            {"step": 2, "action": "Interim custody / visitation order", "details": "Court can grant interim custody to one parent and visitation rights to the other while the case is pending.", "authority": "Family Court"},
            {"step": 3, "action": "Court welfare report", "details": "Court may appoint a counsellor to assess the child's living conditions with each parent and submit a welfare report.", "authority": "Family Court Counsellor"},
            {"step": 4, "action": "Final custody order", "details": "Court decides based on: child's age and wishes (if old enough), emotional and educational needs, capacity of each parent, any history of abuse. Mother usually preferred for children under 5.", "authority": "Family Court"},
        ],
        "relevant_law": [
            {"act": "HMA", "section": "26"},
            {"act": "GWA", "section": "7"},
            {"act": "GWA", "section": "17"},
        ],
        "tags": ["child custody", "guardianship", "visitation", "family court", "welfare"],
        "source": "curated",
    },
    {
        "id": "proc_fam_05",
        "question": "How to obtain a legal heir certificate?",
        "answer": (
            "A legal heir certificate establishes who the legal heirs of a deceased "
            "person are. It is issued by the Tehsildar (revenue authority) and is "
            "needed for claiming pension, insurance, bank deposits, etc."
        ),
        "domain": "family",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Apply at Tehsildar's office", "details": "Submit application with: death certificate, Aadhaar of deceased and all legal heirs, ration card, self-declaration listing all legal heirs.", "authority": "Tehsildar / Revenue Office", "fees": "Rs. 50-200"},
            {"step": 2, "action": "Verification", "details": "Revenue officer may conduct local enquiry and publish notice for objections (7-15 days).", "authority": "Revenue Inspector", "time_limit": "15-30 days"},
            {"step": 3, "action": "Certificate issued", "details": "If no objection, Tehsildar issues legal heir certificate listing all legal heirs and their relationship to the deceased.", "authority": "Tehsildar"},
        ],
        "relevant_law": [
            {"act": "HSA", "section": "8"},
            {"act": "ISA", "section": "57"},
        ],
        "tags": ["legal heir certificate", "death", "pension", "insurance", "succession"],
        "source": "curated",
    },

    # ── Consumer (3 more → total 4 with NALSA's 1) ───────────────────────
    {
        "id": "proc_cons_02",
        "question": "How to claim insurance and what to do if claim is rejected?",
        "answer": (
            "If your insurance claim is rejected or delayed unreasonably, you can "
            "approach the Insurance Ombudsman or file a consumer complaint."
        ),
        "domain": "consumer",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "File claim with the insurance company", "details": "Submit all required documents (claim form, medical records/bills, FIR for accident/theft, death certificate for life insurance) within the policy's notification period.", "authority": "Insurance Company", "time_limit": "As per policy (usually 30-90 days for intimation)"},
            {"step": 2, "action": "If rejected, send written complaint to insurer", "details": "Write to the Grievance Redressal Officer of the insurance company. They must respond within 15 days.", "time_limit": "15 days for response"},
            {"step": 3, "action": "Approach Insurance Ombudsman", "details": "If unsatisfied, file complaint with the Insurance Ombudsman (claims up to Rs. 30 lakh for life, Rs. 20 lakh for general). Free of charge. Decision is binding on insurer.", "authority": "Insurance Ombudsman", "fees": "Free"},
            {"step": 4, "action": "File consumer complaint", "details": "Alternatively, file a complaint before the Consumer Commission for deficiency in service. Can claim compensation beyond just the claim amount.", "authority": "Consumer Commission"},
        ],
        "relevant_law": [
            {"act": "CPA", "section": "2(7)"},
            {"act": "CPA", "section": "35"},
        ],
        "tags": ["insurance claim", "rejected claim", "ombudsman", "health insurance", "life insurance"],
        "source": "curated",
    },
    {
        "id": "proc_cons_03",
        "question": "How to file a complaint for medical negligence?",
        "answer": (
            "Medical negligence by a hospital or doctor can be complained about "
            "to the medical council, consumer forum, or through a criminal complaint."
        ),
        "domain": "consumer",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Collect medical records", "details": "Under Section 1.3 of Indian Medical Council Regulations, you have the right to your complete medical records. Request copies of all prescriptions, reports, discharge summary, operation notes."},
            {"step": 2, "action": "Get expert medical opinion", "details": "Consult another doctor to get an independent opinion on whether there was negligence or deviation from standard medical practice."},
            {"step": 3, "action": "File complaint with State Medical Council", "details": "Lodge complaint with the State Medical Council against the doctor. Council can suspend or revoke their licence.", "authority": "State Medical Council"},
            {"step": 4, "action": "File consumer complaint", "details": "File before the Consumer Commission for compensation (medical services are 'services' under CPA 2019). Attach expert opinion, medical records, and bills.", "authority": "Consumer Commission", "fees": "Nominal court fee"},
            {"step": 5, "action": "Criminal complaint if death occurred", "details": "If patient died due to negligence, file FIR under Section 304A IPC (death by negligence) or Section 338 IPC (grievous hurt by negligence).", "authority": "Police"},
        ],
        "relevant_law": [
            {"act": "CPA", "section": "2(42)"},
            {"act": "IPC", "section": "304A"},
            {"act": "IPC", "section": "338"},
        ],
        "tags": ["medical negligence", "hospital", "doctor", "malpractice", "compensation"],
        "source": "curated",
    },
    {
        "id": "proc_cons_04",
        "question": "How to get a refund for a defective product bought online?",
        "answer": (
            "E-commerce purchases are covered under the Consumer Protection Act, 2019 "
            "and the Consumer Protection (E-Commerce) Rules, 2020. You can claim "
            "refund, replacement, or compensation."
        ),
        "domain": "consumer",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Request return/refund on the platform", "details": "Use the e-commerce platform's return/refund mechanism within the return window. Document the defect with photos/videos."},
            {"step": 2, "action": "Email the grievance officer", "details": "Under E-Commerce Rules, every platform must have a grievance officer who must acknowledge within 48 hours and resolve within 1 month.", "time_limit": "48 hours acknowledgement, 1 month resolution"},
            {"step": 3, "action": "File on National Consumer Helpline", "details": "Call 1915 or file at consumerhelpline.gov.in. They mediate with the company for resolution.", "authority": "National Consumer Helpline", "fees": "Free"},
            {"step": 4, "action": "File consumer complaint", "details": "If unresolved, file at edaakhil.nic.in against both the seller and the e-commerce platform (they are jointly liable under CPA 2019).", "authority": "Consumer Commission", "fees": "Rs. 100-500"},
        ],
        "relevant_law": [
            {"act": "CPA", "section": "2(7)"},
            {"act": "CPA", "section": "84"},
            {"act": "CPA", "section": "87"},
        ],
        "tags": ["refund", "online shopping", "e-commerce", "defective product", "Amazon", "Flipkart"],
        "source": "curated",
    },

    # ── Labour (2 more → total 4 with NALSA's 2) ─────────────────────────
    {
        "id": "proc_lab_03",
        "question": "How to claim EPF (Provident Fund) withdrawal?",
        "answer": (
            "Employees can withdraw PF after leaving a job. Full withdrawal is "
            "allowed after 2 months of unemployment or at retirement (58 years). "
            "Partial withdrawal is allowed for specific purposes."
        ),
        "domain": "labour",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Check your UAN and link Aadhaar", "details": "Log in to unifiedportal-mem.epfindia.gov.in. Ensure your UAN (Universal Account Number) is active and Aadhaar, PAN, and bank account are linked and verified."},
            {"step": 2, "action": "Submit online claim", "details": "Go to Online Services > Claim (Form 19 for full withdrawal, Form 31 for partial). Select claim type, enter bank details, and submit.", "authority": "EPFO Online Portal", "fees": "Free"},
            {"step": 3, "action": "Employer attestation (if applicable)", "details": "If filing through employer, the claim needs employer's digital attestation. If UAN-Aadhaar linked, you can file directly without employer.", "authority": "Employer / EPFO"},
            {"step": 4, "action": "Track and receive", "details": "Track status on the EPFO portal. Processing takes 10-20 days. Amount credited directly to linked bank account.", "time_limit": "10-20 days processing"},
        ],
        "relevant_law": [
            {"act": "EPF Act", "section": "7"},
            {"act": "SS Code", "section": "16"},
        ],
        "tags": ["PF withdrawal", "EPF", "provident fund", "EPFO", "UAN", "retirement"],
        "source": "curated",
    },
    {
        "id": "proc_lab_04",
        "question": "How to claim maternity benefit?",
        "answer": (
            "Under the Maternity Benefit Act, 1961 (and Code on Social Security, 2020), "
            "women employees are entitled to 26 weeks of paid maternity leave "
            "for the first two children."
        ),
        "domain": "labour",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Notify employer in writing", "details": "Inform your employer about the expected delivery date and your intention to take maternity leave. Provide medical certificate.", "time_limit": "Notify at least 8 weeks before expected delivery"},
            {"step": 2, "action": "Employer grants leave and pay", "details": "Employer must grant 26 weeks of leave (8 weeks before delivery + 18 weeks after) at average daily wage. Cannot be denied or terminated during this period.", "authority": "Employer"},
            {"step": 3, "action": "If employer refuses", "details": "File complaint with the Inspector appointed under the Maternity Benefit Act. Inspector can order the employer to pay.", "authority": "Inspector under MB Act"},
            {"step": 4, "action": "Claim through ESI (if applicable)", "details": "If covered under ESI scheme, claim maternity benefit through ESI by submitting Form 19 at the local ESI office.", "authority": "ESIC", "fees": "Free"},
        ],
        "relevant_law": [
            {"act": "Maternity Benefit Act", "section": "5"},
            {"act": "Maternity Benefit Act", "section": "12"},
            {"act": "ESI Act", "section": "50"},
        ],
        "tags": ["maternity leave", "pregnancy", "maternity benefit", "26 weeks", "ESI"],
        "source": "curated",
    },

    # ── Constitutional (3 more → total 5 with NALSA's 2) ─────────────────
    {
        "id": "proc_const_03",
        "question": "How to file a PIL (Public Interest Litigation)?",
        "answer": (
            "A PIL can be filed directly before the Supreme Court (Article 32) "
            "or High Court (Article 226) for enforcement of fundamental rights "
            "or public interest issues."
        ),
        "domain": "constitutional",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Identify the public interest issue", "details": "PIL must concern a matter of public interest — not a private grievance. Environmental damage, violation of fundamental rights of a group, government inaction on public health, etc."},
            {"step": 2, "action": "Draft the PIL petition", "details": "Include: facts of the public interest issue, fundamental rights violated, prayer/relief sought, supporting documents. A letter to the Chief Justice can also be treated as PIL."},
            {"step": 3, "action": "File before High Court or Supreme Court", "details": "File before the High Court of the state (Article 226) or directly before the Supreme Court (Article 32) if fundamental rights are involved. Court fees are nominal for PILs.", "authority": "High Court / Supreme Court", "fees": "Nominal (Rs. 50-500)"},
            {"step": 4, "action": "Court hearing", "details": "Court examines whether the petition raises genuine public interest. If admitted, court issues notice to respondents (usually government bodies) and monitors compliance with orders.", "authority": "Court"},
        ],
        "relevant_law": [
            {"act": "Constitution of India", "section": "32"},
            {"act": "Constitution of India", "section": "226"},
        ],
        "tags": ["PIL", "public interest litigation", "Article 32", "Article 226", "writ petition"],
        "source": "curated",
    },
    {
        "id": "proc_const_04",
        "question": "How to file a complaint with the Lokpal / Lokayukta?",
        "answer": (
            "Complaints about corruption by public servants can be filed with the "
            "Lokpal (central) or Lokayukta (state) under the Lokpal and Lokayuktas "
            "Act, 2013."
        ),
        "domain": "constitutional",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Prepare the complaint", "details": "Write a complaint with: name of the public servant, their designation, nature of corruption/misconduct, supporting evidence (documents, recordings, etc.)."},
            {"step": 2, "action": "File online or by post", "details": "File at lokpal.gov.in for central officials, or the state Lokayukta website for state officials. Can also send by registered post.", "authority": "Lokpal / Lokayukta", "fees": "Free"},
            {"step": 3, "action": "Preliminary inquiry", "details": "Lokpal conducts preliminary inquiry within 60 days. If prima facie case exists, orders investigation by CBI or other agency.", "authority": "Lokpal", "time_limit": "60 days for preliminary inquiry"},
            {"step": 4, "action": "Investigation and prosecution", "details": "Investigation must be completed within 6 months (extendable by 6 months). Special court trials corruption cases.", "time_limit": "6 months investigation, extendable by 6 months"},
        ],
        "relevant_law": [
            {"act": "Lokpal Act", "section": "14"},
            {"act": "Lokpal Act", "section": "20"},
            {"act": "PCA", "section": "7"},
        ],
        "tags": ["Lokpal", "Lokayukta", "corruption", "public servant", "bribery"],
        "source": "curated",
    },
    {
        "id": "proc_const_05",
        "question": "How to file a human rights complaint with NHRC?",
        "answer": (
            "The National Human Rights Commission (NHRC) investigates complaints "
            "of human rights violations by public servants. File within one year "
            "of the violation."
        ),
        "domain": "constitutional",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "File complaint online or by post", "details": "File at hrcnet.nic.in or send by post to NHRC, Manav Adhikar Bhawan, Block-C, GPO Complex, New Delhi - 110023. Include detailed facts, names of violators, evidence.", "authority": "NHRC", "fees": "Free", "time_limit": "Within 1 year of violation"},
            {"step": 2, "action": "NHRC inquiry", "details": "NHRC examines the complaint. May call for report from government authority, conduct investigation through its own team, or direct state authority to investigate.", "authority": "NHRC"},
            {"step": 3, "action": "Recommendations", "details": "NHRC can recommend compensation to the victim, prosecution of the violator, or policy changes. While not binding, government must respond within one month.", "authority": "NHRC / Government", "time_limit": "Government must respond within 1 month"},
        ],
        "relevant_law": [
            {"act": "Constitution of India", "section": "21"},
        ],
        "tags": ["NHRC", "human rights", "custodial death", "police brutality", "fundamental rights"],
        "source": "curated",
    },

    # ── Corporate (2 more → total 3 with NALSA's 1) ──────────────────────
    {
        "id": "proc_corp_02",
        "question": "How to file for insolvency under IBC?",
        "answer": (
            "Under the Insolvency and Bankruptcy Code, 2016, creditors can initiate "
            "insolvency proceedings against a defaulting company before the NCLT "
            "if the default is Rs. 1 crore or more."
        ),
        "domain": "corporate",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "File application before NCLT", "details": "Financial creditor files under Section 7, operational creditor under Section 9. Include proof of default, demand notice (for operational creditors), and proposed insolvency resolution professional.", "authority": "NCLT", "fees": "Rs. 25,000 (for financial creditor)"},
            {"step": 2, "action": "NCLT admits or rejects", "details": "NCLT must admit or reject within 14 days. If admitted, moratorium is imposed — no suits, no recovery, no transfer of assets.", "authority": "NCLT", "time_limit": "14 days for admission"},
            {"step": 3, "action": "Corporate Insolvency Resolution Process (CIRP)", "details": "IRP/RP takes over management. Committee of Creditors formed. Resolution plan invited. CIRP must complete within 330 days.", "authority": "Resolution Professional / CoC", "time_limit": "330 days maximum"},
            {"step": 4, "action": "Resolution or liquidation", "details": "If resolution plan approved by 66% of CoC and NCLT, company continues under new management. If no plan, company goes into liquidation.", "authority": "NCLT"},
        ],
        "relevant_law": [
            {"act": "IBC", "section": "7"},
            {"act": "IBC", "section": "9"},
            {"act": "IBC", "section": "12"},
        ],
        "tags": ["IBC", "insolvency", "bankruptcy", "NCLT", "CIRP", "default"],
        "source": "curated",
    },
    {
        "id": "proc_corp_03",
        "question": "How to register a trademark?",
        "answer": (
            "Trademark registration protects your brand name, logo, or slogan. "
            "Register online at ipindiaonline.gov.in. Registration gives you "
            "exclusive right to use the mark for 10 years (renewable)."
        ),
        "domain": "ip",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Conduct trademark search", "details": "Search the Trade Marks Registry at ipindiaonline.gov.in to check if your desired mark is already registered or applied for.", "authority": "Trade Marks Registry"},
            {"step": 2, "action": "File application online", "details": "File Form TM-A at ipindiaonline.gov.in. Choose the right class (Nice Classification). Individual applicants get 50% fee concession.", "authority": "Trade Marks Registry", "fees": "Rs. 4,500 (individual) / Rs. 9,000 (others) per class"},
            {"step": 3, "action": "Examination and objections", "details": "Examiner reviews the application. May raise objections (similar existing marks, descriptive marks). You must respond within 30 days. Hearing if needed.", "authority": "Trade Marks Examiner", "time_limit": "30 days to respond to objections"},
            {"step": 4, "action": "Publication and opposition", "details": "If accepted, mark is published in Trade Marks Journal for 4 months. Anyone can oppose during this period.", "time_limit": "4 months opposition period"},
            {"step": 5, "action": "Registration certificate", "details": "If no opposition (or opposition fails), registration certificate is issued. Valid for 10 years from filing date, renewable.", "authority": "Trade Marks Registry", "time_limit": "10 years (renewable)"},
        ],
        "relevant_law": [
            {"act": "Trade Marks Act", "section": "18"},
            {"act": "Trade Marks Act", "section": "25"},
        ],
        "tags": ["trademark", "brand registration", "logo", "IP", "intellectual property"],
        "source": "curated",
    },

    # ── Environmental (2 new) ─────────────────────────────────────────────
    {
        "id": "proc_env_01",
        "question": "How to file a complaint with the National Green Tribunal?",
        "answer": (
            "The NGT hears cases related to environmental protection, forest "
            "conservation, and pollution. Any person can file within 6 months "
            "of the cause of action (extendable by 60 days)."
        ),
        "domain": "environmental",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Identify the environmental violation", "details": "Illegal construction in eco-sensitive zone, industrial pollution, illegal tree felling, groundwater contamination, noise pollution, etc."},
            {"step": 2, "action": "File application at NGT", "details": "File at the NGT bench having jurisdiction (principal bench in Delhi, circuit benches in Bhopal, Pune, Kolkata, Chennai). No lawyer mandatory — you can appear in person.", "authority": "National Green Tribunal", "fees": "Rs. 1,000 for application", "time_limit": "Within 6 months of cause of action"},
            {"step": 3, "action": "Hearing and directions", "details": "NGT issues notice to polluter/violator. Can order interim relief (stop construction, close factory). Final order with compensation.", "authority": "NGT"},
        ],
        "relevant_law": [
            {"act": "NGT Act", "section": "14"},
            {"act": "NGT Act", "section": "15"},
            {"act": "EPA", "section": "19"},
        ],
        "tags": ["NGT", "pollution", "environment", "green tribunal", "forest", "eco-sensitive"],
        "source": "curated",
    },
    {
        "id": "proc_env_02",
        "question": "How to file a pollution complaint against a factory or construction site?",
        "answer": (
            "Pollution complaints can be filed with the State Pollution Control Board "
            "(SPCB), local municipal corporation, or the NGT."
        ),
        "domain": "environmental",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Document the pollution", "details": "Take photos/videos of the pollution source. Note the type (air, water, noise, solid waste), location, time, and impact on the area."},
            {"step": 2, "action": "Complain to SPCB/CPCB", "details": "File complaint online at the state PCB website or at cpcb.nic.in. Include evidence and location details.", "authority": "State Pollution Control Board", "fees": "Free"},
            {"step": 3, "action": "Complain to Municipal Corporation", "details": "For construction noise, garbage burning, or local nuisance, file complaint with the municipal corporation. They can issue closure/stop-work notice.", "authority": "Municipal Corporation"},
            {"step": 4, "action": "File at NGT if no action", "details": "If SPCB/municipal corporation takes no action within 30 days, file application before the National Green Tribunal.", "authority": "NGT", "time_limit": "If no action in 30 days"},
        ],
        "relevant_law": [
            {"act": "EPA", "section": "19"},
            {"act": "Water Act", "section": "24"},
            {"act": "Air Act", "section": "22"},
        ],
        "tags": ["pollution", "factory", "noise", "SPCB", "CPCB", "environment"],
        "source": "curated",
    },

    # ── Motor Accident / Insurance (2 new) ────────────────────────────────
    {
        "id": "proc_misc_01",
        "question": "How to file a motor accident claim?",
        "answer": (
            "Victims of road accidents (or their families) can claim compensation "
            "from the Motor Accident Claims Tribunal (MACT) under the Motor "
            "Vehicles Act, 1988."
        ),
        "domain": "consumer",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "File FIR and get accident report", "details": "Ensure FIR is filed at the nearest police station. Obtain a copy of the FIR and the Motor Vehicle Accident Report."},
            {"step": 2, "action": "Gather medical records", "details": "Collect all medical bills, treatment records, disability certificate (if permanent disability), and death certificate (if fatal accident)."},
            {"step": 3, "action": "File claim petition at MACT", "details": "File before the MACT having jurisdiction (where the accident occurred or where the claimant resides). Claim against the vehicle owner, driver, and insurance company.", "authority": "Motor Accident Claims Tribunal", "fees": "No court fee for accident victims"},
            {"step": 4, "action": "Tribunal hearing and award", "details": "MACT assesses compensation based on: age, income, nature of injury/death, dependents. Award includes medical expenses + loss of income + pain and suffering. Insurance company pays.", "authority": "MACT"},
        ],
        "relevant_law": [
            {"act": "Motor Vehicles Act", "section": "166"},
            {"act": "Motor Vehicles Act", "section": "163A"},
        ],
        "tags": ["motor accident", "road accident", "MACT", "compensation", "insurance claim"],
        "source": "curated",
    },
    {
        "id": "proc_misc_02",
        "question": "How to apply for a passport and what to do if it is delayed or rejected?",
        "answer": (
            "Passport applications are filed online at passportindia.gov.in. If "
            "delayed beyond 30 days (normal) or 7 days (tatkal), you can file "
            "grievances or approach the court."
        ),
        "domain": "constitutional",
        "jurisdiction": "central",
        "steps": [
            {"step": 1, "action": "Apply online", "details": "Fill the application form on passportindia.gov.in, pay fee online, and book an appointment at the nearest Passport Seva Kendra (PSK).", "authority": "Passport Seva Kendra", "fees": "Rs. 1,500 (normal) / Rs. 3,500 (tatkal)"},
            {"step": 2, "action": "Visit PSK with documents", "details": "Bring original documents (Aadhaar, PAN, address proof, old passport if renewal). Biometrics are captured at the PSK."},
            {"step": 3, "action": "Police verification", "details": "Police conduct address verification. If clear, passport dispatched within 7-30 days.", "time_limit": "7 days (tatkal) / 30 days (normal)"},
            {"step": 4, "action": "If delayed or rejected, file grievance", "details": "File grievance on passportindia.gov.in. If still unresolved, approach the Regional Passport Officer. As last resort, file writ petition before the High Court under Article 226.", "authority": "Regional Passport Officer / High Court"},
        ],
        "relevant_law": [
            {"act": "Passports Act", "section": "5"},
            {"act": "Constitution of India", "section": "21"},
        ],
        "tags": ["passport", "delayed passport", "PSK", "police verification", "tatkal"],
        "source": "curated",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Builder Functions
# ═══════════════════════════════════════════════════════════════════════════════


def load_nalsa_faqs() -> list[dict]:
    """Load FAQ entries from the NALSA scraper module."""
    try:
        from data.scrapers.nalsa_scraper import ALL_FAQS
        return ALL_FAQS
    except ImportError:
        print("Warning: Could not import NALSA FAQs. Using additional procedures only.")
        return []


def build_all_procedures() -> list[dict]:
    """
    Combine NALSA FAQs + additional procedures into the full 50-procedure set.

    Deduplicates by ID and validates each entry.
    """
    nalsa = load_nalsa_faqs()
    combined = nalsa + ADDITIONAL_PROCEDURES

    # Deduplicate by ID
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for proc in combined:
        pid = proc.get("id", "")
        if pid not in seen_ids:
            seen_ids.add(pid)
            unique.append(proc)

    # Validate
    valid: list[dict] = []
    for proc in unique:
        errors = []
        if not proc.get("id"):
            errors.append("missing id")
        if not proc.get("question"):
            errors.append("missing question")
        if not proc.get("steps"):
            errors.append("missing steps")
        if not proc.get("domain"):
            errors.append("missing domain")
        if errors:
            print(f"  Warning: {proc.get('id', '?')} — {', '.join(errors)}")
        else:
            valid.append(proc)

    return valid


def export_procedures_json(procedures: list[dict]) -> Path:
    """Export the complete procedure set as JSON."""
    PROCEDURES_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROCEDURES_JSON.write_text(
        json.dumps(procedures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return PROCEDURES_JSON


def export_embedding_chunks(procedures: list[dict]) -> Path:
    """
    Export procedures as pre-chunked data for the embedding pipeline.

    Each procedure becomes a single chunk with question + answer + steps
    combined into one text block for embedding.
    """
    chunks = []
    for proc in procedures:
        answer_text = proc.get("answer", "")
        steps = proc.get("steps", [])
        if steps:
            steps_text = "\n".join(
                f"Step {s['step']}: {s['action']} — {s.get('details', '')}"
                for s in steps
            )
            answer_text = f"{answer_text}\n\n{steps_text}"

        chunks.append({
            "chunk_id": f"{proc['id']}_main",
            "text": f"Q: {proc['question']}\n\nA: {answer_text}",
            "metadata": {
                "source": "procedure",
                "procedure_id": proc["id"],
                "domain": proc["domain"],
                "jurisdiction": proc.get("jurisdiction", "central"),
                "tags": proc.get("tags", []),
                "relevant_law": proc.get("relevant_law", []),
                "num_steps": len(steps),
                "chunk_type": "procedure",
            },
        })

    CHUNKS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_OUTPUT.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return CHUNKS_OUTPUT


def print_summary(procedures: list[dict]) -> None:
    """Print a summary of the procedure knowledge base."""
    print(f"\n{'=' * 65}")
    print("  NyayaMitra — Procedural Knowledge Base Summary")
    print(f"{'=' * 65}\n")
    print(f"  Total procedures: {len(procedures)}\n")

    domains: dict[str, list[str]] = {}
    total_steps = 0
    total_laws = 0

    for proc in procedures:
        d = proc["domain"]
        if d not in domains:
            domains[d] = []
        domains[d].append(proc["id"])
        total_steps += len(proc.get("steps", []))
        total_laws += len(proc.get("relevant_law", []))

    print(f"  {'Domain':<20} {'Count':>5}  {'IDs'}")
    print("  " + "─" * 60)
    for domain in sorted(domains.keys()):
        ids = domains[domain]
        id_preview = ", ".join(ids[:3])
        if len(ids) > 3:
            id_preview += f" +{len(ids) - 3} more"
        print(f"  {domain:<20} {len(ids):>5}  {id_preview}")

    print(f"\n  Total procedure steps:  {total_steps}")
    print(f"  Total law references:   {total_laws}")
    print(f"  Avg steps/procedure:    {total_steps / len(procedures):.1f}")
    print(f"\n{'=' * 65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra Procedural Knowledge Base Builder (Sprint 7)",
    )
    parser.add_argument("--export-only", action="store_true",
                        help="Only export, don't rebuild")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary only")
    args = parser.parse_args()

    # Build
    procedures = build_all_procedures()

    if args.summary:
        print_summary(procedures)
        return

    # Export
    json_path = export_procedures_json(procedures)
    print(f"Procedures JSON: {json_path} ({len(procedures)} entries)")

    chunks_path = export_embedding_chunks(procedures)
    print(f"Embedding chunks: {chunks_path} ({len(procedures)} chunks)")

    print_summary(procedures)


if __name__ == "__main__":
    main()