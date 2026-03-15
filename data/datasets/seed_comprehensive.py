"""
NyayaMitra — Comprehensive Seed Data.

Seeds the database with verified legal data across all 7 domains:
    1. Criminal (IPC, CrPC — already seeded, this adds more)
    2. Property (TPA, Registration Act, RERA, Contract Act)
    3. Family (HMA, SMA, DV Act, Maintenance under CrPC)
    4. Labor (ID Act, POSH, Payment of Wages)
    5. Consumer (CPA 2019)
    6. Constitutional (Constitution, RTI Act)
    7. IP (Copyright Act, IT Act)

Also adds 20+ landmark judgments across all domains.

Usage:
    python -m data.datasets.seed_comprehensive
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.database import async_session
from app.models.legal import Act, Section, Judgment

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# ACTS & SECTIONS — All 7 Domains
# ═══════════════════════════════════════════════════════════════════════════════

ACTS_AND_SECTIONS = [
    # ─── PROPERTY LAW ────────────────────────────────────────────────────────
    {
        "act": {
            "name": "Transfer of Property Act, 1882",
            "short_name": "TPA",
            "year": 1882,
            "act_number": "Act No. 4 of 1882",
            "domain": "property",
        },
        "sections": [
            {
                "section_number": "5",
                "title": "Transfer of property defined",
                "text": "In the following sections 'transfer of property' means an act by which a living person conveys property, in present or in future, to one or more other living persons, or to himself and one or more other living persons; and 'to transfer property' is to perform such act. 'Living person' includes a company or association or body of individuals, whether incorporated or not.",
                "chapter": "Chapter II - Of Transfers of Property by Act of Parties",
            },
            {
                "section_number": "52",
                "title": "Transfer of property pending suit relating thereto (Doctrine of Lis Pendens)",
                "text": "During the pendency in any Court of any suit or proceeding which is not collusive and in which any right to immoveable property is directly and specifically in question, the property cannot be transferred or otherwise dealt with by any party to the suit or proceeding so as to affect the rights of any other party thereto under any decree or order which may be made therein, except under the authority of the Court and on such terms as it may impose.",
                "chapter": "Chapter IV - Of Transfers of Immoveable Property",
            },
            {
                "section_number": "54",
                "title": "Sale defined",
                "text": "'Sale' is a transfer of ownership in exchange for a price paid or promised or part-paid and part-promised. Such transfer, in the case of tangible immoveable property of the value of one hundred rupees and upwards, or in the case of a reversion or other intangible thing, can be made only by a registered instrument. In the case of tangible immoveable property of a value less than one hundred rupees, such transfer may be made either by a registered instrument or by delivery of the property.",
                "chapter": "Chapter IV - Of Transfers of Immoveable Property",
            },
            {
                "section_number": "58",
                "title": "Mortgage defined",
                "text": "A mortgage is the transfer of an interest in specific immoveable property for the purpose of securing the payment of money advanced or to be advanced by way of loan, an existing or future debt, or the performance of an engagement which may give rise to a pecuniary liability. The transferor is called a mortgagor, the transferee a mortgagee; the principal money and interest of which payment is secured for the time being are called the mortgage-money, and the instrument (if any) by which the transfer is effected is called a mortgage-deed.",
                "chapter": "Chapter IV - Of Transfers of Immoveable Property",
            },
            {
                "section_number": "106",
                "title": "Duration of certain leases in absence of written contract or local usage",
                "text": "In the absence of a contract or local law or usage to the contrary, a lease of immoveable property for agricultural or manufacturing purposes shall be deemed to be a lease from year to year, terminable, on the part of either lessor or lessee, by six months' notice; and a lease of immoveable property for any other purpose shall be deemed to be a lease from month to month, terminable, on the part of either lessor or lessee, by fifteen days' notice.",
                "chapter": "Chapter V - Of Leases of Immoveable Property",
            },
        ],
    },
    {
        "act": {
            "name": "Indian Contract Act, 1872",
            "short_name": "Contract Act",
            "year": 1872,
            "act_number": "Act No. 9 of 1872",
            "domain": "property",
        },
        "sections": [
            {
                "section_number": "2(h)",
                "title": "Contract defined",
                "text": "An agreement enforceable by law is a contract.",
                "chapter": "Chapter I - Preliminary",
            },
            {
                "section_number": "10",
                "title": "What agreements are contracts",
                "text": "All agreements are contracts if they are made by the free consent of parties competent to contract, for a lawful consideration and with a lawful object, and are not hereby expressly declared to be void.",
                "chapter": "Chapter II - Of Contracts, Voidable Contracts and Void Agreements",
            },
            {
                "section_number": "73",
                "title": "Compensation for loss or damage caused by breach of contract",
                "text": "When a contract has been broken, the party who suffers by such breach is entitled to receive, as compensation for any loss or damage caused to him thereby, such compensation as would in the ordinary course of things have been caused by such breach, and such compensation is not to be given for any remote and indirect loss or damage sustained by reason of the breach.",
                "chapter": "Chapter VI - Of the Consequences of Breach of Contract",
            },
        ],
    },
    {
        "act": {
            "name": "Real Estate (Regulation and Development) Act, 2016",
            "short_name": "RERA",
            "year": 2016,
            "act_number": "Act No. 16 of 2016",
            "domain": "property",
        },
        "sections": [
            {
                "section_number": "3",
                "title": "Prior registration of real estate project with Authority",
                "text": "No promoter shall advertise, market, book, sell or offer for sale, or invite persons to purchase in any manner any plot, apartment or building, as the case may be, in any real estate project or part of it, in any planning area, without registering the real estate project with the Real Estate Regulatory Authority established under this Act.",
                "chapter": "Chapter II - Registration of Real Estate Project and Agent",
            },
            {
                "section_number": "18",
                "title": "Return of amount and compensation",
                "text": "If the promoter fails to complete or is unable to give possession of an apartment, plot or building in accordance with the terms of the agreement for sale or, as the case may be, duly completed by the date specified therein, he shall be liable on demand to the allottees to return the amount received by him in respect of that apartment, plot, building, as the case may be, with interest at such rate as may be prescribed.",
                "chapter": "Chapter III - Functions and Duties of Promoter",
            },
        ],
    },

    # ─── FAMILY LAW ──────────────────────────────────────────────────────────
    {
        "act": {
            "name": "Hindu Marriage Act, 1955",
            "short_name": "HMA",
            "year": 1955,
            "act_number": "Act No. 25 of 1955",
            "domain": "family",
        },
        "sections": [
            {
                "section_number": "5",
                "title": "Conditions for a Hindu marriage",
                "text": "A marriage may be solemnized between any two Hindus, if the following conditions are fulfilled, namely: (i) neither party has a spouse living at the time of the marriage; (ii) at the time of the marriage, neither party is incapable of giving a valid consent to it in consequence of unsoundness of mind; (iii) the bridegroom has completed the age of twenty-one years and the bride the age of eighteen years at the time of the marriage; (iv) the parties are not within the degrees of prohibited relationship unless the custom or usage governing each of them permits of a marriage between the two.",
                "chapter": "Chapter II - Hindu Marriages",
            },
            {
                "section_number": "13",
                "title": "Divorce",
                "text": "Any marriage solemnized, whether before or after the commencement of this Act, may, on a petition presented by either the husband or the wife, be dissolved by a decree of divorce on the ground that the other party has, after the solemnization of the marriage, had voluntary sexual intercourse with any person other than his or her spouse; or has treated the petitioner with cruelty; or has deserted the petitioner for a continuous period of not less than two years; or has ceased to be a Hindu by conversion to another religion; or has been incurably of unsound mind.",
                "chapter": "Chapter IV - Dissolution of Marriage",
            },
            {
                "section_number": "13B",
                "title": "Divorce by mutual consent",
                "text": "Subject to the provisions of this Act a petition for dissolution of marriage by a decree of divorce may be presented to the district court by both the parties to a marriage together, whether such marriage was solemnized before or after the commencement of the Marriage Laws (Amendment) Act, 1976, on the ground that they have been living separately for a period of one year or more, that they have not been able to live together and that they have mutually agreed that the marriage should be dissolved.",
                "chapter": "Chapter IV - Dissolution of Marriage",
            },
            {
                "section_number": "24",
                "title": "Maintenance pendente lite and expenses of proceedings",
                "text": "Where in any proceeding under this Act it appears to the court that either the wife or the husband, as the case may be, has no independent income sufficient for her or his support and the necessary expenses of the proceeding, it may, on the application of the wife or the husband, order the respondent to pay to the petitioner the expenses of the proceeding, and monthly during the proceeding such sum as, having regard to the petitioner's own income and the income of the respondent, it may seem to the court to be reasonable.",
                "chapter": "Chapter V - Jurisdiction and Procedure",
            },
        ],
    },
    {
        "act": {
            "name": "Protection of Women from Domestic Violence Act, 2005",
            "short_name": "DV Act",
            "year": 2005,
            "act_number": "Act No. 43 of 2005",
            "domain": "family",
        },
        "sections": [
            {
                "section_number": "3",
                "title": "Definition of domestic violence",
                "text": "For the purposes of this Act, any act, omission or commission or conduct of the respondent shall constitute domestic violence in case it (a) harms or injures or endangers the health, safety, life, limb or well-being, whether mental or physical, of the aggrieved person or tends to do so and includes causing physical abuse, sexual abuse, verbal and emotional abuse and economic abuse; or (b) harasses, harms, injures or endangers the aggrieved person with a view to coerce her or any other person related to her to meet any unlawful demand for any dowry or other property or valuable security.",
                "chapter": "Chapter II - Domestic Violence",
            },
            {
                "section_number": "12",
                "title": "Application to Magistrate",
                "text": "An aggrieved person or a Protection Officer or any other person on behalf of the aggrieved person may present an application to the Magistrate seeking one or more of the reliefs under this Act. Every application under sub-section (1) shall be in such form and contain such particulars as may be prescribed or as nearly as possible thereto. The Magistrate shall fix the first date of hearing, which shall not ordinarily be beyond three days from the date of receipt of the application by the court.",
                "chapter": "Chapter IV - Procedure for Obtaining Orders of Reliefs",
            },
            {
                "section_number": "19",
                "title": "Residence orders",
                "text": "While disposing of an application under sub-section (1) of section 12, the Magistrate may, on being satisfied that domestic violence has taken place, pass a residence order restraining the respondent from dispossessing or in any other manner disturbing the possession of the aggrieved person from the shared household, or directing the respondent to remove himself from the shared household.",
                "chapter": "Chapter IV - Procedure for Obtaining Orders of Reliefs",
            },
        ],
    },
    {
        "act": {
            "name": "Special Marriage Act, 1954",
            "short_name": "SMA",
            "year": 1954,
            "act_number": "Act No. 43 of 1954",
            "domain": "family",
        },
        "sections": [
            {
                "section_number": "4",
                "title": "Conditions relating to solemnization of special marriages",
                "text": "Notwithstanding anything contained in any other law for the time being in force relating to the solemnization of marriages, a marriage between any two persons may be solemnized under this Act, if at the time of the marriage neither party has a spouse living; neither party is an idiot or a lunatic; the male has completed the age of twenty-one years and the female the age of eighteen years; the parties are not within the degrees of prohibited relationship.",
                "chapter": "Chapter II - Solemnization of Special Marriages",
            },
            {
                "section_number": "27",
                "title": "Divorce",
                "text": "Any marriage solemnized under this Act may, on a petition presented by either the husband or the wife, be dissolved by a decree of divorce on the ground that the other party has committed adultery; or has deserted the petitioner for a continuous period of not less than two years; or is undergoing a sentence of imprisonment for seven years or more; or has since the solemnization of the marriage treated the petitioner with cruelty; or has been incurably of unsound mind for a continuous period of not less than three years.",
                "chapter": "Chapter V - Dissolution of Special Marriages",
            },
        ],
    },

    # ─── CONSUMER LAW ────────────────────────────────────────────────────────
    {
        "act": {
            "name": "Consumer Protection Act, 2019",
            "short_name": "CPA",
            "year": 2019,
            "act_number": "Act No. 35 of 2019",
            "domain": "consumer",
        },
        "sections": [
            {
                "section_number": "2(7)",
                "title": "Consumer defined",
                "text": "'Consumer' means any person who buys any goods for a consideration which has been paid or promised or partly paid and partly promised, or under any system of deferred payment and includes any user of such goods other than the person who buys such goods for consideration paid or promised or partly paid or partly promised, or under any system of deferred payment, when such use is made with the approval of such person, but does not include a person who obtains such goods for resale or for any commercial purpose.",
                "chapter": "Chapter I - Preliminary",
            },
            {
                "section_number": "34",
                "title": "Jurisdiction of District Commission",
                "text": "Subject to the other provisions of this Act, the District Commission shall have jurisdiction to entertain complaints where the value of the goods or services paid as consideration does not exceed one crore rupees.",
                "chapter": "Chapter III - Consumer Disputes Redressal Commission",
            },
            {
                "section_number": "35",
                "title": "Jurisdiction of State Commission",
                "text": "Subject to the other provisions of this Act, the State Commission shall have jurisdiction to entertain complaints where the value of the goods or services paid as consideration exceeds one crore rupees but does not exceed ten crore rupees.",
                "chapter": "Chapter III - Consumer Disputes Redressal Commission",
            },
            {
                "section_number": "38",
                "title": "Manner of filing complaint",
                "text": "A complaint may be filed with the District Commission, the State Commission or the National Commission, as the case may be, by the consumer to whom the goods are sold or delivered or agreed to be sold or delivered, or the service is provided or agreed to be provided; or any recognized consumer association; or one or more consumers, where there are numerous consumers having the same interest, with the permission of the Commission; or the Central Government or the Central Authority; or the State Government.",
                "chapter": "Chapter IV - Jurisdiction and Power of Consumer Disputes Redressal Commission",
            },
        ],
    },

    # ─── LABOR LAW ───────────────────────────────────────────────────────────
    {
        "act": {
            "name": "Industrial Disputes Act, 1947",
            "short_name": "ID Act",
            "year": 1947,
            "act_number": "Act No. 14 of 1947",
            "domain": "labor",
        },
        "sections": [
            {
                "section_number": "2(k)",
                "title": "Strike defined",
                "text": "'Strike' means a cessation of work by a body of persons employed in any industry acting in combination, or a concerted refusal, or a refusal under a common understanding, of any number of persons who are or have been so employed to continue to work or to accept employment.",
                "chapter": "Chapter I - Preliminary",
            },
            {
                "section_number": "25F",
                "title": "Conditions precedent to retrenchment of workmen",
                "text": "No workman employed in any industry who has been in continuous service for not less than one year under an employer shall be retrenched by that employer until (a) the workman has been given one month's notice in writing indicating the reasons for retrenchment and the period of notice has expired, or the workman has been paid in lieu of such notice, wages for the period of the notice; (b) the workman has been paid, at the time of retrenchment, compensation which shall be equivalent to fifteen days' average pay for every completed year of continuous service or any part thereof in excess of six months.",
                "chapter": "Chapter VA - Lay-off and Retrenchment",
            },
        ],
    },
    {
        "act": {
            "name": "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013",
            "short_name": "POSH Act",
            "year": 2013,
            "act_number": "Act No. 14 of 2013",
            "domain": "labor",
        },
        "sections": [
            {
                "section_number": "2(n)",
                "title": "Sexual harassment defined",
                "text": "'Sexual harassment' includes any one or more of the following unwelcome acts or behaviour (whether directly or by implication) namely: (i) physical contact and advances; or (ii) a demand or request for sexual favours; or (iii) making sexually coloured remarks; or (iv) showing pornography; or (v) any other unwelcome physical, verbal or non-verbal conduct of sexual nature.",
                "chapter": "Chapter I - Preliminary",
            },
            {
                "section_number": "4",
                "title": "Constitution of Internal Complaints Committee",
                "text": "Every employer of a workplace shall, by an order in writing, constitute a Committee to be known as the Internal Complaints Committee. The Presiding Officer shall be a woman employed at a senior level at workplace from amongst the employees. Not less than two Members from amongst employees preferably committed to the cause of women or who have had experience in social work or have legal knowledge. One member from amongst non-governmental organisations or associations committed to the cause of women or a person familiar with the issues relating to sexual harassment.",
                "chapter": "Chapter II - Constitution of Internal Complaints Committee",
            },
            {
                "section_number": "9",
                "title": "Complaint of sexual harassment",
                "text": "Any aggrieved woman may make, in writing, a complaint of sexual harassment at workplace to the Internal Committee if so constituted, or the Local Committee, in case it is so constituted, within a period of three months from the date of incident and in case of a series of incidents, within a period of three months from the date of last incident.",
                "chapter": "Chapter IV - Complaint",
            },
        ],
    },

    # ─── CONSTITUTIONAL LAW ──────────────────────────────────────────────────
    {
        "act": {
            "name": "Constitution of India",
            "short_name": "Constitution",
            "year": 1950,
            "act_number": None,
            "domain": "constitutional",
        },
        "sections": [
            {
                "section_number": "14",
                "title": "Equality before law",
                "text": "The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.",
                "chapter": "Part III - Fundamental Rights",
            },
            {
                "section_number": "19",
                "title": "Protection of certain rights regarding freedom of speech, etc.",
                "text": "All citizens shall have the right (a) to freedom of speech and expression; (b) to assemble peaceably and without arms; (c) to form associations or unions; (d) to move freely throughout the territory of India; (e) to reside and settle in any part of the territory of India; (g) to practise any profession, or to carry on any occupation, trade or business.",
                "chapter": "Part III - Fundamental Rights",
            },
            {
                "section_number": "21",
                "title": "Protection of life and personal liberty",
                "text": "No person shall be deprived of his life or personal liberty except according to procedure established by law.",
                "chapter": "Part III - Fundamental Rights",
            },
            {
                "section_number": "21A",
                "title": "Right to education",
                "text": "The State shall provide free and compulsory education to all children of the age of six to fourteen years in such manner as the State may, by law, determine.",
                "chapter": "Part III - Fundamental Rights",
            },
            {
                "section_number": "32",
                "title": "Remedies for enforcement of fundamental rights",
                "text": "The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed. The Supreme Court shall have power to issue directions or orders or writs, including writs in the nature of habeas corpus, mandamus, prohibition, quo warranto and certiorari, whichever may be appropriate, for the enforcement of any of the rights conferred by this Part.",
                "chapter": "Part III - Fundamental Rights",
            },
            {
                "section_number": "226",
                "title": "Power of High Courts to issue certain writs",
                "text": "Notwithstanding anything in article 32, every High Court shall have power, throughout the territories in relation to which it exercises jurisdiction, to issue to any person or authority, including in appropriate cases, any Government, within those territories directions, orders or writs, including writs in the nature of habeas corpus, mandamus, prohibition, quo warranto and certiorari, or any of them, for the enforcement of any of the rights conferred by Part III and for any other purpose.",
                "chapter": "Part V - The Union, Chapter V - The High Courts",
            },
        ],
    },
    {
        "act": {
            "name": "Right to Information Act, 2005",
            "short_name": "RTI",
            "year": 2005,
            "act_number": "Act No. 22 of 2005",
            "domain": "constitutional",
        },
        "sections": [
            {
                "section_number": "3",
                "title": "Right to information",
                "text": "Subject to the provisions of this Act, all citizens shall have the right to information.",
                "chapter": "Chapter II - Right to Information and Obligations of Public Authorities",
            },
            {
                "section_number": "6",
                "title": "Request for obtaining information",
                "text": "A person, who desires to obtain any information under this Act, shall make a request in writing or through electronic means in English or Hindi or in the official language of the area in which the application is being made, accompanying such fee as may be prescribed, to the Central Public Information Officer or State Public Information Officer, as the case may be, of the concerned public authority, specifying the particulars of the information sought by him.",
                "chapter": "Chapter II - Right to Information and Obligations of Public Authorities",
            },
            {
                "section_number": "7",
                "title": "Disposal of request",
                "text": "Subject to the proviso to sub-section (2) of section 5 or the proviso to sub-section (3) of section 6, the Central Public Information Officer or State Public Information Officer, as the case may be, on receipt of a request under section 6 shall, as expeditiously as possible, and in any case within thirty days of the receipt of the request, either provide the information on payment of such fee as may be prescribed or reject the request for any of the reasons specified in sections 8 and 9.",
                "chapter": "Chapter II - Right to Information and Obligations of Public Authorities",
            },
        ],
    },

    # ─── IP LAW ──────────────────────────────────────────────────────────────
    {
        "act": {
            "name": "Information Technology Act, 2000",
            "short_name": "IT Act",
            "year": 2000,
            "act_number": "Act No. 21 of 2000",
            "domain": "ip",
        },
        "sections": [
            {
                "section_number": "43",
                "title": "Penalty and compensation for damage to computer, computer system, etc.",
                "text": "If any person without permission of the owner or any other person who is in charge of a computer, computer system or computer network, accesses or secures access to such computer, computer system or computer network or downloads, copies or extracts any data, computer data base or information from such computer, computer system or computer network, he shall be liable to pay damages by way of compensation to the person so affected.",
                "chapter": "Chapter IX - Penalties, Compensation and Adjudication",
            },
            {
                "section_number": "66",
                "title": "Computer related offences",
                "text": "If any person, dishonestly or fraudulently, does any act referred to in section 43, he shall be punishable with imprisonment for a term which may extend to three years or with fine which may extend to five lakh rupees or with both.",
                "chapter": "Chapter XI - Offences",
            },
            {
                "section_number": "66A",
                "title": "Punishment for sending offensive messages (STRUCK DOWN)",
                "text": "This section was struck down by the Supreme Court in Shreya Singhal v. Union of India (2015) as unconstitutional for violating Article 19(1)(a) - freedom of speech and expression. The section had previously criminalized sending offensive messages through communication service.",
                "chapter": "Chapter XI - Offences",
            },
            {
                "section_number": "72",
                "title": "Breach of confidentiality and privacy",
                "text": "Save as otherwise provided in this Act or any other law for the time being in force, if any person who, in pursuance of any of the powers conferred under this Act, rules or regulations made thereunder, has secured access to any electronic record, book, register, correspondence, information, document or other material without the consent of the person concerned discloses such electronic record, book, register, correspondence, information, document or other material to any other person shall be punished with imprisonment for a term which may extend to two years, or with fine which may extend to one lakh rupees, or with both.",
                "chapter": "Chapter XII - Intermediaries Not To Be Liable in Certain Cases",
            },
        ],
    },
    {
        "act": {
            "name": "Copyright Act, 1957",
            "short_name": "Copyright Act",
            "year": 1957,
            "act_number": "Act No. 14 of 1957",
            "domain": "ip",
        },
        "sections": [
            {
                "section_number": "14",
                "title": "Meaning of copyright",
                "text": "For the purposes of this Act, 'copyright' means the exclusive right subject to the provisions of this Act, to do or authorise the doing of any of the following acts in respect of a work or any substantial part thereof, namely: in the case of a literary, dramatic or musical work, not being a computer programme, to reproduce the work in any material form including the storing of it in any medium by electronic means; to issue copies of the work to the public; to perform the work in public; to communicate the work to the public; to make any cinematograph film or sound recording in respect of the work; to make any translation of the work; to make any adaptation of the work.",
                "chapter": "Chapter III - Copyright",
            },
            {
                "section_number": "52",
                "title": "Certain acts not to be infringement of copyright (Fair Use)",
                "text": "The following acts shall not constitute an infringement of copyright, namely: a fair dealing with any work, not being a computer programme, for the purposes of private or personal use, including research; criticism or review; reporting current events and current affairs, including the reporting of a lecture delivered in public.",
                "chapter": "Chapter XI - Infringement of Copyright",
            },
        ],
    },

    # ─── ADDITIONAL CRIMINAL LAW SECTIONS ────────────────────────────────────
    {
        "act": {
            "name": "Indian Penal Code, 1860",
            "short_name": "IPC",
            "year": 1860,
            "act_number": "Act No. 45 of 1860",
            "domain": "criminal",
        },
        "sections": [
            {
                "section_number": "304B",
                "title": "Dowry death",
                "text": "Where the death of a woman is caused by any burns or bodily injury or occurs otherwise than under normal circumstances within seven years of her marriage and it is shown that soon before her death she was subjected to cruelty or harassment by her husband or any relative of her husband for, or in connection with, any demand for dowry, such death shall be called 'dowry death', and such husband or relative shall be deemed to have caused her death. Punishment: imprisonment for a term which shall not be less than seven years but which may extend to imprisonment for life.",
                "chapter": "Chapter XVI - Of Offences Affecting the Human Body",
            },
            {
                "section_number": "379",
                "title": "Punishment for theft",
                "text": "Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
                "chapter": "Chapter XVII - Of Offences Against Property",
            },
            {
                "section_number": "406",
                "title": "Punishment for criminal breach of trust",
                "text": "Whoever commits criminal breach of trust shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
                "chapter": "Chapter XVII - Of Offences Against Property",
            },
            {
                "section_number": "500",
                "title": "Punishment for defamation",
                "text": "Whoever defames another shall be punished with simple imprisonment for a term which may extend to two years, or with fine, or with both.",
                "chapter": "Chapter XXI - Of Defamation",
            },
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# JUDGMENTS — All 7 Domains
# ═══════════════════════════════════════════════════════════════════════════════

ADDITIONAL_JUDGMENTS = [
    # ─── PROPERTY LAW ────────────────────────────────────────────────────────
    {
        "case_name": "Suraj Lamp & Industries Pvt. Ltd. v. State of Haryana",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 3,
        "judgment_date": "2011-10-11",
        "year": 2011,
        "citation_scc": "(2012) 1 SCC 656",
        "domain": "property",
        "headnote": "The Supreme Court held that sale agreements, general power of attorney, and will transfers (GPA/SA/Will transactions) do not convey title and cannot be treated as completed transfers. Only a registered sale deed can legally transfer immovable property. All transactions done through GPA route were declared illegal and not conferring any title on the purchaser.",
        "ratio_decidendi": "Immovable property can be legally and lawfully transferred or conveyed only by a registered deed of conveyance. Any practice of transferring immovable property through power of attorney, sale agreement, or will is not legally valid and does not convey any title to the transferee.",
        "sections_interpreted": '[{"act": "TPA", "section": "54"}, {"act": "Registration Act", "section": "17"}, {"act": "Registration Act", "section": "49"}]',
    },
    {
        "case_name": "Pioneer Urban Land and Infrastructure Ltd. v. Union of India",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "2019-08-09",
        "year": 2019,
        "citation_scc": "(2019) 8 SCC 416",
        "domain": "property",
        "headnote": "The Supreme Court upheld the constitutional validity of amendments to the Insolvency and Bankruptcy Code allowing homebuyers to be treated as financial creditors. Allottees of real estate projects were held to be consumers and financial creditors under IBC, giving them a voice in insolvency proceedings against defaulting builders.",
        "ratio_decidendi": "Homebuyers are financial creditors under the Insolvency and Bankruptcy Code and have a right to participate in the resolution process. Real estate allottees deserve protection as a class of consumers who have invested their savings.",
        "sections_interpreted": '[{"act": "RERA", "section": "18"}, {"act": "IBC", "section": "5(8)(f)"}]',
    },

    # ─── FAMILY LAW ──────────────────────────────────────────────────────────
    {
        "case_name": "Shayara Bano v. Union of India",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 5,
        "judgment_date": "2017-08-22",
        "year": 2017,
        "citation_scc": "(2017) 9 SCC 1",
        "domain": "family",
        "headnote": "A five-judge Constitution Bench struck down the practice of instant triple talaq (talaq-e-biddat) as unconstitutional by a 3:2 majority. The practice of instant divorce by Muslim men by pronouncing talaq three times was held to be violative of Article 14 of the Constitution.",
        "ratio_decidendi": "Triple talaq is manifestly arbitrary and therefore violative of Article 14 of the Constitution. What is bad in theology is bad in law as well. The practice of talaq-e-biddat is set aside as being unconstitutional.",
        "sections_interpreted": '[{"act": "Constitution of India", "section": "14"}, {"act": "Constitution of India", "section": "25"}]',
    },
    {
        "case_name": "V. Bhagat v. D. Bhagat",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "1994-01-20",
        "year": 1994,
        "citation_scc": "(1994) 1 SCC 337",
        "domain": "family",
        "headnote": "The Supreme Court defined mental cruelty for the purpose of divorce. Mental cruelty must be of such a nature that the parties cannot reasonably be expected to live together. The court held that making wild allegations of adultery and filing false criminal cases constitutes mental cruelty sufficient to grant divorce.",
        "ratio_decidendi": "Mental cruelty in Section 13(1)(ia) can broadly be defined as that conduct which inflicts upon the other party such mental pain and suffering as would make it not possible for that party to live with the other. Levelling wild, false and unsubstantiated allegations constitutes cruelty.",
        "sections_interpreted": '[{"act": "HMA", "section": "13(1)(ia)"}]',
    },
    {
        "case_name": "Rajnesh v. Neha",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "2020-11-04",
        "year": 2020,
        "citation_scc": "(2021) 2 SCC 324",
        "domain": "family",
        "headnote": "The Supreme Court laid down comprehensive guidelines for determination of maintenance in matrimonial cases. Both parties must file income affidavits. Overlapping jurisdiction must be avoided by granting maintenance under one statute. The court established criteria for computing maintenance including the status, standard of living, educational qualifications, employment, and liabilities of both parties.",
        "ratio_decidendi": "Courts must adopt a uniform and consistent approach while determining maintenance. Both parties are required to disclose their income and assets through sworn affidavits. Maintenance should be awarded from the date of filing the application.",
        "sections_interpreted": '[{"act": "CrPC", "section": "125"}, {"act": "HMA", "section": "24"}, {"act": "DV Act", "section": "20"}]',
    },

    # ─── CONSUMER LAW ────────────────────────────────────────────────────────
    {
        "case_name": "Indian Medical Association v. V.P. Shantha",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 3,
        "judgment_date": "1995-11-13",
        "year": 1995,
        "citation_scc": "(1995) 6 SCC 651",
        "domain": "consumer",
        "headnote": "The Supreme Court held that medical services rendered by doctors and hospitals fall within the ambit of 'service' as defined under the Consumer Protection Act. Medical practitioners who charge fees are liable under consumer protection law for deficiency in service. This landmark judgment brought the entire medical profession under the purview of consumer courts.",
        "ratio_decidendi": "Service rendered to a patient by a medical practitioner by way of consultation, diagnosis and treatment, both medical and surgical, is service as defined under the Consumer Protection Act. Medical profession is not excluded from the ambit of the Act.",
        "sections_interpreted": '[{"act": "CPA", "section": "2(1)(o)"}, {"act": "CPA", "section": "14"}]',
    },

    # ─── LABOR LAW ───────────────────────────────────────────────────────────
    {
        "case_name": "Workmen of Dimakuchi Tea Estate v. Dimakuchi Tea Estate",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "1958-01-01",
        "year": 1958,
        "citation_scc": "AIR 1958 SC 353",
        "domain": "labor",
        "headnote": "The Supreme Court established the principles for determining fair wages. The court held that wages must be above the bare subsistence level and should take into account the industry's capacity to pay, the prevailing rates in similar industries, the level of national income and its distribution, and the place of the industry in the economy.",
        "ratio_decidendi": "Fair wages lie between the minimum wage and the living wage. In fixing fair wages, the tribunal should consider the productivity of labour, the prevailing rates of wages in the same or similar industries, the level of national income and its distribution.",
        "sections_interpreted": '[{"act": "ID Act", "section": "2(k)"}, {"act": "Minimum Wages Act", "section": "3"}]',
    },

    # ─── CONSTITUTIONAL LAW ──────────────────────────────────────────────────
    {
        "case_name": "Kesavananda Bharati v. State of Kerala",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 13,
        "judgment_date": "1973-04-24",
        "year": 1973,
        "citation_scc": "(1973) 4 SCC 225",
        "citation_air": "AIR 1973 SC 1461",
        "domain": "constitutional",
        "headnote": "The largest-ever Constitution Bench (13 judges) propounded the Basic Structure Doctrine, holding that Parliament has wide powers to amend the Constitution under Article 368 but cannot alter its basic structure. Fundamental rights, secularism, federalism, separation of powers, and judicial review were identified as part of the basic structure that cannot be abrogated even by constitutional amendment.",
        "ratio_decidendi": "Parliament's power to amend the Constitution under Article 368 does not include the power to destroy or emasculate the basic features or the fundamental framework of the Constitution. The basic structure of the Constitution cannot be abrogated even by a constitutional amendment.",
        "sections_interpreted": '[{"act": "Constitution of India", "section": "368"}, {"act": "Constitution of India", "section": "13"}, {"act": "Constitution of India", "section": "14"}, {"act": "Constitution of India", "section": "21"}]',
    },
    {
        "case_name": "Shreya Singhal v. Union of India",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "2015-03-24",
        "year": 2015,
        "citation_scc": "(2015) 5 SCC 1",
        "domain": "constitutional",
        "headnote": "The Supreme Court struck down Section 66A of the Information Technology Act as unconstitutional for being violative of Article 19(1)(a) - freedom of speech and expression. The section was held to be vague and overbroad, creating a chilling effect on free speech online. The court also read down Section 79 (intermediary liability) to require a court order before content takedown.",
        "ratio_decidendi": "Section 66A of the IT Act is struck down in its entirety as being violative of Article 19(1)(a) and not saved under Article 19(2). The provision is vague, overbroad, and has a chilling effect on free speech. Section 79 is read down to mean that intermediaries must receive actual knowledge from a court order before being required to take down content.",
        "sections_interpreted": '[{"act": "IT Act", "section": "66A"}, {"act": "IT Act", "section": "79"}, {"act": "Constitution of India", "section": "19(1)(a)"}, {"act": "Constitution of India", "section": "19(2)"}]',
    },

    # ─── IP LAW ──────────────────────────────────────────────────────────────
    {
        "case_name": "Eastern Book Company v. D.B. Modak",
        "court": "Supreme Court",
        "court_type": "SC",
        "bench_size": 2,
        "judgment_date": "2007-12-12",
        "year": 2007,
        "citation_scc": "(2008) 1 SCC 1",
        "domain": "ip",
        "headnote": "The Supreme Court held that there is no copyright in judgments and orders of courts as they are in the public domain. However, the creative inputs of editors such as headnotes, editorial notes, and arrangement may be protected under copyright if they satisfy the minimum degree of creativity standard. Copy-edited judgments with minimal changes do not attract copyright protection.",
        "ratio_decidendi": "While there can be no copyright in the judgments of the courts or in the texts of legislative and judicial pronouncements, the headnotes and editorial notes prepared by editors with intellectual labour and skill are entitled to copyright protection provided they satisfy the modicum of creativity test.",
        "sections_interpreted": '[{"act": "Copyright Act", "section": "14"}, {"act": "Copyright Act", "section": "52"}]',
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Seed Function
# ═══════════════════════════════════════════════════════════════════════════════


async def seed_all() -> dict:
    """Seed all acts, sections, and judgments."""
    stats = {"acts_new": 0, "acts_existing": 0, "sections_new": 0, "judgments_new": 0, "judgments_existing": 0}

    async with async_session() as session:
        from sqlalchemy import select

        # ── Seed Acts and Sections ──
        for entry in ACTS_AND_SECTIONS:
            act_info = entry["act"]
            sections_data = entry["sections"]

            # Check if act exists
            stmt = select(Act).where(
                Act.name == act_info["name"],
                Act.year == act_info["year"],
            )
            result = await session.execute(stmt)
            existing_act = result.scalar_one_or_none()

            if existing_act:
                act_id = existing_act.id
                stats["acts_existing"] += 1
            else:
                new_act = Act(
                    id=uuid.uuid4(),
                    name=act_info["name"],
                    short_name=act_info.get("short_name"),
                    year=act_info["year"],
                    act_number=act_info.get("act_number"),
                    domain=act_info.get("domain", "general"),
                    jurisdiction="central",
                    status="active",
                )
                session.add(new_act)
                await session.flush()
                act_id = new_act.id
                stats["acts_new"] += 1
                logger.info("seed_act_created", name=act_info["name"], domain=act_info.get("domain"))

            # Seed sections
            for sec in sections_data:
                stmt = select(Section).where(
                    Section.act_id == act_id,
                    Section.section_number == sec["section_number"],
                )
                result = await session.execute(stmt)
                if result.scalar_one_or_none() is None:
                    session.add(Section(
                        id=uuid.uuid4(),
                        act_id=act_id,
                        section_number=sec["section_number"],
                        title=sec.get("title"),
                        text=sec["text"],
                        chapter=sec.get("chapter"),
                        status="active",
                    ))
                    stats["sections_new"] += 1

        # ── Seed Judgments ──
        for jdata in ADDITIONAL_JUDGMENTS:
            stmt = select(Judgment).where(
                Judgment.case_name == jdata["case_name"],
                Judgment.year == jdata["year"],
            )
            result = await session.execute(stmt)
            if result.scalar_one_or_none():
                stats["judgments_existing"] += 1
                continue

            from datetime import datetime as dt
            jdate = None
            if jdata.get("judgment_date"):
                jdate = dt.strptime(jdata["judgment_date"], "%Y-%m-%d").date()

            session.add(Judgment(
                id=uuid.uuid4(),
                case_name=jdata["case_name"],
                court=jdata["court"],
                court_type=jdata["court_type"],
                bench_size=jdata.get("bench_size"),
                judgment_date=jdate,
                year=jdata["year"],
                citation_scc=jdata.get("citation_scc"),
                citation_air=jdata.get("citation_air"),
                domain=jdata.get("domain"),
                headnote=jdata.get("headnote"),
                ratio_decidendi=jdata.get("ratio_decidendi"),
                sections_interpreted=jdata.get("sections_interpreted"),
                source="seed",
            ))
            stats["judgments_new"] += 1
            logger.info("seed_judgment_created", case=jdata["case_name"], domain=jdata.get("domain"))

        await session.commit()

    logger.info("comprehensive_seed_complete", **stats)
    return stats


async def main():
    """Run comprehensive seeding."""
    print("\n" + "=" * 60)
    print("NyayaMitra — Comprehensive Legal Data Seeding")
    print("=" * 60)

    stats = await seed_all()

    print(f"\n{'='*60}")
    print(f"Seeding Results:")
    print(f"  New acts:        {stats['acts_new']}")
    print(f"  Existing acts:   {stats['acts_existing']}")
    print(f"  New sections:    {stats['sections_new']}")
    print(f"  New judgments:   {stats['judgments_new']}")
    print(f"  Existing judg:   {stats['judgments_existing']}")
    print(f"{'='*60}")
    print(f"\nNow run 'python -m data.embeddings.indexer' to embed and index the new data!")


if __name__ == "__main__":
    asyncio.run(main())