"""
NyayaMitra — Knowledge Graph Loader.

Reads acts, sections, and judgments from PostgreSQL and populates
the Neo4j knowledge graph with nodes and relationships.

Loading stages:
    1. Courts      — Create Court nodes (Supreme Court + 25 High Courts)
    2. Acts        — Create Act nodes from acts table
    3. Sections    — Create Section nodes + BELONGS_TO relationships
    4. Judgments   — Create Judgment nodes + DECIDED_BY relationships
    5. Interprets  — Parse sections_interpreted JSON -> INTERPRETS edges
    6. Overrules   — Parse is_overruled/overruled_by -> OVERRULES edges
    7. Principles  — Extract LegalPrinciple nodes from ratio_decidendi
    8. References  — Cross-reference detection between sections

Uses MERGE (not CREATE) so the loader is idempotent.

Usage:
    python -m data.knowledge_graph.graph_loader
    python -m data.knowledge_graph.graph_loader --stage acts
    python -m data.knowledge_graph.graph_loader --stats
    python -m data.knowledge_graph.graph_loader --drop-reload
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

logger = structlog.get_logger()

ACT_ABBREVIATION_MAP = {
    "IPC": "Indian Penal Code, 1860",
    "Indian Penal Code": "Indian Penal Code, 1860",
    "CrPC": "Code of Criminal Procedure, 1973",
    "Cr.P.C.": "Code of Criminal Procedure, 1973",
    "Code of Criminal Procedure": "Code of Criminal Procedure, 1973",
    "CPC": "Code of Civil Procedure, 1908",
    "Constitution of India": "Constitution of India",
    "Constitution": "Constitution of India",
    "TPA": "Transfer of Property Act, 1882",
    "Transfer of Property Act": "Transfer of Property Act, 1882",
    "HMA": "Hindu Marriage Act, 1955",
    "Hindu Marriage Act": "Hindu Marriage Act, 1955",
    "SMA": "Special Marriage Act, 1954",
    "Special Marriage Act": "Special Marriage Act, 1954",
    "DV Act": "Protection of Women from Domestic Violence Act, 2005",
    "Domestic Violence Act": "Protection of Women from Domestic Violence Act, 2005",
    "CPA": "Consumer Protection Act, 2019",
    "Consumer Protection Act": "Consumer Protection Act, 2019",
    "POSH": "Sexual Harassment of Women at Workplace Act, 2013",
    "ID Act": "Industrial Disputes Act, 1947",
    "Industrial Disputes Act": "Industrial Disputes Act, 1947",
    "RERA": "Real Estate (Regulation and Development) Act, 2016",
    "RTI": "Right to Information Act, 2005",
    "Right to Information Act": "Right to Information Act, 2005",
    "IT Act": "Information Technology Act, 2000",
    "Information Technology Act": "Information Technology Act, 2000",
    "Copyright Act": "Copyright Act, 1957",
    "Contract Act": "Indian Contract Act, 1872",
    "Indian Contract Act": "Indian Contract Act, 1872",
    "Indian Evidence Act": "Indian Evidence Act, 1872",
    "Evidence Act": "Indian Evidence Act, 1872",
    "Dowry Prohibition Act": "Dowry Prohibition Act, 1961",
    "Passports Act": "Passports Act, 1967",
    "Payment of Wages Act": "Payment of Wages Act, 1936",
    "NI Act": "Negotiable Instruments Act, 1881",
}


class GraphLoader:
    """Loads legal data from PostgreSQL into Neo4j knowledge graph."""

    def __init__(self):
        self.driver = None
        self.stats = {
            "courts": 0, "acts": 0, "sections": 0, "judgments": 0,
            "interprets": 0, "overrules": 0, "principles": 0,
            "references": 0, "errors": 0,
        }

    def connect(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        self.driver.verify_connectivity()
        logger.info("graph_loader_connected", uri=settings.neo4j_uri)

    def close(self):
        if self.driver:
            self.driver.close()

    def load_courts(self):
        courts = [
            {"name": "Supreme Court", "court_type": "SC", "state": None, "established": 1950},
            {"name": "Bombay High Court", "court_type": "HC", "state": "Maharashtra", "established": 1862},
            {"name": "Delhi High Court", "court_type": "HC", "state": "Delhi", "established": 1966},
            {"name": "Madras High Court", "court_type": "HC", "state": "Tamil Nadu", "established": 1862},
            {"name": "Calcutta High Court", "court_type": "HC", "state": "West Bengal", "established": 1862},
            {"name": "Karnataka High Court", "court_type": "HC", "state": "Karnataka", "established": 1884},
            {"name": "Allahabad High Court", "court_type": "HC", "state": "Uttar Pradesh", "established": 1866},
            {"name": "Kerala High Court", "court_type": "HC", "state": "Kerala", "established": 1956},
            {"name": "Punjab and Haryana High Court", "court_type": "HC", "state": "Punjab", "established": 1947},
            {"name": "Gujarat High Court", "court_type": "HC", "state": "Gujarat", "established": 1960},
            {"name": "Rajasthan High Court", "court_type": "HC", "state": "Rajasthan", "established": 1949},
            {"name": "Andhra Pradesh High Court", "court_type": "HC", "state": "Andhra Pradesh", "established": 1954},
            {"name": "Telangana High Court", "court_type": "HC", "state": "Telangana", "established": 2019},
            {"name": "Patna High Court", "court_type": "HC", "state": "Bihar", "established": 1916},
            {"name": "Jharkhand High Court", "court_type": "HC", "state": "Jharkhand", "established": 2000},
            {"name": "Gauhati High Court", "court_type": "HC", "state": "Assam", "established": 1948},
            {"name": "Orissa High Court", "court_type": "HC", "state": "Odisha", "established": 1948},
            {"name": "Chhattisgarh High Court", "court_type": "HC", "state": "Chhattisgarh", "established": 2000},
            {"name": "Uttarakhand High Court", "court_type": "HC", "state": "Uttarakhand", "established": 2000},
            {"name": "Himachal Pradesh High Court", "court_type": "HC", "state": "Himachal Pradesh", "established": 1971},
            {"name": "Jammu and Kashmir High Court", "court_type": "HC", "state": "J&K", "established": 1928},
            {"name": "Tripura High Court", "court_type": "HC", "state": "Tripura", "established": 2013},
            {"name": "Meghalaya High Court", "court_type": "HC", "state": "Meghalaya", "established": 2013},
            {"name": "Manipur High Court", "court_type": "HC", "state": "Manipur", "established": 2013},
            {"name": "Sikkim High Court", "court_type": "HC", "state": "Sikkim", "established": 1975},
        ]
        count = 0
        with self.driver.session() as session:
            for court in courts:
                session.run(
                    "MERGE (c:Court {name: $name}) "
                    "SET c.court_type = $court_type, c.state = $state, c.established = $established",
                    **court,
                )
                count += 1
        self.stats["courts"] = count
        logger.info("courts_loaded", count=count)
        return count

    async def load_acts(self):
        from app.database import async_session
        from app.models.legal import Act
        from sqlalchemy import select

        async with async_session() as db:
            result = await db.execute(select(Act))
            acts = result.scalars().all()

        count = 0
        with self.driver.session() as session:
            for act in acts:
                session.run(
                    "MERGE (a:Act {name: $name}) "
                    "SET a.short_name = $short_name, a.year = $year, a.act_number = $act_number, "
                    "a.domain = $domain, a.jurisdiction = $jurisdiction, a.status = $status, a.pg_id = $pg_id",
                    name=act.name, short_name=act.short_name or "", year=act.year,
                    act_number=act.act_number or "", domain=act.domain or "",
                    jurisdiction=act.jurisdiction or "central", status=act.status or "active",
                    pg_id=str(act.id),
                )
                count += 1
        self.stats["acts"] = count
        logger.info("acts_loaded", count=count)
        return count

    async def load_sections(self):
        from app.database import async_session
        from app.models.legal import Act
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        async with async_session() as db:
            result = await db.execute(select(Act).options(selectinload(Act.sections)))
            acts = result.scalars().all()

        count = 0
        with self.driver.session() as session:
            for act in acts:
                for sec in act.sections:
                    uid = f"{act.name}/{sec.section_number}"
                    session.run(
                        "MERGE (s:Section {uid: $uid}) "
                        "SET s.section_number = $section_number, s.title = $title, "
                        "s.text = $text, s.chapter = $chapter, s.act_name = $act_name, "
                        "s.act_short_name = $act_short_name, s.domain = $domain, "
                        "s.status = $status, s.pg_id = $pg_id "
                        "WITH s "
                        "MERGE (a:Act {name: $act_name}) "
                        "MERGE (s)-[:BELONGS_TO]->(a)",
                        uid=uid, section_number=sec.section_number, title=sec.title or "",
                        text=(sec.text or "")[:2000], chapter=sec.chapter or "",
                        act_name=act.name, act_short_name=act.short_name or "",
                        domain=act.domain or "", status=sec.status or "active",
                        pg_id=str(sec.id),
                    )
                    count += 1
        self.stats["sections"] = count
        logger.info("sections_loaded", count=count)
        return count

    async def load_judgments(self):
        from app.database import async_session
        from app.models.legal import Judgment
        from sqlalchemy import select

        async with async_session() as db:
            result = await db.execute(select(Judgment))
            judgments = result.scalars().all()

        count = 0
        with self.driver.session() as session:
            for j in judgments:
                court_name = "Supreme Court" if j.court_type == "SC" else (j.court or "Supreme Court")
                session.run(
                    "MERGE (j:Judgment {case_name: $case_name}) "
                    "SET j.case_number = $case_number, j.court = $court, j.court_type = $court_type, "
                    "j.bench = $bench, j.bench_size = $bench_size, j.year = $year, "
                    "j.citation_scc = $citation_scc, j.citation_air = $citation_air, "
                    "j.domain = $domain, j.headnote = $headnote, j.ratio_decidendi = $ratio, "
                    "j.is_overruled = $is_overruled, j.overruled_by = $overruled_by, j.pg_id = $pg_id "
                    "WITH j "
                    "MERGE (c:Court {name: $court_name}) "
                    "MERGE (j)-[:DECIDED_BY]->(c)",
                    case_name=j.case_name, case_number=j.case_number or "",
                    court=j.court or "", court_type=j.court_type or "",
                    bench=j.bench or "", bench_size=j.bench_size or 0, year=j.year,
                    citation_scc=j.citation_scc or "", citation_air=j.citation_air or "",
                    domain=j.domain or "", headnote=(j.headnote or "")[:2000],
                    ratio=(j.ratio_decidendi or "")[:2000],
                    is_overruled=j.is_overruled or False, overruled_by=j.overruled_by or "",
                    pg_id=str(j.id), court_name=court_name,
                )
                count += 1
        self.stats["judgments"] = count
        logger.info("judgments_loaded", count=count)
        return count

    async def load_interprets(self):
        from app.database import async_session
        from app.models.legal import Judgment
        from sqlalchemy import select

        async with async_session() as db:
            result = await db.execute(select(Judgment))
            judgments = result.scalars().all()

        count = 0
        errors = 0
        with self.driver.session() as session:
            for j in judgments:
                if not j.sections_interpreted:
                    continue
                try:
                    interpreted = json.loads(j.sections_interpreted)
                except (json.JSONDecodeError, TypeError):
                    errors += 1
                    continue

                for entry in interpreted:
                    act_abbr = entry.get("act", "")
                    section_num = entry.get("section", "")
                    if not act_abbr or not section_num:
                        continue

                    canonical_act = ACT_ABBREVIATION_MAP.get(
                        act_abbr, ACT_ABBREVIATION_MAP.get(act_abbr.strip(), act_abbr)
                    )
                    section_uid = f"{canonical_act}/{section_num}"

                    session.run(
                        "MATCH (j:Judgment {case_name: $case_name}) "
                        "MERGE (s:Section {uid: $section_uid}) "
                        "ON CREATE SET s.section_number = $section_num, s.act_name = $act_name, s.domain = j.domain "
                        "MERGE (j)-[:INTERPRETS {act_abbreviation: $act_abbr}]->(s)",
                        case_name=j.case_name, section_uid=section_uid,
                        section_num=section_num, act_name=canonical_act, act_abbr=act_abbr,
                    )
                    count += 1

        self.stats["interprets"] = count
        self.stats["errors"] += errors
        logger.info("interprets_loaded", edges=count, parse_errors=errors)
        return count

    async def load_overrules(self):
        from app.database import async_session
        from app.models.legal import Judgment
        from sqlalchemy import select

        async with async_session() as db:
            result = await db.execute(select(Judgment).where(Judgment.is_overruled == True))
            overruled = result.scalars().all()

        count = 0
        with self.driver.session() as session:
            for j in overruled:
                if not j.overruled_by:
                    continue
                session.run(
                    "MATCH (overruled:Judgment {case_name: $overruled_case}) "
                    "MATCH (overruling:Judgment) WHERE overruling.case_name CONTAINS $overruling_name "
                    "MERGE (overruling)-[:OVERRULES]->(overruled)",
                    overruled_case=j.case_name, overruling_name=j.overruled_by,
                )
                count += 1
        self.stats["overrules"] = count
        logger.info("overrules_loaded", edges=count)
        return count

    async def load_principles(self):
        from app.database import async_session
        from app.models.legal import Judgment
        from sqlalchemy import select

        async with async_session() as db:
            result = await db.execute(select(Judgment))
            judgments = result.scalars().all()

        count = 0
        with self.driver.session() as session:
            for j in judgments:
                if not j.ratio_decidendi:
                    continue
                principle_uid = f"principle:{j.case_name}"
                session.run(
                    "MERGE (p:LegalPrinciple {uid: $uid}) "
                    "SET p.text = $text, p.source_case = $case_name, p.domain = $domain, p.year = $year "
                    "WITH p "
                    "MATCH (j:Judgment {case_name: $case_name}) "
                    "MERGE (j)-[:ESTABLISHES]->(p)",
                    uid=principle_uid, text=(j.ratio_decidendi or "")[:1000],
                    case_name=j.case_name, domain=j.domain or "", year=j.year,
                )
                count += 1

                if j.sections_interpreted:
                    try:
                        interpreted = json.loads(j.sections_interpreted)
                        for entry in interpreted:
                            act_abbr = entry.get("act", "")
                            section_num = entry.get("section", "")
                            if not act_abbr or not section_num:
                                continue
                            canonical_act = ACT_ABBREVIATION_MAP.get(act_abbr, act_abbr)
                            section_uid = f"{canonical_act}/{section_num}"
                            session.run(
                                "MATCH (p:LegalPrinciple {uid: $principle_uid}) "
                                "MATCH (s:Section {uid: $section_uid}) "
                                "MERGE (p)-[:DERIVED_FROM]->(s)",
                                principle_uid=principle_uid, section_uid=section_uid,
                            )
                    except (json.JSONDecodeError, TypeError):
                        pass
        self.stats["principles"] = count
        logger.info("principles_loaded", count=count)
        return count

    def load_references(self):
        import re
        ref_pattern = re.compile(r"(?:section|Section)\s+(\d+[A-Za-z]?)")

        count = 0
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Section) WHERE s.text IS NOT NULL "
                "RETURN s.uid AS uid, s.text AS text, s.act_name AS act_name"
            )
            sections = list(result)

            for record in sections:
                uid = record["uid"]
                text = record["text"] or ""
                act_name = record["act_name"] or ""
                own_section = uid.split("/")[-1] if "/" in uid else ""

                for ref_section in set(ref_pattern.findall(text)):
                    if ref_section == own_section:
                        continue
                    target_uid = f"{act_name}/{ref_section}"
                    session.run(
                        "MATCH (source:Section {uid: $source_uid}) "
                        "MATCH (target:Section {uid: $target_uid}) "
                        "WHERE source <> target "
                        "MERGE (source)-[:REFERENCES]->(target)",
                        source_uid=uid, target_uid=target_uid,
                    )
                    count += 1
        self.stats["references"] = count
        logger.info("references_loaded", edges=count)
        return count

    async def load_all(self, stages=None):
        all_stages = [
            ("courts", self.load_courts),
            ("acts", self.load_acts),
            ("sections", self.load_sections),
            ("judgments", self.load_judgments),
            ("interprets", self.load_interprets),
            ("overrules", self.load_overrules),
            ("principles", self.load_principles),
            ("references", self.load_references),
        ]
        start = time.time()
        for name, func in all_stages:
            if stages and name not in stages:
                continue
            logger.info("loading_stage", stage=name)
            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
            except Exception as e:
                logger.error("stage_error", stage=name, error=str(e))
                self.stats["errors"] += 1
        self.stats["duration_seconds"] = round(time.time() - start, 2)
        logger.info("graph_load_complete", **self.stats)
        return self.stats

    def get_stats(self):
        stats = {}
        with self.driver.session() as session:
            for label in ["Act", "Section", "Judgment", "Court", "LegalPrinciple"]:
                r = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                stats[label.lower()] = r.single()["cnt"]
            for rel in ["BELONGS_TO", "INTERPRETS", "DECIDED_BY", "OVERRULES",
                        "DISTINGUISHES", "ESTABLISHES", "DERIVED_FROM", "REFERENCES"]:
                r = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt")
                stats[rel.lower()] = r.single()["cnt"]
            r = session.run("MATCH (n) RETURN count(n) AS cnt")
            stats["total_nodes"] = r.single()["cnt"]
            r = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            stats["total_relationships"] = r.single()["cnt"]
        return stats


async def main():
    parser = argparse.ArgumentParser(description="NyayaMitra Knowledge Graph Loader")
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--drop-reload", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  NyayaMitra — Knowledge Graph Loader")
    print("=" * 60 + "\n")

    loader = GraphLoader()
    loader.connect()

    try:
        if args.stats:
            stats = loader.get_stats()
            print("  Graph Statistics:")
            print("  " + "-" * 40)
            print(f"    Total nodes:         {stats['total_nodes']}")
            print(f"    Total relationships: {stats['total_relationships']}")
            print("\n  Nodes:")
            for label in ["act", "section", "judgment", "court", "legalprinciple"]:
                print(f"    {label:<20} {stats.get(label, 0)}")
            print("\n  Relationships:")
            for rel in ["belongs_to", "interprets", "decided_by", "overrules",
                        "distinguishes", "establishes", "derived_from", "references"]:
                print(f"    {rel:<20} {stats.get(rel, 0)}")
            return

        if args.drop_reload:
            print("  Dropping all graph data...")
            from data.knowledge_graph.schema import GraphSchemaManager
            schema_mgr = GraphSchemaManager()
            schema_mgr.driver = loader.driver
            schema_mgr.drop_all()
            schema_mgr.create_schema()
            print()

        stages = [args.stage] if args.stage else None
        stats = await loader.load_all(stages=stages)

        print("\n  Loading Results:")
        print("  " + "-" * 40)
        for key in ["courts", "acts", "sections", "judgments", "interprets",
                     "overrules", "principles", "references", "errors"]:
            print(f"    {key:<16} {stats[key]}")
        print(f"    {'duration':<16} {stats.get('duration_seconds', 0)}s")

        final = loader.get_stats()
        print(f"\n  Final: {final['total_nodes']} nodes, {final['total_relationships']} relationships\n")
    finally:
        loader.close()


if __name__ == "__main__":
    asyncio.run(main())