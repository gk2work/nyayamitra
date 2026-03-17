"""
NyayaMitra — Knowledge Graph Loader (Sprint 7 — Full Corpus).

Sprint 7: UNWIND batch Cypher for 100x faster bulk loading,
streams from PostgreSQL, all courts from registry.

Usage:
    python -m data.knowledge_graph.graph_loader
    python -m data.knowledge_graph.graph_loader --stage judgments
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

try:
    from data.processors.judgment_parser import ACT_ABBREVIATIONS as ACT_ABBREVIATION_MAP
except ImportError:
    ACT_ABBREVIATION_MAP = {"IPC": "Indian Penal Code, 1860", "CrPC": "Code of Criminal Procedure, 1973"}

BATCH_SIZE = 500


class GraphLoader:
    """Loads legal data from PostgreSQL into Neo4j. Uses UNWIND batch Cypher."""

    def __init__(self):
        self.driver = None
        self.stats = {"courts": 0, "acts": 0, "sections": 0, "judgments": 0,
                      "interprets": 0, "overrules": 0, "principles": 0,
                      "references": 0, "errors": 0, "duration_seconds": 0}

    def connect(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        self.driver.verify_connectivity()
        logger.info("neo4j_connected", uri=settings.neo4j_uri)

    def close(self):
        if self.driver:
            self.driver.close()

    def _batch_execute(self, items, cypher, label):
        total = 0
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i:i + BATCH_SIZE]
            with self.driver.session() as session:
                session.run(cypher, batch=batch)
            total += len(batch)
            if total % 2000 == 0 or total == len(items):
                logger.info("batch_progress", stage=label, processed=total, total=len(items))
        return total

    def load_courts(self):
        from data.config.courts_registry import get_all_courts
        items = [{"name": c.name, "short_name": c.short_name, "court_type": c.court_type.value,
                  "principal_seat": c.principal_seat, "states": c.states} for c in get_all_courts()]
        cypher = """UNWIND $batch AS item
        MERGE (c:Court {name: item.name})
        SET c.short_name = item.short_name, c.court_type = item.court_type,
            c.principal_seat = item.principal_seat, c.states = item.states"""
        self.stats["courts"] = self._batch_execute(items, cypher, "courts")
        logger.info("courts_loaded", count=self.stats["courts"])
        return self.stats["courts"]

    async def load_acts(self):
        from app.database import async_session
        from app.models.legal import Act
        from sqlalchemy import select
        async with async_session() as db:
            acts = (await db.execute(select(Act))).scalars().all()
        items = [{"name": a.name, "short_name": a.short_name or "", "year": a.year,
                  "act_number": a.act_number or "", "domain": a.domain or "",
                  "jurisdiction": a.jurisdiction or "central", "status": a.status or "active",
                  "replaced_by": a.replaced_by or "", "pg_id": str(a.id)} for a in acts]
        cypher = """UNWIND $batch AS item
        MERGE (a:Act {name: item.name})
        SET a.short_name = item.short_name, a.year = item.year, a.act_number = item.act_number,
            a.domain = item.domain, a.jurisdiction = item.jurisdiction, a.status = item.status,
            a.replaced_by = item.replaced_by, a.pg_id = item.pg_id"""
        self.stats["acts"] = self._batch_execute(items, cypher, "acts")
        logger.info("acts_loaded", count=self.stats["acts"])
        return self.stats["acts"]

    async def load_sections(self):
        from app.database import async_session
        from app.models.legal import Act, Section
        from sqlalchemy import select, func
        async with async_session() as db:
            total = (await db.execute(select(func.count(Section.id)))).scalar() or 0
        logger.info("loading_sections_start", total=total)
        count, offset, page = 0, 0, 1000
        while offset < total:
            async with async_session() as db:
                rows = (await db.execute(
                    select(Section, Act).join(Act, Section.act_id == Act.id)
                    .order_by(Section.id).offset(offset).limit(page)
                )).all()
            if not rows:
                break
            items = [{"uid": f"{act.name}/{sec.section_number}", "section_number": sec.section_number,
                      "title": sec.title or "", "text": (sec.text or "")[:2000], "chapter": sec.chapter or "",
                      "act_name": act.name, "act_short_name": act.short_name or "",
                      "domain": act.domain or "", "status": sec.status or "active",
                      "pg_id": str(sec.id)} for sec, act in rows]
            cypher = """UNWIND $batch AS item
            MERGE (s:Section {uid: item.uid})
            SET s.section_number = item.section_number, s.title = item.title, s.text = item.text,
                s.chapter = item.chapter, s.act_name = item.act_name, s.act_short_name = item.act_short_name,
                s.domain = item.domain, s.status = item.status, s.pg_id = item.pg_id
            WITH s, item MERGE (a:Act {name: item.act_name}) MERGE (s)-[:BELONGS_TO]->(a)"""
            count += self._batch_execute(items, cypher, "sections")
            offset += page
        self.stats["sections"] = count
        logger.info("sections_loaded", count=count)
        return count

    async def load_judgments(self):
        from app.database import async_session
        from app.models.legal import Judgment
        from sqlalchemy import select, func
        async with async_session() as db:
            total = (await db.execute(select(func.count(Judgment.id)))).scalar() or 0
        logger.info("loading_judgments_start", total=total)
        count, offset, page = 0, 0, 500
        while offset < total:
            async with async_session() as db:
                judgments = (await db.execute(
                    select(Judgment).order_by(Judgment.id).offset(offset).limit(page)
                )).scalars().all()
            if not judgments:
                break
            items = [{"case_name": j.case_name, "court": j.court or "Supreme Court",
                      "court_type": j.court_type or "SC", "year": j.year or 0,
                      "citation_scc": j.citation_scc or "", "citation_air": j.citation_air or "",
                      "domain": j.domain or "", "bench_size": j.bench_size or 0,
                      "headnote": (j.headnote or "")[:1000], "ratio": (j.ratio_decidendi or "")[:1000],
                      "is_overruled": getattr(j, "is_overruled", False) or False,
                      "pg_id": str(j.id)} for j in judgments]
            cypher_j = """UNWIND $batch AS item
            MERGE (j:Judgment {case_name: item.case_name})
            SET j.court = item.court, j.court_type = item.court_type, j.year = item.year,
                j.citation_scc = item.citation_scc, j.citation_air = item.citation_air,
                j.domain = item.domain, j.bench_size = item.bench_size, j.headnote = item.headnote,
                j.ratio_decidendi = item.ratio, j.is_overruled = item.is_overruled, j.pg_id = item.pg_id"""
            self._batch_execute(items, cypher_j, "judgments")
            cypher_d = """UNWIND $batch AS item
            MATCH (j:Judgment {case_name: item.case_name})
            MERGE (c:Court {name: item.court}) MERGE (j)-[:DECIDED_BY]->(c)"""
            self._batch_execute(items, cypher_d, "decided_by")
            count += len(items)
            offset += page
        self.stats["judgments"] = count
        logger.info("judgments_loaded", count=count)
        return count

    async def load_interprets(self):
        from app.database import async_session
        from app.models.legal import Judgment
        from sqlalchemy import select
        offset, page, count, errors = 0, 500, 0, 0
        while True:
            async with async_session() as db:
                judgments = (await db.execute(
                    select(Judgment).where(Judgment.sections_interpreted.isnot(None))
                    .order_by(Judgment.id).offset(offset).limit(page)
                )).scalars().all()
            if not judgments:
                break
            pairs = []
            for j in judgments:
                try:
                    interpreted = json.loads(j.sections_interpreted)
                except (json.JSONDecodeError, TypeError):
                    errors += 1; continue
                for e in interpreted:
                    act_abbr, sec_num = e.get("act", ""), e.get("section", "")
                    if not act_abbr or not sec_num:
                        continue
                    canonical = ACT_ABBREVIATION_MAP.get(act_abbr, ACT_ABBREVIATION_MAP.get(act_abbr.strip(), act_abbr))
                    pairs.append({"case_name": j.case_name, "section_uid": f"{canonical}/{sec_num}",
                                  "section_num": sec_num, "act_name": canonical, "act_abbr": act_abbr})
            if pairs:
                cypher = """UNWIND $batch AS item
                MATCH (j:Judgment {case_name: item.case_name})
                MERGE (s:Section {uid: item.section_uid})
                ON CREATE SET s.section_number = item.section_num, s.act_name = item.act_name, s.domain = j.domain
                MERGE (j)-[:INTERPRETS {act_abbreviation: item.act_abbr}]->(s)"""
                count += self._batch_execute(pairs, cypher, "interprets")
            offset += page
        self.stats["interprets"] = count
        self.stats["errors"] += errors
        logger.info("interprets_loaded", edges=count, parse_errors=errors)
        return count

    async def load_overrules(self):
        from app.database import async_session
        from app.models.legal import Judgment
        from sqlalchemy import select
        async with async_session() as db:
            overruled = (await db.execute(select(Judgment).where(Judgment.is_overruled == True))).scalars().all()
        count = 0
        with self.driver.session() as session:
            for j in overruled:
                if not j.overruled_by:
                    continue
                try:
                    session.run("MATCH (o:Judgment {case_name: $oc}) MATCH (r:Judgment) WHERE r.case_name CONTAINS $rn MERGE (r)-[:OVERRULES]->(o)",
                                oc=j.case_name, rn=j.overruled_by)
                    count += 1
                except Exception:
                    pass
        self.stats["overrules"] = count
        logger.info("overrules_loaded", edges=count)
        return count

    async def load_principles(self):
        from app.database import async_session
        from app.models.legal import Judgment
        from sqlalchemy import select
        offset, page, count = 0, 500, 0
        while True:
            async with async_session() as db:
                judgments = (await db.execute(
                    select(Judgment).where(Judgment.ratio_decidendi.isnot(None))
                    .order_by(Judgment.id).offset(offset).limit(page)
                )).scalars().all()
            if not judgments:
                break
            items = [{"uid": f"principle:{j.case_name}", "text": (j.ratio_decidendi or "")[:1000],
                      "case_name": j.case_name, "domain": j.domain or "", "year": j.year or 0}
                     for j in judgments if j.ratio_decidendi and len(j.ratio_decidendi.strip()) >= 20]
            if items:
                cypher = """UNWIND $batch AS item
                MERGE (p:LegalPrinciple {uid: item.uid})
                SET p.text = item.text, p.source_case = item.case_name, p.domain = item.domain, p.year = item.year
                WITH p, item MATCH (j:Judgment {case_name: item.case_name}) MERGE (j)-[:ESTABLISHES]->(p)"""
                count += self._batch_execute(items, cypher, "principles")
            offset += page
        self.stats["principles"] = count
        logger.info("principles_loaded", count=count)
        return count

    def load_references(self):
        import re
        ref_pattern = re.compile(r"(?:section|Section)\s+(\d+[A-Za-z]?)")
        count = 0
        with self.driver.session() as session:
            result = session.run("MATCH (s:Section) WHERE s.text IS NOT NULL RETURN s.uid AS uid, s.text AS text, s.act_name AS act_name LIMIT 10000")
            refs = []
            for r in result:
                for sec_num in ref_pattern.findall(r["text"] or ""):
                    target = f"{r['act_name'] or ''}/{sec_num}"
                    if target != r["uid"]:
                        refs.append({"from_uid": r["uid"], "to_uid": target})
            if refs:
                cypher = """UNWIND $batch AS item
                MATCH (f:Section {uid: item.from_uid}) MATCH (t:Section {uid: item.to_uid})
                MERGE (f)-[:REFERENCES]->(t)"""
                count = self._batch_execute(refs, cypher, "references")
        self.stats["references"] = count
        logger.info("references_loaded", edges=count)
        return count

    async def load_all(self, stages=None):
        all_stages = [("courts", self.load_courts), ("acts", self.load_acts),
                      ("sections", self.load_sections), ("judgments", self.load_judgments),
                      ("interprets", self.load_interprets), ("overrules", self.load_overrules),
                      ("principles", self.load_principles), ("references", self.load_references)]
        start = time.time()
        for name, func in all_stages:
            if stages and name not in stages:
                continue
            logger.info("loading_stage", stage=name)
            try:
                await func() if asyncio.iscoroutinefunction(func) else func()
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
                stats[label.lower()] = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            for rel in ["BELONGS_TO", "INTERPRETS", "DECIDED_BY", "OVERRULES", "DISTINGUISHES", "ESTABLISHES", "DERIVED_FROM", "REFERENCES"]:
                stats[rel.lower()] = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
            stats["total_nodes"] = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            stats["total_relationships"] = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return stats


async def main():
    parser = argparse.ArgumentParser(description="NyayaMitra Knowledge Graph Loader (Sprint 7)")
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--drop-reload", action="store_true")
    args = parser.parse_args()
    print(f"\n{'='*60}\n  NyayaMitra — Knowledge Graph Loader (Sprint 7)\n{'='*60}\n")
    loader = GraphLoader()
    loader.connect()
    try:
        if args.stats:
            s = loader.get_stats()
            print(f"  Nodes: {s['total_nodes']:,}  Rels: {s['total_relationships']:,}")
            for l in ["act","section","judgment","court","legalprinciple"]:
                print(f"    {l:<20} {s.get(l,0):,}")
            for r in ["belongs_to","interprets","decided_by","overrules","establishes","references"]:
                print(f"    {r:<20} {s.get(r,0):,}")
            return
        if args.drop_reload:
            from data.knowledge_graph.schema import GraphSchemaManager
            mgr = GraphSchemaManager(); mgr.driver = loader.driver; mgr.drop_all(); mgr.create_schema()
        stages = [args.stage] if args.stage else None
        stats = await loader.load_all(stages=stages)
        print("\n  Results:")
        for k in ["courts","acts","sections","judgments","interprets","overrules","principles","references","errors"]:
            print(f"    {k:<16} {stats[k]:,}")
        print(f"    duration         {stats['duration_seconds']}s")
        f = loader.get_stats()
        print(f"\n  Final: {f['total_nodes']:,} nodes, {f['total_relationships']:,} rels\n")
    finally:
        loader.close()

if __name__ == "__main__":
    asyncio.run(main())