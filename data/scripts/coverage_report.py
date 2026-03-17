"""
NyayaMitra — Data Coverage Report (Sprint 7).

Analyzes the current state of ingested data and identifies gaps:
    - Acts per domain (ingested vs registry target)
    - Sections per act (are any acts empty?)
    - Judgments per court (ingested vs target)
    - Judgments per year (distribution)
    - Domain coverage (which domains are underrepresented?)
    - Qdrant collection sizes
    - Neo4j graph node/edge counts
    - Missing acts (in registry but not ingested)

Outputs:
    - Console summary
    - Markdown report (data/raw/coverage_report.md)
    - JSON report (data/raw/coverage_report.json)

Usage:
    python -m data.scripts.coverage_report
    python -m data.scripts.coverage_report --json-only
    python -m data.scripts.coverage_report --check  # Exit 1 if targets not met
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

logger = structlog.get_logger()

REPORT_DIR = PROJECT_ROOT / "data" / "raw"

# Sprint 7 acceptance criteria thresholds
TARGET_ACTS = 100
TARGET_JUDGMENTS = 100_000
TARGET_RETRIEVAL_RECALL = 0.80


# ═══════════════════════════════════════════════════════════════════════════════
# Data Collection
# ═══════════════════════════════════════════════════════════════════════════════


async def collect_postgres_stats() -> dict:
    """Gather statistics from PostgreSQL tables."""
    from sqlalchemy import select, func, distinct
    from app.database import async_session
    from app.models.legal import Act, Section, Judgment

    stats = {}

    async with async_session() as session:
        # ── Acts ──────────────────────────────────────────────────────
        result = await session.execute(select(func.count(Act.id)))
        stats["total_acts"] = result.scalar() or 0

        result = await session.execute(
            select(Act.domain, func.count(Act.id)).group_by(Act.domain)
        )
        stats["acts_by_domain"] = {row[0] or "unknown": row[1] for row in result.all()}

        result = await session.execute(
            select(Act.status, func.count(Act.id)).group_by(Act.status)
        )
        stats["acts_by_status"] = {row[0] or "unknown": row[1] for row in result.all()}

        # ── Sections ──────────────────────────────────────────────────
        result = await session.execute(select(func.count(Section.id)))
        stats["total_sections"] = result.scalar() or 0

        result = await session.execute(
            select(func.count(Section.id)).where(Section.is_indexed == True)
        )
        stats["sections_indexed"] = result.scalar() or 0

        # Sections per act (top 20 + bottom 20)
        result = await session.execute(
            select(
                Act.short_name, Act.name, Act.domain,
                func.count(Section.id).label("section_count"),
            )
            .join(Section, Act.id == Section.act_id)
            .group_by(Act.id, Act.short_name, Act.name, Act.domain)
            .order_by(func.count(Section.id).desc())
        )
        all_act_sections = [
            {
                "short_name": row[0] or "",
                "name": row[1],
                "domain": row[2] or "",
                "sections": row[3],
            }
            for row in result.all()
        ]
        stats["sections_per_act"] = all_act_sections

        # Acts with zero sections
        result = await session.execute(
            select(Act.short_name, Act.name, Act.domain)
            .outerjoin(Section, Act.id == Section.act_id)
            .group_by(Act.id, Act.short_name, Act.name, Act.domain)
            .having(func.count(Section.id) == 0)
        )
        stats["acts_with_no_sections"] = [
            {"short_name": row[0] or "", "name": row[1], "domain": row[2] or ""}
            for row in result.all()
        ]

        # ── Judgments ─────────────────────────────────────────────────
        result = await session.execute(select(func.count(Judgment.id)))
        stats["total_judgments"] = result.scalar() or 0

        result = await session.execute(
            select(func.count(Judgment.id)).where(Judgment.is_indexed == True)
        )
        stats["judgments_indexed"] = result.scalar() or 0

        # By court type
        result = await session.execute(
            select(Judgment.court_type, func.count(Judgment.id))
            .group_by(Judgment.court_type)
        )
        stats["judgments_by_court_type"] = {
            row[0] or "unknown": row[1] for row in result.all()
        }

        # By court (top 20)
        result = await session.execute(
            select(Judgment.court, func.count(Judgment.id).label("cnt"))
            .group_by(Judgment.court)
            .order_by(func.count(Judgment.id).desc())
            .limit(30)
        )
        stats["judgments_by_court"] = [
            {"court": row[0] or "unknown", "count": row[1]}
            for row in result.all()
        ]

        # By year
        result = await session.execute(
            select(Judgment.year, func.count(Judgment.id))
            .group_by(Judgment.year)
            .order_by(Judgment.year.desc())
        )
        stats["judgments_by_year"] = {
            row[0]: row[1] for row in result.all() if row[0]
        }

        # By domain
        result = await session.execute(
            select(Judgment.domain, func.count(Judgment.id))
            .group_by(Judgment.domain)
        )
        stats["judgments_by_domain"] = {
            row[0] or "unknown": row[1] for row in result.all()
        }

        # Judgments with no headnote or ratio
        result = await session.execute(
            select(func.count(Judgment.id)).where(
                Judgment.headnote.is_(None),
                Judgment.ratio_decidendi.is_(None),
            )
        )
        stats["judgments_no_content"] = result.scalar() or 0

    return stats


def collect_registry_stats() -> dict:
    """Gather statistics from the acts and courts registries."""
    from data.config.acts_registry import (
        get_all_acts, get_acts_by_priority, get_acts_by_domain,
        ACTS_REGISTRY,
    )
    from data.config.courts_registry import (
        get_all_courts, get_courts_for_ingestion,
        SUPREME_COURT, ALL_HIGH_COURTS, TRIBUNALS,
    )

    all_acts = get_all_acts()

    return {
        "registry_total_acts": len(all_acts),
        "registry_acts_p0": len(get_acts_by_priority("P0")),
        "registry_acts_p1": len(get_acts_by_priority("P1")),
        "registry_acts_p2": len(get_acts_by_priority("P2")),
        "registry_acts_by_domain": {
            domain: len(acts) for domain, acts in ACTS_REGISTRY.items()
        },
        "registry_total_courts": len(get_all_courts()),
        "registry_hcs": len(ALL_HIGH_COURTS),
        "registry_tribunals": len(TRIBUNALS),
        "registry_target_judgments_p0": sum(
            c.target_judgments for c in get_courts_for_ingestion("P0")
        ),
        "registry_target_judgments_all": sum(
            c.target_judgments for c in get_all_courts()
        ),
    }


def collect_qdrant_stats() -> dict:
    """Gather Qdrant collection statistics."""
    stats = {}
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

        for col_name in ["legal_sections", "legal_judgments", "legal_procedures"]:
            try:
                info = client.get_collection(col_name)
                stats[col_name] = {
                    "points": info.points_count,
                    "vectors": info.vectors_count,
                    "status": info.status.name,
                }
            except Exception:
                stats[col_name] = {"points": 0, "vectors": 0, "status": "missing"}

    except Exception as e:
        logger.warning("qdrant_stats_unavailable", error=str(e))

    return stats


def collect_neo4j_stats() -> dict:
    """Gather Neo4j graph statistics."""
    stats = {}
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        driver.verify_connectivity()

        with driver.session() as session:
            for label in ["Act", "Section", "Judgment", "Court",
                          "LegalPrinciple", "Procedure"]:
                r = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                stats[f"node_{label.lower()}"] = r.single()["cnt"]

            for rel in ["BELONGS_TO", "INTERPRETS", "DECIDED_BY", "OVERRULES",
                        "ESTABLISHES", "DERIVED_FROM", "REFERENCES"]:
                r = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt")
                stats[f"rel_{rel.lower()}"] = r.single()["cnt"]

            r = session.run("MATCH (n) RETURN count(n) AS cnt")
            stats["total_nodes"] = r.single()["cnt"]
            r = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            stats["total_relationships"] = r.single()["cnt"]

        driver.close()
    except Exception as e:
        logger.warning("neo4j_stats_unavailable", error=str(e))

    return stats


def find_missing_acts(pg_stats: dict, registry_stats: dict) -> list[dict]:
    """Find acts that are in the registry but not yet ingested."""
    from data.config.acts_registry import get_all_acts

    ingested_names = {
        a["short_name"] for a in pg_stats.get("sections_per_act", [])
    }
    # Also check acts with no sections
    ingested_names.update(
        a["short_name"] for a in pg_stats.get("acts_with_no_sections", [])
    )

    missing = []
    for act in get_all_acts():
        if act.short_name not in ingested_names:
            missing.append({
                "short_name": act.short_name,
                "name": act.name,
                "domain": act.domain.value,
                "priority": act.priority.value,
            })

    return sorted(missing, key=lambda a: (a["priority"], a["domain"]))


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_coverage_report() -> dict:
    """
    Generate the full coverage report.

    Returns a dict with all statistics + gap analysis.
    """
    logger.info("coverage_report_start")
    start = time.time()

    report = {"generated_at": datetime.utcnow().isoformat()}

    # Collect all stats
    try:
        report["postgres"] = await collect_postgres_stats()
    except Exception as e:
        logger.error("postgres_stats_error", error=str(e))
        report["postgres"] = {"error": str(e)}

    report["registry"] = collect_registry_stats()

    try:
        report["qdrant"] = collect_qdrant_stats()
    except Exception as e:
        report["qdrant"] = {"error": str(e)}

    try:
        report["neo4j"] = collect_neo4j_stats()
    except Exception as e:
        report["neo4j"] = {"error": str(e)}

    # Gap analysis
    pg = report.get("postgres", {})
    reg = report.get("registry", {})

    report["missing_acts"] = find_missing_acts(pg, reg)

    # Acceptance criteria check
    total_acts = pg.get("total_acts", 0)
    total_judgments = pg.get("total_judgments", 0)
    total_sections = pg.get("total_sections", 0)

    report["acceptance"] = {
        "acts_target": TARGET_ACTS,
        "acts_actual": total_acts,
        "acts_pass": total_acts >= TARGET_ACTS,
        "judgments_target": TARGET_JUDGMENTS,
        "judgments_actual": total_judgments,
        "judgments_pass": total_judgments >= TARGET_JUDGMENTS,
        "sections_total": total_sections,
        "missing_acts_count": len(report["missing_acts"]),
    }

    report["duration_seconds"] = round(time.time() - start, 2)
    logger.info("coverage_report_complete", duration=report["duration_seconds"])
    return report


def export_json(report: dict) -> Path:
    """Export report as JSON."""
    path = REPORT_DIR / "coverage_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def export_markdown(report: dict) -> Path:
    """Export report as a readable markdown file."""
    pg = report.get("postgres", {})
    reg = report.get("registry", {})
    qd = report.get("qdrant", {})
    neo = report.get("neo4j", {})
    acc = report.get("acceptance", {})
    missing = report.get("missing_acts", [])

    lines = []
    lines.append("# NyayaMitra — Sprint 7 Data Coverage Report")
    lines.append(f"\nGenerated: {report.get('generated_at', 'N/A')}\n")

    # ── Summary ───────────────────────────────────────────────────────
    lines.append("## Summary\n")
    lines.append(f"| Metric | Actual | Target | Status |")
    lines.append(f"|--------|--------|--------|--------|")

    acts_icon = "PASS" if acc.get("acts_pass") else "FAIL"
    jdg_icon = "PASS" if acc.get("judgments_pass") else "FAIL"
    lines.append(f"| Acts ingested | {acc.get('acts_actual', 0)} | {acc.get('acts_target', 0)}+ | {acts_icon} |")
    lines.append(f"| Total sections | {acc.get('sections_total', 0)} | — | — |")
    lines.append(f"| Judgments ingested | {acc.get('judgments_actual', 0):,} | {acc.get('judgments_target', 0):,}+ | {jdg_icon} |")
    lines.append(f"| Missing acts | {acc.get('missing_acts_count', 0)} | 0 | {'PASS' if acc.get('missing_acts_count', 0) == 0 else 'GAP'} |")

    # ── Acts by Domain ────────────────────────────────────────────────
    lines.append("\n## Acts by Domain\n")
    lines.append("| Domain | Ingested | Registry | Coverage |")
    lines.append("|--------|----------|----------|----------|")

    acts_by_domain = pg.get("acts_by_domain", {})
    reg_by_domain = reg.get("registry_acts_by_domain", {})
    all_domains = sorted(set(list(acts_by_domain.keys()) + list(reg_by_domain.keys())))

    for domain in all_domains:
        ingested = acts_by_domain.get(domain, 0)
        registry = reg_by_domain.get(domain, 0)
        pct = f"{ingested / registry * 100:.0f}%" if registry > 0 else "—"
        lines.append(f"| {domain} | {ingested} | {registry} | {pct} |")

    # ── Top Acts by Section Count ─────────────────────────────────────
    lines.append("\n## Top 20 Acts by Section Count\n")
    lines.append("| Act | Domain | Sections |")
    lines.append("|-----|--------|----------|")

    for act in pg.get("sections_per_act", [])[:20]:
        lines.append(f"| {act['short_name']} | {act['domain']} | {act['sections']} |")

    # ── Acts with No Sections ─────────────────────────────────────────
    empty_acts = pg.get("acts_with_no_sections", [])
    if empty_acts:
        lines.append(f"\n## Acts with No Sections ({len(empty_acts)})\n")
        for act in empty_acts[:20]:
            lines.append(f"- {act['short_name']}: {act['name']} ({act['domain']})")
        if len(empty_acts) > 20:
            lines.append(f"- ... and {len(empty_acts) - 20} more")

    # ── Judgments by Court ────────────────────────────────────────────
    lines.append("\n## Judgments by Court (Top 20)\n")
    lines.append("| Court | Count |")
    lines.append("|-------|-------|")

    for entry in pg.get("judgments_by_court", [])[:20]:
        lines.append(f"| {entry['court']} | {entry['count']:,} |")

    # ── Judgments by Year ─────────────────────────────────────────────
    lines.append("\n## Judgments by Year\n")
    lines.append("| Year | Count |")
    lines.append("|------|-------|")

    for year in sorted(pg.get("judgments_by_year", {}).keys(), reverse=True)[:10]:
        lines.append(f"| {year} | {pg['judgments_by_year'][year]:,} |")

    # ── Qdrant ────────────────────────────────────────────────────────
    lines.append("\n## Qdrant Collections\n")
    lines.append("| Collection | Points | Vectors | Status |")
    lines.append("|------------|--------|---------|--------|")

    for col_name in ["legal_sections", "legal_judgments", "legal_procedures"]:
        info = qd.get(col_name, {})
        lines.append(
            f"| {col_name} | {info.get('points', '?'):,} | "
            f"{info.get('vectors', '?'):,} | {info.get('status', '?')} |"
        )

    # ── Neo4j ─────────────────────────────────────────────────────────
    if neo and not neo.get("error"):
        lines.append("\n## Neo4j Knowledge Graph\n")
        lines.append("| Node/Rel | Count |")
        lines.append("|----------|-------|")

        for key in ["node_act", "node_section", "node_judgment", "node_court",
                     "node_legalprinciple", "node_procedure"]:
            label = key.replace("node_", "").capitalize()
            lines.append(f"| {label} | {neo.get(key, 0):,} |")

        lines.append(f"| **Total Nodes** | **{neo.get('total_nodes', 0):,}** |")

        for key in ["rel_belongs_to", "rel_interprets", "rel_decided_by",
                     "rel_overrules", "rel_establishes", "rel_derived_from",
                     "rel_references"]:
            label = key.replace("rel_", "").upper()
            lines.append(f"| {label} | {neo.get(key, 0):,} |")

        lines.append(f"| **Total Relationships** | **{neo.get('total_relationships', 0):,}** |")

    # ── Missing Acts ──────────────────────────────────────────────────
    if missing:
        lines.append(f"\n## Missing Acts ({len(missing)} not yet ingested)\n")
        lines.append("| Priority | Domain | Short Name | Full Name |")
        lines.append("|----------|--------|------------|-----------|")

        for act in missing[:30]:
            lines.append(
                f"| {act['priority']} | {act['domain']} | "
                f"{act['short_name']} | {act['name']} |"
            )
        if len(missing) > 30:
            lines.append(f"\n*... and {len(missing) - 30} more*")

    # ── Write ─────────────────────────────────────────────────────────
    md_text = "\n".join(lines)
    path = REPORT_DIR / "coverage_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md_text, encoding="utf-8")
    return path


def print_console_summary(report: dict) -> None:
    """Print a compact summary to the console."""
    pg = report.get("postgres", {})
    reg = report.get("registry", {})
    acc = report.get("acceptance", {})
    qd = report.get("qdrant", {})
    neo = report.get("neo4j", {})

    print(f"\n{'═' * 65}")
    print(f"  NyayaMitra — Sprint 7 Coverage Report")
    print(f"{'═' * 65}\n")

    # Acceptance criteria
    def check(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    acts_ok = acc.get("acts_pass", False)
    jdg_ok = acc.get("judgments_pass", False)

    print(f"  {'Metric':<30} {'Actual':>10} {'Target':>10} {'Status':>8}")
    print(f"  {'─' * 62}")
    print(f"  {'Acts ingested':<30} {acc.get('acts_actual', 0):>10} {acc.get('acts_target', 0):>10}+ {check(acts_ok):>8}")
    print(f"  {'Total sections':<30} {acc.get('sections_total', 0):>10} {'—':>10} {'—':>8}")
    print(f"  {'Judgments ingested':<30} {acc.get('judgments_actual', 0):>10,} {acc.get('judgments_target', 0):>10,}+ {check(jdg_ok):>8}")
    print(f"  {'Missing acts':<30} {acc.get('missing_acts_count', 0):>10} {'0':>10} {check(acc.get('missing_acts_count', 0) == 0):>8}")

    # Domain breakdown
    print(f"\n  Acts by Domain:")
    acts_by_domain = pg.get("acts_by_domain", {})
    reg_by_domain = reg.get("registry_acts_by_domain", {})
    for domain in sorted(set(list(acts_by_domain.keys()) + list(reg_by_domain.keys()))):
        ingested = acts_by_domain.get(domain, 0)
        registry = reg_by_domain.get(domain, 0)
        bar = "█" * min(ingested, 20) + "░" * max(0, min(registry - ingested, 20))
        print(f"    {domain:<16} {ingested:>3}/{registry:<3}  {bar}")

    # Judgments by type
    jbt = pg.get("judgments_by_court_type", {})
    if jbt:
        print(f"\n  Judgments by Court Type:")
        for ct, count in sorted(jbt.items(), key=lambda x: -x[1]):
            print(f"    {ct:<12} {count:>8,}")

    # Qdrant
    if qd and not qd.get("error"):
        print(f"\n  Qdrant:")
        for col in ["legal_sections", "legal_judgments", "legal_procedures"]:
            info = qd.get(col, {})
            print(f"    {col:<24} {info.get('points', 0):>8,} points")

    # Neo4j
    if neo and not neo.get("error"):
        print(f"\n  Neo4j: {neo.get('total_nodes', 0):,} nodes, "
              f"{neo.get('total_relationships', 0):,} relationships")

    # Missing acts summary
    missing = report.get("missing_acts", [])
    if missing:
        print(f"\n  Missing acts: {len(missing)}")
        p0_missing = [a for a in missing if a["priority"] == "P0"]
        if p0_missing:
            print(f"    P0 (critical): {len(p0_missing)} — "
                  f"{', '.join(a['short_name'] for a in p0_missing[:5])}"
                  f"{'...' if len(p0_missing) > 5 else ''}")

    # Overall verdict
    all_pass = acts_ok and jdg_ok
    print(f"\n  {'═' * 50}")
    if all_pass:
        print(f"  ║  VERDICT: PASS — Sprint 7 acceptance criteria met  ║")
    else:
        print(f"  ║  VERDICT: GAPS REMAIN — See report for details      ║")
    print(f"  {'═' * 50}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra Data Coverage Report (Sprint 7)",
    )
    parser.add_argument("--json-only", action="store_true",
                        help="Only output JSON report (no console/markdown)")
    parser.add_argument("--check", action="store_true",
                        help="Exit with code 1 if acceptance criteria not met (for CI)")

    args = parser.parse_args()

    report = await generate_coverage_report()

    # Export
    json_path = export_json(report)

    if not args.json_only:
        md_path = export_markdown(report)
        print_console_summary(report)
        print(f"  Reports saved:")
        print(f"    JSON:     {json_path}")
        print(f"    Markdown: {md_path}")
        print()
    else:
        print(json.dumps(report, indent=2, default=str))

    # CI check mode
    if args.check:
        acc = report.get("acceptance", {})
        if not (acc.get("acts_pass") and acc.get("judgments_pass")):
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())