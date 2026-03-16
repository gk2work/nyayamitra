"""
NyayaMitra — Data Pipeline Orchestrator.

Single entry point to run the full data ingestion workflow in the correct
order. Replaces running each scraper/seeder script manually.

Pipeline stages (in order):
    1. Database init: ensure tables exist
    2. Seed acts: IPC, CrPC seed data (india_code)
    3. Seed judgments: landmark SC judgments (indian_kanoon + sci_scraper)
    4. Seed comprehensive: all 7 domains (seed_comprehensive)
    5. Scrape acts: India Code website (if --scrape)
    6. Scrape judgments: Indian Kanoon API + SCI website (if --scrape)
    7. Validate: run DataValidator on all DB records
    8. Report: print summary stats

Each stage logs its results and catches errors without stopping the
entire pipeline. If scraping fails, seeded data is still available.

Usage:
    python -m data.pipeline                 # seed only (default, safe)
    python -m data.pipeline --scrape        # seed + live scraping
    python -m data.pipeline --validate-only # validate existing DB data
    python -m data.pipeline --report        # print current DB stats
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings
from app.exceptions import PipelineError

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Stages
# ═══════════════════════════════════════════════════════════════════════════════


async def stage_init_db() -> dict:
    """Stage 1: Initialize database tables."""
    logger.info("pipeline_stage", stage="init_db", status="starting")

    from app.database import init_db

    await init_db()

    return {"status": "success"}


async def stage_seed_acts() -> dict:
    """Stage 2: Seed curated act and section data (IPC, CrPC)."""
    logger.info("pipeline_stage", stage="seed_acts", status="starting")

    from data.scrapers.india_code import seed_initial_data

    stats = await seed_initial_data()

    logger.info("pipeline_stage", stage="seed_acts", status="complete", **stats)
    return stats


async def stage_seed_judgments() -> dict:
    """Stage 3: Seed landmark SC judgments from Indian Kanoon + SCI sources."""
    logger.info("pipeline_stage", stage="seed_judgments", status="starting")

    combined_stats = {"ik_new": 0, "ik_existing": 0, "sci_new": 0, "sci_existing": 0}

    # Indian Kanoon landmark judgments
    from data.scrapers.indian_kanoon import seed_landmark_judgments

    ik_stats = await seed_landmark_judgments()
    combined_stats["ik_new"] = ik_stats.get("new", 0)
    combined_stats["ik_existing"] = ik_stats.get("existing", 0)

    # SCI seed judgments
    from data.scrapers.sci_scraper import seed_sci_judgments

    sci_stats = await seed_sci_judgments()
    combined_stats["sci_new"] = sci_stats.get("new", 0)
    combined_stats["sci_existing"] = sci_stats.get("existing", 0)

    logger.info("pipeline_stage", stage="seed_judgments", status="complete", **combined_stats)
    return combined_stats


async def stage_seed_comprehensive() -> dict:
    """Stage 4: Seed comprehensive data across all 7 legal domains."""
    logger.info("pipeline_stage", stage="seed_comprehensive", status="starting")

    from data.datasets.seed_comprehensive import seed_all

    stats = await seed_all()

    logger.info("pipeline_stage", stage="seed_comprehensive", status="complete", **stats)
    return stats


async def stage_scrape_acts() -> dict:
    """Stage 5: Scrape acts from India Code website."""
    logger.info("pipeline_stage", stage="scrape_acts", status="starting")

    from data.scrapers.india_code import IndiaCodeScraper

    async with IndiaCodeScraper() as scraper:
        stats = await scraper.scrape_priority_acts()

    logger.info("pipeline_stage", stage="scrape_acts", status="complete", **stats)
    return stats


async def stage_scrape_judgments() -> dict:
    """Stage 6: Scrape judgments from Indian Kanoon API and SCI website."""
    logger.info("pipeline_stage", stage="scrape_judgments", status="starting")

    combined_stats = {
        "ik_fetched": 0, "ik_new": 0, "ik_errors": 0,
        "sci_found": 0, "sci_new": 0, "sci_errors": 0,
    }

    # Indian Kanoon API (requires token)
    if settings.INDIAN_KANOON_API_TOKEN:
        from data.scrapers.indian_kanoon import IndianKanoonClient

        async with IndianKanoonClient() as client:
            ik_stats = await client.scrape_court_judgments(
                court="supremecourt",
                max_pages=5,
            )
            combined_stats["ik_fetched"] = ik_stats.get("judgments_fetched", 0)
            combined_stats["ik_new"] = ik_stats.get("judgments_new", 0)
            combined_stats["ik_errors"] = ik_stats.get("errors", 0)
    else:
        logger.warning(
            "pipeline_skip",
            stage="scrape_judgments",
            reason="INDIAN_KANOON_API_TOKEN not set, skipping Indian Kanoon scraping",
        )

    # SCI website scraping
    from data.scrapers.sci_scraper import SCIScraper

    async with SCIScraper() as scraper:
        sci_stats = await scraper.scrape_recent_judgments(max_pages=5)
        combined_stats["sci_found"] = sci_stats.get("judgments_found", 0)
        combined_stats["sci_new"] = sci_stats.get("judgments_new", 0)
        combined_stats["sci_errors"] = sci_stats.get("fetch_errors", 0)

    logger.info("pipeline_stage", stage="scrape_judgments", status="complete", **combined_stats)
    return combined_stats


async def stage_validate() -> dict:
    """Stage 7: Validate all records currently in the database."""
    logger.info("pipeline_stage", stage="validate", status="starting")

    from sqlalchemy import select, func

    from app.database import async_session
    from app.models.legal import Act, Section, Judgment
    from data.processors.validator import DataValidator

    validator = DataValidator()
    stats = {
        "acts_checked": 0, "acts_valid": 0, "acts_warnings": 0,
        "sections_checked": 0, "sections_valid": 0, "sections_warnings": 0,
        "judgments_checked": 0, "judgments_valid": 0, "judgments_warnings": 0,
    }

    async with async_session() as session:
        # Validate acts
        result = await session.execute(select(Act))
        acts = result.scalars().all()
        for act in acts:
            act_data = {
                "name": act.name,
                "year": act.year,
                "domain": act.domain,
                "status": act.status,
            }
            vr = validator.validate_act(act_data)
            stats["acts_checked"] += 1
            if vr.is_valid:
                stats["acts_valid"] += 1
            if vr.warnings:
                stats["acts_warnings"] += len(vr.warnings)

        # Validate sections
        result = await session.execute(select(Section))
        sections = result.scalars().all()
        for section in sections:
            sec_data = {
                "section_number": section.section_number,
                "text": section.text,
                "act_id": section.act_id,
            }
            vr = validator.validate_section(sec_data, act_id=section.act_id)
            stats["sections_checked"] += 1
            if vr.is_valid:
                stats["sections_valid"] += 1
            if vr.warnings:
                stats["sections_warnings"] += len(vr.warnings)

        # Validate judgments
        result = await session.execute(select(Judgment))
        judgments = result.scalars().all()
        for judgment in judgments:
            j_data = {
                "case_name": judgment.case_name,
                "year": judgment.year,
                "court": judgment.court,
                "court_type": judgment.court_type,
                "citation_scc": judgment.citation_scc,
                "citation_air": judgment.citation_air,
                "bench_size": judgment.bench_size,
                "domain": judgment.domain,
                "headnote": judgment.headnote,
                "sections_interpreted": judgment.sections_interpreted,
            }
            vr = validator.validate_judgment(j_data)
            stats["judgments_checked"] += 1
            if vr.is_valid:
                stats["judgments_valid"] += 1
            if vr.warnings:
                stats["judgments_warnings"] += len(vr.warnings)

    logger.info("pipeline_stage", stage="validate", status="complete", **stats)
    return stats


async def stage_report() -> dict:
    """Stage 8: Print summary statistics of the database."""
    logger.info("pipeline_stage", stage="report", status="starting")

    from sqlalchemy import select, func, distinct

    from app.database import async_session
    from app.models.legal import Act, Section, Judgment, IngestionLog

    stats = {}

    async with async_session() as session:
        # Acts count
        result = await session.execute(select(func.count(Act.id)))
        stats["total_acts"] = result.scalar_one()

        # Acts by domain
        result = await session.execute(
            select(Act.domain, func.count(Act.id)).group_by(Act.domain)
        )
        stats["acts_by_domain"] = {row[0]: row[1] for row in result.all()}

        # Sections count
        result = await session.execute(select(func.count(Section.id)))
        stats["total_sections"] = result.scalar_one()

        # Judgments count
        result = await session.execute(select(func.count(Judgment.id)))
        stats["total_judgments"] = result.scalar_one()

        # Judgments by domain
        result = await session.execute(
            select(Judgment.domain, func.count(Judgment.id)).group_by(Judgment.domain)
        )
        stats["judgments_by_domain"] = {
            (row[0] or "unclassified"): row[1] for row in result.all()
        }

        # Judgments by source
        result = await session.execute(
            select(Judgment.source, func.count(Judgment.id)).group_by(Judgment.source)
        )
        stats["judgments_by_source"] = {
            (row[0] or "unknown"): row[1] for row in result.all()
        }

        # Indexed counts
        result = await session.execute(
            select(func.count(Section.id)).where(Section.is_indexed.is_(True))
        )
        stats["sections_indexed"] = result.scalar_one()

        result = await session.execute(
            select(func.count(Judgment.id)).where(Judgment.is_indexed.is_(True))
        )
        stats["judgments_indexed"] = result.scalar_one()

        # Recent ingestion logs
        result = await session.execute(
            select(IngestionLog)
            .order_by(IngestionLog.started_at.desc())
            .limit(5)
        )
        recent_logs = result.scalars().all()
        stats["recent_ingestions"] = [
            {
                "source": log.source,
                "task": log.task,
                "status": log.status,
                "items_new": log.items_new,
                "started_at": log.started_at.isoformat() if log.started_at else None,
            }
            for log in recent_logs
        ]

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════════


async def run_pipeline(scrape: bool = False, validate_only: bool = False, report_only: bool = False):
    """
    Run the full data ingestion pipeline.

    Args:
        scrape: If True, run live scraping in addition to seeding.
        validate_only: If True, only validate existing DB data.
        report_only: If True, only print current DB stats.
    """
    start_time = time.time()
    stage_results: dict[str, dict] = {}
    stage_errors: dict[str, str] = {}

    print("\n" + "=" * 70)
    print("  NyayaMitra — Data Pipeline Orchestrator")
    print("=" * 70)

    if report_only:
        print("\n[Stage] Report: Current database statistics")
        stats = await stage_report()
        _print_report(stats)
        return

    # ── Stage 1: Init DB ──
    try:
        print("\n[Stage 1/7] Initializing database tables...")
        stage_results["init_db"] = await stage_init_db()
        print("  ✓ Database tables ready")
    except Exception as e:
        stage_errors["init_db"] = str(e)
        print(f"  ✗ Database init failed: {e}")
        print("  Cannot continue without database. Aborting.")
        return

    if validate_only:
        print("\n[Stage] Validating existing database records...")
        stage_results["validate"] = await stage_validate()
        _print_validation(stage_results["validate"])
        return

    # ── Stage 2: Seed Acts ──
    try:
        print("\n[Stage 2/7] Seeding curated act data (IPC, CrPC)...")
        stage_results["seed_acts"] = await stage_seed_acts()
        acts = stage_results["seed_acts"].get("acts", 0)
        sections = stage_results["seed_acts"].get("sections", 0)
        print(f"  ✓ {acts} new acts, {sections} new sections")
    except Exception as e:
        stage_errors["seed_acts"] = str(e)
        print(f"  ✗ Act seeding failed: {e}")

    # ── Stage 3: Seed Judgments ──
    try:
        print("\n[Stage 3/7] Seeding landmark SC judgments...")
        stage_results["seed_judgments"] = await stage_seed_judgments()
        s = stage_results["seed_judgments"]
        total_new = s.get("ik_new", 0) + s.get("sci_new", 0)
        total_existing = s.get("ik_existing", 0) + s.get("sci_existing", 0)
        print(f"  ✓ {total_new} new, {total_existing} already existing")
    except Exception as e:
        stage_errors["seed_judgments"] = str(e)
        print(f"  ✗ Judgment seeding failed: {e}")

    # ── Stage 4: Seed Comprehensive ──
    try:
        print("\n[Stage 4/7] Seeding comprehensive data (all 7 domains)...")
        stage_results["seed_comprehensive"] = await stage_seed_comprehensive()
        s = stage_results["seed_comprehensive"]
        print(
            f"  ✓ {s.get('acts_new', 0)} acts, {s.get('sections_new', 0)} sections, "
            f"{s.get('judgments_new', 0)} judgments"
        )
    except Exception as e:
        stage_errors["seed_comprehensive"] = str(e)
        print(f"  ✗ Comprehensive seeding failed: {e}")

    # ── Stage 5-6: Scraping (optional) ──
    if scrape:
        try:
            print("\n[Stage 5/7] Scraping acts from India Code...")
            stage_results["scrape_acts"] = await stage_scrape_acts()
            s = stage_results["scrape_acts"]
            print(f"  ✓ {s.get('acts_new', 0)} new acts, {s.get('sections_created', 0)} sections")
        except Exception as e:
            stage_errors["scrape_acts"] = str(e)
            print(f"  ✗ Act scraping failed: {e}")

        try:
            print("\n[Stage 6/7] Scraping judgments (Indian Kanoon + SCI)...")
            stage_results["scrape_judgments"] = await stage_scrape_judgments()
            s = stage_results["scrape_judgments"]
            print(
                f"  ✓ IK: {s.get('ik_new', 0)} new | SCI: {s.get('sci_new', 0)} new"
            )
        except Exception as e:
            stage_errors["scrape_judgments"] = str(e)
            print(f"  ✗ Judgment scraping failed: {e}")
    else:
        print("\n[Stage 5-6] Skipping live scraping (use --scrape to enable)")

    # ── Stage 7: Validate ──
    try:
        print("\n[Stage 7/7] Validating all database records...")
        stage_results["validate"] = await stage_validate()
        _print_validation(stage_results["validate"])
    except Exception as e:
        stage_errors["validate"] = str(e)
        print(f"  ✗ Validation failed: {e}")

    # ── Final Report ──
    print("\n[Report] Current database statistics:")
    try:
        report_stats = await stage_report()
        _print_report(report_stats)
    except Exception as e:
        print(f"  ✗ Report generation failed: {e}")

    # ── Summary ──
    duration = round(time.time() - start_time, 2)
    print(f"\n{'=' * 70}")
    print(f"  Pipeline completed in {duration}s")
    if stage_errors:
        print(f"  Stages with errors: {list(stage_errors.keys())}")
    else:
        print(f"  All stages completed successfully")
    print(f"{'=' * 70}\n")


def _print_validation(stats: dict) -> None:
    """Print validation results."""
    print(f"  Acts:      {stats['acts_valid']}/{stats['acts_checked']} valid ({stats['acts_warnings']} warnings)")
    print(f"  Sections:  {stats['sections_valid']}/{stats['sections_checked']} valid ({stats['sections_warnings']} warnings)")
    print(f"  Judgments: {stats['judgments_valid']}/{stats['judgments_checked']} valid ({stats['judgments_warnings']} warnings)")


def _print_report(stats: dict) -> None:
    """Print database report."""
    print(f"\n  Total acts:       {stats.get('total_acts', 0)}")
    print(f"  Total sections:   {stats.get('total_sections', 0)}")
    print(f"  Total judgments:  {stats.get('total_judgments', 0)}")
    print(f"  Sections indexed: {stats.get('sections_indexed', 0)}")
    print(f"  Judgments indexed: {stats.get('judgments_indexed', 0)}")

    acts_by_domain = stats.get("acts_by_domain", {})
    if acts_by_domain:
        print(f"\n  Acts by domain:")
        for domain, count in sorted(acts_by_domain.items()):
            print(f"    {domain:20s} {count}")

    judgments_by_domain = stats.get("judgments_by_domain", {})
    if judgments_by_domain:
        print(f"\n  Judgments by domain:")
        for domain, count in sorted(judgments_by_domain.items()):
            print(f"    {domain:20s} {count}")

    judgments_by_source = stats.get("judgments_by_source", {})
    if judgments_by_source:
        print(f"\n  Judgments by source:")
        for source, count in sorted(judgments_by_source.items()):
            print(f"    {source:20s} {count}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Parse args and run the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NyayaMitra — Data Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m data.pipeline                 # Seed all data (default)
  python -m data.pipeline --scrape        # Seed + live scraping
  python -m data.pipeline --validate-only # Validate existing DB records
  python -m data.pipeline --report        # Print DB statistics only
        """,
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Enable live web scraping (India Code + Indian Kanoon + SCI)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing database records (no seeding/scraping)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Only print current database statistics",
    )
    args = parser.parse_args()

    await run_pipeline(
        scrape=args.scrape,
        validate_only=args.validate_only,
        report_only=args.report,
    )


if __name__ == "__main__":
    asyncio.run(main())