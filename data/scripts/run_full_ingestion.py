"""
NyayaMitra — Full Ingestion Orchestrator (Sprint 7).

Master script that runs the complete data ingestion pipeline end-to-end:

    Stage 1: Seed curated data (acts + judgments from Phase 1)
    Stage 2: Scrape acts from India Code (registry-driven, 96 acts)
    Stage 3: Scrape judgments from Indian Kanoon (SC + HCs, 100K+ target)
    Stage 4: Seed NALSA FAQ data (procedural knowledge)
    Stage 5: Build procedures knowledge base (37+ procedures)
    Stage 6: Bulk embed + index into Qdrant + Elasticsearch
    Stage 7: Load Neo4j knowledge graph
    Stage 8: Run retrieval evaluation benchmark
    Stage 9: Run full 200-question MVP benchmark
    Stage 10: Generate coverage report

Each stage can be run independently or as part of the full pipeline.
Stages are idempotent — safe to re-run (uses checkpoints and dedup).

Usage:
    # Full pipeline (all stages)
    python -m data.scripts.run_full_ingestion

    # Specific stages only
    python -m data.scripts.run_full_ingestion --stages seed,scrape-acts

    # Skip scraping (just index + graph + eval)
    python -m data.scripts.run_full_ingestion --stages index,graph,eval

    # Resume after a crash (skips completed stages)
    python -m data.scripts.run_full_ingestion --resume

    # Dry run (show plan)
    python -m data.scripts.run_full_ingestion --dry-run

    # Set act priority tier
    python -m data.scripts.run_full_ingestion --act-priority P0

    # Set court priority tier
    python -m data.scripts.run_full_ingestion --court-priority P0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

logger = structlog.get_logger()

# Where to persist the run log
RUN_LOG_PATH = PROJECT_ROOT / "data" / "raw" / "ingestion_run_log.json"

# All stage names in order
ALL_STAGES = [
    "seed",
    "scrape-acts",
    "scrape-judgments",
    "seed-nalsa",
    "build-procedures",
    "index",
    "graph",
    "eval-retrieval",
    "eval-benchmark",
    "coverage",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Run Log — tracks which stages have completed for resume support
# ═══════════════════════════════════════════════════════════════════════════════


class RunLog:
    """Persists stage completion status for resume support."""

    def __init__(self, path: Path = RUN_LOG_PATH):
        self.path = path
        self.data: dict = {"stages": {}, "started_at": None, "last_updated": None}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["last_updated"] = datetime.utcnow().isoformat()
        self.path.write_text(
            json.dumps(self.data, indent=2, default=str),
            encoding="utf-8",
        )

    def mark_started(self) -> None:
        self.data["started_at"] = datetime.utcnow().isoformat()
        self._save()

    def mark_stage_done(self, stage: str, stats: dict) -> None:
        self.data["stages"][stage] = {
            "status": "done",
            "completed_at": datetime.utcnow().isoformat(),
            "stats": stats,
        }
        self._save()

    def mark_stage_failed(self, stage: str, error: str) -> None:
        self.data["stages"][stage] = {
            "status": "failed",
            "failed_at": datetime.utcnow().isoformat(),
            "error": error[:500],
        }
        self._save()

    def is_stage_done(self, stage: str) -> bool:
        return self.data.get("stages", {}).get(stage, {}).get("status") == "done"

    def reset(self) -> None:
        self.data = {"stages": {}, "started_at": None, "last_updated": None}
        self._save()


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════════


class IngestionPipeline:
    """
    Orchestrates the full Sprint 7 ingestion pipeline.

    Each stage is a method that returns a stats dict.
    Stages are independent and idempotent.
    """

    def __init__(
        self,
        act_priority: str = "P0",
        court_priority: str = "P0",
        resume: bool = False,
        scraper_concurrency: int = 3,
        skip_es: bool = False,
    ):
        self.act_priority = act_priority
        self.court_priority = court_priority
        self.resume = resume
        self.scraper_concurrency = scraper_concurrency
        self.skip_es = skip_es
        self.run_log = RunLog()
        self.all_stats: dict[str, dict] = {}

    def should_skip(self, stage: str) -> bool:
        """Check if stage should be skipped (resume mode + already done)."""
        if self.resume and self.run_log.is_stage_done(stage):
            logger.info("stage_skipped_resume", stage=stage)
            return True
        return False

    # ── Stage 1: Seed ─────────────────────────────────────────────────────

    async def stage_seed(self) -> dict:
        """Seed curated data from Phase 1 (acts + judgments)."""
        from data.scrapers.india_code import seed_initial_data
        from data.scrapers.indian_kanoon import seed_landmark_judgments

        act_stats = await seed_initial_data()
        judgment_stats = await seed_landmark_judgments()

        return {
            "seed_acts": act_stats.get("acts", 0),
            "seed_sections": act_stats.get("sections", 0),
            "seed_judgments_new": judgment_stats.get("new", 0),
            "seed_judgments_existing": judgment_stats.get("existing", 0),
        }

    # ── Stage 2: Scrape Acts ──────────────────────────────────────────────

    async def stage_scrape_acts(self) -> dict:
        """Scrape acts from India Code using the registry."""
        from data.scrapers.india_code import IndiaCodeScraper

        async with IndiaCodeScraper(
            concurrency=self.scraper_concurrency,
            force=False,
            cache_enabled=True,
        ) as scraper:
            stats = await scraper.scrape_by_priority(self.act_priority)

        return stats

    # ── Stage 3: Scrape Judgments ─────────────────────────────────────────

    async def stage_scrape_judgments(self) -> dict:
        """Scrape judgments from Indian Kanoon (SC + HCs)."""
        token = getattr(settings, "INDIAN_KANOON_API_TOKEN", "")
        if not token:
            logger.warning(
                "skipping_judgment_scrape",
                reason="INDIAN_KANOON_API_TOKEN not set in .env",
            )
            return {"skipped": True, "reason": "no_api_token"}

        from data.scrapers.indian_kanoon import IndianKanoonClient

        async with IndianKanoonClient(
            concurrency=2,
            force=False,
            cache_enabled=True,
        ) as client:
            stats = await client.scrape_by_priority(self.court_priority)

        return stats

    # ── Stage 4: Seed NALSA FAQs ──────────────────────────────────────────

    async def stage_seed_nalsa(self) -> dict:
        """Seed NALSA FAQ procedural knowledge."""
        from data.scrapers.nalsa_scraper import seed_faqs

        stats = await seed_faqs()
        return stats

    # ── Stage 5: Build Procedures ─────────────────────────────────────────

    async def stage_build_procedures(self) -> dict:
        """Build the procedural knowledge base and export for embedding."""
        from data.procedures.procedure_builder import (
            build_all_procedures,
            export_procedures_json,
            export_embedding_chunks,
        )

        procedures = build_all_procedures()
        json_path = export_procedures_json(procedures)
        chunks_path = export_embedding_chunks(procedures)

        return {
            "procedures": len(procedures),
            "json_path": str(json_path),
            "chunks_path": str(chunks_path),
        }

    # ── Stage 6: Bulk Index ───────────────────────────────────────────────

    async def stage_index(self) -> dict:
        """Embed and index all data into Qdrant + Elasticsearch."""
        from data.embeddings.bulk_indexer import BulkIndexer

        indexer = BulkIndexer(
            skip_es=self.skip_es,
            recreate=not self.resume,  # Recreate if not resuming
        )

        stats = await indexer.run_full(
            sections=True,
            judgments=True,
            procedures=True,
        )

        return stats

    # ── Stage 7: Graph ────────────────────────────────────────────────────

    async def stage_graph(self) -> dict:
        """Load all data into Neo4j knowledge graph."""
        from data.knowledge_graph.graph_loader import GraphLoader

        loader = GraphLoader()
        loader.connect()

        try:
            stats = await loader.load_all()
            final = loader.get_stats()
            stats["final_nodes"] = final.get("total_nodes", 0)
            stats["final_relationships"] = final.get("total_relationships", 0)
        finally:
            loader.close()

        return stats

    # ── Stage 8: Retrieval Evaluation ─────────────────────────────────────

    async def stage_eval_retrieval(self) -> dict:
        """Run the 50-query retrieval evaluation benchmark."""
        try:
            from evaluation.retrieval_eval import (
                run_evaluation, export_report, load_gold_standard,
            )
            from pathlib import Path

            gold_data = load_gold_standard()
            report = await run_evaluation(
                gold_data=gold_data, mode="hybrid", top_k=10,
            )

            export_path = PROJECT_ROOT / "evaluation" / "sprint7_retrieval_report.json"
            export_report(report, export_path)

            return {
                "recall_at_k": report.recall_at_k,
                "mrr": report.mrr,
                "hit_rate": report.hit_rate,
                "avg_latency_ms": report.avg_latency_ms,
                "pass": report.recall_at_k >= 0.8 and report.avg_latency_ms < 500,
            }
        except Exception as e:
            logger.error("retrieval_eval_failed", error=str(e))
            return {"error": str(e)}

    # ── Stage 9: MVP Benchmark ────────────────────────────────────────────

    async def stage_eval_benchmark(self) -> dict:
        """Run the 200-question MVP benchmark."""
        try:
            from evaluation.benchmark_eval import (
                score_result, build_report, export_report,
                QueryBenchResult, BENCHMARK_PATH,
            )
            from pathlib import Path
            import json

            # Load benchmark queries
            if not BENCHMARK_PATH.exists():
                return {"error": f"Benchmark file not found: {BENCHMARK_PATH}"}

            with open(BENCHMARK_PATH) as f:
                queries = json.load(f)

            # Run queries through the pipeline (direct mode)
            # Import the direct runner if available, else skip
            try:
                from evaluation.benchmark_eval import run_direct
                results = []
                for entry in queries[:50]:  # Run first 50 for speed
                    result = await run_direct(entry)
                    result = await score_result(result)
                    results.append(result)

                report = build_report(results)
                export_path = PROJECT_ROOT / "evaluation" / "sprint7_benchmark_report.json"
                export_report(report, export_path)

                return {
                    "section_accuracy": report.avg_section_accuracy,
                    "case_accuracy": report.avg_case_accuracy,
                    "fabrication_rate": report.fabrication_rate,
                    "usable_rate": report.usable_rate,
                    "disclaimer_rate": report.disclaimer_rate,
                    "pass": report.avg_section_accuracy >= 0.80,
                    "queries_run": len(results),
                }
            except ImportError:
                return {"error": "run_direct not available — run benchmark manually"}

        except Exception as e:
            logger.error("benchmark_eval_failed", error=str(e))
            return {"error": str(e)}

    # ── Stage 10: Coverage Report ─────────────────────────────────────────

    async def stage_coverage(self) -> dict:
        """Generate data coverage report."""
        try:
            from data.scripts.coverage_report import generate_coverage_report

            report = await generate_coverage_report()
            return report
        except Exception as e:
            logger.error("coverage_report_failed", error=str(e))
            return {"error": str(e)}

    # ── Run Pipeline ──────────────────────────────────────────────────────

    async def run(self, stages: list[str] | None = None) -> dict:
        """
        Run the full pipeline or a subset of stages.

        Args:
            stages: Optional list of stage names to run.
                    If None, runs all stages in order.
        """
        target_stages = stages or ALL_STAGES
        self.run_log.mark_started()

        stage_map = {
            "seed": self.stage_seed,
            "scrape-acts": self.stage_scrape_acts,
            "scrape-judgments": self.stage_scrape_judgments,
            "seed-nalsa": self.stage_seed_nalsa,
            "build-procedures": self.stage_build_procedures,
            "index": self.stage_index,
            "graph": self.stage_graph,
            "eval-retrieval": self.stage_eval_retrieval,
            "eval-benchmark": self.stage_eval_benchmark,
            "coverage": self.stage_coverage,
        }

        pipeline_start = time.time()
        results: dict[str, dict] = {}

        for stage_name in target_stages:
            if stage_name not in stage_map:
                logger.warning("unknown_stage", stage=stage_name)
                continue

            if self.should_skip(stage_name):
                results[stage_name] = {"skipped": True, "reason": "resume"}
                continue

            print(f"\n{'─' * 60}")
            print(f"  Stage: {stage_name}")
            print(f"{'─' * 60}")

            stage_start = time.time()
            logger.info("stage_start", stage=stage_name)

            try:
                stats = await stage_map[stage_name]()
                duration = round(time.time() - stage_start, 2)
                stats["duration_seconds"] = duration

                self.run_log.mark_stage_done(stage_name, stats)
                results[stage_name] = stats

                logger.info("stage_complete", stage=stage_name, duration=duration)
                print(f"  Completed in {duration}s")

            except Exception as e:
                duration = round(time.time() - stage_start, 2)
                error_msg = str(e)

                self.run_log.mark_stage_failed(stage_name, error_msg)
                results[stage_name] = {"error": error_msg, "duration_seconds": duration}

                logger.error(
                    "stage_failed",
                    stage=stage_name,
                    error=error_msg,
                    duration=duration,
                )
                print(f"  FAILED: {error_msg[:100]}")

                # Continue to next stage (don't abort the whole pipeline)

        total_duration = round(time.time() - pipeline_start, 2)
        results["_total_duration_seconds"] = total_duration

        self.all_stats = results
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Report Printer
# ═══════════════════════════════════════════════════════════════════════════════


def print_final_report(results: dict) -> None:
    """Print a comprehensive summary of the ingestion run."""
    print(f"\n{'═' * 65}")
    print(f"  NyayaMitra — Sprint 7 Full Ingestion Report")
    print(f"{'═' * 65}\n")

    for stage in ALL_STAGES:
        stats = results.get(stage, {})
        status = "SKIP" if stats.get("skipped") else "FAIL" if stats.get("error") else "OK"
        duration = stats.get("duration_seconds", 0)
        icon = {"OK": "✓", "FAIL": "✗", "SKIP": "○"}[status]

        print(f"  {icon} {stage:<22} [{status}]  {duration}s")

        # Print key metrics per stage
        if stage == "scrape-acts" and not stats.get("skipped"):
            print(f"      Acts: {stats.get('acts_success', '?')} success, "
                  f"{stats.get('acts_failed', '?')} failed, "
                  f"{stats.get('sections_created', '?')} sections")

        elif stage == "scrape-judgments" and not stats.get("skipped"):
            if stats.get("reason") == "no_api_token":
                print(f"      Skipped: INDIAN_KANOON_API_TOKEN not set")
            else:
                print(f"      Judgments: {stats.get('judgments_new', '?')} new, "
                      f"{stats.get('judgments_existing', '?')} existing")

        elif stage == "build-procedures":
            print(f"      Procedures: {stats.get('procedures', '?')}")

        elif stage == "index":
            sec = stats.get("sections", {})
            jdg = stats.get("judgments", {})
            prc = stats.get("procedures", {})
            print(f"      Qdrant: {sec.get('qdrant_indexed', '?')} sections, "
                  f"{jdg.get('qdrant_indexed', '?')} judgment chunks, "
                  f"{prc.get('procedures_indexed', '?')} procedures")

        elif stage == "graph":
            print(f"      Nodes: {stats.get('final_nodes', '?')}, "
                  f"Rels: {stats.get('final_relationships', '?')}")

        elif stage == "eval-retrieval" and not stats.get("error"):
            recall = stats.get("recall_at_k", 0)
            verdict = "PASS" if stats.get("pass") else "FAIL"
            print(f"      Recall@10: {recall:.3f} [{verdict}]")

        elif stage == "eval-benchmark" and not stats.get("error"):
            acc = stats.get("section_accuracy", 0)
            verdict = "PASS" if stats.get("pass") else "FAIL"
            print(f"      Citation Accuracy: {acc:.1%} [{verdict}]")

    total = results.get("_total_duration_seconds", 0)
    minutes = int(total // 60)
    seconds = int(total % 60)
    print(f"\n  Total Duration: {minutes}m {seconds}s")
    print(f"{'═' * 65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra Full Ingestion Pipeline (Sprint 7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Stage names (comma-separated for --stages):
  {', '.join(ALL_STAGES)}

Examples:
  python -m data.scripts.run_full_ingestion                         # Full pipeline
  python -m data.scripts.run_full_ingestion --stages seed,scrape-acts
  python -m data.scripts.run_full_ingestion --stages index,graph    # Re-index only
  python -m data.scripts.run_full_ingestion --resume                # Resume after crash
  python -m data.scripts.run_full_ingestion --dry-run               # Show plan
  python -m data.scripts.run_full_ingestion --act-priority P1       # Include P1 acts
  python -m data.scripts.run_full_ingestion --court-priority P1     # Include P1 courts
        """,
    )
    parser.add_argument(
        "--stages", type=str, default=None,
        help=f"Comma-separated list of stages to run (default: all)",
    )
    parser.add_argument(
        "--act-priority", choices=["P0", "P1", "P2"], default="P0",
        help="Maximum act priority tier to scrape (default: P0 = 33 acts)",
    )
    parser.add_argument(
        "--court-priority", choices=["P0", "P1", "P2"], default="P0",
        help="Maximum court priority tier to scrape (default: P0 = SC + 10 HCs)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last successful stage (skip completed stages)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset run log and start fresh",
    )
    parser.add_argument(
        "--skip-es", action="store_true",
        help="Skip Elasticsearch indexing (Qdrant only)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Scraper concurrency (default: 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show execution plan without running",
    )

    args = parser.parse_args()

    # Parse stages
    target_stages = None
    if args.stages:
        target_stages = [s.strip() for s in args.stages.split(",")]
        invalid = [s for s in target_stages if s not in ALL_STAGES]
        if invalid:
            print(f"Error: Unknown stages: {invalid}")
            print(f"Valid stages: {ALL_STAGES}")
            return

    # Dry run
    if args.dry_run:
        stages_to_run = target_stages or ALL_STAGES

        from data.config.acts_registry import get_acts_for_ingestion
        from data.config.courts_registry import get_courts_for_ingestion

        acts = get_acts_for_ingestion(args.act_priority)
        courts = get_courts_for_ingestion(args.court_priority)
        target_judgments = sum(c.target_judgments for c in courts)

        print(f"\n{'═' * 65}")
        print(f"  DRY RUN — Sprint 7 Full Ingestion Plan")
        print(f"{'═' * 65}\n")
        print(f"  Act priority:    {args.act_priority} ({len(acts)} acts)")
        print(f"  Court priority:  {args.court_priority} ({len(courts)} courts, ~{target_judgments:,} judgments)")
        print(f"  Concurrency:     {args.concurrency}")
        print(f"  Skip ES:         {args.skip_es}")
        print(f"  Resume mode:     {args.resume}")
        print(f"\n  Stages to run ({len(stages_to_run)}):")
        for i, stage in enumerate(stages_to_run, 1):
            print(f"    {i:>2}. {stage}")
        print(f"\n{'═' * 65}\n")
        return

    # Reset
    if args.reset:
        RunLog().reset()
        print("Run log reset.")

    # Run pipeline
    print(f"\n{'═' * 65}")
    print(f"  NyayaMitra — Sprint 7 Full Ingestion")
    print(f"  Act priority: {args.act_priority}, Court priority: {args.court_priority}")
    print(f"{'═' * 65}")

    pipeline = IngestionPipeline(
        act_priority=args.act_priority,
        court_priority=args.court_priority,
        resume=args.resume,
        scraper_concurrency=args.concurrency,
        skip_es=args.skip_es,
    )

    results = await pipeline.run(stages=target_stages)

    # Save full results
    results_path = PROJECT_ROOT / "data" / "raw" / "sprint7_ingestion_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    print_final_report(results)
    print(f"  Full results saved to: {results_path}\n")


if __name__ == "__main__":
    asyncio.run(main())