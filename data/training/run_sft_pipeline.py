"""
NyayaMitra — SFT Dataset Pipeline Orchestrator (Sprint 8).

Master script that runs the complete SFT dataset construction pipeline:

    Stage 1: Extract headnote pairs from judgments (target: 25K)
    Stage 2: Generate synthetic pairs via Anthropic API (target: 15K)
    Stage 3: Convert NALSA/procedures to SFT format (target: 500+)
    Stage 4: Import all pairs into annotation queue
    Stage 5: Validate all pairs (format, citations, length, quality)
    Stage 6: Deduplicate by question similarity
    Stage 7: Create train/val/test splits (80/10/10)
    Stage 8: Generate dataset statistics report
    Stage 9: Run quality audit sample

Each stage can be run independently or as part of the full pipeline.
Stages are idempotent — safe to re-run.

Usage:
    # Full pipeline
    python -m data.training.run_sft_pipeline

    # Specific stages only
    python -m data.training.run_sft_pipeline --stages headnote,validate,split

    # Skip synthetic (no API key / save budget)
    python -m data.training.run_sft_pipeline --skip-synthetic

    # Limit headnote extraction (for testing)
    python -m data.training.run_sft_pipeline --headnote-limit 50

    # Dry run
    python -m data.training.run_sft_pipeline --dry-run
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

from data.training.sft_config import (
    SFT_PATHS,
    QUALITY_THRESHOLDS,
    DOMAIN_TARGETS,
)

logger = structlog.get_logger()

ALL_STAGES = [
    "headnote",
    "synthetic",
    "nalsa",
    "import",
    "validate",
    "split",
    "stats",
    "audit-sample",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════════


class SFTPipeline:
    """Orchestrates the complete SFT dataset construction."""

    def __init__(
        self,
        headnote_limit: int | None = None,
        synthetic_limit: int | None = None,
        skip_synthetic: bool = False,
        skip_dedup: bool = False,
        dry_run: bool = False,
    ):
        self.headnote_limit = headnote_limit
        self.synthetic_limit = synthetic_limit
        self.skip_synthetic = skip_synthetic
        self.skip_dedup = skip_dedup
        self.dry_run = dry_run
        self.results: dict[str, dict] = {}

    # ── Stage 1: Headnote Extraction ──────────────────────────────────────

    async def stage_headnote(self) -> dict:
        """Extract SFT pairs from judgment headnotes."""
        from data.training.headnote_extractor import (
            extract_headnote_pairs,
            export_pairs,
        )

        pairs = await extract_headnote_pairs(limit=self.headnote_limit)

        if not self.dry_run:
            path = export_pairs(pairs)
        else:
            path = "(dry run — not saved)"

        return {
            "pairs": len(pairs),
            "output": str(path),
        }

    # ── Stage 2: Synthetic Generation ─────────────────────────────────────

    async def stage_synthetic(self) -> dict:
        """Generate synthetic pairs via Anthropic API."""
        if self.skip_synthetic:
            return {"skipped": True, "reason": "skip_synthetic flag"}

        from data.training.synthetic_generator import (
            generate_synthetic_pairs,
            export_pairs,
        )

        pairs = await generate_synthetic_pairs(
            limit=self.synthetic_limit,
            dry_run=self.dry_run,
        )

        if not self.dry_run and pairs:
            path = export_pairs(pairs)
        else:
            path = "(dry run — not saved)"

        return {
            "pairs": len(pairs),
            "output": str(path),
        }

    # ── Stage 3: NALSA/Procedure Conversion ───────────────────────────────

    async def stage_nalsa(self) -> dict:
        """Convert procedures and NALSA FAQs to SFT format."""
        from data.training.nalsa_converter import convert_procedures, export_pairs

        pairs = convert_procedures()

        if not self.dry_run:
            path = export_pairs(pairs)
        else:
            path = "(dry run — not saved)"

        return {
            "pairs": len(pairs),
            "output": str(path),
        }

    # ── Stage 4: Import into Annotation Queue ─────────────────────────────

    async def stage_import(self) -> dict:
        """Import all raw pairs into the annotation queue."""
        from data.training.annotation_manager import AnnotationQueue

        queue = AnnotationQueue()

        raw_files = [
            SFT_PATHS.headnote_raw,
            SFT_PATHS.synthetic_raw,
            SFT_PATHS.nalsa_raw,
            SFT_PATHS.procedural_raw,
        ]
        existing = [f for f in raw_files if f.exists()]

        if not existing:
            return {"error": "No raw files found", "files_checked": len(raw_files)}

        if self.dry_run:
            return {"files_found": len(existing), "dry_run": True}

        stats = queue.import_raw_pairs(existing)
        return stats

    # ── Stage 5: Validate ─────────────────────────────────────────────────

    async def stage_validate(self) -> dict:
        """Validate all pairs for format, citations, quality."""
        from data.training.sft_validator import (
            validate_pairs,
            load_pairs_from_jsonl,
            save_pairs_to_jsonl,
            save_report,
        )

        # Load from all raw sources + accepted annotations
        all_pairs = []
        sources = [
            SFT_PATHS.headnote_raw,
            SFT_PATHS.synthetic_raw,
            SFT_PATHS.nalsa_raw,
            SFT_PATHS.procedural_raw,
            SFT_PATHS.annotation_dir / "accepted.jsonl",
        ]
        for src in sources:
            if src.exists():
                loaded = load_pairs_from_jsonl(src)
                all_pairs.extend(loaded)
                logger.info("loaded_for_validation", source=src.name, count=len(loaded))

        if not all_pairs:
            return {"error": "No pairs to validate"}

        if self.dry_run:
            return {"total_loaded": len(all_pairs), "dry_run": True}

        valid_pairs, report = await validate_pairs(
            all_pairs,
            skip_dedup=self.skip_dedup,
        )

        save_pairs_to_jsonl(valid_pairs, SFT_PATHS.validated_all)

        report_path = SFT_PATHS.audit_dir / "validation_report.json"
        save_report(report, report_path)

        return {
            "total_input": report.total,
            "valid": report.valid,
            "invalid": report.invalid,
            "duplicates_removed": report.duplicates_removed,
            "valid_rate": round(report.valid_rate, 4),
            "output": str(SFT_PATHS.validated_all),
            "report": str(report_path),
        }

    # ── Stage 6: Split ────────────────────────────────────────────────────

    async def stage_split(self) -> dict:
        """Create train/val/test splits."""
        from data.training.sft_splitter import (
            stratified_split,
            check_domain_balance,
            export_splits,
            export_stats,
            load_pairs,
        )

        if not SFT_PATHS.validated_all.exists():
            return {"error": "No validated pairs file. Run validate first."}

        pairs = load_pairs(SFT_PATHS.validated_all)

        if not pairs:
            return {"error": "No validated pairs found"}

        if self.dry_run:
            return {"total_validated": len(pairs), "dry_run": True}

        splits = stratified_split(pairs)
        balance = check_domain_balance(splits)
        paths = export_splits(splits)
        stats_path = export_stats(splits, balance)

        return {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": sum(len(s) for s in splits.values()),
            "balance_issues": sum(
                len(b.get("issues", [])) for b in balance.values()
            ),
            "output_dir": str(SFT_PATHS.splits_dir),
            "stats": str(stats_path),
        }

    # ── Stage 7: Stats Report ─────────────────────────────────────────────

    async def stage_stats(self) -> dict:
        """Generate comprehensive dataset statistics."""
        from collections import Counter

        stats = {"generated_at": datetime.utcnow().isoformat()}

        # Count pairs per source file
        for name, path in [
            ("headnote_raw", SFT_PATHS.headnote_raw),
            ("synthetic_raw", SFT_PATHS.synthetic_raw),
            ("nalsa_raw", SFT_PATHS.nalsa_raw),
            ("procedural_raw", SFT_PATHS.procedural_raw),
            ("validated", SFT_PATHS.validated_all),
            ("train", SFT_PATHS.train),
            ("val", SFT_PATHS.val),
            ("test", SFT_PATHS.test),
        ]:
            if path.exists():
                with open(path) as f:
                    count = sum(1 for line in f if line.strip())
                stats[name] = count
            else:
                stats[name] = 0

        # Domain distribution from validated file
        if SFT_PATHS.validated_all.exists():
            domains = Counter()
            sources = Counter()
            qtypes = Counter()
            with open(SFT_PATHS.validated_all) as f:
                for line in f:
                    if not line.strip():
                        continue
                    pair = json.loads(line)
                    domains[pair.get("domain", "unknown")] += 1
                    sources[pair.get("source", "unknown")] += 1
                    qtypes[pair.get("query_type", "unknown")] += 1
            stats["by_domain"] = dict(domains)
            stats["by_source"] = dict(sources)
            stats["by_query_type"] = dict(qtypes)

        # Acceptance criteria
        total_valid = stats.get("validated", 0)
        stats["target"] = QUALITY_THRESHOLDS.min_total_pairs
        stats["target_met"] = total_valid >= QUALITY_THRESHOLDS.min_total_pairs

        # Save
        report_path = SFT_PATHS.stats_report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        return stats

    # ── Stage 8: Audit Sample ─────────────────────────────────────────────

    async def stage_audit_sample(self) -> dict:
        """Generate quality audit sample for expert review."""
        from data.training.quality_audit import (
            load_validated_pairs,
            stratified_sample,
            export_sample,
        )

        pairs = load_validated_pairs()
        if not pairs:
            return {"error": "No validated pairs for audit"}

        if self.dry_run:
            return {"validated_pairs": len(pairs), "dry_run": True}

        sample = stratified_sample(pairs)
        path = export_sample(sample)

        return {
            "validated_pairs": len(pairs),
            "sample_size": len(sample),
            "sample_pct": round(len(sample) / len(pairs) * 100, 1),
            "output": str(path),
        }

    # ── Full Pipeline ─────────────────────────────────────────────────────

    async def run(self, stages: list[str] | None = None) -> dict:
        """Run the full pipeline or selected stages."""
        target_stages = stages or ALL_STAGES

        stage_map = {
            "headnote": self.stage_headnote,
            "synthetic": self.stage_synthetic,
            "nalsa": self.stage_nalsa,
            "import": self.stage_import,
            "validate": self.stage_validate,
            "split": self.stage_split,
            "stats": self.stage_stats,
            "audit-sample": self.stage_audit_sample,
        }

        pipeline_start = time.time()

        for stage_name in target_stages:
            if stage_name not in stage_map:
                logger.warning("unknown_stage", stage=stage_name)
                continue

            print(f"\n{'─' * 60}")
            print(f"  Stage: {stage_name}")
            print(f"{'─' * 60}")

            stage_start = time.time()

            try:
                stats = await stage_map[stage_name]()
                duration = round(time.time() - stage_start, 2)
                stats["duration_seconds"] = duration
                self.results[stage_name] = stats

                logger.info("stage_complete", stage=stage_name, duration=duration)
                print(f"  Completed in {duration}s")

                # Print key metrics
                if "pairs" in stats:
                    print(f"  Pairs: {stats['pairs']}")
                if "valid" in stats:
                    print(f"  Valid: {stats['valid']} ({stats.get('valid_rate', 0):.1%})")
                if "train" in stats:
                    print(f"  Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
                if stats.get("error"):
                    print(f"  Error: {stats['error']}")
                if stats.get("skipped"):
                    print(f"  Skipped: {stats.get('reason', '')}")

            except Exception as e:
                duration = round(time.time() - stage_start, 2)
                self.results[stage_name] = {
                    "error": str(e),
                    "duration_seconds": duration,
                }
                logger.error("stage_failed", stage=stage_name, error=str(e))
                print(f"  FAILED: {str(e)[:100]}")

        total_duration = round(time.time() - pipeline_start, 2)
        self.results["_total_duration_seconds"] = total_duration

        return self.results


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def print_final_report(results: dict) -> None:
    """Print pipeline results summary."""
    print(f"\n{'═' * 65}")
    print(f"  NyayaMitra — Sprint 8 SFT Pipeline Report")
    print(f"{'═' * 65}\n")

    for stage in ALL_STAGES:
        stats = results.get(stage, {})
        has_error = bool(stats.get("error"))
        skipped = bool(stats.get("skipped"))
        dry_run = bool(stats.get("dry_run"))

        if has_error:
            icon, status = "✗", "FAIL"
        elif skipped:
            icon, status = "○", "SKIP"
        elif dry_run:
            icon, status = "◇", "DRY"
        else:
            icon, status = "✓", "OK"

        duration = stats.get("duration_seconds", 0)
        print(f"  {icon} {stage:<18} [{status}]  {duration}s")

        # Key metrics
        if stage == "headnote" and stats.get("pairs"):
            print(f"      Pairs generated: {stats['pairs']}")
        elif stage == "synthetic":
            if stats.get("skipped"):
                print(f"      Reason: {stats.get('reason', '')}")
            elif stats.get("pairs"):
                print(f"      Pairs generated: {stats['pairs']}")
        elif stage == "nalsa" and stats.get("pairs"):
            print(f"      Pairs generated: {stats['pairs']}")
        elif stage == "import" and stats.get("imported"):
            print(f"      Imported: {stats['imported']}")
        elif stage == "validate":
            if stats.get("valid"):
                print(f"      Valid: {stats['valid']} / {stats['total_input']} "
                      f"({stats.get('valid_rate', 0):.1%})")
                print(f"      Duplicates removed: {stats.get('duplicates_removed', 0)}")
        elif stage == "split":
            if stats.get("train"):
                print(f"      Train: {stats['train']}, Val: {stats['val']}, "
                      f"Test: {stats['test']}")
        elif stage == "stats":
            if stats.get("target_met") is not None:
                target = stats.get("target", 50000)
                actual = stats.get("validated", 0)
                met = "PASS" if stats["target_met"] else "GAP"
                print(f"      Target: {target:,}, Actual: {actual:,} [{met}]")
        elif stage == "audit-sample" and stats.get("sample_size"):
            print(f"      Sample: {stats['sample_size']} pairs "
                  f"({stats.get('sample_pct', 0)}%)")

    # Total pairs summary
    headnote_pairs = results.get("headnote", {}).get("pairs", 0)
    synthetic_pairs = results.get("synthetic", {}).get("pairs", 0)
    nalsa_pairs = results.get("nalsa", {}).get("pairs", 0)
    validated = results.get("validate", {}).get("valid", 0)

    print(f"\n  Pipeline Summary:")
    print(f"    Headnote pairs:    {headnote_pairs:,}")
    print(f"    Synthetic pairs:   {synthetic_pairs:,}")
    print(f"    NALSA pairs:       {nalsa_pairs:,}")
    print(f"    Total raw:         {headnote_pairs + synthetic_pairs + nalsa_pairs:,}")
    print(f"    After validation:  {validated:,}")

    target = QUALITY_THRESHOLDS.min_total_pairs
    if validated >= target:
        print(f"\n  ║  TARGET MET: {validated:,} >= {target:,}  ║")
    else:
        gap = target - validated
        print(f"\n  ║  GAP: need {gap:,} more pairs ({validated:,} / {target:,})  ║")

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
        description="NyayaMitra SFT Dataset Pipeline (Sprint 8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Stage names (comma-separated for --stages):
  {', '.join(ALL_STAGES)}

Examples:
  python -m data.training.run_sft_pipeline                              # Full pipeline
  python -m data.training.run_sft_pipeline --stages headnote,nalsa      # Specific stages
  python -m data.training.run_sft_pipeline --skip-synthetic             # No API calls
  python -m data.training.run_sft_pipeline --headnote-limit 50 --dry-run
  python -m data.training.run_sft_pipeline --stages validate,split      # Re-validate + re-split
        """,
    )
    parser.add_argument(
        "--stages", type=str, default=None,
        help="Comma-separated list of stages to run",
    )
    parser.add_argument(
        "--headnote-limit", type=int, default=None,
        help="Max judgments for headnote extraction",
    )
    parser.add_argument(
        "--synthetic-limit", type=int, default=None,
        help="Max sections for synthetic generation",
    )
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Skip synthetic generation (saves API budget)",
    )
    parser.add_argument(
        "--skip-dedup", action="store_true",
        help="Skip deduplication in validation",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without writing files or calling APIs",
    )

    args = parser.parse_args()

    # Parse stages
    target_stages = None
    if args.stages:
        target_stages = [s.strip() for s in args.stages.split(",")]
        invalid = [s for s in target_stages if s not in ALL_STAGES]
        if invalid:
            print(f"Error: Unknown stages: {invalid}")
            print(f"Valid: {ALL_STAGES}")
            return

    # Ensure directories exist
    SFT_PATHS.ensure_dirs()

    print(f"\n{'═' * 65}")
    print(f"  NyayaMitra — Sprint 8 SFT Dataset Pipeline")
    if args.dry_run:
        print(f"  (DRY RUN — no files written, no API calls)")
    print(f"{'═' * 65}")

    pipeline = SFTPipeline(
        headnote_limit=args.headnote_limit,
        synthetic_limit=args.synthetic_limit,
        skip_synthetic=args.skip_synthetic,
        skip_dedup=args.skip_dedup,
        dry_run=args.dry_run,
    )

    results = await pipeline.run(stages=target_stages)

    # Save results
    results_path = SFT_PATHS.raw_dir.parent / "sft_pipeline_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    print_final_report(results)
    print(f"  Results saved to: {results_path}\n")


if __name__ == "__main__":
    asyncio.run(main())