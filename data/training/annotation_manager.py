"""
NyayaMitra — Annotation Manager (Sprint 8).

Manages the human annotation workflow for SFT dataset verification.
Assigns synthetic/headnote pairs to annotators in batches, tracks
review status, collects corrections, and computes quality metrics.

Workflow:
    1. Import raw pairs (from headnote_extractor / synthetic_generator)
    2. Assign batches of 50 to annotators
    3. Track status: draft → needs_review → assigned → accepted/rejected
    4. Collect annotator decisions + corrections
    5. Compute inter-annotator agreement (IAA) on overlap pairs
    6. Export accepted pairs for validation pipeline

Storage:
    All annotation state is stored in JSONL files under
    data/datasets/sft/annotation/ — no database dependency.
    Each annotator gets their own batch file.

Usage:
    # Import raw pairs into the annotation queue
    python -m data.training.annotation_manager import-raw

    # Assign a batch to an annotator
    python -m data.training.annotation_manager assign --annotator alice

    # Record an annotator's decisions
    python -m data.training.annotation_manager submit --batch batch_alice_001.jsonl

    # View status dashboard
    python -m data.training.annotation_manager status

    # Export all accepted pairs
    python -m data.training.annotation_manager export

    # Compute inter-annotator agreement
    python -m data.training.annotation_manager iaa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from data.training.sft_config import (
    SFTPair,
    SFT_PATHS,
    QUALITY_THRESHOLDS,
    ANNOTATION_BATCH_SIZE,
    ANNOTATION_OVERLAP_PERCENTAGE,
    ANNOTATION_STATUS_FLOW,
)

logger = structlog.get_logger()

# ── Paths ─────────────────────────────────────────────────────────────────
QUEUE_FILE = SFT_PATHS.annotation_dir / "queue.jsonl"
ASSIGNMENTS_FILE = SFT_PATHS.annotation_dir / "assignments.json"
ACCEPTED_FILE = SFT_PATHS.annotation_dir / "accepted.jsonl"
REJECTED_FILE = SFT_PATHS.annotation_dir / "rejected.jsonl"
IAA_REPORT_FILE = SFT_PATHS.annotation_dir / "iaa_report.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Annotation Queue
# ═══════════════════════════════════════════════════════════════════════════════


class AnnotationQueue:
    """
    Manages the pool of pairs waiting for annotation.

    Pairs flow through: queue → assigned → accepted/rejected.
    State is persisted in JSONL files for simplicity.
    """

    def __init__(self):
        self.pairs: dict[str, dict] = {}  # pair_id → pair dict
        self.assignments: dict = {
            "batches": {},          # batch_id → {annotator, pair_ids, status, ...}
            "annotator_stats": {},  # annotator_id → {assigned, accepted, rejected, ...}
        }
        SFT_PATHS.annotation_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Load queue and assignments from disk."""
        # Load pairs
        if QUEUE_FILE.exists():
            with open(QUEUE_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pair = json.loads(line)
                        self.pairs[pair["pair_id"]] = pair

        # Load assignments
        if ASSIGNMENTS_FILE.exists():
            self.assignments = json.loads(
                ASSIGNMENTS_FILE.read_text(encoding="utf-8")
            )

    def _save_queue(self) -> None:
        """Persist queue to disk."""
        with open(QUEUE_FILE, "w", encoding="utf-8") as f:
            for pair in self.pairs.values():
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    def _save_assignments(self) -> None:
        """Persist assignments to disk."""
        ASSIGNMENTS_FILE.write_text(
            json.dumps(self.assignments, indent=2, default=str),
            encoding="utf-8",
        )

    # ── Import ────────────────────────────────────────────────────────────

    def import_raw_pairs(self, source_paths: list[Path]) -> dict:
        """
        Import raw pairs from JSONL files into the annotation queue.

        Sets status to "needs_review" for all imported pairs.
        Skips pairs already in the queue (by pair_id).
        """
        stats = {"imported": 0, "skipped_duplicate": 0, "skipped_accepted": 0, "files": 0}

        for path in source_paths:
            if not path.exists():
                logger.warning("import_file_not_found", path=str(path))
                continue

            stats["files"] += 1
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    pair = json.loads(line)
                    pid = pair.get("pair_id", "")

                    if pid in self.pairs:
                        # Skip if already accepted
                        if self.pairs[pid].get("status") == "accepted":
                            stats["skipped_accepted"] += 1
                        else:
                            stats["skipped_duplicate"] += 1
                        continue

                    pair["status"] = "needs_review"
                    pair["imported_at"] = datetime.utcnow().isoformat()
                    self.pairs[pid] = pair
                    stats["imported"] += 1

        self._save_queue()
        logger.info("import_complete", **stats)
        return stats

    # ── Assignment ────────────────────────────────────────────────────────

    def assign_batch(
        self,
        annotator_id: str,
        batch_size: int = ANNOTATION_BATCH_SIZE,
        domain: str | None = None,
    ) -> dict:
        """
        Assign a batch of pairs to an annotator.

        Selects pairs with status "needs_review", creates a batch,
        and marks them as "assigned".

        Args:
            annotator_id: Unique annotator identifier.
            batch_size: Number of pairs per batch.
            domain: Optional domain filter.

        Returns:
            Batch metadata dict.
        """
        # Find eligible pairs
        eligible = [
            pid for pid, pair in self.pairs.items()
            if pair.get("status") == "needs_review"
            and (domain is None or pair.get("domain") == domain)
        ]

        if not eligible:
            logger.info("no_pairs_available", annotator=annotator_id, domain=domain)
            return {"error": "No pairs available for assignment"}

        # Select batch (with optional overlap for IAA)
        selected = eligible[:batch_size]

        # Add overlap pairs (already assigned to another annotator)
        overlap_count = max(1, int(batch_size * ANNOTATION_OVERLAP_PERCENTAGE / 100))
        assigned_pairs = [
            pid for pid, pair in self.pairs.items()
            if pair.get("status") == "assigned"
            and pair.get("annotator_id") != annotator_id
        ]
        if assigned_pairs:
            overlap = random.sample(
                assigned_pairs,
                min(overlap_count, len(assigned_pairs)),
            )
            # Don't count overlap against batch size
            selected = selected[:batch_size]  # Keep main batch at batch_size

        # Create batch
        batch_count = len(self.assignments.get("batches", {})) + 1
        batch_id = f"batch_{annotator_id}_{batch_count:03d}"

        # Update pair statuses
        for pid in selected:
            self.pairs[pid]["status"] = "assigned"
            self.pairs[pid]["annotator_id"] = annotator_id
            self.pairs[pid]["assigned_at"] = datetime.utcnow().isoformat()
            self.pairs[pid]["batch_id"] = batch_id

        # Record batch
        if "batches" not in self.assignments:
            self.assignments["batches"] = {}
        self.assignments["batches"][batch_id] = {
            "annotator_id": annotator_id,
            "pair_ids": selected,
            "batch_size": len(selected),
            "domain": domain,
            "assigned_at": datetime.utcnow().isoformat(),
            "status": "assigned",
            "submitted_at": None,
        }

        # Update annotator stats
        if "annotator_stats" not in self.assignments:
            self.assignments["annotator_stats"] = {}
        if annotator_id not in self.assignments["annotator_stats"]:
            self.assignments["annotator_stats"][annotator_id] = {
                "batches_assigned": 0,
                "batches_submitted": 0,
                "total_assigned": 0,
                "total_accepted": 0,
                "total_rejected": 0,
                "total_corrected": 0,
            }
        astats = self.assignments["annotator_stats"][annotator_id]
        astats["batches_assigned"] += 1
        astats["total_assigned"] += len(selected)

        # Export batch file for annotator
        batch_path = SFT_PATHS.annotation_dir / f"{batch_id}.jsonl"
        with open(batch_path, "w", encoding="utf-8") as f:
            for pid in selected:
                f.write(json.dumps(self.pairs[pid], ensure_ascii=False) + "\n")

        self._save_queue()
        self._save_assignments()

        logger.info(
            "batch_assigned",
            batch_id=batch_id,
            annotator=annotator_id,
            pairs=len(selected),
            file=str(batch_path),
        )

        return {
            "batch_id": batch_id,
            "annotator_id": annotator_id,
            "pairs": len(selected),
            "file": str(batch_path),
        }

    # ── Submission ────────────────────────────────────────────────────────

    def submit_batch(self, batch_path: Path) -> dict:
        """
        Process a submitted annotation batch.

        Reads the annotated JSONL file where the annotator has set
        status to "accepted" or "rejected" on each pair, and optionally
        modified the response.

        Expected fields per pair:
            - pair_id: unchanged
            - status: "accepted" or "rejected"
            - response: possibly corrected
            - review_notes: annotator comments
            - rejection_reason: if rejected (R1-R8)
        """
        if not batch_path.exists():
            return {"error": f"File not found: {batch_path}"}

        stats = {"accepted": 0, "rejected": 0, "corrected": 0, "errors": 0}

        with open(batch_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    annotated = json.loads(line)
                except json.JSONDecodeError:
                    stats["errors"] += 1
                    continue

                pid = annotated.get("pair_id", "")
                if pid not in self.pairs:
                    stats["errors"] += 1
                    continue

                new_status = annotated.get("status", "")
                if new_status not in ("accepted", "rejected"):
                    stats["errors"] += 1
                    continue

                # Update the pair
                original = self.pairs[pid]

                if new_status == "accepted":
                    # Check if response was modified (correction)
                    if annotated.get("response", "") != original.get("response", ""):
                        stats["corrected"] += 1
                        original["response"] = annotated["response"]
                    stats["accepted"] += 1

                elif new_status == "rejected":
                    stats["rejected"] += 1

                original["status"] = new_status
                original["review_notes"] = annotated.get("review_notes", "")
                original["rejection_reason"] = annotated.get("rejection_reason", "")
                original["reviewed_at"] = datetime.utcnow().isoformat()

        # Update batch status
        batch_name = batch_path.stem
        if batch_name in self.assignments.get("batches", {}):
            self.assignments["batches"][batch_name]["status"] = "submitted"
            self.assignments["batches"][batch_name]["submitted_at"] = datetime.utcnow().isoformat()

            # Update annotator stats
            annotator = self.assignments["batches"][batch_name].get("annotator_id")
            if annotator and annotator in self.assignments.get("annotator_stats", {}):
                astats = self.assignments["annotator_stats"][annotator]
                astats["batches_submitted"] += 1
                astats["total_accepted"] += stats["accepted"]
                astats["total_rejected"] += stats["rejected"]
                astats["total_corrected"] += stats["corrected"]

        self._save_queue()
        self._save_assignments()

        logger.info("batch_submitted", batch=batch_name, **stats)
        return stats

    # ── Export ────────────────────────────────────────────────────────────

    def export_accepted(self) -> dict:
        """Export all accepted pairs to a single JSONL file."""
        accepted = [
            pair for pair in self.pairs.values()
            if pair.get("status") == "accepted"
        ]

        with open(ACCEPTED_FILE, "w", encoding="utf-8") as f:
            for pair in accepted:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        # Also export rejected for analysis
        rejected = [
            pair for pair in self.pairs.values()
            if pair.get("status") == "rejected"
        ]
        with open(REJECTED_FILE, "w", encoding="utf-8") as f:
            for pair in rejected:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        stats = {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "accepted_path": str(ACCEPTED_FILE),
            "rejected_path": str(REJECTED_FILE),
        }
        logger.info("export_complete", **stats)
        return stats

    # ── Inter-Annotator Agreement ─────────────────────────────────────────

    def compute_iaa(self) -> dict:
        """
        Compute inter-annotator agreement on overlap pairs.

        Finds pairs reviewed by 2+ annotators and computes:
        - Raw agreement: % where both annotators made the same decision
        - Cohen's kappa: agreement adjusted for chance
        """
        # Find pairs with multiple reviews (overlap pairs)
        # For simplicity, check pairs that appear in multiple submitted batches
        pair_decisions: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for batch_id, batch in self.assignments.get("batches", {}).items():
            if batch.get("status") != "submitted":
                continue
            annotator = batch.get("annotator_id", "")
            for pid in batch.get("pair_ids", []):
                if pid in self.pairs:
                    status = self.pairs[pid].get("status", "")
                    if status in ("accepted", "rejected"):
                        pair_decisions[pid].append((annotator, status))

        # Only pairs with 2+ decisions
        overlap_pairs = {
            pid: decisions
            for pid, decisions in pair_decisions.items()
            if len(decisions) >= 2
        }

        if not overlap_pairs:
            return {
                "overlap_pairs": 0,
                "raw_agreement": 0.0,
                "cohens_kappa": 0.0,
                "note": "No overlap pairs found",
            }

        # Compute raw agreement
        agree = 0
        total = len(overlap_pairs)

        for pid, decisions in overlap_pairs.items():
            statuses = [d[1] for d in decisions[:2]]  # Take first 2
            if statuses[0] == statuses[1]:
                agree += 1

        raw_agreement = agree / total if total > 0 else 0.0

        # Compute Cohen's kappa
        # Count all decisions
        all_decisions = []
        for decisions in overlap_pairs.values():
            for _, status in decisions[:2]:
                all_decisions.append(status)

        n = total
        if n > 0:
            p_accept = all_decisions.count("accepted") / len(all_decisions)
            p_reject = all_decisions.count("rejected") / len(all_decisions)
            pe = p_accept ** 2 + p_reject ** 2  # Expected agreement by chance
            kappa = (raw_agreement - pe) / (1 - pe) if pe < 1.0 else 0.0
        else:
            kappa = 0.0

        report = {
            "overlap_pairs": total,
            "agreements": agree,
            "raw_agreement": round(raw_agreement, 4),
            "cohens_kappa": round(kappa, 4),
            "threshold": QUALITY_THRESHOLDS.min_inter_annotator_agreement,
            "pass": kappa >= QUALITY_THRESHOLDS.min_inter_annotator_agreement,
        }

        # Save report
        IAA_REPORT_FILE.write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )

        logger.info("iaa_computed", **report)
        return report

    # ── Status Dashboard ──────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get comprehensive status of the annotation pipeline."""
        status_counts = Counter(p.get("status", "unknown") for p in self.pairs.values())
        domain_counts = Counter(p.get("domain", "unknown") for p in self.pairs.values())
        source_counts = Counter(p.get("source", "unknown") for p in self.pairs.values())

        # Acceptance rate
        reviewed = status_counts.get("accepted", 0) + status_counts.get("rejected", 0)
        acceptance_rate = (
            status_counts.get("accepted", 0) / reviewed if reviewed > 0 else 0.0
        )

        # Rejection reasons
        rejection_reasons = Counter(
            p.get("rejection_reason", "unknown")
            for p in self.pairs.values()
            if p.get("status") == "rejected" and p.get("rejection_reason")
        )

        return {
            "total_pairs": len(self.pairs),
            "by_status": dict(status_counts),
            "by_domain": dict(domain_counts),
            "by_source": dict(source_counts),
            "reviewed": reviewed,
            "acceptance_rate": round(acceptance_rate, 4),
            "rejection_reasons": dict(rejection_reasons),
            "batches": len(self.assignments.get("batches", {})),
            "annotators": len(self.assignments.get("annotator_stats", {})),
            "annotator_stats": self.assignments.get("annotator_stats", {}),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def print_status(status: dict) -> None:
    """Print a formatted status dashboard."""
    print(f"\n{'═' * 60}")
    print(f"  Annotation Pipeline — Status Dashboard")
    print(f"{'═' * 60}\n")

    print(f"  Total pairs in queue: {status['total_pairs']:,}")
    print(f"  Reviewed: {status['reviewed']:,}")
    print(f"  Acceptance rate: {status['acceptance_rate']:.1%}")

    print(f"\n  By status:")
    for s, c in sorted(status["by_status"].items()):
        bar = "█" * min(c // 10, 30)
        print(f"    {s:<16} {c:>6,}  {bar}")

    print(f"\n  By domain:")
    for d, c in sorted(status["by_domain"].items(), key=lambda x: -x[1]):
        print(f"    {d:<16} {c:>6,}")

    print(f"\n  By source:")
    for s, c in sorted(status["by_source"].items(), key=lambda x: -x[1]):
        print(f"    {s:<16} {c:>6,}")

    if status.get("rejection_reasons"):
        print(f"\n  Top rejection reasons:")
        for reason, count in status["rejection_reasons"].most_common(5) if hasattr(status["rejection_reasons"], "most_common") else sorted(status["rejection_reasons"].items(), key=lambda x: -x[1])[:5]:
            print(f"    {reason:<8} {count:>4}")

    print(f"\n  Batches: {status['batches']}")
    print(f"  Annotators: {status['annotators']}")

    if status.get("annotator_stats"):
        print(f"\n  Annotator leaderboard:")
        print(f"    {'Name':<16} {'Assigned':>8} {'Accepted':>8} {'Rejected':>8} {'Rate':>8}")
        print(f"    {'─' * 52}")
        for name, astats in sorted(
            status["annotator_stats"].items(),
            key=lambda x: -x[1].get("total_accepted", 0),
        ):
            total = astats.get("total_accepted", 0) + astats.get("total_rejected", 0)
            rate = astats["total_accepted"] / total if total > 0 else 0
            print(
                f"    {name:<16} {astats.get('total_assigned', 0):>8} "
                f"{astats.get('total_accepted', 0):>8} "
                f"{astats.get('total_rejected', 0):>8} "
                f"{rate:>7.0%}"
            )

    print(f"\n{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra Annotation Manager (Sprint 8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  import-raw    Import raw pairs from JSONL files into annotation queue
  assign        Assign a batch to an annotator
  submit        Process a submitted annotated batch
  status        Show annotation pipeline status
  export        Export all accepted pairs
  iaa           Compute inter-annotator agreement
        """,
    )
    parser.add_argument("command", choices=[
        "import-raw", "assign", "submit", "status", "export", "iaa",
    ])
    parser.add_argument("--annotator", type=str, default=None,
                        help="Annotator ID (for assign)")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch file path (for submit)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Domain filter (for assign)")
    parser.add_argument("--batch-size", type=int, default=ANNOTATION_BATCH_SIZE,
                        help=f"Batch size (default: {ANNOTATION_BATCH_SIZE})")

    args = parser.parse_args()

    queue = AnnotationQueue()

    if args.command == "import-raw":
        # Import all raw pair files
        raw_files = [
            SFT_PATHS.headnote_raw,
            SFT_PATHS.synthetic_raw,
            SFT_PATHS.nalsa_raw,
            SFT_PATHS.procedural_raw,
        ]
        existing = [f for f in raw_files if f.exists()]
        if not existing:
            print("No raw pair files found. Run the extraction pipelines first.")
            return
        stats = queue.import_raw_pairs(existing)
        print(f"\n  Imported: {stats['imported']}")
        print(f"  Skipped (duplicate): {stats['skipped_duplicate']}")
        print(f"  Files processed: {stats['files']}\n")

    elif args.command == "assign":
        if not args.annotator:
            print("Error: --annotator is required for assign")
            return
        result = queue.assign_batch(
            annotator_id=args.annotator,
            batch_size=args.batch_size,
            domain=args.domain,
        )
        if "error" in result:
            print(f"\n  {result['error']}\n")
        else:
            print(f"\n  Batch assigned: {result['batch_id']}")
            print(f"  Annotator: {result['annotator_id']}")
            print(f"  Pairs: {result['pairs']}")
            print(f"  File: {result['file']}\n")

    elif args.command == "submit":
        if not args.batch:
            print("Error: --batch is required for submit")
            return
        result = queue.submit_batch(Path(args.batch))
        if "error" in result:
            print(f"\n  {result['error']}\n")
        else:
            print(f"\n  Accepted: {result['accepted']}")
            print(f"  Rejected: {result['rejected']}")
            print(f"  Corrected: {result['corrected']}")
            print(f"  Errors: {result['errors']}\n")

    elif args.command == "status":
        status = queue.get_status()
        print_status(status)

    elif args.command == "export":
        result = queue.export_accepted()
        print(f"\n  Accepted pairs: {result['accepted']}")
        print(f"  Rejected pairs: {result['rejected']}")
        print(f"  Accepted file: {result['accepted_path']}")
        print(f"  Rejected file: {result['rejected_path']}\n")

    elif args.command == "iaa":
        report = queue.compute_iaa()
        print(f"\n  Inter-Annotator Agreement:")
        print(f"    Overlap pairs: {report.get('overlap_pairs', 0)}")
        print(f"    Raw agreement: {report.get('raw_agreement', 0):.1%}")
        print(f"    Cohen's kappa: {report.get('cohens_kappa', 0):.3f}")
        threshold = report.get("threshold", 0.8)
        verdict = "PASS" if report.get("pass") else "FAIL"
        print(f"    Threshold:     {threshold}")
        print(f"    Verdict:       {verdict}\n")


if __name__ == "__main__":
    main()