"""
NyayaMitra — SFT Quality Audit Tool (Sprint 8).

Final quality gate before the SFT dataset is used for training.
A legal expert reviews a random 5% stratified sample and rates
each pair as pass/fail. If the batch acceptance rate drops below
90%, the entire dataset is flagged for rework.

Workflow:
    1. Sample 5% of validated pairs, stratified by domain
    2. Export sample as a reviewable JSONL (one pair per line)
    3. Expert reviews each pair: pass / fail + notes
    4. Import expert decisions
    5. Compute acceptance rate per domain and overall
    6. Generate audit report
    7. Gate: PASS if >=90% accepted, FAIL otherwise

Usage:
    # Generate audit sample from validated pairs
    python -m data.training.quality_audit sample

    # Import expert decisions after review
    python -m data.training.quality_audit submit --file audit_sample_reviewed.jsonl

    # View audit report
    python -m data.training.quality_audit report

    # Quick audit — interactive CLI review (for solo developer)
    python -m data.training.quality_audit interactive --limit 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
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
    LEGAL_DOMAINS,
)

logger = structlog.get_logger()

# ── Paths ─────────────────────────────────────────────────────────────────
AUDIT_DIR = SFT_PATHS.audit_dir
AUDIT_SAMPLE_FILE = AUDIT_DIR / "audit_sample.jsonl"
AUDIT_REVIEWED_FILE = AUDIT_DIR / "audit_sample_reviewed.jsonl"
AUDIT_REPORT_FILE = SFT_PATHS.audit_report

# ── Config ────────────────────────────────────────────────────────────────
SAMPLE_PERCENTAGE = 5.0         # 5% of dataset
MIN_SAMPLE_PER_DOMAIN = 3       # At least 3 pairs from each domain
PASS_THRESHOLD = QUALITY_THRESHOLDS.min_batch_acceptance_rate  # 0.90


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════════


def stratified_sample(
    pairs: list[SFTPair],
    sample_pct: float = SAMPLE_PERCENTAGE,
    min_per_domain: int = MIN_SAMPLE_PER_DOMAIN,
    seed: int = 42,
) -> list[SFTPair]:
    """
    Draw a stratified random sample from validated pairs.

    Ensures each domain gets at least min_per_domain pairs in the sample,
    with the rest drawn proportionally.
    """
    rng = random.Random(seed)

    # Group by domain
    by_domain: dict[str, list[SFTPair]] = {}
    for p in pairs:
        domain = p.domain or "general"
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(p)

    total_target = max(int(len(pairs) * sample_pct / 100), 10)
    sample: list[SFTPair] = []

    # First pass: guarantee minimum per domain
    remaining_target = total_target
    for domain, domain_pairs in by_domain.items():
        n = min(min_per_domain, len(domain_pairs))
        selected = rng.sample(domain_pairs, n)
        sample.extend(selected)
        remaining_target -= n

    # Second pass: fill remaining proportionally
    if remaining_target > 0:
        # Pool of unselected pairs
        selected_ids = {p.pair_id for p in sample}
        pool = [p for p in pairs if p.pair_id not in selected_ids]
        rng.shuffle(pool)
        sample.extend(pool[:remaining_target])

    rng.shuffle(sample)
    return sample


# ═══════════════════════════════════════════════════════════════════════════════
# Audit Report
# ═══════════════════════════════════════════════════════════════════════════════


def compute_audit_report(reviewed_pairs: list[dict]) -> dict:
    """
    Compute the audit report from expert-reviewed pairs.

    Each reviewed pair should have:
        - pair_id: str
        - audit_decision: "pass" or "fail"
        - audit_notes: str (optional)
        - domain: str
    """
    total = len(reviewed_pairs)
    if total == 0:
        return {"error": "No reviewed pairs found"}

    passed = sum(1 for p in reviewed_pairs if p.get("audit_decision") == "pass")
    failed = sum(1 for p in reviewed_pairs if p.get("audit_decision") == "fail")
    undecided = total - passed - failed

    acceptance_rate = passed / (passed + failed) if (passed + failed) > 0 else 0.0

    # Per-domain breakdown
    domain_stats = {}
    by_domain: dict[str, list[dict]] = {}
    for p in reviewed_pairs:
        d = p.get("domain", "unknown")
        if d not in by_domain:
            by_domain[d] = []
        by_domain[d].append(p)

    for domain, dpairs in by_domain.items():
        d_passed = sum(1 for p in dpairs if p.get("audit_decision") == "pass")
        d_failed = sum(1 for p in dpairs if p.get("audit_decision") == "fail")
        d_total = d_passed + d_failed
        domain_stats[domain] = {
            "total": len(dpairs),
            "passed": d_passed,
            "failed": d_failed,
            "acceptance_rate": round(d_passed / d_total, 4) if d_total > 0 else 0.0,
        }

    # Common failure notes
    failure_notes = [
        p.get("audit_notes", "").strip()
        for p in reviewed_pairs
        if p.get("audit_decision") == "fail" and p.get("audit_notes")
    ]

    report = {
        "audit_date": datetime.utcnow().isoformat(),
        "total_reviewed": total,
        "passed": passed,
        "failed": failed,
        "undecided": undecided,
        "acceptance_rate": round(acceptance_rate, 4),
        "threshold": PASS_THRESHOLD,
        "verdict": "PASS" if acceptance_rate >= PASS_THRESHOLD else "FAIL",
        "per_domain": domain_stats,
        "failure_notes": failure_notes[:20],
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive Audit (solo developer mode)
# ═══════════════════════════════════════════════════════════════════════════════


def run_interactive_audit(
    pairs: list[SFTPair],
    limit: int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    Run an interactive CLI-based audit for solo developers.

    Presents each pair and asks for pass/fail + optional notes.
    No external file exchange needed.
    """
    sample = stratified_sample(pairs, sample_pct=100, seed=seed)[:limit]

    print(f"\n{'═' * 65}")
    print(f"  Interactive Quality Audit — {len(sample)} pairs")
    print(f"  Commands: p=pass, f=fail, s=skip, q=quit")
    print(f"{'═' * 65}")

    reviewed = []

    for i, pair in enumerate(sample, 1):
        print(f"\n{'─' * 65}")
        print(f"  [{i}/{len(sample)}] ID: {pair.pair_id}  Domain: {pair.domain}  Source: {pair.source}")
        print(f"{'─' * 65}")
        print(f"\n  Q: {pair.instruction}\n")

        # Show response (truncated for readability)
        response_lines = pair.response.split("\n")
        for line in response_lines[:25]:
            print(f"  {line}")
        if len(response_lines) > 25:
            print(f"  ... ({len(response_lines) - 25} more lines)")

        print()
        print(f"  Cited sections: {pair.cited_sections}")
        print(f"  Cited cases: {pair.cited_cases}")
        print()

        while True:
            try:
                decision = input("  Decision (p=pass, f=fail, s=skip, q=quit): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                decision = "q"

            if decision in ("p", "f", "s", "q"):
                break
            print("  Invalid input. Use p/f/s/q.")

        if decision == "q":
            print("\n  Audit ended early by user.")
            break

        if decision == "s":
            continue

        notes = ""
        if decision == "f":
            try:
                notes = input("  Failure reason (optional): ").strip()
            except (EOFError, KeyboardInterrupt):
                notes = ""

        reviewed.append({
            "pair_id": pair.pair_id,
            "domain": pair.domain,
            "source": pair.source,
            "instruction": pair.instruction,
            "audit_decision": "pass" if decision == "p" else "fail",
            "audit_notes": notes,
        })

    return reviewed


# ═══════════════════════════════════════════════════════════════════════════════
# File I/O
# ═══════════════════════════════════════════════════════════════════════════════


def load_validated_pairs() -> list[SFTPair]:
    """Load validated pairs from the standard path."""
    path = SFT_PATHS.validated_all
    if not path.exists():
        logger.error("validated_pairs_not_found", path=str(path))
        return []

    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(SFTPair.from_dict(json.loads(line)))

    return pairs


def export_sample(sample: list[SFTPair]) -> Path:
    """Export audit sample as JSONL for expert review."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    with open(AUDIT_SAMPLE_FILE, "w", encoding="utf-8") as f:
        for pair in sample:
            record = pair.to_dict()
            # Add audit fields for the expert to fill in
            record["audit_decision"] = ""  # "pass" or "fail"
            record["audit_notes"] = ""
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return AUDIT_SAMPLE_FILE


def load_reviewed(path: Path) -> list[dict]:
    """Load expert-reviewed pairs from JSONL."""
    reviewed = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reviewed.append(json.loads(line))
    return reviewed


def save_report(report: dict) -> Path:
    """Save audit report as JSON."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_FILE.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return AUDIT_REPORT_FILE


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def print_report(report: dict) -> None:
    """Print audit report to console."""
    print(f"\n{'═' * 60}")
    print(f"  Quality Audit Report")
    print(f"{'═' * 60}\n")

    print(f"  Date:             {report.get('audit_date', 'N/A')}")
    print(f"  Pairs reviewed:   {report.get('total_reviewed', 0)}")
    print(f"  Passed:           {report.get('passed', 0)}")
    print(f"  Failed:           {report.get('failed', 0)}")
    print(f"  Undecided:        {report.get('undecided', 0)}")
    print(f"  Acceptance rate:  {report.get('acceptance_rate', 0):.1%}")
    print(f"  Threshold:        {report.get('threshold', 0.9):.0%}")

    verdict = report.get("verdict", "UNKNOWN")
    if verdict == "PASS":
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  VERDICT: PASS — Dataset approved    ║")
        print(f"  ╚══════════════════════════════════════╝")
    else:
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  VERDICT: FAIL — Rework required     ║")
        print(f"  ╚══════════════════════════════════════╝")

    # Per-domain breakdown
    domain_stats = report.get("per_domain", {})
    if domain_stats:
        print(f"\n  Per-domain breakdown:")
        print(f"    {'Domain':<16} {'Passed':>8} {'Failed':>8} {'Rate':>8}")
        print(f"    {'─' * 44}")
        for domain, ds in sorted(domain_stats.items()):
            rate = ds.get("acceptance_rate", 0)
            flag = " ⚠" if rate < PASS_THRESHOLD else ""
            print(
                f"    {domain:<16} {ds.get('passed', 0):>8} "
                f"{ds.get('failed', 0):>8} {rate:>7.0%}{flag}"
            )

    # Failure notes
    notes = report.get("failure_notes", [])
    if notes:
        print(f"\n  Failure reasons (top {min(len(notes), 10)}):")
        for note in notes[:10]:
            print(f"    - {note[:80]}")

    print(f"\n{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra SFT Quality Audit (Sprint 8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  sample        Generate stratified audit sample from validated pairs
  submit        Import expert decisions and compute acceptance rate
  report        View the latest audit report
  interactive   Run interactive CLI audit (solo developer mode)
        """,
    )
    parser.add_argument("command", choices=["sample", "submit", "report", "interactive"])
    parser.add_argument("--file", type=str, default=None,
                        help="Reviewed file path (for submit)")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of pairs for interactive audit (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")

    args = parser.parse_args()

    if args.command == "sample":
        pairs = load_validated_pairs()
        if not pairs:
            print("\n  No validated pairs found. Run the validator first.\n")
            return

        sample = stratified_sample(pairs, seed=args.seed)
        path = export_sample(sample)

        domains = Counter(p.domain for p in sample)
        print(f"\n  Audit sample generated:")
        print(f"    Total pairs: {len(pairs):,}")
        print(f"    Sample size: {len(sample)} ({len(sample) / len(pairs) * 100:.1f}%)")
        print(f"    Domains: {dict(domains)}")
        print(f"    File: {path}")
        print(f"\n  Send this file to the legal expert for review.")
        print(f"  Expert should set 'audit_decision' to 'pass' or 'fail' for each pair.\n")

    elif args.command == "submit":
        reviewed_path = Path(args.file) if args.file else AUDIT_REVIEWED_FILE
        if not reviewed_path.exists():
            print(f"\n  File not found: {reviewed_path}")
            print(f"  Expert should review {AUDIT_SAMPLE_FILE} and save as {AUDIT_REVIEWED_FILE}\n")
            return

        reviewed = load_reviewed(reviewed_path)
        report = compute_audit_report(reviewed)
        report_path = save_report(report)

        print_report(report)
        print(f"  Report saved to: {report_path}\n")

    elif args.command == "report":
        if not AUDIT_REPORT_FILE.exists():
            print(f"\n  No audit report found. Run 'sample' then 'submit' first.\n")
            return

        report = json.loads(AUDIT_REPORT_FILE.read_text(encoding="utf-8"))
        print_report(report)

    elif args.command == "interactive":
        pairs = load_validated_pairs()
        if not pairs:
            print("\n  No validated pairs found. Run the validator first.\n")
            return

        reviewed = run_interactive_audit(pairs, limit=args.limit, seed=args.seed)

        if reviewed:
            report = compute_audit_report(reviewed)
            report_path = save_report(report)

            # Also save reviewed pairs
            AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_REVIEWED_FILE, "w", encoding="utf-8") as f:
                for r in reviewed:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            print_report(report)
            print(f"  Report saved to: {report_path}")
            print(f"  Reviewed pairs saved to: {AUDIT_REVIEWED_FILE}\n")
        else:
            print("\n  No pairs reviewed.\n")


if __name__ == "__main__":
    main()