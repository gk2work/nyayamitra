"""
NyayaMitra — Retrieval Evaluation Framework.

Evaluates the hybrid retrieval pipeline against a gold-standard dataset
of 50 test queries with known relevant documents.

Metrics:
    - Recall@K: fraction of gold-standard docs found in top-K results
    - MRR (Mean Reciprocal Rank): average of 1/rank of first relevant result
    - Per-domain breakdown: same metrics grouped by legal domain
    - Latency: average query time in milliseconds

Gold standard: evaluation/datasets/retrieval_gold.json
Each entry maps a natural-language legal question to the section numbers
and/or case names that MUST appear in the top-K results.

Acceptance criteria (Sprint 3):
    Recall@10 > 0.8 (40+ of 50 queries return relevant results)
    Average query latency < 500ms

Usage:
    python -m evaluation.retrieval_eval                     # full eval
    python -m evaluation.retrieval_eval --domain criminal   # single domain
    python -m evaluation.retrieval_eval --top-k 5           # stricter cutoff
    python -m evaluation.retrieval_eval --mode dense-only   # Qdrant only
    python -m evaluation.retrieval_eval --mode sparse-only  # ES only
    python -m evaluation.retrieval_eval --mode hybrid       # default: full pipeline
    python -m evaluation.retrieval_eval --verbose           # show per-query results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

logger = structlog.get_logger()

GOLD_FILE = PROJECT_ROOT / "evaluation" / "datasets" / "retrieval_gold.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QueryEvalResult:
    """Evaluation result for a single query."""

    query_id: str
    query: str
    domain: str
    recall: float = 0.0
    reciprocal_rank: float = 0.0
    latency_ms: float = 0.0
    retrieved_relevant: list[str] = field(default_factory=list)
    missed_relevant: list[str] = field(default_factory=list)
    top_k_texts: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    total_queries: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    queries_with_hit: int = 0
    hit_rate: float = 0.0
    per_domain: dict[str, dict] = field(default_factory=dict)
    per_query: list[QueryEvalResult] = field(default_factory=list)
    mode: str = "hybrid"
    top_k: int = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Gold Standard Loader
# ═══════════════════════════════════════════════════════════════════════════════


def load_gold_standard(path: Path | None = None) -> list[dict]:
    """
    Load the gold-standard evaluation dataset.

    Each entry:
        {
            "id": "crim_01",
            "query": "Can police arrest without a warrant?",
            "domain": "criminal",
            "relevant_sections": ["CrPC/41", "CrPC/41A"],
            "relevant_cases": ["Arnesh Kumar v. State of Bihar"],
            "notes": "optional explanation"
        }

    relevant_sections use format "ActShortName/SectionNumber".
    relevant_cases use the case_name field.
    """
    path = path or GOLD_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Gold standard file not found: {path}\n"
            f"Generate it with: evaluation/datasets/retrieval_gold.json"
        )

    with open(path) as f:
        data = json.load(f)

    logger.info("gold_standard_loaded", count=len(data), path=str(path))
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Matching Logic
# ═══════════════════════════════════════════════════════════════════════════════


def result_matches_gold(result, gold_sections: list[str], gold_cases: list[str]) -> bool:
    """
    Check if a single retrieval result matches any gold-standard reference.

    Section matching: "ActShortName/SectionNumber"
        e.g. "CrPC/41" matches a result with act_short_name="CrPC", section_number="41"

    Case matching: substring match on case_name (case-insensitive).
        e.g. "Arnesh Kumar" matches "Arnesh Kumar v. State of Bihar"
    """
    # Check section match
    if result.source_type == "act" and result.act_short_name and result.section_number:
        section_id = f"{result.act_short_name}/{result.section_number}"
        if section_id in gold_sections:
            return True

    # Check case match
    if result.source_type == "judgment" and result.case_name:
        case_lower = result.case_name.lower()
        for gold_case in gold_cases:
            if gold_case.lower() in case_lower or case_lower in gold_case.lower():
                return True

    return False


def get_match_id(result) -> str:
    """Get a human-readable ID for a matched result."""
    if result.source_type == "act" and result.act_short_name:
        return f"{result.act_short_name}/{result.section_number}"
    elif result.source_type == "judgment" and result.case_name:
        return result.case_name
    return f"unknown({result.chunk_type})"


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Runner
# ═══════════════════════════════════════════════════════════════════════════════


async def evaluate_single_query(
    service,
    entry: dict,
    top_k: int = 10,
    mode: str = "hybrid",
) -> QueryEvalResult:
    """
    Evaluate a single query against gold-standard references.

    Args:
        service: RetrievalService instance.
        entry: Gold standard entry with query, domain, relevant_sections/cases.
        top_k: Number of results to evaluate.
        mode: "hybrid", "dense-only", or "sparse-only".

    Returns:
        QueryEvalResult with recall, MRR, and latency.
    """
    query = entry["query"]
    domain = entry.get("domain")
    gold_sections = entry.get("relevant_sections", [])
    gold_cases = entry.get("relevant_cases", [])
    total_relevant = len(gold_sections) + len(gold_cases)

    if total_relevant == 0:
        logger.warning("no_gold_references", query_id=entry["id"])
        return QueryEvalResult(
            query_id=entry["id"],
            query=query,
            domain=domain or "unknown",
        )

    # NOTE: We intentionally do NOT pass domain as a filter to search().
    # The evaluation should measure raw retrieval quality — whether the
    # pipeline can find relevant documents across all domains. Domain
    # filtering is the query router's job (Sprint 5) and is tested
    # separately. Many gold references are cross-domain (e.g., a
    # "criminal" query whose landmark judgment is tagged "constitutional").
    use_reranker = mode == "hybrid"

    start = time.time()

    results = await service.search(
        query=query,
        top_k=top_k,
        use_reranker=use_reranker,
    )

    latency_ms = round((time.time() - start) * 1000, 2)

    # For dense-only or sparse-only, we'd need to modify the service.
    # For now, mode selection is handled via use_reranker flag.
    # Full mode isolation (disable ES or Qdrant) is a future enhancement.

    # Evaluate matches
    retrieved_relevant = []
    first_relevant_rank = None

    for rank, result in enumerate(results, start=1):
        if result_matches_gold(result, gold_sections, gold_cases):
            match_id = get_match_id(result)
            retrieved_relevant.append(match_id)
            if first_relevant_rank is None:
                first_relevant_rank = rank

    # Determine what was missed
    all_gold = set(gold_sections + gold_cases)
    missed = all_gold - set(retrieved_relevant)

    recall = len(retrieved_relevant) / total_relevant if total_relevant > 0 else 0.0
    rr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    return QueryEvalResult(
        query_id=entry["id"],
        query=query,
        domain=domain or "unknown",
        recall=round(recall, 4),
        reciprocal_rank=round(rr, 4),
        latency_ms=latency_ms,
        retrieved_relevant=retrieved_relevant,
        missed_relevant=list(missed),
    )


async def run_evaluation(
    gold_data: list[dict],
    top_k: int = 10,
    mode: str = "hybrid",
    domain_filter: str | None = None,
    verbose: bool = False,
) -> EvalReport:
    """
    Run the full evaluation suite.

    Args:
        gold_data: Gold standard query-document pairs.
        top_k: Number of results to evaluate per query.
        mode: "hybrid", "dense-only", or "sparse-only".
        domain_filter: If set, only evaluate queries in this domain.
        verbose: Print per-query results.

    Returns:
        EvalReport with aggregate and per-domain metrics.
    """
    from app.services.retrieval import RetrievalService

    # Filter by domain if specified
    if domain_filter:
        gold_data = [e for e in gold_data if e.get("domain") == domain_filter]

    if not gold_data:
        print(f"No queries found for domain: {domain_filter}")
        return EvalReport()

    # Initialize retrieval service
    print("Initializing retrieval service...")
    service = RetrievalService()
    await service.initialize()

    print(f"Running evaluation: {len(gold_data)} queries, top_k={top_k}, mode={mode}")
    print("─" * 70)

    # Warmup: run one throwaway query to prime caches, JIT, and
    # connection pools. This eliminates the cold-start penalty
    # (~1000ms) that would otherwise inflate the first query's latency.
    await service.search(query="warmup query", top_k=1, use_reranker=(mode == "hybrid"))
    print("  (warmup complete)")

    report = EvalReport(
        total_queries=len(gold_data),
        mode=mode,
        top_k=top_k,
    )

    all_recalls = []
    all_rrs = []
    all_latencies = []
    domain_results: dict[str, list[QueryEvalResult]] = {}

    for i, entry in enumerate(gold_data, 1):
        result = await evaluate_single_query(service, entry, top_k=top_k, mode=mode)
        report.per_query.append(result)

        all_recalls.append(result.recall)
        all_rrs.append(result.reciprocal_rank)
        all_latencies.append(result.latency_ms)

        if result.recall > 0:
            report.queries_with_hit += 1

        # Group by domain
        d = result.domain
        if d not in domain_results:
            domain_results[d] = []
        domain_results[d].append(result)

        if verbose:
            status = "✓" if result.recall > 0 else "✗"
            print(
                f"  [{i:2d}/{len(gold_data)}] {status} "
                f"R={result.recall:.2f} "
                f"RR={result.reciprocal_rank:.2f} "
                f"({result.latency_ms:.0f}ms) "
                f"| {result.query[:60]}..."
            )
            if result.missed_relevant:
                print(f"           Missed: {result.missed_relevant}")

    # Aggregate metrics
    report.recall_at_k = round(sum(all_recalls) / len(all_recalls), 4) if all_recalls else 0.0
    report.mrr = round(sum(all_rrs) / len(all_rrs), 4) if all_rrs else 0.0
    report.avg_latency_ms = round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0.0
    report.hit_rate = round(report.queries_with_hit / report.total_queries, 4) if report.total_queries else 0.0

    # Per-domain breakdown
    for domain, results in domain_results.items():
        recalls = [r.recall for r in results]
        rrs = [r.reciprocal_rank for r in results]
        latencies = [r.latency_ms for r in results]
        hits = sum(1 for r in results if r.recall > 0)

        report.per_domain[domain] = {
            "count": len(results),
            "recall_at_k": round(sum(recalls) / len(recalls), 4),
            "mrr": round(sum(rrs) / len(rrs), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "hit_rate": round(hits / len(results), 4),
        }

    # Close connections to avoid 'Unclosed connector' warnings
    await service.close()

    return report


def print_report(report: EvalReport) -> None:
    """Print a formatted evaluation report to stdout."""
    print()
    print("=" * 70)
    print("  NyayaMitra — Retrieval Evaluation Report")
    print("=" * 70)
    print()
    print(f"  Mode:              {report.mode}")
    print(f"  Top-K:             {report.top_k}")
    print(f"  Total queries:     {report.total_queries}")
    print()

    # Acceptance criteria check
    recall_pass = report.recall_at_k >= 0.8
    latency_pass = report.avg_latency_ms < 500
    recall_icon = "✓ PASS" if recall_pass else "✗ FAIL"
    latency_icon = "✓ PASS" if latency_pass else "✗ FAIL"

    print("  ┌──────────────────────────────────────────────┐")
    print(f"  │  Recall@{report.top_k:<2d}:      {report.recall_at_k:.4f}  [{recall_icon}]  (target > 0.8)  │")
    print(f"  │  MRR:            {report.mrr:.4f}                         │")
    print(f"  │  Hit Rate:       {report.hit_rate:.4f}  ({report.queries_with_hit}/{report.total_queries} queries)          │")
    print(f"  │  Avg Latency:    {report.avg_latency_ms:.1f}ms  [{latency_icon}]  (target < 500ms)│")
    print("  └──────────────────────────────────────────────┘")
    print()

    # Per-domain breakdown
    if report.per_domain:
        print("  Per-Domain Breakdown:")
        print("  ─────────────────────────────────────────────────")
        print(f"  {'Domain':<16} {'Count':>5} {'Recall@K':>9} {'MRR':>7} {'Hit%':>7} {'Latency':>8}")
        print("  ─────────────────────────────────────────────────")

        for domain in sorted(report.per_domain.keys()):
            d = report.per_domain[domain]
            print(
                f"  {domain:<16} {d['count']:>5} "
                f"{d['recall_at_k']:>9.4f} {d['mrr']:>7.4f} "
                f"{d['hit_rate'] * 100:>6.1f}% "
                f"{d['avg_latency_ms']:>7.1f}ms"
            )

        print("  ─────────────────────────────────────────────────")
    print()

    # Failure analysis
    failures = [q for q in report.per_query if q.recall == 0]
    if failures:
        print(f"  Failed Queries ({len(failures)}):")
        for f in failures[:10]:
            print(f"    ✗ [{f.query_id}] {f.query[:60]}...")
            print(f"      Expected: {f.missed_relevant}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
        print()

    # Overall verdict
    if recall_pass and latency_pass:
        print("  ══════════════════════════════════════════════")
        print("  ║  VERDICT: PASS — Sprint 3 criteria met     ║")
        print("  ══════════════════════════════════════════════")
    else:
        print("  ══════════════════════════════════════════════")
        print("  ║  VERDICT: FAIL — See failures above         ║")
        print("  ══════════════════════════════════════════════")
    print()


def export_report(report: EvalReport, path: Path) -> None:
    """Export evaluation report as JSON for CI/CD integration."""
    export = {
        "mode": report.mode,
        "top_k": report.top_k,
        "total_queries": report.total_queries,
        "recall_at_k": report.recall_at_k,
        "mrr": report.mrr,
        "avg_latency_ms": report.avg_latency_ms,
        "hit_rate": report.hit_rate,
        "queries_with_hit": report.queries_with_hit,
        "acceptance_criteria": {
            "recall_pass": report.recall_at_k >= 0.8,
            "latency_pass": report.avg_latency_ms < 500,
            "overall_pass": report.recall_at_k >= 0.8 and report.avg_latency_ms < 500,
        },
        "per_domain": report.per_domain,
        "per_query": [
            {
                "query_id": q.query_id,
                "query": q.query,
                "domain": q.domain,
                "recall": q.recall,
                "reciprocal_rank": q.reciprocal_rank,
                "latency_ms": q.latency_ms,
                "retrieved_relevant": q.retrieved_relevant,
                "missed_relevant": q.missed_relevant,
            }
            for q in report.per_query
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"Report exported to: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra Retrieval Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of results to evaluate (default: 10)",
    )
    parser.add_argument(
        "--mode", choices=["hybrid", "dense-only", "sparse-only"], default="hybrid",
        help="Retrieval mode to evaluate (default: hybrid)",
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Evaluate only queries in this domain",
    )
    parser.add_argument(
        "--gold-file", type=str, default=None,
        help="Path to gold standard JSON file",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export report to JSON file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-query results",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold_file) if args.gold_file else None
    gold_data = load_gold_standard(gold_path)

    report = await run_evaluation(
        gold_data=gold_data,
        top_k=args.top_k,
        mode=args.mode,
        domain_filter=args.domain,
        verbose=args.verbose,
    )

    print_report(report)

    if args.export:
        export_report(report, Path(args.export))


if __name__ == "__main__":
    asyncio.run(main())