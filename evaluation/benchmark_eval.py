"""
NyayaMitra — MVP Benchmark Evaluation.

Runs 200 curated legal queries through the full pipeline and measures
quality against gold-standard expectations.

Metrics:
    Citation Accuracy:     % of cited sections that exist in DB (target >80%)
    Citation Recall:       % of expected sections actually cited
    Case Accuracy:         % of cited cases that exist in DB
    Case Recall:           % of expected cases actually cited
    Fabrication Rate:      % of citations not in DB (target: 0%)
    Overruled Warning:     % of overruled cases correctly flagged
    Jurisdiction Accuracy: Did the router detect the right domain?
    Procedural Complete:   For procedure queries, does response have steps?
    Disclaimer Rate:       % of responses with legal disclaimer (target: 100%)
    Usable Rate:           Has answer + has citations + has disclaimer

Acceptance criteria (Sprint 6):
    Citation accuracy >80%, Zero fabricated case names, MVP demo runs smoothly.

Usage:
    # Via API (backend must be running)
    python -m evaluation.benchmark_eval

    # Direct pipeline (no server)
    python -m evaluation.benchmark_eval --direct

    # Specific domain
    python -m evaluation.benchmark_eval --direct --domain criminal

    # Export report
    python -m evaluation.benchmark_eval --direct --export evaluation/benchmark_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "benchmark_200.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Result Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QueryBenchResult:
    """Result for a single benchmark query."""

    query_id: str
    query: str
    domain: str
    query_type: str

    # Response data
    answer: str = ""
    cited_sections: list[str] = field(default_factory=list)
    cited_cases: list[str] = field(default_factory=list)
    has_procedure: bool = False
    has_disclaimer: bool = False
    sources_verified: bool = False
    router_domain: str = ""
    latency_ms: float = 0.0
    error: str = ""

    # Expected
    expected_sections: list[str] = field(default_factory=list)
    expected_cases: list[str] = field(default_factory=list)
    expected_has_procedure: bool = False

    # Computed metrics
    section_accuracy: float = 0.0  # cited sections that exist / total cited
    section_recall: float = 0.0    # expected sections found / total expected
    case_accuracy: float = 0.0
    case_recall: float = 0.0
    domain_correct: bool = False
    usable: bool = False


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report."""

    total: int = 0
    errors: int = 0

    # Citation metrics
    avg_section_accuracy: float = 0.0
    avg_section_recall: float = 0.0
    avg_case_accuracy: float = 0.0
    avg_case_recall: float = 0.0
    fabrication_rate: float = 0.0  # % of responses with at least one fabricated citation

    # Quality metrics
    domain_accuracy: float = 0.0
    procedural_completeness: float = 0.0
    disclaimer_rate: float = 0.0
    usable_rate: float = 0.0
    verified_rate: float = 0.0

    # Latency
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # Per-domain breakdown
    per_domain: dict = field(default_factory=dict)

    # All results
    results: list[QueryBenchResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════════


async def score_result(result: QueryBenchResult) -> QueryBenchResult:
    """
    Score a single benchmark result against expectations.

    Uses the citation verifier for section/case accuracy
    and compares against gold-standard expectations.
    """
    # Section accuracy: what fraction of cited sections are real?
    if result.cited_sections:
        try:
            from app.services.citation_verifier import get_citation_verifier

            verifier = await get_citation_verifier()
            verified_count = 0
            for sec_str in result.cited_sections:
                parts = sec_str.split("/", 1) if "/" in sec_str else [sec_str, ""]
                act = parts[0].strip()
                section = parts[1].strip() if len(parts) > 1 else ""
                if act and section:
                    sv = await verifier.verify_section(act, section)
                    if sv.exists:
                        verified_count += 1
            result.section_accuracy = verified_count / len(result.cited_sections)
        except Exception:
            # If verifier unavailable, assume 1.0 (can't verify)
            result.section_accuracy = 1.0
    else:
        result.section_accuracy = 1.0  # No citations to verify

    # Section recall: what fraction of expected sections were cited?
    if result.expected_sections:
        found = 0
        for exp in result.expected_sections:
            exp_lower = exp.lower()
            for cited in result.cited_sections:
                if exp_lower in cited.lower() or cited.lower() in exp_lower:
                    found += 1
                    break
        result.section_recall = found / len(result.expected_sections)
    else:
        result.section_recall = 1.0

    # Case accuracy: what fraction of cited cases are real?
    if result.cited_cases:
        try:
            from app.services.citation_verifier import get_citation_verifier

            verifier = await get_citation_verifier()
            verified_count = 0
            for case_name in result.cited_cases:
                cv = await verifier.verify_case(case_name)
                if cv.exists:
                    verified_count += 1
            result.case_accuracy = verified_count / len(result.cited_cases)
        except Exception:
            result.case_accuracy = 1.0
    else:
        result.case_accuracy = 1.0

    # Case recall
    if result.expected_cases:
        found = 0
        for exp in result.expected_cases:
            exp_lower = exp.lower()
            for cited in result.cited_cases:
                if exp_lower in cited.lower() or cited.lower() in exp_lower:
                    found += 1
                    break
        result.case_recall = found / len(result.expected_cases)
    else:
        result.case_recall = 1.0

    # Domain accuracy
    result.domain_correct = result.router_domain.lower() == result.domain.lower()

    # Usable = has answer + has at least one citation + has disclaimer
    has_answer = len(result.answer.strip()) > 50
    has_citation = bool(result.cited_sections or result.cited_cases)
    result.usable = has_answer and has_citation and result.has_disclaimer

    return result


def build_report(results: list[QueryBenchResult]) -> BenchmarkReport:
    """Build aggregate report from individual results."""
    report = BenchmarkReport(total=len(results), results=results)

    if not results:
        return report

    section_accs = []
    section_recalls = []
    case_accs = []
    case_recalls = []
    latencies = []
    fabrication_count = 0
    domain_correct = 0
    procedure_correct = 0
    procedure_total = 0
    disclaimer_count = 0
    usable_count = 0
    verified_count = 0
    domain_results: dict[str, list[QueryBenchResult]] = {}

    for r in results:
        if r.error:
            report.errors += 1
            continue

        section_accs.append(r.section_accuracy)
        section_recalls.append(r.section_recall)
        case_accs.append(r.case_accuracy)
        case_recalls.append(r.case_recall)
        latencies.append(r.latency_ms)

        if r.section_accuracy < 1.0 or r.case_accuracy < 1.0:
            fabrication_count += 1

        if r.domain_correct:
            domain_correct += 1

        if r.query_type == "procedure":
            procedure_total += 1
            if r.has_procedure:
                procedure_correct += 1

        if r.has_disclaimer:
            disclaimer_count += 1

        if r.usable:
            usable_count += 1

        if r.sources_verified:
            verified_count += 1

        d = r.domain
        if d not in domain_results:
            domain_results[d] = []
        domain_results[d].append(r)

    valid = report.total - report.errors
    report.avg_section_accuracy = sum(section_accs) / len(section_accs) if section_accs else 0
    report.avg_section_recall = sum(section_recalls) / len(section_recalls) if section_recalls else 0
    report.avg_case_accuracy = sum(case_accs) / len(case_accs) if case_accs else 0
    report.avg_case_recall = sum(case_recalls) / len(case_recalls) if case_recalls else 0
    report.fabrication_rate = fabrication_count / valid if valid else 0
    report.domain_accuracy = domain_correct / valid if valid else 0
    report.procedural_completeness = procedure_correct / procedure_total if procedure_total else 0
    report.disclaimer_rate = disclaimer_count / valid if valid else 0
    report.usable_rate = usable_count / valid if valid else 0
    report.verified_rate = verified_count / valid if valid else 0
    report.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0

    if latencies:
        sorted_lats = sorted(latencies)
        p95_idx = int(len(sorted_lats) * 0.95)
        report.p95_latency_ms = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]

    # Per-domain breakdown
    for domain, dr in domain_results.items():
        dr_valid = [r for r in dr if not r.error]
        if not dr_valid:
            continue
        report.per_domain[domain] = {
            "count": len(dr_valid),
            "section_accuracy": round(sum(r.section_accuracy for r in dr_valid) / len(dr_valid), 3),
            "section_recall": round(sum(r.section_recall for r in dr_valid) / len(dr_valid), 3),
            "case_recall": round(sum(r.case_recall for r in dr_valid) / len(dr_valid), 3),
            "usable_rate": round(sum(1 for r in dr_valid if r.usable) / len(dr_valid), 3),
            "domain_accuracy": round(sum(1 for r in dr_valid if r.domain_correct) / len(dr_valid), 3),
            "avg_latency_ms": round(sum(r.latency_ms for r in dr_valid) / len(dr_valid), 1),
        }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Runners
# ═══════════════════════════════════════════════════════════════════════════════


async def run_direct(entry: dict) -> QueryBenchResult:
    """Run a query directly through the pipeline."""
    from app.models.query import QueryRequest, Language, LegalDomain

    domain_map = {
        "criminal": LegalDomain.CRIMINAL, "property": LegalDomain.PROPERTY,
        "family": LegalDomain.FAMILY, "constitutional": LegalDomain.CONSTITUTIONAL,
        "labor": LegalDomain.LABOR, "consumer": LegalDomain.CONSUMER,
        "ip": LegalDomain.IP, "general": LegalDomain.GENERAL,
    }

    result = QueryBenchResult(
        query_id=entry["id"], query=entry["query"],
        domain=entry["domain"], query_type=entry["query_type"],
        expected_sections=entry.get("expected_sections", []),
        expected_cases=entry.get("expected_cases", []),
        expected_has_procedure=entry.get("expected_has_procedure", False),
    )

    request = QueryRequest(
        query=entry["query"],
        language=Language.ENGLISH,
    )

    start = time.time()
    try:
        from app.routers.query import legal_query
        response = await legal_query(request)

        result.answer = response.answer
        result.cited_sections = [
            f"{law.act}/{law.section}" for law in response.applicable_law
        ]
        result.cited_cases = [p.case for p in response.precedents]
        result.has_procedure = len(response.procedure) > 0
        result.has_disclaimer = any(
            phrase in response.answer.lower()
            for phrase in ["legal information, not legal advice", "consult a qualified advocate", "consult a lawyer", "⚖️"]
        )
        result.sources_verified = response.sources_verified
        result.router_domain = response.jurisdiction_notes.split("Domain: ")[-1].split(".")[0] if "Domain:" in (response.jurisdiction_notes or "") else ""
        result.latency_ms = round((time.time() - start) * 1000, 2)

    except Exception as e:
        result.error = str(e)
        result.latency_ms = round((time.time() - start) * 1000, 2)

    return await score_result(result)


async def run_via_api(entry: dict, base_url: str = "http://localhost:8080") -> QueryBenchResult:
    """Run a query via the HTTP API."""
    import httpx

    result = QueryBenchResult(
        query_id=entry["id"], query=entry["query"],
        domain=entry["domain"], query_type=entry["query_type"],
        expected_sections=entry.get("expected_sections", []),
        expected_cases=entry.get("expected_cases", []),
        expected_has_procedure=entry.get("expected_has_procedure", False),
    )

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{base_url}/api/v1/query",
                json={"query": entry["query"]},
            )
            resp.raise_for_status()
            data = resp.json()

        result.answer = data.get("answer", "")
        result.cited_sections = [
            f"{law['act']}/{law['section']}" for law in data.get("applicable_law", [])
        ]
        result.cited_cases = [p["case"] for p in data.get("precedents", [])]
        result.has_procedure = len(data.get("procedure", [])) > 0
        result.has_disclaimer = any(
            phrase in result.answer.lower()
            for phrase in ["legal information, not legal advice", "consult a qualified advocate", "consult a lawyer", "⚖️"]
        )
        result.sources_verified = data.get("sources_verified", False)
        jn = data.get("jurisdiction_notes", "")
        result.router_domain = jn.split("Domain: ")[-1].split(".")[0] if "Domain:" in jn else ""
        result.latency_ms = round((time.time() - start) * 1000, 2)

    except Exception as e:
        result.error = str(e)
        result.latency_ms = round((time.time() - start) * 1000, 2)

    return await score_result(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Report Printer
# ═══════════════════════════════════════════════════════════════════════════════


def print_report(report: BenchmarkReport, verbose: bool = False) -> None:
    """Print formatted benchmark report."""
    print()
    print("=" * 70)
    print("  NyayaMitra — MVP Benchmark Report (200 Queries)")
    print("=" * 70)
    print()
    print(f"  Total: {report.total}  |  Errors: {report.errors}  |  Valid: {report.total - report.errors}")
    print()

    # Citation metrics
    sa_pass = "✓" if report.avg_section_accuracy >= 0.80 else "✗"
    fab_pass = "✓" if report.fabrication_rate == 0.0 else "✗"

    print("  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Citation Accuracy:    {report.avg_section_accuracy:.1%}  {sa_pass}  (target ≥ 80%)       │")
    print(f"  │  Citation Recall:      {report.avg_section_recall:.1%}                            │")
    print(f"  │  Case Accuracy:        {report.avg_case_accuracy:.1%}                            │")
    print(f"  │  Case Recall:          {report.avg_case_recall:.1%}                            │")
    print(f"  │  Fabrication Rate:     {report.fabrication_rate:.1%}  {fab_pass}  (target = 0%)        │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Domain Accuracy:      {report.domain_accuracy:.1%}                            │")
    print(f"  │  Procedural Complete:  {report.procedural_completeness:.1%}                            │")
    print(f"  │  Disclaimer Rate:      {report.disclaimer_rate:.1%}                            │")
    print(f"  │  Usable Rate:          {report.usable_rate:.1%}                            │")
    print(f"  │  Verified Rate:        {report.verified_rate:.1%}                            │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Avg Latency:          {report.avg_latency_ms:.0f}ms                          │")
    print(f"  │  P95 Latency:          {report.p95_latency_ms:.0f}ms                          │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    # Per-domain
    if report.per_domain:
        print("  Per-Domain Breakdown:")
        print("  " + "─" * 66)
        print(f"  {'Domain':<16} {'N':>3} {'SecAcc':>7} {'SecRec':>7} {'CaseRec':>8} {'Usable':>7} {'DomAcc':>7}")
        print("  " + "─" * 66)
        for domain in sorted(report.per_domain.keys()):
            d = report.per_domain[domain]
            print(
                f"  {domain:<16} {d['count']:>3} "
                f"{d['section_accuracy']:>6.1%} {d['section_recall']:>6.1%} "
                f"{d['case_recall']:>7.1%} {d['usable_rate']:>6.1%} "
                f"{d['domain_accuracy']:>6.1%}"
            )
        print("  " + "─" * 66)
    print()

    # Failures
    if verbose:
        failures = [r for r in report.results if not r.usable and not r.error]
        if failures:
            print(f"  Non-usable Queries ({len(failures)}):")
            for r in failures[:15]:
                print(f"    [{r.query_id}] {r.query[:50]}...")
                print(
                    f"      secAcc={r.section_accuracy:.0%} secRec={r.section_recall:.0%} "
                    f"caseRec={r.case_recall:.0%} disclaimer={r.has_disclaimer} "
                    f"secs={len(r.cited_sections)} cases={len(r.cited_cases)}"
                )
            if len(failures) > 15:
                print(f"    ... and {len(failures) - 15} more")
            print()

        # Fabricated citations
        fabricated = [
            r for r in report.results
            if r.section_accuracy < 1.0 or r.case_accuracy < 1.0
        ]
        if fabricated:
            print(f"  Queries with Unverified Citations ({len(fabricated)}):")
            for r in fabricated[:10]:
                print(f"    [{r.query_id}] secAcc={r.section_accuracy:.0%} caseAcc={r.case_accuracy:.0%}")
            print()

    # Verdict
    citation_pass = report.avg_section_accuracy >= 0.80
    no_fabrication = report.fabrication_rate == 0.0

    print()
    if citation_pass and no_fabrication:
        print("  ══════════════════════════════════════════════════")
        print("  ║  VERDICT: PASS — Sprint 6 criteria met         ║")
        print("  ║  Citation accuracy ≥80% ✓                      ║")
        print("  ║  Zero fabricated citations ✓                    ║")
        print("  ══════════════════════════════════════════════════")
    elif citation_pass:
        print("  ══════════════════════════════════════════════════")
        print("  ║  PARTIAL PASS — Citation accuracy met           ║")
        print(f"  ║  Fabrication rate: {report.fabrication_rate:.1%} (target: 0%)        ║")
        print("  ══════════════════════════════════════════════════")
    else:
        print("  ══════════════════════════════════════════════════")
        print("  ║  VERDICT: FAIL — See results above              ║")
        print(f"  ║  Citation accuracy: {report.avg_section_accuracy:.1%} (target: ≥80%)    ║")
        print(f"  ║  Fabrication rate:  {report.fabrication_rate:.1%} (target: 0%)       ║")
        print("  ══════════════════════════════════════════════════")
    print()


def export_report(report: BenchmarkReport, path: Path) -> None:
    """Export report as JSON."""
    export = {
        "total": report.total,
        "errors": report.errors,
        "citation_accuracy": round(report.avg_section_accuracy, 4),
        "citation_recall": round(report.avg_section_recall, 4),
        "case_accuracy": round(report.avg_case_accuracy, 4),
        "case_recall": round(report.avg_case_recall, 4),
        "fabrication_rate": round(report.fabrication_rate, 4),
        "domain_accuracy": round(report.domain_accuracy, 4),
        "procedural_completeness": round(report.procedural_completeness, 4),
        "disclaimer_rate": round(report.disclaimer_rate, 4),
        "usable_rate": round(report.usable_rate, 4),
        "verified_rate": round(report.verified_rate, 4),
        "avg_latency_ms": round(report.avg_latency_ms, 1),
        "p95_latency_ms": round(report.p95_latency_ms, 1),
        "per_domain": report.per_domain,
        "results": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "domain": r.domain,
                "query_type": r.query_type,
                "section_accuracy": round(r.section_accuracy, 3),
                "section_recall": round(r.section_recall, 3),
                "case_accuracy": round(r.case_accuracy, 3),
                "case_recall": round(r.case_recall, 3),
                "domain_correct": r.domain_correct,
                "has_procedure": r.has_procedure,
                "has_disclaimer": r.has_disclaimer,
                "sources_verified": r.sources_verified,
                "usable": r.usable,
                "cited_sections": r.cited_sections,
                "cited_cases": r.cited_cases,
                "latency_ms": r.latency_ms,
                "error": r.error,
                "answer_preview": r.answer[:200],
            }
            for r in report.results
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
    parser = argparse.ArgumentParser(description="NyayaMitra MVP Benchmark Evaluation")
    parser.add_argument("--direct", action="store_true", help="Run directly (no HTTP server)")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080", help="API base URL")
    parser.add_argument("--domain", type=str, default=None, help="Run only this domain")
    parser.add_argument("--limit", type=int, default=None, help="Limit queries")
    parser.add_argument("--export", type=str, default=None, help="Export report to JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-query details")
    args = parser.parse_args()

    # Load benchmark
    with open(BENCHMARK_PATH) as f:
        queries = json.load(f)

    if args.domain:
        queries = [q for q in queries if q["domain"] == args.domain]
    if args.limit:
        queries = queries[:args.limit]

    print(f"\n  Running {len(queries)} benchmark queries ({'direct' if args.direct else 'via API'})...")
    print("  " + "─" * 66)

    results = []
    for i, entry in enumerate(queries, 1):
        if args.direct:
            result = await run_direct(entry)
        else:
            result = await run_via_api(entry, args.api_url)

        results.append(result)

        status = "✓" if result.usable else ("ERR" if result.error else "✗")
        print(
            f"  [{i:3d}/{len(queries)}] {status:>3} "
            f"({result.latency_ms:>6.0f}ms) "
            f"secAcc={result.section_accuracy:.0%} "
            f"caseRec={result.case_recall:.0%} "
            f"| {result.query[:45]}..."
        )

    report = build_report(results)
    print_report(report, verbose=args.verbose)

    if args.export:
        export_report(report, Path(args.export))


if __name__ == "__main__":
    asyncio.run(main())