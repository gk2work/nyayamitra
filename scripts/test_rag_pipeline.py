"""
NyayaMitra — RAG Pipeline Test Harness.

Runs 100 legal test queries through the full pipeline (retrieval →
LLM generation → response parsing) and evaluates answer quality.

Quality scoring (per query):
    - has_answer:       Response contains a non-empty answer (0/1)
    - has_citations:    Response cites at least one section or case (0/1)
    - has_sections:     applicable_law list is non-empty (0/1)
    - has_precedents:   precedents list is non-empty (0/1)
    - has_procedure:    procedure list is non-empty (0/1)
    - has_disclaimer:   Answer contains a legal disclaimer (0/1)
    - no_hallucination: Answer does NOT contain "I don't have" or obvious refusals (0/1)
    - usable:           has_answer AND has_citations AND no_hallucination (0/1)

Acceptance criteria (Sprint 4):
    60%+ of queries produce usable answers.

Usage:
    # Against the live API server (must be running on port 8080)
    python -m scripts.test_rag_pipeline

    # Direct pipeline call (no server needed)
    python -m scripts.test_rag_pipeline --direct

    # With specific LLM provider
    python -m scripts.test_rag_pipeline --direct --verbose

    # Export report
    python -m scripts.test_rag_pipeline --direct --export results.json
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


# ═══════════════════════════════════════════════════════════════════════════════
# 100 Test Queries — All 7 Domains
# ═══════════════════════════════════════════════════════════════════════════════

TEST_QUERIES = [
    # ─── Criminal Law (20 queries) ───────────────────────────────────────
    {"id": "crim_01", "query": "Can police arrest me without a warrant?", "domain": "criminal"},
    {"id": "crim_02", "query": "What is the punishment for murder under IPC?", "domain": "criminal"},
    {"id": "crim_03", "query": "How do I file an FIR at a police station?", "domain": "criminal"},
    {"id": "crim_04", "query": "What is the law on dowry death in India?", "domain": "criminal"},
    {"id": "crim_05", "query": "What are my rights if I am arrested by the police?", "domain": "criminal"},
    {"id": "crim_06", "query": "What happens if police refuse to register my FIR?", "domain": "criminal"},
    {"id": "crim_07", "query": "What is the penalty for theft under Indian law?", "domain": "criminal"},
    {"id": "crim_08", "query": "Is adultery still a crime in India?", "domain": "criminal"},
    {"id": "crim_09", "query": "When can bail be granted in non-bailable offences?", "domain": "criminal"},
    {"id": "crim_10", "query": "What is punishment for sexual assault or rape?", "domain": "criminal"},
    {"id": "crim_11", "query": "Is Section 377 still valid for same-sex relations?", "domain": "criminal"},
    {"id": "crim_12", "query": "What are the guidelines to prevent misuse of 498A?", "domain": "criminal"},
    {"id": "crim_13", "query": "What is the difference between cognizable and non-cognizable offences?", "domain": "criminal"},
    {"id": "crim_14", "query": "Can a woman be arrested at night?", "domain": "criminal"},
    {"id": "crim_15", "query": "What is anticipatory bail and how to apply for it?", "domain": "criminal"},
    {"id": "crim_16", "query": "What is the procedure for filing a criminal complaint?", "domain": "criminal"},
    {"id": "crim_17", "query": "What is causing death by negligence?", "domain": "criminal"},
    {"id": "crim_18", "query": "What is the punishment for cheating someone?", "domain": "criminal"},
    {"id": "crim_19", "query": "How long can police keep someone in custody without producing before a magistrate?", "domain": "criminal"},
    {"id": "crim_20", "query": "What is cruelty by husband under Section 498A?", "domain": "criminal"},

    # ─── Property Law (15 queries) ───────────────────────────────────────
    {"id": "prop_01", "query": "What is the definition of transfer of property?", "domain": "property"},
    {"id": "prop_02", "query": "What is the doctrine of lis pendens?", "domain": "property"},
    {"id": "prop_03", "query": "What notice period is required to terminate a lease?", "domain": "property"},
    {"id": "prop_04", "query": "What is a valid contract under Indian law?", "domain": "property"},
    {"id": "prop_05", "query": "How do I get compensation for breach of contract?", "domain": "property"},
    {"id": "prop_06", "query": "Do real estate builders need RERA registration?", "domain": "property"},
    {"id": "prop_07", "query": "What remedy do I have if my builder delays possession?", "domain": "property"},
    {"id": "prop_08", "query": "What is the legal definition of sale of property?", "domain": "property"},
    {"id": "prop_09", "query": "What are my rights as a tenant in India?", "domain": "property"},
    {"id": "prop_10", "query": "Can a landlord evict me without notice?", "domain": "property"},
    {"id": "prop_11", "query": "What is the meaning of immovable property?", "domain": "property"},
    {"id": "prop_12", "query": "How is stamp duty calculated on property transfer?", "domain": "property"},
    {"id": "prop_13", "query": "What are the elements needed for a valid agreement?", "domain": "property"},
    {"id": "prop_14", "query": "What is the difference between sale and gift of property?", "domain": "property"},
    {"id": "prop_15", "query": "Can a minor enter into a contract?", "domain": "property"},

    # ─── Family Law (15 queries) ─────────────────────────────────────────
    {"id": "fam_01", "query": "What are the grounds for divorce under Hindu law?", "domain": "family"},
    {"id": "fam_02", "query": "Can I get maintenance during divorce proceedings?", "domain": "family"},
    {"id": "fam_03", "query": "What protection does the domestic violence law provide?", "domain": "family"},
    {"id": "fam_04", "query": "Can people of different religions get married in India?", "domain": "family"},
    {"id": "fam_05", "query": "How much maintenance can a wife claim under Section 125 CrPC?", "domain": "family"},
    {"id": "fam_06", "query": "What is the minimum age for marriage in India?", "domain": "family"},
    {"id": "fam_07", "query": "What constitutes domestic violence?", "domain": "family"},
    {"id": "fam_08", "query": "How to file for mutual consent divorce?", "domain": "family"},
    {"id": "fam_09", "query": "What are child custody rights for mothers?", "domain": "family"},
    {"id": "fam_10", "query": "Can a husband claim maintenance from his wife?", "domain": "family"},
    {"id": "fam_11", "query": "What is the procedure for court marriage in India?", "domain": "family"},
    {"id": "fam_12", "query": "Is dowry demand a criminal offence?", "domain": "family"},
    {"id": "fam_13", "query": "What rights does a woman have to her husband's property?", "domain": "family"},
    {"id": "fam_14", "query": "How to get a protection order under domestic violence act?", "domain": "family"},
    {"id": "fam_15", "query": "What is the waiting period after divorce before remarriage?", "domain": "family"},

    # ─── Constitutional Law (15 queries) ─────────────────────────────────
    {"id": "const_01", "query": "What is the right to equality under the Indian Constitution?", "domain": "constitutional"},
    {"id": "const_02", "query": "What is the right to life and personal liberty?", "domain": "constitutional"},
    {"id": "const_03", "query": "What is the scope of freedom of speech in India?", "domain": "constitutional"},
    {"id": "const_04", "query": "Is privacy a fundamental right in India?", "domain": "constitutional"},
    {"id": "const_05", "query": "How to file a PIL in the Supreme Court?", "domain": "constitutional"},
    {"id": "const_06", "query": "How to get information from government under RTI?", "domain": "constitutional"},
    {"id": "const_07", "query": "Can the government restrict free speech online?", "domain": "constitutional"},
    {"id": "const_08", "query": "What is the right to education in India?", "domain": "constitutional"},
    {"id": "const_09", "query": "What are directive principles of state policy?", "domain": "constitutional"},
    {"id": "const_10", "query": "Can fundamental rights be suspended during emergency?", "domain": "constitutional"},
    {"id": "const_11", "query": "What is Article 370 and its current status?", "domain": "constitutional"},
    {"id": "const_12", "query": "What is the right against exploitation?", "domain": "constitutional"},
    {"id": "const_13", "query": "How does judicial review work in India?", "domain": "constitutional"},
    {"id": "const_14", "query": "What is the difference between fundamental rights and DPSP?", "domain": "constitutional"},
    {"id": "const_15", "query": "What is the procedure to amend the Constitution?", "domain": "constitutional"},

    # ─── Labor Law (12 queries) ──────────────────────────────────────────
    {"id": "labor_01", "query": "What is the legal definition of a strike?", "domain": "labor"},
    {"id": "labor_02", "query": "Can an employer terminate a worker without notice?", "domain": "labor"},
    {"id": "labor_03", "query": "What is the law against sexual harassment at workplace?", "domain": "labor"},
    {"id": "labor_04", "query": "What are fair wage principles?", "domain": "labor"},
    {"id": "labor_05", "query": "How to file a complaint about workplace harassment?", "domain": "labor"},
    {"id": "labor_06", "query": "What is the maximum working hours per week in India?", "domain": "labor"},
    {"id": "labor_07", "query": "What are the rules for overtime payment?", "domain": "labor"},
    {"id": "labor_08", "query": "Can an employer deduct wages without consent?", "domain": "labor"},
    {"id": "labor_09", "query": "What is the procedure for raising an industrial dispute?", "domain": "labor"},
    {"id": "labor_10", "query": "What are maternity leave entitlements in India?", "domain": "labor"},
    {"id": "labor_11", "query": "What are the rights of contract workers?", "domain": "labor"},
    {"id": "labor_12", "query": "What is the role of the Internal Complaints Committee?", "domain": "labor"},

    # ─── Consumer Law (12 queries) ───────────────────────────────────────
    {"id": "consumer_01", "query": "Where do I file a consumer complaint in India?", "domain": "consumer"},
    {"id": "consumer_02", "query": "What is the definition of consumer under CPA 2019?", "domain": "consumer"},
    {"id": "consumer_03", "query": "Is medical negligence covered under consumer law?", "domain": "consumer"},
    {"id": "consumer_04", "query": "What is the maximum claim in district consumer court?", "domain": "consumer"},
    {"id": "consumer_05", "query": "What rights do consumers have against defective products?", "domain": "consumer"},
    {"id": "consumer_06", "query": "How long do I have to file a consumer complaint?", "domain": "consumer"},
    {"id": "consumer_07", "query": "Can I file a consumer complaint online?", "domain": "consumer"},
    {"id": "consumer_08", "query": "What are unfair trade practices?", "domain": "consumer"},
    {"id": "consumer_09", "query": "What compensation can I get for deficient services?", "domain": "consumer"},
    {"id": "consumer_10", "query": "Can a builder be held liable under consumer protection?", "domain": "consumer"},
    {"id": "consumer_11", "query": "What is the difference between district, state, and national consumer commission?", "domain": "consumer"},
    {"id": "consumer_12", "query": "Can I claim refund for a cancelled flight under consumer law?", "domain": "consumer"},

    # ─── IP Law (11 queries) ─────────────────────────────────────────────
    {"id": "ip_01", "query": "What is the meaning of copyright under Indian law?", "domain": "ip"},
    {"id": "ip_02", "query": "What constitutes fair use under copyright law?", "domain": "ip"},
    {"id": "ip_03", "query": "Can I be prosecuted for offensive social media posts?", "domain": "ip"},
    {"id": "ip_04", "query": "Are court judgments protected by copyright?", "domain": "ip"},
    {"id": "ip_05", "query": "What is intermediary liability for user content?", "domain": "ip"},
    {"id": "ip_06", "query": "How do I register a trademark in India?", "domain": "ip"},
    {"id": "ip_07", "query": "What is the punishment for online fraud?", "domain": "ip"},
    {"id": "ip_08", "query": "Can someone use my photo without permission?", "domain": "ip"},
    {"id": "ip_09", "query": "What are the data protection laws in India?", "domain": "ip"},
    {"id": "ip_10", "query": "What is the penalty for software piracy?", "domain": "ip"},
    {"id": "ip_11", "query": "What is the IT Act Section 66A controversy?", "domain": "ip"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Quality Scorer
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QueryResult:
    """Result and quality scores for a single test query."""

    query_id: str
    query: str
    domain: str
    answer: str = ""
    num_laws: int = 0
    num_precedents: int = 0
    num_steps: int = 0
    confidence: str = ""
    llm_used: bool = False
    latency_ms: float = 0.0
    error: str = ""

    # Quality scores (0 or 1)
    has_answer: int = 0
    has_citations: int = 0
    has_sections: int = 0
    has_precedents: int = 0
    has_procedure: int = 0
    has_disclaimer: int = 0
    no_hallucination: int = 0
    usable: int = 0


def score_response(result: QueryResult) -> QueryResult:
    """Score a query result for quality metrics."""
    answer = result.answer.lower()

    result.has_answer = 1 if len(result.answer.strip()) > 50 else 0
    result.has_sections = 1 if result.num_laws > 0 else 0
    result.has_precedents = 1 if result.num_precedents > 0 else 0
    result.has_procedure = 1 if result.num_steps > 0 else 0
    result.has_citations = 1 if (result.has_sections or result.has_precedents) else 0

    # Check for disclaimer
    disclaimer_phrases = [
        "legal information, not legal advice",
        "consult a qualified advocate",
        "consult a lawyer",
        "seek professional legal",
        "⚖️",
    ]
    result.has_disclaimer = 1 if any(p in answer for p in disclaimer_phrases) else 0

    # Check for hallucination / refusal
    refusal_phrases = [
        "i don't have enough information",
        "i cannot provide",
        "i'm unable to",
        "i do not have access",
        "no relevant legal provisions",
        "could not find specific",
    ]
    has_refusal = any(p in answer for p in refusal_phrases)
    result.no_hallucination = 0 if has_refusal else 1

    # Usable = has a real answer with at least one citation
    result.usable = 1 if (result.has_answer and result.has_citations and result.no_hallucination) else 0

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════════


async def run_direct(query_entry: dict) -> QueryResult:
    """Run a query directly through the pipeline (no HTTP server)."""
    from app.models.query import QueryRequest, Language, LegalDomain

    # Map domain string to enum
    domain_map = {
        "criminal": LegalDomain.CRIMINAL,
        "property": LegalDomain.PROPERTY,
        "family": LegalDomain.FAMILY,
        "constitutional": LegalDomain.CONSTITUTIONAL,
        "labor": LegalDomain.LABOR,
        "consumer": LegalDomain.CONSUMER,
        "ip": LegalDomain.IP,
    }

    request = QueryRequest(
        query=query_entry["query"],
        domain_hint=domain_map.get(query_entry.get("domain")),
        language=Language.ENGLISH,
    )

    result = QueryResult(
        query_id=query_entry["id"],
        query=query_entry["query"],
        domain=query_entry.get("domain", "general"),
    )

    start = time.time()

    try:
        from app.routers.query import legal_query

        response = await legal_query(request)

        result.answer = response.answer
        result.num_laws = len(response.applicable_law)
        result.num_precedents = len(response.precedents)
        result.num_steps = len(response.procedure)
        result.confidence = response.confidence.value
        result.latency_ms = round((time.time() - start) * 1000, 2)

    except Exception as e:
        result.error = str(e)
        result.latency_ms = round((time.time() - start) * 1000, 2)

    return score_response(result)


async def run_via_api(query_entry: dict, base_url: str = "http://localhost:8080") -> QueryResult:
    """Run a query via the HTTP API."""
    import httpx

    result = QueryResult(
        query_id=query_entry["id"],
        query=query_entry["query"],
        domain=query_entry.get("domain", "general"),
    )

    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{base_url}/api/v1/query",
                json={
                    "query": query_entry["query"],
                    "domain_hint": query_entry.get("domain"),
                },
            )
            resp.raise_for_status()
            data = resp.json()

        result.answer = data.get("answer", "")
        result.num_laws = len(data.get("applicable_law", []))
        result.num_precedents = len(data.get("precedents", []))
        result.num_steps = len(data.get("procedure", []))
        result.confidence = data.get("confidence", "")
        result.latency_ms = round((time.time() - start) * 1000, 2)

    except Exception as e:
        result.error = str(e)
        result.latency_ms = round((time.time() - start) * 1000, 2)

    return score_response(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TestReport:
    """Aggregated test report."""

    total: int = 0
    usable: int = 0
    usable_pct: float = 0.0
    has_answer: int = 0
    has_citations: int = 0
    has_disclaimer: int = 0
    no_hallucination: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    avg_laws: float = 0.0
    avg_precedents: float = 0.0
    per_domain: dict = field(default_factory=dict)
    results: list[QueryResult] = field(default_factory=list)


def build_report(results: list[QueryResult]) -> TestReport:
    """Build aggregated report from individual results."""
    report = TestReport(total=len(results), results=results)

    latencies = []
    domain_results: dict[str, list[QueryResult]] = {}

    for r in results:
        report.usable += r.usable
        report.has_answer += r.has_answer
        report.has_citations += r.has_citations
        report.has_disclaimer += r.has_disclaimer
        report.no_hallucination += r.no_hallucination
        if r.error:
            report.errors += 1
        latencies.append(r.latency_ms)

        d = r.domain
        if d not in domain_results:
            domain_results[d] = []
        domain_results[d].append(r)

    report.usable_pct = round(report.usable / report.total * 100, 1) if report.total else 0
    report.avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else 0
    report.avg_laws = round(sum(r.num_laws for r in results) / len(results), 2) if results else 0
    report.avg_precedents = round(sum(r.num_precedents for r in results) / len(results), 2) if results else 0

    for domain, dr in domain_results.items():
        usable = sum(r.usable for r in dr)
        report.per_domain[domain] = {
            "count": len(dr),
            "usable": usable,
            "usable_pct": round(usable / len(dr) * 100, 1),
            "avg_latency_ms": round(sum(r.latency_ms for r in dr) / len(dr), 1),
        }

    return report


def print_report(report: TestReport, verbose: bool = False) -> None:
    """Print formatted test report."""
    print()
    print("=" * 70)
    print("  NyayaMitra — RAG Pipeline Test Report")
    print("=" * 70)
    print()
    print(f"  Total queries:     {report.total}")
    print(f"  Errors:            {report.errors}")
    print()

    pass_icon = "✓ PASS" if report.usable_pct >= 60 else "✗ FAIL"
    print("  ┌──────────────────────────────────────────────┐")
    print(f"  │  Usable answers:  {report.usable}/{report.total} ({report.usable_pct}%)  [{pass_icon}]  (target ≥ 60%)  │")
    print(f"  │  Has answer:      {report.has_answer}/{report.total}                              │")
    print(f"  │  Has citations:   {report.has_citations}/{report.total}                              │")
    print(f"  │  Has disclaimer:  {report.has_disclaimer}/{report.total}                              │")
    print(f"  │  No refusal:      {report.no_hallucination}/{report.total}                              │")
    print(f"  │  Avg latency:     {report.avg_latency_ms}ms                          │")
    print(f"  │  Avg laws/query:  {report.avg_laws}                              │")
    print(f"  │  Avg precs/query: {report.avg_precedents}                              │")
    print("  └──────────────────────────────────────────────┘")
    print()

    if report.per_domain:
        print("  Per-Domain Breakdown:")
        print("  ─────────────────────────────────────────────────")
        print(f"  {'Domain':<16} {'Count':>5} {'Usable':>7} {'Rate':>7} {'Latency':>9}")
        print("  ─────────────────────────────────────────────────")
        for domain in sorted(report.per_domain.keys()):
            d = report.per_domain[domain]
            print(
                f"  {domain:<16} {d['count']:>5} "
                f"{d['usable']:>7} {d['usable_pct']:>6.1f}% "
                f"{d['avg_latency_ms']:>8.1f}ms"
            )
        print("  ─────────────────────────────────────────────────")
    print()

    # Show failures
    failures = [r for r in report.results if not r.usable and not r.error]
    errors = [r for r in report.results if r.error]

    if errors:
        print(f"  Errors ({len(errors)}):")
        for r in errors[:5]:
            print(f"    ✗ [{r.query_id}] {r.query[:55]}...")
            print(f"      Error: {r.error[:80]}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")
        print()

    if failures and verbose:
        print(f"  Non-usable Queries ({len(failures)}):")
        for r in failures[:10]:
            print(f"    [{r.query_id}] {r.query[:55]}...")
            print(
                f"      answer={r.has_answer} citations={r.has_citations} "
                f"refusal={1 - r.no_hallucination} laws={r.num_laws} precs={r.num_precedents}"
            )
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
        print()

    if report.usable_pct >= 60:
        print("  ══════════════════════════════════════════════")
        print("  ║  VERDICT: PASS — Sprint 4 criteria met     ║")
        print("  ══════════════════════════════════════════════")
    else:
        print("  ══════════════════════════════════════════════")
        print("  ║  VERDICT: FAIL — See results above          ║")
        print("  ══════════════════════════════════════════════")
    print()


def export_report(report: TestReport, path: Path) -> None:
    """Export report as JSON."""
    export = {
        "total": report.total,
        "usable": report.usable,
        "usable_pct": report.usable_pct,
        "has_answer": report.has_answer,
        "has_citations": report.has_citations,
        "has_disclaimer": report.has_disclaimer,
        "no_hallucination": report.no_hallucination,
        "errors": report.errors,
        "avg_latency_ms": report.avg_latency_ms,
        "per_domain": report.per_domain,
        "results": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "domain": r.domain,
                "usable": r.usable,
                "has_answer": r.has_answer,
                "has_citations": r.has_citations,
                "num_laws": r.num_laws,
                "num_precedents": r.num_precedents,
                "num_steps": r.num_steps,
                "confidence": r.confidence,
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
    parser = argparse.ArgumentParser(description="NyayaMitra RAG Pipeline Test")
    parser.add_argument(
        "--direct", action="store_true",
        help="Run queries directly through pipeline (no HTTP server needed)",
    )
    parser.add_argument(
        "--api-url", type=str, default="http://localhost:8080",
        help="API base URL (when not using --direct)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of queries to run (for quick testing)",
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Run only queries for this domain",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export report to JSON file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed per-query results",
    )
    args = parser.parse_args()

    queries = TEST_QUERIES
    if args.domain:
        queries = [q for q in queries if q.get("domain") == args.domain]
    if args.limit:
        queries = queries[:args.limit]

    print(f"\nRunning {len(queries)} test queries ({'direct' if args.direct else 'via API'})...")
    print("─" * 70)

    results = []
    for i, entry in enumerate(queries, 1):
        if args.direct:
            result = await run_direct(entry)
        else:
            result = await run_via_api(entry, args.api_url)

        results.append(result)

        status = "✓" if result.usable else ("✗ ERR" if result.error else "✗")
        print(
            f"  [{i:3d}/{len(queries)}] {status} "
            f"({result.latency_ms:.0f}ms) "
            f"L={result.num_laws} P={result.num_precedents} "
            f"| {result.query[:50]}..."
        )

    report = build_report(results)
    print_report(report, verbose=args.verbose)

    if args.export:
        export_report(report, Path(args.export))


if __name__ == "__main__":
    asyncio.run(main())