"""
NyayaMitra — MVP Demo Query Runner.

Runs 15 curated demo queries that showcase NyayaMitra's best capabilities.
Saves formatted output as markdown files for a 5-minute MVP walkthrough.

Demo scenarios:
    Criminal  (5): Arrest rights, FIR filing, bail, 498A guidelines, murder
    Property  (5): Tenant rights, RERA delay, sale deed, eviction, contract
    Consumer  (5): Defective product, medical negligence, online refund,
                   misleading ads, consumer court filing

Usage:
    # Via API (backend must be running)
    python -m scripts.demo_queries

    # Direct pipeline
    python -m scripts.demo_queries --direct

    # Custom output directory
    python -m scripts.demo_queries --output evaluation/demo_output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "demo_output"

DEMO_QUERIES = [
    # ─── Criminal (5) ────────────────────────────────────────────────────
    {
        "id": "demo_crim_01",
        "title": "Arrest Without Warrant",
        "query": "What are my rights if police arrest me without a warrant?",
        "scenario": "A citizen stopped by police wants to know their rights.",
    },
    {
        "id": "demo_crim_02",
        "title": "FIR Filing Procedure",
        "query": "How do I file an FIR at a police station? What if they refuse?",
        "scenario": "A victim of theft needs to file a police complaint.",
    },
    {
        "id": "demo_crim_03",
        "title": "Bail in Non-Bailable Offence",
        "query": "My brother has been arrested for a non-bailable offence. How can I get him bail?",
        "scenario": "A family member seeking bail for an arrested relative.",
    },
    {
        "id": "demo_crim_04",
        "title": "498A Misuse Guidelines",
        "query": "What are the Supreme Court guidelines to prevent misuse of Section 498A?",
        "scenario": "A person accused under 498A wants to know about protective guidelines.",
    },
    {
        "id": "demo_crim_05",
        "title": "Punishment for Murder",
        "query": "What is the punishment for murder under Indian law and how is it different from culpable homicide?",
        "scenario": "A law student researching criminal provisions.",
    },

    # ─── Property (5) ────────────────────────────────────────────────────
    {
        "id": "demo_prop_01",
        "title": "Tenant Rights",
        "query": "What are my rights as a tenant? Can my landlord evict me without notice?",
        "scenario": "A tenant facing pressure from landlord to vacate.",
    },
    {
        "id": "demo_prop_02",
        "title": "Builder Delay Compensation",
        "query": "My builder has delayed flat possession by 2 years. What remedy do I have under RERA?",
        "scenario": "A homebuyer waiting for delayed possession.",
    },
    {
        "id": "demo_prop_03",
        "title": "Property Sale Process",
        "query": "What is the legal process to sell property in India? What documents are needed?",
        "scenario": "A property owner planning to sell.",
    },
    {
        "id": "demo_prop_04",
        "title": "Lease Termination Notice",
        "query": "What notice period is required to terminate a lease agreement?",
        "scenario": "A landlord wanting to end a lease legally.",
    },
    {
        "id": "demo_prop_05",
        "title": "Breach of Contract",
        "query": "How do I get compensation for breach of contract? What does the law say?",
        "scenario": "A business owner dealing with a broken agreement.",
    },

    # ─── Consumer (5) ────────────────────────────────────────────────────
    {
        "id": "demo_cons_01",
        "title": "Defective Product Complaint",
        "query": "I bought a defective TV and the company refuses to replace it. How do I file a consumer complaint?",
        "scenario": "A consumer with a faulty product seeking redressal.",
    },
    {
        "id": "demo_cons_02",
        "title": "Medical Negligence",
        "query": "Is medical negligence covered under consumer protection law? How to file a complaint against a doctor?",
        "scenario": "A patient who suffered due to wrong treatment.",
    },
    {
        "id": "demo_cons_03",
        "title": "Online Shopping Refund",
        "query": "I ordered something online but received a different product. The seller refuses to refund. What can I do?",
        "scenario": "An online shopper dealing with a fraudulent seller.",
    },
    {
        "id": "demo_cons_04",
        "title": "Misleading Advertisement",
        "query": "A company made false claims in their advertisement and I bought the product based on that. What are my rights?",
        "scenario": "A consumer misled by false advertising.",
    },
    {
        "id": "demo_cons_05",
        "title": "Consumer Court Guide",
        "query": "Where do I file a consumer complaint in India? What is the maximum claim in district consumer court?",
        "scenario": "A first-time consumer court filer seeking guidance.",
    },
]


def format_demo_markdown(
    entry: dict,
    answer: str,
    laws: list,
    precedents: list,
    procedure: list,
    metadata: dict,
    latency_ms: float,
) -> str:
    """Format a demo query result as a clean markdown document."""
    lines = [
        f"# {entry['title']}",
        "",
        f"> **Scenario:** {entry['scenario']}",
        "",
        f"**User Query:** {entry['query']}",
        "",
        "---",
        "",
    ]

    # Answer
    lines.append("## Answer")
    lines.append("")
    lines.append(answer)
    lines.append("")

    # Applicable Law
    if laws:
        lines.append("## Applicable Law")
        lines.append("")
        for law in laws:
            act = law.get("act", "") if isinstance(law, dict) else getattr(law, "act", "")
            section = law.get("section", "") if isinstance(law, dict) else getattr(law, "section", "")
            text = law.get("text", "") if isinstance(law, dict) else getattr(law, "text", "")
            status = law.get("status", "active") if isinstance(law, dict) else getattr(law, "status", "active")
            lines.append(f"- **Section {section} of {act}** [{status}]")
            if text:
                lines.append(f"  {text[:200]}...")
            lines.append("")

    # Precedents
    if precedents:
        lines.append("## Key Precedents")
        lines.append("")
        for prec in precedents:
            case = prec.get("case", "") if isinstance(prec, dict) else getattr(prec, "case", "")
            year = prec.get("year", "") if isinstance(prec, dict) else getattr(prec, "year", "")
            court = prec.get("court", "") if isinstance(prec, dict) else getattr(prec, "court", "")
            citation = prec.get("citation", "") if isinstance(prec, dict) else getattr(prec, "citation", "")
            relevance = prec.get("relevance", "") if isinstance(prec, dict) else getattr(prec, "relevance", "")
            lines.append(f"- **{case} ({year})** — {court}")
            if citation:
                lines.append(f"  Citation: {citation}")
            if relevance:
                lines.append(f"  {relevance[:200]}")
            lines.append("")

    # Procedure
    if procedure:
        lines.append("## What You Should Do")
        lines.append("")
        for step in procedure:
            num = step.get("step", 0) if isinstance(step, dict) else getattr(step, "step", 0)
            action = step.get("action", "") if isinstance(step, dict) else getattr(step, "action", "")
            details = step.get("details", "") if isinstance(step, dict) else getattr(step, "details", "")
            lines.append(f"{num}. **{action}**")
            if details:
                lines.append(f"   {details}")
            lines.append("")

    # Metadata
    lines.append("---")
    lines.append("")
    lines.append("## Response Metadata")
    lines.append("")

    confidence = metadata.get("confidence", "unknown") if metadata else "unknown"
    verified = metadata.get("sources_verified", False) if metadata else False
    domain = metadata.get("domain", "") if metadata else ""
    v_acc = metadata.get("verification_accuracy") if metadata else None

    lines.append(f"- **Confidence:** {confidence}")
    lines.append(f"- **Domain:** {domain}")
    lines.append(f"- **Sources Verified:** {'Yes' if verified else 'No'}")
    if v_acc is not None:
        lines.append(f"- **Verification Accuracy:** {v_acc:.0%}")
    lines.append(f"- **Latency:** {latency_ms:.0f}ms")
    lines.append(f"- **Laws Cited:** {len(laws)}")
    lines.append(f"- **Cases Cited:** {len(precedents)}")
    lines.append(f"- **Procedure Steps:** {len(procedure)}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*This is legal information, not legal advice. For case-specific advice, consult a qualified advocate.*")
    lines.append("")

    return "\n".join(lines)


async def run_demo_direct(entry: dict) -> dict:
    """Run a single demo query directly through the pipeline."""
    from app.models.query import QueryRequest, Language

    request = QueryRequest(query=entry["query"], language=Language.ENGLISH)

    start = time.time()
    try:
        from app.routers.query import legal_query

        response = await legal_query(request)
        latency = round((time.time() - start) * 1000, 2)

        return {
            "answer": response.answer,
            "laws": [l.model_dump() for l in response.applicable_law],
            "precedents": [p.model_dump() for p in response.precedents],
            "procedure": [s.model_dump() for s in response.procedure],
            "metadata": {
                "confidence": response.confidence.value,
                "sources_verified": response.sources_verified,
                "domain": "",
            },
            "latency_ms": latency,
            "error": "",
        }
    except Exception as e:
        return {
            "answer": "", "laws": [], "precedents": [], "procedure": [],
            "metadata": {}, "latency_ms": round((time.time() - start) * 1000, 2),
            "error": str(e),
        }


async def run_demo_api(entry: dict, base_url: str = "http://localhost:8080") -> dict:
    """Run a single demo query via the HTTP API."""
    import httpx

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{base_url}/api/v1/query",
                json={"query": entry["query"]},
            )
            resp.raise_for_status()
            data = resp.json()

        latency = round((time.time() - start) * 1000, 2)
        return {
            "answer": data.get("answer", ""),
            "laws": data.get("applicable_law", []),
            "precedents": data.get("precedents", []),
            "procedure": data.get("procedure", []),
            "metadata": {
                "confidence": data.get("confidence", ""),
                "sources_verified": data.get("sources_verified", False),
                "domain": "",
            },
            "latency_ms": latency,
            "error": "",
        }
    except Exception as e:
        return {
            "answer": "", "laws": [], "precedents": [], "procedure": [],
            "metadata": {}, "latency_ms": round((time.time() - start) * 1000, 2),
            "error": str(e),
        }


async def main():
    parser = argparse.ArgumentParser(description="NyayaMitra MVP Demo Runner")
    parser.add_argument("--direct", action="store_true", help="Run directly (no server)")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  NyayaMitra — MVP Demo Runner")
    print("=" * 60)
    print(f"  Mode: {'direct' if args.direct else 'API'}")
    print(f"  Queries: {len(DEMO_QUERIES)}")
    print(f"  Output: {output_dir}")
    print("  " + "-" * 56)
    print()

    summary = []

    for i, entry in enumerate(DEMO_QUERIES, 1):
        print(f"  [{i:2d}/{len(DEMO_QUERIES)}] {entry['title']}...")

        if args.direct:
            result = await run_demo_direct(entry)
        else:
            result = await run_demo_api(entry, args.api_url)

        if result["error"]:
            print(f"         ERROR: {result['error'][:60]}")
            status = "ERROR"
        else:
            n_laws = len(result["laws"])
            n_precs = len(result["precedents"])
            n_steps = len(result["procedure"])
            latency = result["latency_ms"]
            print(f"         {latency:.0f}ms | {n_laws} laws, {n_precs} cases, {n_steps} steps")
            status = "OK"

            # Save markdown
            md = format_demo_markdown(
                entry=entry,
                answer=result["answer"],
                laws=result["laws"],
                precedents=result["precedents"],
                procedure=result["procedure"],
                metadata=result["metadata"],
                latency_ms=result["latency_ms"],
            )
            md_path = output_dir / f"{entry['id']}.md"
            with open(md_path, "w") as f:
                f.write(md)

        summary.append({
            "id": entry["id"],
            "title": entry["title"],
            "status": status,
            "laws": len(result["laws"]),
            "precedents": len(result["precedents"]),
            "procedure": len(result["procedure"]),
            "latency_ms": result["latency_ms"],
        })

    # Summary
    print()
    print("  " + "=" * 56)
    print("  Demo Summary:")
    print("  " + "-" * 56)

    ok_count = sum(1 for s in summary if s["status"] == "OK")
    err_count = sum(1 for s in summary if s["status"] == "ERROR")
    avg_latency = sum(s["latency_ms"] for s in summary) / len(summary) if summary else 0

    print(f"  Successful: {ok_count}/{len(summary)}")
    print(f"  Errors: {err_count}")
    print(f"  Avg Latency: {avg_latency:.0f}ms")
    print(f"  Output: {output_dir}/")
    print()

    # Save summary JSON
    summary_path = output_dir / "demo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if ok_count == len(summary):
        print("  All demo queries passed. MVP demo is ready.")
    else:
        print(f"  {err_count} queries failed. Check errors above.")

    print()


if __name__ == "__main__":
    asyncio.run(main())