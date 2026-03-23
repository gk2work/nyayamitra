"""
NyayaMitra — SFT Dataset Splitter (Sprint 8).

Creates train/validation/test splits from validated SFT pairs.

Split strategy:
    - 80% train / 10% validation / 10% test
    - Stratified by domain (each domain represented proportionally)
    - Grouped by source judgment (all pairs from one judgment stay in
      the same split — prevents data leakage)
    - Shuffled within each split

Output:
    data/datasets/sft/splits/train.jsonl
    data/datasets/sft/splits/val.jsonl
    data/datasets/sft/splits/test.jsonl

Each line is a JSON object with a "text" field containing the full
Llama 3.1 formatted conversation, ready for tokenization.

Usage:
    # Split validated pairs
    python -m data.training.sft_splitter

    # Custom input
    python -m data.training.sft_splitter --input my_pairs.jsonl

    # Custom ratios
    python -m data.training.sft_splitter --train 0.85 --val 0.10 --test 0.05

    # Preview split stats without writing
    python -m data.training.sft_splitter --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
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
    DOMAIN_MIN_PERCENTAGE,
)

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Grouping — prevent data leakage
# ═══════════════════════════════════════════════════════════════════════════════


def group_pairs(pairs: list[SFTPair]) -> dict[str, list[SFTPair]]:
    """
    Group pairs by their source entity to prevent data leakage.

    All pairs derived from the same judgment (same source_judgment_id)
    or same procedure (same source_procedure_id) must stay in the
    same split. Otherwise the model could memorize a judgment's headnote
    in training and be tested on a different question from the same judgment.

    Pairs without a group key are each treated as their own group.
    """
    groups: dict[str, list[SFTPair]] = defaultdict(list)

    for pair in pairs:
        if pair.source_judgment_id:
            key = f"judgment:{pair.source_judgment_id}"
        elif pair.source_procedure_id:
            key = f"procedure:{pair.source_procedure_id}"
        elif pair.source_section_id:
            key = f"section:{pair.source_section_id}"
        else:
            # No group — standalone pair
            key = f"standalone:{pair.pair_id}"

        groups[key].append(pair)

    return dict(groups)


# ═══════════════════════════════════════════════════════════════════════════════
# Stratified Split
# ═══════════════════════════════════════════════════════════════════════════════


def stratified_split(
    pairs: list[SFTPair],
    train_ratio: float = QUALITY_THRESHOLDS.train_split,
    val_ratio: float = QUALITY_THRESHOLDS.val_split,
    test_ratio: float = QUALITY_THRESHOLDS.test_split,
    seed: int = 42,
) -> dict[str, list[SFTPair]]:
    """
    Split pairs into train/val/test with domain stratification and
    source-entity grouping (no data leakage).

    Algorithm:
    1. Group pairs by source entity (judgment/procedure/section)
    2. For each domain, collect its groups
    3. Shuffle groups within each domain
    4. Assign groups to train/val/test by ratio
    5. Flatten groups back into pair lists

    This ensures:
    - Each domain is proportionally represented in each split
    - All pairs from one judgment are in the same split
    - Splits are reproducible (fixed seed)
    """
    rng = random.Random(seed)

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Group pairs
    groups = group_pairs(pairs)

    # Organize groups by domain (based on first pair's domain)
    domain_groups: dict[str, list[tuple[str, list[SFTPair]]]] = defaultdict(list)
    for group_key, group_pairs_list in groups.items():
        domain = group_pairs_list[0].domain or "general"
        domain_groups[domain].append((group_key, group_pairs_list))

    # Split each domain's groups proportionally
    train_pairs: list[SFTPair] = []
    val_pairs: list[SFTPair] = []
    test_pairs: list[SFTPair] = []

    for domain, dgroups in domain_groups.items():
        rng.shuffle(dgroups)

        # Count total pairs in this domain
        total_domain_pairs = sum(len(g[1]) for g in dgroups)

        # Calculate split boundaries by pair count
        train_target = int(total_domain_pairs * train_ratio)
        val_target = int(total_domain_pairs * val_ratio)

        # Assign groups to splits
        running_count = 0
        for group_key, group_list in dgroups:
            if running_count < train_target:
                train_pairs.extend(group_list)
            elif running_count < train_target + val_target:
                val_pairs.extend(group_list)
            else:
                test_pairs.extend(group_list)

            running_count += len(group_list)

    # Shuffle within each split
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    rng.shuffle(test_pairs)

    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Domain Balance Check
# ═══════════════════════════════════════════════════════════════════════════════


def check_domain_balance(splits: dict[str, list[SFTPair]]) -> dict:
    """
    Check domain balance across splits.

    Verifies no core domain drops below the minimum percentage threshold.
    """
    results = {}

    for split_name, split_pairs in splits.items():
        total = len(split_pairs)
        if total == 0:
            results[split_name] = {"total": 0, "domains": {}, "issues": ["empty split"]}
            continue

        domain_counts = Counter(p.domain for p in split_pairs)
        issues = []

        domains = {}
        for domain in LEGAL_DOMAINS:
            count = domain_counts.get(domain, 0)
            pct = (count / total) * 100
            domains[domain] = {"count": count, "pct": round(pct, 1)}

            if pct < DOMAIN_MIN_PERCENTAGE and count > 0:
                issues.append(f"{domain} underrepresented: {pct:.1f}% (min {DOMAIN_MIN_PERCENTAGE}%)")

        # Check for domains with zero pairs
        for domain in LEGAL_DOMAINS:
            if domain_counts.get(domain, 0) == 0:
                issues.append(f"{domain} has zero pairs")

        results[split_name] = {
            "total": total,
            "domains": domains,
            "issues": issues,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════


def export_splits(
    splits: dict[str, list[SFTPair]],
    output_dir: Path | None = None,
    training_format: bool = True,
) -> dict[str, Path]:
    """
    Export splits as JSONL files.

    Args:
        splits: Dict with "train", "val", "test" → list of SFTPair.
        output_dir: Output directory (default: SFT_PATHS.splits_dir).
        training_format: If True, export in Llama 3.1 format with "text" key.
                         If False, export raw SFTPair dicts.

    Returns:
        Dict mapping split name to file path.
    """
    out_dir = output_dir or SFT_PATHS.splits_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    for split_name, split_pairs in splits.items():
        path = out_dir / f"{split_name}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for pair in split_pairs:
                if training_format:
                    record = pair.to_training_format()
                else:
                    record = pair.to_dict()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        paths[split_name] = path
        logger.info(
            "split_exported",
            split=split_name,
            pairs=len(split_pairs),
            path=str(path),
        )

    return paths


def export_stats(
    splits: dict[str, list[SFTPair]],
    balance: dict,
    output_path: Path | None = None,
) -> Path:
    """Export dataset statistics as JSON."""
    path = output_path or SFT_PATHS.stats_report
    path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_pairs": sum(len(s) for s in splits.values()),
        "splits": {},
    }

    for split_name, split_pairs in splits.items():
        domain_counts = Counter(p.domain for p in split_pairs)
        source_counts = Counter(p.source for p in split_pairs)
        qtype_counts = Counter(p.query_type for p in split_pairs)

        stats["splits"][split_name] = {
            "count": len(split_pairs),
            "by_domain": dict(domain_counts),
            "by_source": dict(source_counts),
            "by_query_type": dict(qtype_counts),
            "balance": balance.get(split_name, {}),
        }

    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def print_split_summary(splits: dict[str, list[SFTPair]], balance: dict) -> None:
    """Print split summary to console."""
    total = sum(len(s) for s in splits.values())

    print(f"\n{'═' * 65}")
    print(f"  SFT Dataset Split Summary")
    print(f"{'═' * 65}\n")
    print(f"  Total pairs: {total:,}\n")

    # Per-split summary
    print(f"  {'Split':<10} {'Count':>8} {'%':>8}  Domains")
    print(f"  {'─' * 60}")

    for split_name in ["train", "val", "test"]:
        pairs = splits[split_name]
        pct = len(pairs) / total * 100 if total else 0
        domains = Counter(p.domain for p in pairs)
        top_3 = ", ".join(f"{d}:{c}" for d, c in domains.most_common(3))
        print(f"  {split_name:<10} {len(pairs):>8,} {pct:>7.1f}%  {top_3}")

    # Domain balance per split
    print(f"\n  Domain Balance:")
    print(f"  {'Domain':<16}", end="")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name:>10}", end="")
    print()
    print(f"  {'─' * 50}")

    all_domains = set()
    for split_pairs in splits.values():
        for p in split_pairs:
            all_domains.add(p.domain)

    for domain in sorted(all_domains):
        print(f"  {domain:<16}", end="")
        for split_name in ["train", "val", "test"]:
            info = balance.get(split_name, {}).get("domains", {}).get(domain, {})
            count = info.get("count", 0) if isinstance(info, dict) else 0
            pct = info.get("pct", 0.0) if isinstance(info, dict) else 0.0
            print(f"  {count:>5} ({pct:>4.1f}%)", end="")
        print()

    # Issues
    all_issues = []
    for split_name, bdata in balance.items():
        for issue in bdata.get("issues", []):
            all_issues.append(f"  [{split_name}] {issue}")

    if all_issues:
        print(f"\n  Balance Issues:")
        for issue in all_issues:
            print(f"    ⚠ {issue}")

    # Leakage check
    train_ids = {p.source_judgment_id for p in splits["train"] if p.source_judgment_id}
    val_ids = {p.source_judgment_id for p in splits["val"] if p.source_judgment_id}
    test_ids = {p.source_judgment_id for p in splits["test"] if p.source_judgment_id}

    leak_tv = train_ids & val_ids
    leak_tt = train_ids & test_ids
    leak_vt = val_ids & test_ids

    if leak_tv or leak_tt or leak_vt:
        print(f"\n  ⚠ DATA LEAKAGE DETECTED:")
        if leak_tv:
            print(f"    Train↔Val: {len(leak_tv)} shared judgments")
        if leak_tt:
            print(f"    Train↔Test: {len(leak_tt)} shared judgments")
        if leak_vt:
            print(f"    Val↔Test: {len(leak_vt)} shared judgments")
    else:
        print(f"\n  ✓ No data leakage (judgment groups isolated)")

    print(f"\n{'═' * 65}\n")


def load_pairs(input_path: Path) -> list[SFTPair]:
    """Load SFTPairs from JSONL."""
    pairs = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(SFTPair.from_dict(json.loads(line)))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra SFT Dataset Splitter (Sprint 8)",
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL of validated pairs (default: all_validated.jsonl)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for splits")
    parser.add_argument("--train", type=float, default=QUALITY_THRESHOLDS.train_split,
                        help=f"Train ratio (default: {QUALITY_THRESHOLDS.train_split})")
    parser.add_argument("--val", type=float, default=QUALITY_THRESHOLDS.val_split,
                        help=f"Val ratio (default: {QUALITY_THRESHOLDS.val_split})")
    parser.add_argument("--test", type=float, default=QUALITY_THRESHOLDS.test_split,
                        help=f"Test ratio (default: {QUALITY_THRESHOLDS.test_split})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--raw-format", action="store_true",
                        help="Export raw SFTPair dicts instead of Llama training format")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show split stats without writing files")

    args = parser.parse_args()

    # Load pairs
    input_path = Path(args.input) if args.input else SFT_PATHS.validated_all
    if not input_path.exists():
        print(f"\n  Input file not found: {input_path}")
        print(f"  Run the validator first: python -m data.training.sft_validator\n")
        return

    pairs = load_pairs(input_path)
    print(f"\n  Loaded {len(pairs):,} validated pairs from {input_path.name}")

    # Split
    start = time.time()
    splits = stratified_split(
        pairs,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
    duration = round(time.time() - start, 2)

    # Check balance
    balance = check_domain_balance(splits)

    # Print summary
    print_split_summary(splits, balance)

    if args.dry_run:
        print(f"  (Dry run — splits computed in {duration}s, not saved)\n")
        return

    # Export
    output_dir = Path(args.output_dir) if args.output_dir else None
    paths = export_splits(
        splits,
        output_dir=output_dir,
        training_format=not args.raw_format,
    )

    # Export stats
    stats_path = export_stats(splits, balance)

    print(f"  Files exported:")
    for name, path in paths.items():
        print(f"    {name}: {path} ({len(splits[name]):,} pairs)")
    print(f"    stats: {stats_path}")
    print(f"\n  Duration: {duration}s\n")


if __name__ == "__main__":
    main()