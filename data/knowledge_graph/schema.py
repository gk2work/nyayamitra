"""
NyayaMitra — Neo4j Knowledge Graph Schema.

Defines and creates the graph schema for the legal knowledge graph:

Nodes:
    Act         — Legislation (IPC, CrPC, TPA, HMA, etc.)
    Section     — Individual provision within an Act
    Judgment    — Court decision (SC, HC)
    Court       — Judicial institution (Supreme Court, High Courts)
    LegalPrinciple — Binding principle established by a judgment

Relationships:
    (Section)-[:BELONGS_TO]->(Act)
    (Judgment)-[:INTERPRETS]->(Section)
    (Judgment)-[:DECIDED_BY]->(Court)
    (Judgment)-[:OVERRULES]->(Judgment)
    (Judgment)-[:DISTINGUISHES]->(Judgment)
    (Judgment)-[:ESTABLISHES]->(LegalPrinciple)
    (LegalPrinciple)-[:DERIVED_FROM]->(Section)
    (Section)-[:REFERENCES]->(Section)  — cross-references between sections

Usage:
    python -m data.knowledge_graph.schema              # create schema
    python -m data.knowledge_graph.schema --drop        # drop all and recreate
    python -m data.knowledge_graph.schema --verify      # verify schema exists
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Schema Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Uniqueness constraints (also create implicit indexes)
CONSTRAINTS = [
    # Act: unique by name
    "CREATE CONSTRAINT act_name_unique IF NOT EXISTS FOR (a:Act) REQUIRE a.name IS UNIQUE",
    # Section: unique by act_name + section_number combination
    "CREATE CONSTRAINT section_uid_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.uid IS UNIQUE",
    # Judgment: unique by case_name
    "CREATE CONSTRAINT judgment_case_unique IF NOT EXISTS FOR (j:Judgment) REQUIRE j.case_name IS UNIQUE",
    # Court: unique by name
    "CREATE CONSTRAINT court_name_unique IF NOT EXISTS FOR (c:Court) REQUIRE c.name IS UNIQUE",
    # LegalPrinciple: unique by uid (generated from source judgment + index)
    "CREATE CONSTRAINT principle_uid_unique IF NOT EXISTS FOR (p:LegalPrinciple) REQUIRE p.uid IS UNIQUE",
]

# Additional indexes for query performance
INDEXES = [
    # Act indexes
    "CREATE INDEX act_domain_idx IF NOT EXISTS FOR (a:Act) ON (a.domain)",
    "CREATE INDEX act_year_idx IF NOT EXISTS FOR (a:Act) ON (a.year)",
    "CREATE INDEX act_short_name_idx IF NOT EXISTS FOR (a:Act) ON (a.short_name)",
    # Section indexes
    "CREATE INDEX section_number_idx IF NOT EXISTS FOR (s:Section) ON (s.section_number)",
    "CREATE INDEX section_domain_idx IF NOT EXISTS FOR (s:Section) ON (s.domain)",
    "CREATE INDEX section_act_name_idx IF NOT EXISTS FOR (s:Section) ON (s.act_name)",
    # Judgment indexes
    "CREATE INDEX judgment_year_idx IF NOT EXISTS FOR (j:Judgment) ON (j.year)",
    "CREATE INDEX judgment_domain_idx IF NOT EXISTS FOR (j:Judgment) ON (j.domain)",
    "CREATE INDEX judgment_court_idx IF NOT EXISTS FOR (j:Judgment) ON (j.court)",
    "CREATE INDEX judgment_overruled_idx IF NOT EXISTS FOR (j:Judgment) ON (j.is_overruled)",
    # Court indexes
    "CREATE INDEX court_type_idx IF NOT EXISTS FOR (c:Court) ON (c.court_type)",
    # LegalPrinciple indexes
    "CREATE INDEX principle_domain_idx IF NOT EXISTS FOR (p:LegalPrinciple) ON (p.domain)",
    # Full-text search indexes for natural language queries
    """CREATE FULLTEXT INDEX section_text_ft IF NOT EXISTS
       FOR (s:Section) ON EACH [s.title, s.text]""",
    """CREATE FULLTEXT INDEX judgment_text_ft IF NOT EXISTS
       FOR (j:Judgment) ON EACH [j.case_name, j.headnote, j.ratio_decidendi]""",
]

# Drop all data and schema (USE WITH CAUTION)
DROP_ALL = [
    "MATCH (n) DETACH DELETE n",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Schema Manager
# ═══════════════════════════════════════════════════════════════════════════════


class GraphSchemaManager:
    """
    Manages the Neo4j knowledge graph schema.

    Creates constraints, indexes, and full-text search indexes.
    Uses the Neo4j Python driver with bolt protocol.
    """

    def __init__(self):
        self.driver = None

    def connect(self) -> None:
        """Connect to Neo4j via bolt driver."""
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )

        # Verify connectivity
        self.driver.verify_connectivity()
        server_info = self.driver.get_server_info()
        logger.info(
            "neo4j_connected",
            uri=settings.neo4j_uri,
            server=str(server_info.agent),
            protocol_version=str(server_info.protocol_version),
        )

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()
            logger.info("neo4j_closed")

    def create_schema(self) -> dict:
        """
        Create all constraints and indexes.

        Returns stats dict with counts of created constraints and indexes.
        """
        stats = {"constraints": 0, "indexes": 0, "errors": []}

        with self.driver.session() as session:
            # Create constraints
            for cypher in CONSTRAINTS:
                try:
                    session.run(cypher)
                    stats["constraints"] += 1
                    logger.info("constraint_created", cypher=cypher[:60])
                except Exception as e:
                    if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                        stats["constraints"] += 1
                        logger.info("constraint_exists", cypher=cypher[:60])
                    else:
                        stats["errors"].append(str(e))
                        logger.error("constraint_error", cypher=cypher[:60], error=str(e))

            # Create indexes
            for cypher in INDEXES:
                try:
                    session.run(cypher)
                    stats["indexes"] += 1
                    logger.info("index_created", cypher=cypher[:60])
                except Exception as e:
                    if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                        stats["indexes"] += 1
                        logger.info("index_exists", cypher=cypher[:60])
                    else:
                        stats["errors"].append(str(e))
                        logger.error("index_error", cypher=cypher[:60], error=str(e))

        logger.info(
            "schema_created",
            constraints=stats["constraints"],
            indexes=stats["indexes"],
            errors=len(stats["errors"]),
        )
        return stats

    def drop_all(self) -> None:
        """
        Drop all nodes, relationships, constraints, and indexes.

        USE WITH CAUTION — destroys all data.
        """
        with self.driver.session() as session:
            # Drop data
            result = session.run("MATCH (n) DETACH DELETE n")
            summary = result.consume()
            logger.info(
                "graph_data_dropped",
                nodes_deleted=summary.counters.nodes_deleted,
                relationships_deleted=summary.counters.relationships_deleted,
            )

            # Drop constraints
            constraints = session.run("SHOW CONSTRAINTS").data()
            for c in constraints:
                name = c.get("name", "")
                if name:
                    try:
                        session.run(f"DROP CONSTRAINT {name} IF EXISTS")
                        logger.info("constraint_dropped", name=name)
                    except Exception:
                        pass

            # Drop indexes (excluding lookup indexes)
            indexes = session.run("SHOW INDEXES").data()
            for idx in indexes:
                name = idx.get("name", "")
                idx_type = idx.get("type", "")
                if name and idx_type not in ("LOOKUP",):
                    try:
                        session.run(f"DROP INDEX {name} IF EXISTS")
                        logger.info("index_dropped", name=name)
                    except Exception:
                        pass

        logger.info("schema_dropped")

    def verify_schema(self) -> dict:
        """
        Verify that the schema exists and report statistics.

        Returns dict with constraint/index counts and node/relationship counts.
        """
        stats = {}

        with self.driver.session() as session:
            # Count constraints
            constraints = session.run("SHOW CONSTRAINTS").data()
            stats["constraints"] = len(constraints)

            # Count indexes
            indexes = session.run("SHOW INDEXES").data()
            stats["indexes"] = len([i for i in indexes if i.get("type") != "LOOKUP"])

            # Count nodes by label
            for label in ["Act", "Section", "Judgment", "Court", "LegalPrinciple"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                stats[f"node_{label.lower()}"] = result.single()["cnt"]

            # Count relationships by type
            for rel_type in ["BELONGS_TO", "INTERPRETS", "DECIDED_BY", "OVERRULES",
                             "DISTINGUISHES", "ESTABLISHES", "DERIVED_FROM", "REFERENCES"]:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt")
                stats[f"rel_{rel_type.lower()}"] = result.single()["cnt"]

            # Total
            result = session.run("MATCH (n) RETURN count(n) AS nodes")
            stats["total_nodes"] = result.single()["nodes"]
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS rels")
            stats["total_relationships"] = result.single()["rels"]

        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="NyayaMitra Neo4j Schema Manager")
    parser.add_argument("--drop", action="store_true", help="Drop all data and recreate schema")
    parser.add_argument("--verify", action="store_true", help="Verify schema and report stats")
    args = parser.parse_args()

    print()
    print("═" * 60)
    print("  NyayaMitra — Knowledge Graph Schema")
    print("═" * 60)
    print()

    manager = GraphSchemaManager()
    manager.connect()

    try:
        if args.drop:
            print("  Dropping all data and schema...")
            manager.drop_all()
            print()

        if args.verify and not args.drop:
            stats = manager.verify_schema()
            print("  Schema Verification:")
            print("  ─────────────────────────────────────")
            print(f"    Constraints:       {stats['constraints']}")
            print(f"    Indexes:           {stats['indexes']}")
            print(f"    Total nodes:       {stats['total_nodes']}")
            print(f"    Total rels:        {stats['total_relationships']}")
            print()
            print("  Nodes:")
            for label in ["act", "section", "judgment", "court", "legalprinciple"]:
                print(f"    {label:<20} {stats.get(f'node_{label}', 0)}")
            print()
            print("  Relationships:")
            for rel in ["belongs_to", "interprets", "decided_by", "overrules",
                         "distinguishes", "establishes", "derived_from", "references"]:
                print(f"    {rel:<20} {stats.get(f'rel_{rel}', 0)}")
        else:
            print("  Creating schema (constraints + indexes)...")
            stats = manager.create_schema()
            print()
            print(f"  Constraints: {stats['constraints']}")
            print(f"  Indexes:     {stats['indexes']}")
            print(f"  Errors:      {len(stats['errors'])}")

            if stats["errors"]:
                for err in stats["errors"]:
                    print(f"    ✗ {err[:80]}")

    finally:
        manager.close()

    print()
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()