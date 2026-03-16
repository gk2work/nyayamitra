"""
NyayaMitra — Knowledge Graph Query Library.

Cypher query functions for traversing the legal knowledge graph.
Used by the graph_service to enrich retrieval results with
relational context that vector search alone cannot provide.

Key queries:
    - Given a section → find all judgments that interpreted it
    - Given a judgment → find all sections it interpreted
    - Given a section → find the legal principles derived from it
    - Given a judgment → find related judgments (same sections, overruled, etc.)
    - Given a domain → find all legal principles in that domain
    - Path queries: how are two legal concepts connected?

Usage:
    from data.knowledge_graph.graph_queries import GraphQueryExecutor

    executor = GraphQueryExecutor(neo4j_driver)
    judgments = executor.get_interpreting_judgments("IPC", "302")
    related = executor.get_related_sections("Arnesh Kumar v. State of Bihar")
    principles = executor.get_legal_principles(domain="criminal")
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Query Executor
# ═══════════════════════════════════════════════════════════════════════════════


class GraphQueryExecutor:
    """
    Executes Cypher queries against the Neo4j knowledge graph.

    Accepts an initialized Neo4j driver. All methods are synchronous
    (Neo4j Python driver uses synchronous bolt protocol). The async
    GraphService wrapper handles thread offloading.
    """

    def __init__(self, driver):
        self.driver = driver

    # ─── Section → Judgments ─────────────────────────────────────────────

    def get_interpreting_judgments(
        self,
        act_short_name: str,
        section_number: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Find all judgments that interpreted a given section.

        Args:
            act_short_name: Act abbreviation (e.g., "IPC", "CrPC").
            section_number: Section number (e.g., "302", "41A").
            limit: Maximum results.

        Returns:
            List of judgment dicts with case_name, year, court,
            citation, headnote, ratio_decidendi, domain.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (j:Judgment)-[:INTERPRETS]->(s:Section)
                WHERE s.section_number = $section_number
                  AND (s.act_short_name = $act_short_name
                       OR s.act_name CONTAINS $act_short_name)
                RETURN j.case_name AS case_name,
                       j.year AS year,
                       j.court AS court,
                       j.court_type AS court_type,
                       j.citation_scc AS citation_scc,
                       j.citation_air AS citation_air,
                       j.headnote AS headnote,
                       j.ratio_decidendi AS ratio_decidendi,
                       j.domain AS domain,
                       j.bench_size AS bench_size,
                       j.is_overruled AS is_overruled
                ORDER BY j.bench_size DESC, j.year DESC
                LIMIT $limit
                """,
                act_short_name=act_short_name,
                section_number=section_number,
                limit=limit,
            )
            return [dict(record) for record in result]

    # ─── Judgment → Sections ─────────────────────────────────────────────

    def get_interpreted_sections(
        self,
        case_name: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Find all sections that a given judgment interpreted.

        Args:
            case_name: Full or partial case name.
            limit: Maximum results.

        Returns:
            List of section dicts with section_number, act_name,
            act_short_name, title, text, domain.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (j:Judgment)-[:INTERPRETS]->(s:Section)
                WHERE j.case_name CONTAINS $case_name
                OPTIONAL MATCH (s)-[:BELONGS_TO]->(a:Act)
                RETURN s.section_number AS section_number,
                       s.title AS title,
                       s.text AS text,
                       s.act_name AS act_name,
                       s.act_short_name AS act_short_name,
                       s.domain AS domain,
                       a.year AS act_year,
                       s.status AS status
                ORDER BY a.year, s.section_number
                LIMIT $limit
                """,
                case_name=case_name,
                limit=limit,
            )
            return [dict(record) for record in result]

    # ─── Section → Legal Principles ──────────────────────────────────────

    def get_section_principles(
        self,
        act_short_name: str,
        section_number: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Find legal principles derived from a section.

        Traverses: Section ←[:DERIVED_FROM]- LegalPrinciple ←[:ESTABLISHES]- Judgment

        Returns:
            List of dicts with principle_text, source_case, year, domain.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:LegalPrinciple)-[:DERIVED_FROM]->(s:Section)
                WHERE s.section_number = $section_number
                  AND (s.act_short_name = $act_short_name
                       OR s.act_name CONTAINS $act_short_name)
                OPTIONAL MATCH (j:Judgment)-[:ESTABLISHES]->(p)
                RETURN p.text AS principle_text,
                       p.source_case AS source_case,
                       p.domain AS domain,
                       p.year AS year,
                       j.case_name AS judgment_name,
                       j.citation_scc AS citation
                ORDER BY j.bench_size DESC, j.year DESC
                LIMIT $limit
                """,
                act_short_name=act_short_name,
                section_number=section_number,
                limit=limit,
            )
            return [dict(record) for record in result]

    # ─── Section with Full Context ───────────────────────────────────────

    def get_section_with_context(
        self,
        act_short_name: str,
        section_number: str,
    ) -> dict | None:
        """
        Get a section with all its graph context:
        - The act it belongs to
        - Judgments that interpreted it
        - Legal principles derived from it
        - Sibling sections in the same chapter

        Returns a single enriched dict or None if not found.
        """
        with self.driver.session() as session:
            # Get the section itself
            result = session.run(
                """
                MATCH (s:Section)
                WHERE s.section_number = $section_number
                  AND (s.act_short_name = $act_short_name
                       OR s.act_name CONTAINS $act_short_name)
                OPTIONAL MATCH (s)-[:BELONGS_TO]->(a:Act)
                RETURN s, a.name AS act_name, a.year AS act_year,
                       a.domain AS act_domain, s.chapter AS chapter
                LIMIT 1
                """,
                act_short_name=act_short_name,
                section_number=section_number,
            )
            record = result.single()
            if not record:
                return None

            section_node = dict(record["s"])
            context = {
                "section": section_node,
                "act_name": record["act_name"],
                "act_year": record["act_year"],
                "act_domain": record["act_domain"],
                "chapter": record["chapter"],
            }

            # Get interpreting judgments
            context["interpreting_judgments"] = self.get_interpreting_judgments(
                act_short_name, section_number, limit=5
            )

            # Get legal principles
            context["legal_principles"] = self.get_section_principles(
                act_short_name, section_number, limit=5
            )

            # Get sibling sections in the same chapter
            if record["chapter"]:
                siblings = session.run(
                    """
                    MATCH (s:Section)-[:BELONGS_TO]->(a:Act {name: $act_name})
                    WHERE s.chapter = $chapter
                      AND s.section_number <> $section_number
                    RETURN s.section_number AS section_number,
                           s.title AS title
                    ORDER BY s.section_number
                    LIMIT 10
                    """,
                    act_name=record["act_name"],
                    chapter=record["chapter"],
                    section_number=section_number,
                )
                context["sibling_sections"] = [dict(r) for r in siblings]
            else:
                context["sibling_sections"] = []

            return context

    # ─── Judgment → Related Judgments ─────────────────────────────────────

    def get_related_judgments(
        self,
        case_name: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Find judgments related to a given judgment.

        Related means: interprets any of the same sections,
        or has OVERRULES/DISTINGUISHES relationship.

        Returns list of related judgment dicts with relationship type.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                // Find judgments that share interpreted sections
                MATCH (j1:Judgment {case_name: $case_name})-[:INTERPRETS]->(s:Section)
                      <-[:INTERPRETS]-(j2:Judgment)
                WHERE j2.case_name <> $case_name
                WITH j2, collect(DISTINCT s.section_number) AS shared_sections,
                     'shared_section' AS rel_type
                RETURN j2.case_name AS case_name,
                       j2.year AS year,
                       j2.court AS court,
                       j2.domain AS domain,
                       j2.citation_scc AS citation,
                       j2.is_overruled AS is_overruled,
                       rel_type,
                       shared_sections
                ORDER BY size(shared_sections) DESC, j2.year DESC
                LIMIT $limit

                UNION

                // Find overruling/overruled judgments
                MATCH (j1:Judgment {case_name: $case_name})-[r:OVERRULES|DISTINGUISHES]-(j2:Judgment)
                RETURN j2.case_name AS case_name,
                       j2.year AS year,
                       j2.court AS court,
                       j2.domain AS domain,
                       j2.citation_scc AS citation,
                       j2.is_overruled AS is_overruled,
                       type(r) AS rel_type,
                       [] AS shared_sections
                LIMIT $limit
                """,
                case_name=case_name,
                limit=limit,
            )
            return [dict(record) for record in result]

    # ─── Domain → Legal Principles ───────────────────────────────────────

    def get_legal_principles(
        self,
        domain: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Find all legal principles in a domain.

        Args:
            domain: Legal domain (criminal, property, family, etc.).
            limit: Maximum results.

        Returns:
            List of principle dicts with text, source_case, year.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:LegalPrinciple)
                WHERE p.domain = $domain
                OPTIONAL MATCH (j:Judgment)-[:ESTABLISHES]->(p)
                RETURN p.text AS principle_text,
                       p.source_case AS source_case,
                       p.year AS year,
                       j.case_name AS judgment_name,
                       j.citation_scc AS citation
                ORDER BY j.bench_size DESC, j.year DESC
                LIMIT $limit
                """,
                domain=domain,
                limit=limit,
            )
            return [dict(record) for record in result]

    # ─── Act → Full Structure ────────────────────────────────────────────

    def get_act_structure(
        self,
        act_name: str,
    ) -> dict | None:
        """
        Get the full structure of an act with all its sections
        and their interpretation counts.

        Returns dict with act metadata and list of sections.
        """
        with self.driver.session() as session:
            # Get act
            act_result = session.run(
                """
                MATCH (a:Act)
                WHERE a.name CONTAINS $act_name OR a.short_name = $act_name
                RETURN a
                LIMIT 1
                """,
                act_name=act_name,
            )
            act_record = act_result.single()
            if not act_record:
                return None

            act_data = dict(act_record["a"])

            # Get sections with interpretation counts
            sections_result = session.run(
                """
                MATCH (s:Section)-[:BELONGS_TO]->(a:Act)
                WHERE a.name CONTAINS $act_name OR a.short_name = $act_name
                OPTIONAL MATCH (j:Judgment)-[:INTERPRETS]->(s)
                WITH s, count(j) AS interpretation_count
                RETURN s.section_number AS section_number,
                       s.title AS title,
                       s.chapter AS chapter,
                       s.status AS status,
                       interpretation_count
                ORDER BY s.section_number
                """,
                act_name=act_name,
            )
            act_data["sections"] = [dict(r) for r in sections_result]

            return act_data

    # ─── Path Finding ────────────────────────────────────────────────────

    def find_connection(
        self,
        from_label: str,
        from_name: str,
        to_label: str,
        to_name: str,
        max_depth: int = 4,
    ) -> list[dict]:
        """
        Find the shortest path between two nodes in the graph.

        Useful for answering "How is Section X related to Case Y?"

        Args:
            from_label: Node label (Act, Section, Judgment, etc.)
            from_name: Name/identifier of the source node.
            to_label: Node label of the target node.
            to_name: Name/identifier of the target node.
            max_depth: Maximum path length.

        Returns:
            List of path steps, each with node info and relationship type.
        """
        # Build the name property based on label
        from_prop = "case_name" if from_label == "Judgment" else "name" if from_label in ("Act", "Court") else "uid" if from_label == "Section" else "name"
        to_prop = "case_name" if to_label == "Judgment" else "name" if to_label in ("Act", "Court") else "uid" if to_label == "Section" else "name"

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH path = shortestPath(
                    (a:{from_label} {{{from_prop}: $from_name}})-[*..{max_depth}]-
                    (b:{to_label} {{{to_prop}: $to_name}})
                )
                UNWIND relationships(path) AS rel
                UNWIND nodes(path) AS node
                WITH DISTINCT node, rel
                RETURN labels(node)[0] AS node_label,
                       coalesce(node.name, node.case_name, node.uid, node.text) AS node_name,
                       type(rel) AS relationship,
                       startNode(rel) = node AS is_start
                """,
                from_name=from_name,
                to_name=to_name,
            )
            return [dict(record) for record in result]

    # ─── Full-Text Search ────────────────────────────────────────────────

    def search_sections_fulltext(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Full-text search across section titles and text.

        Uses the Neo4j full-text index created in schema.py.
        Useful as a supplementary retrieval mechanism.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('section_text_ft', $query)
                YIELD node, score
                OPTIONAL MATCH (node)-[:BELONGS_TO]->(a:Act)
                RETURN node.section_number AS section_number,
                       node.title AS title,
                       node.text AS text,
                       node.act_short_name AS act_short_name,
                       a.name AS act_name,
                       node.domain AS domain,
                       score
                ORDER BY score DESC
                LIMIT $limit
                """,
                query=query,
                limit=limit,
            )
            return [dict(record) for record in result]

    def search_judgments_fulltext(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Full-text search across judgment case names, headnotes,
        and ratio decidendi.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('judgment_text_ft', $query)
                YIELD node, score
                OPTIONAL MATCH (node)-[:DECIDED_BY]->(c:Court)
                RETURN node.case_name AS case_name,
                       node.year AS year,
                       node.court AS court,
                       node.domain AS domain,
                       node.headnote AS headnote,
                       node.ratio_decidendi AS ratio_decidendi,
                       node.citation_scc AS citation,
                       c.name AS court_full_name,
                       score
                ORDER BY score DESC
                LIMIT $limit
                """,
                query=query,
                limit=limit,
            )
            return [dict(record) for record in result]

    # ─── Statistics ──────────────────────────────────────────────────────

    def get_domain_stats(self) -> list[dict]:
        """
        Get node and relationship counts per legal domain.

        Useful for dashboard and monitoring.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Section)
                WITH s.domain AS domain, count(s) AS sections
                OPTIONAL MATCH (j:Judgment {domain: domain})
                WITH domain, sections, count(DISTINCT j) AS judgments
                OPTIONAL MATCH (p:LegalPrinciple {domain: domain})
                RETURN domain,
                       sections,
                       judgments,
                       count(DISTINCT p) AS principles
                ORDER BY sections DESC
                """
            )
            return [dict(record) for record in result]

    def get_most_interpreted_sections(self, limit: int = 20) -> list[dict]:
        """Find sections with the most interpreting judgments."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (j:Judgment)-[:INTERPRETS]->(s:Section)
                WITH s, count(j) AS interpretation_count
                OPTIONAL MATCH (s)-[:BELONGS_TO]->(a:Act)
                RETURN s.section_number AS section_number,
                       s.title AS title,
                       s.act_short_name AS act_short_name,
                       a.name AS act_name,
                       s.domain AS domain,
                       interpretation_count
                ORDER BY interpretation_count DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(record) for record in result]