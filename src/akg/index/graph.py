
from dataclasses import dataclass

from neo4j import GraphDatabase

from akg.config import Settings

RELATIONS = (
    "OWNS",
    "MEMBER_OF",
    "REPORTS_TO",
    "SUPPORTS",
    "SUPPLIES",
    "DEPENDS_ON",
    "AFFECTS",
    "CAUSED_BY",
)

LABELS = ("Project", "Person", "Team", "Vendor", "Risk")


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    source_name: str
    relation: str
    target_id: str
    target_name: str
    component: str | None
    source_file: str
    locator: str
    cid: str
    doc_hash: str
    depth: int


class GraphStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        try:
            driver.verify_connectivity()
        except Exception as e:
            driver.close()
            raise RuntimeError(
                f"Cannot reach Neo4j at {settings.neo4j_uri}: {e}. "
                f"Start it with: docker compose --profile graph up -d"
            ) from e
        self.driver = driver

    def close(self) -> None:
        self.driver.close()

    def _run(self, cypher: str, **params):
        with self.driver.session() as session:
            return list(session.run(cypher, **params))

    def ensure_constraints(self) -> None:
        for label in LABELS:
            self._run(
                f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )

    def reset(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")

    def counts(self) -> tuple[int, int]:
        nodes = self._run("MATCH (n) RETURN count(n) AS c")[0]["c"]
        edges = self._run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        return nodes, edges

    def upsert_node(self, label: str, node_id: str, props: dict) -> None:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}, expected one of {LABELS}")
        self._run(
            f"MERGE (n:{label} {{id: $id}}) SET n += $props",
            id=node_id,
            props=props,
        )

    def upsert_edge(
        self, relation: str, src_id: str, tgt_id: str, props: dict
    ) -> None:
        if relation not in RELATIONS:
            raise ValueError(
                f"unknown relation {relation!r}, expected one of {RELATIONS}"
            )
        self._run(
            f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
            f"MERGE (a)-[r:{relation}]->(b) SET r += $props",
            src=src_id,
            tgt=tgt_id,
            props=props,
        )

    def all_nodes(self) -> list[dict]:
        rows = self._run(
            "MATCH (n) RETURN n.id AS id, labels(n)[0] AS label, "
            "n.name AS name, n.aliases AS aliases, n.description AS description"
        )
        return [dict(r) for r in rows]

    def all_edges(self) -> list[GraphEdge]:
        rows = self._run(
        )
        return [
            GraphEdge(
                source_id=r["source_id"],
                source_name=r["source_name"],
                relation=r["relation"],
                target_id=r["target_id"],
                target_name=r["target_name"],
                component=r["component"],
                source_file=r["source_file"],
                locator=r["locator"],
                cid=r["cid"],
                doc_hash=r["doc_hash"],
                depth=0,
            )
            for r in rows
        ]

    def paths(
        self, seed_ids: list[str], max_depth: int, limit: int = 4000
    ) -> list[list[GraphEdge]]:
        if not seed_ids:
            return []
        if not isinstance(max_depth, int) or not 1 <= max_depth <= 5:
            raise ValueError(f"max_depth must be an int in 1..5, got {max_depth!r}")

        rows = self._run(
            f"""
            MATCH (seed) WHERE seed.id IN $seeds
            MATCH path = (seed)-[*1..{max_depth}]-(other)
            RETURN [r IN relationships(path) | {{
                relation: type(r),
                source_id: startNode(r).id,
                source_name: startNode(r).name,
                target_id: endNode(r).id,
                target_name: endNode(r).name,
                component: r.component,
                source_file: r.source_file,
                locator: r.locator,
                cid: r.cid,
                doc_hash: r.doc_hash
            }}] AS rels
            LIMIT {int(limit)}
            """,
            seeds=seed_ids,
        )
        return [
            [
                GraphEdge(
                    source_id=rel["source_id"],
                    source_name=rel["source_name"],
                    relation=rel["relation"],
                    target_id=rel["target_id"],
                    target_name=rel["target_name"],
                    component=rel["component"],
                    source_file=rel["source_file"],
                    locator=rel["locator"],
                    cid=rel["cid"],
                    doc_hash=rel["doc_hash"],
                    depth=position + 1,
                )
                for position, rel in enumerate(row["rels"])
            ]
            for row in rows
        ]

    def neighbourhood(self, seed_ids: list[str], max_depth: int) -> list[GraphEdge]:
        if not seed_ids:
            return []
        if not isinstance(max_depth, int) or not 1 <= max_depth <= 5:
            raise ValueError(f"max_depth must be an int in 1..5, got {max_depth!r}")

        rows = self._run(
            f"""
            MATCH (seed) WHERE seed.id IN $seeds
            MATCH path = (seed)-[*1..{max_depth}]-(other)
            UNWIND range(0, size(relationships(path)) - 1) AS i
            WITH relationships(path)[i] AS r, i + 1 AS hop
            RETURN type(r) AS relation,
                   startNode(r).id AS source_id,
                   startNode(r).name AS source_name,
                   endNode(r).id AS target_id,
                   endNode(r).name AS target_name,
                   r.component AS component,
                   r.source_file AS source_file,
                   r.locator AS locator,
                   r.cid AS cid,
                   r.doc_hash AS doc_hash,
                   min(hop) AS depth
            """,
            seeds=seed_ids,
        )
        return [
            GraphEdge(
                source_id=r["source_id"],
                source_name=r["source_name"],
                relation=r["relation"],
                target_id=r["target_id"],
                target_name=r["target_name"],
                component=r["component"],
                source_file=r["source_file"],
                locator=r["locator"],
                cid=r["cid"],
                doc_hash=r["doc_hash"],
                depth=r["depth"],
            )
            for r in rows
        ]
