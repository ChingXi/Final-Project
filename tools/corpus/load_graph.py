import argparse
from akg.config import Settings, load_settings
from akg.index.graph import GraphStore
from akg.index.vector import VectorIndex
from akg.models import Chunk
from tools.corpus.documents import RELATION_SOURCE
from tools.corpus.model import Graph, load_graph
from tools.corpus.questions import FAMILY_FILE

LABEL_OF = {
    "projects": "Project",
    "people": "Person",
    "teams": "Team",
    "vendors": "Vendor",
    "risks": "Risk",
}

RELATION_LABEL = {
    "owns": "OWNS",
    "member_of": "MEMBER_OF",
    "reports_to": "REPORTS_TO",
    "supports": "SUPPORTS",
    "supplies": "SUPPLIES",
    "depends_on": "DEPENDS_ON",
    "affects": "AFFECTS",
    "caused_by": "CAUSED_BY",
}


class ProvenanceError(Exception):
    """An edge has no passage behind it."""


def resolve_provenance(
    chunks: list[Chunk], source_file: str, endpoints: list[str]
) -> Chunk:
    for chunk in chunks:
        if chunk.source_file != source_file:
            continue
        text = chunk.text.lower()
        if all(name.lower() in text for name in endpoints):
            return chunk
    raise ProvenanceError(
        f"no chunk of {source_file} names all of {endpoints!r}. "
        f"Has 'akg ingest data/' been run against the current corpus?"
    )


def _node_props(graph: Graph, entity_id: str, entity: dict) -> dict:
    props = {"name": graph.name_of(entity_id)}
    if entity.get("aliases"):
        props["aliases"] = list(entity["aliases"])
    for key in ("description", "severity", "role", "service", "budget",
                "spent", "remaining", "status"):
        if key in entity:
            props[key] = str(entity[key])
    return props


def _all_chunks(index: VectorIndex) -> list[Chunk]:
    from akg.index.vector import _to_chunk

    found, _ = index.client.scroll(
        index.collection, limit=10000, with_payload=True, with_vectors=False
    )
    return [_to_chunk(p.payload) for p in found]


def load(
    settings: Settings,
    graph_path: str,
    store: GraphStore,
    index: VectorIndex,
) -> dict:
    graph = load_graph(graph_path)

    chunks = _all_chunks(index)
    if not chunks:
        raise ProvenanceError(
            "the vector index is empty; run 'akg ingest data/' before loading "
            "the graph, because provenance resolves against indexed chunks"
        )

    store.reset()
    store.ensure_constraints()

    for section, label in LABEL_OF.items():
        for entity_id, entity in getattr(graph, section).items():
            store.upsert_node(label, entity_id, _node_props(graph, entity_id, entity))

    edges = 0
    for relation, pairs in graph.relations.items():
        source_file = FAMILY_FILE[RELATION_SOURCE[relation]]
        for edge in pairs:
            src, tgt = edge[0], edge[1]
            endpoints = [graph.name_of(src), graph.name_of(tgt)]
            chunk = resolve_provenance(chunks, source_file, endpoints)
            props = {
                "source_file": chunk.source_file,
                "locator": chunk.locator,
                "cid": chunk.cid,
                "doc_hash": chunk.doc_hash,
            }
            if relation == "supplies" and len(edge) > 2:
                props["component"] = edge[2]
            store.upsert_edge(RELATION_LABEL[relation], src, tgt, props)
            edges += 1

    nodes, stored_edges = store.counts()
    return {"nodes": nodes, "edges": stored_edges, "edges_written": edges}


def main() -> None:
    ap = argparse.ArgumentParser(description="Load the ground truth into Neo4j")
    ap.add_argument("--graph", default="corpus/graph.yaml")
    args = ap.parse_args()

    settings = load_settings()
    store = GraphStore(settings)
    try:
        stats = load(settings, args.graph, store, VectorIndex(settings))
    except ProvenanceError as e:
        raise SystemExit(f"load failed: {e}") from e
    finally:
        store.close()

    print(f"loaded {stats['nodes']} nodes and {stats['edges']} edges")


if __name__ == "__main__":
    main()
