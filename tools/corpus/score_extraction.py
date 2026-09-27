
import argparse
from collections import defaultdict

from akg.config import load_settings
from akg.extract.graph import slugify
from akg.index.graph import GraphEdge, GraphStore
from tools.corpus.load_graph import RELATION_LABEL
from tools.corpus.model import Graph, load_graph

_normalize = slugify


def _gold_entity_forms(graph: Graph) -> dict[str, str]:
    forms: dict[str, str] = {}
    for entity_id, entity in graph.entities().items():
        candidates = [graph.name_of(entity_id), *(entity.get("aliases") or [])]
        for form in candidates:
            forms[_normalize(form)] = entity_id
    return forms


def score_entities(graph: Graph, extracted_names: list[str]) -> dict:
    forms = _gold_entity_forms(graph)
    matches: dict[str, set[str]] = defaultdict(set) 
    extracted_hits = 0

    for name in extracted_names:
        norm = _normalize(name)
        entity_id = forms.get(norm)
        if entity_id is not None:
            matches[entity_id].add(norm)
            extracted_hits += 1

    gold_total = len(graph.entities())
    extracted_total = len(extracted_names)
    found = len(matches)
    fragmented = sum(1 for seen in matches.values() if len(seen) > 1)

    return {
        "gold_total": gold_total,
        "extracted_total": extracted_total,
        "found": found,
        "fragmented": fragmented,
        "recall": found / gold_total if gold_total else 0.0,
        "precision": extracted_hits / extracted_total if extracted_total else 0.0,
    }


def _gold_relation_triples(graph: Graph) -> set[tuple[str, str, str]]:
    triples = set()
    for relation, edges in graph.relations.items():
        label = RELATION_LABEL[relation]
        for edge in edges:
            src, tgt = edge[0], edge[1]
            triples.add((_normalize(graph.name_of(src)), label, _normalize(graph.name_of(tgt))))
    return triples


def score_relations(graph: Graph, extracted_edges: list[GraphEdge]) -> dict:
    gold = _gold_relation_triples(graph)
    extracted = [
        (_normalize(e.source_name), e.relation, _normalize(e.target_name))
        for e in extracted_edges
    ]
    found = len(set(extracted) & gold)

    gold_total = len(gold)
    extracted_total = len(extracted)
    return {
        "gold_total": gold_total,
        "extracted_total": extracted_total,
        "found": found,
        "recall": found / gold_total if gold_total else 0.0,
        "precision": found / extracted_total if extracted_total else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score an extracted Neo4j graph against corpus/graph.yaml"
    )
    ap.add_argument("--graph", default="corpus/graph.yaml")
    args = ap.parse_args()

    graph = load_graph(args.graph)
    settings = load_settings()
    store = GraphStore(settings)
    try:
        extracted_names = [n["name"] for n in store.all_nodes() if n.get("name")]
        extracted_edges = store.all_edges()
    finally:
        store.close()

    entities = score_entities(graph, extracted_names)
    relations = score_relations(graph, extracted_edges)

    print("entities:")
    print(
        f"  gold {entities['gold_total']}, extracted {entities['extracted_total']}, "
        f"found {entities['found']}, fragmented {entities['fragmented']}"
    )
    print(f"  precision {entities['precision']:.1%}  recall {entities['recall']:.1%}")
    print("relations:")
    print(
        f"  gold {relations['gold_total']}, extracted {relations['extracted_total']}, "
        f"found {relations['found']}"
    )
    print(f"  precision {relations['precision']:.1%}  recall {relations['recall']:.1%}")


if __name__ == "__main__":
    main()
