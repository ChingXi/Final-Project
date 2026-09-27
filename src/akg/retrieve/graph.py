
from collections.abc import Callable

import numpy as np

from akg.index.graph import RELATIONS, GraphEdge
from akg.models import Chunk

EmbedMany = Callable[[list[str]], np.ndarray]

RELATION_WEIGHT = 0.5


def _surface_forms(node: dict) -> list[str]:
    forms = [node.get("name") or "", node.get("description") or ""]
    forms.extend(node.get("aliases") or [])
    return [f for f in forms if f]


def link_entities(
    question: str,
    nodes: list[dict],
    embed_many: EmbedMany | None = None,
    threshold: float = 0.72,
) -> list[str]:
    lowered = question.lower()
    matched = [
        node["id"]
        for node in nodes
        if any(form.lower() in lowered for form in _surface_forms(node))
    ]
    if matched or embed_many is None or not nodes:
        return matched

    names = [node.get("name") or node.get("description") or "" for node in nodes]
    vectors = embed_many([question, *names])
    query, rest = vectors[0], vectors[1:]
    scores = rest @ query / (
        np.linalg.norm(rest, axis=1) * np.linalg.norm(query) + 1e-12
    )
    best = int(np.argmax(scores))
    return [nodes[best]["id"]] if scores[best] >= threshold else []


SENTENCE = {
    "OWNS": "{s} owns {t}.",
    "MEMBER_OF": "{s} is a member of the {t}.",
    "REPORTS_TO": "{s} reports to {t}.",
    "SUPPORTS": "{s} supports {t}.",
    "DEPENDS_ON": "{s} depends on {t}.",
    "AFFECTS": "The risk '{s}' affects {t}.",
    "CAUSED_BY": "The risk '{s}' is attributed to {t}.",
}


def edge_sentence(edge: GraphEdge) -> str:
    if edge.relation == "SUPPLIES":
        component = f" ({edge.component})" if edge.component else ""
        return f"{edge.source_name} supplies {edge.target_name}{component}."
    template = SENTENCE.get(edge.relation, "{s} {r} {t}.")
    return template.format(
        s=edge.source_name, t=edge.target_name, r=edge.relation.lower()
    )


def path_sentences(path: list[GraphEdge]) -> str:
    return " ".join(edge_sentence(e) for e in path)


def _key(edge: GraphEdge) -> tuple:
    return (edge.source_id, edge.relation, edge.target_id)


def _cosine(query: np.ndarray, rest: np.ndarray) -> np.ndarray:
    return rest @ query / (
        np.linalg.norm(rest, axis=1) * np.linalg.norm(query) + 1e-12
    )


def relation_affinity(
    question: str, relations: list[str], embed_many: EmbedMany
) -> dict[str, float]:
    if not relations:
        return {}
    phrases = [r.lower().replace("_", " ") for r in relations]
    vectors = embed_many([question, *phrases])
    scores = _cosine(vectors[0], vectors[1:])
    return {r: float(s) for r, s in zip(relations, scores)}


def select_edges(
    paths: list[list[GraphEdge]],
    question: str,
    embed_many: EmbedMany,
    budget: int,
) -> list[GraphEdge]:
    if not paths:
        return []

    chosen: dict[tuple, GraphEdge] = {}

    def take(path: list[GraphEdge], ceiling: int) -> None:
        for edge in path:
            key = _key(edge)
            if key in chosen:
                if edge.depth < chosen[key].depth:
                    chosen[key] = edge
                continue
            if len(chosen) >= ceiling:
                return
            chosen[key] = edge

    direct = [p for p in paths if len(p) == 1]
    longer = [p for p in paths if len(p) > 1]

    direct_ceiling = max(1, budget // 2) if longer else budget
    for path in direct:
        take(path, direct_ceiling)

    if len(chosen) < budget and longer:
        vectors = embed_many([question, *[path_sentences(p) for p in longer]])
        similarity = _cosine(vectors[0], vectors[1:])

        affinity = relation_affinity(question, list(RELATIONS), embed_many)

        remaining = set(range(len(longer)))
        while len(chosen) < budget and remaining:
            best, best_rank = None, ()
            for i in remaining:
                novel = [e for e in longer[i] if _key(e) not in chosen]
                if not novel:
                    continue
                gain = (1 - RELATION_WEIGHT) * float(similarity[i]) + (
                    RELATION_WEIGHT
                    * sum(affinity.get(e.relation, 0.0) for e in novel)
                    / len(novel)
                )
                rank = (gain, len(novel), -len(longer[i]))
                if best is None or rank > best_rank:
                    best, best_rank = i, rank
            if best is None:
                break
            take(longer[best], budget)
            remaining.discard(best)

    if len(chosen) < budget:
        for path in direct:
            take(path, budget)

    return list(chosen.values())[:budget]


def to_sources(
    edges: list[GraphEdge], index, allowed_scopes: list[str] | None = None
) -> list[Chunk]:
    grouped: dict[tuple[str, str], list[GraphEdge]] = {}
    for edge in edges:
        grouped.setdefault((edge.doc_hash, edge.cid), []).append(edge)

    sources: list[Chunk] = []
    for (doc_hash, cid), group in grouped.items():
        resolved = index.get_chunk(doc_hash, cid)
        if resolved is None:
            continue
        chunk, vector = resolved
        if allowed_scopes is not None and chunk.scope not in (*allowed_scopes, "shared"):
            continue
        source = Chunk(
            cid=cid,
            text=" ".join(edge_sentence(e) for e in group),
            source_file=chunk.source_file,
            modality="graph",
            locator=chunk.locator,
            doc_hash=doc_hash,
        )
        source.vector = vector
        sources.append(source)
    return sources


def graph_sources(
    store, index, question: str, settings, allowed_scopes: list[str] | None = None
) -> list[Chunk]:
    seeds = link_entities(question, store.all_nodes(), index.embed_many)
    if not seeds:
        return []
    paths = store.paths(seeds, settings.graph_max_depth)
    chosen = select_edges(
        paths, question, index.embed_many, settings.graph_max_edges
    )
    return to_sources(chosen, index, allowed_scopes)
