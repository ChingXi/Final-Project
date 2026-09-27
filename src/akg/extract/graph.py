import re
from dataclasses import dataclass

from akg.config import Settings
from akg.index.graph import LABELS, RELATIONS
from akg.models import Chunk
from akg.synthesis.llm import chat

_SLUG_NON_WORD = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    slug = _SLUG_NON_WORD.sub("_", name.strip().lower()).strip("_")
    return slug


@dataclass(frozen=True)
class NodeWrite:
    label: str
    node_id: str
    name: str


@dataclass(frozen=True)
class EdgeWrite:
    relation: str
    source_id: str
    target_id: str
    component: str | None
    source_file: str
    locator: str
    cid: str
    doc_hash: str


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "label": {"type": "string", "enum": list(LABELS)},
                },
                "required": ["name", "label"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "relation": {"type": "string", "enum": list(RELATIONS)},
                    "target": {"type": "string"},
                    "component": {"type": "string"},
                },
                "required": ["source", "relation", "target"],
            },
        },
    },
    "required": ["entities", "relations"],
}

SYSTEM = """

You extract entities and relationships stated in the given passage. Do not use outside knowledge and do not infer a relationship the passage does not state.
Every entity you name in a relation must also appear in the entities list, with its type. An entity mentioned but not related to anything else in this passage should still be listed.
Use a label from Project, Person, Team, Vendor, Risk for every entity. Use a relation from OWNS, MEMBER_OF, REPORTS_TO, SUPPORTS, SUPPLIES, DEPENDS_ON, AFFECTS, CAUSED_BY for every relationship, matching the direction the passage states it in.

For a SUPPLIES relationship, put what is supplied in `component`, and do not also list that component as its own entity: a component is something a vendor supplies, not a Project, Person, Team, Vendor or Risk in its own right.

An entity name is the proper noun that identifies it, never a description of what it does: extract the vendor's name, not its service; the project's name, not its status or budget.

Do not extract a document title, section heading, or the name of the report or register the passage comes from as an entity. Do not extract a bare job title or role (e.g. 'the delivery lead') as an entity unless a specific person is named for it. Do not extract a generic word (team, budget, status, quarter) that names no specific thing.
If the same entity is referred to more than once in the passage under both a full form and a shorter one (a first name alone, a project referred to without 'Project'), use the fullest form as its name.
A Risk is identified by what the risk is, as a short descriptive phrase (what is at risk and why), not by a one- or two-word summary of it.
If the passage names no entities, return empty lists rather than guessing.

"""


RELATION_ROLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "OWNS": ("Person", ("Project",)),
    "MEMBER_OF": ("Person", ("Team",)),
    "REPORTS_TO": ("Person", ("Person",)),
    "SUPPORTS": ("Team", ("Project",)),
    "SUPPLIES": ("Vendor", ("Project",)),
    "DEPENDS_ON": ("Project", ("Project",)),
    "AFFECTS": ("Risk", ("Project",)),
    "CAUSED_BY": ("Risk", ("Vendor", "Team")),
}


def build_prompt(chunk: Chunk) -> str:
    return f"PASSAGE ({chunk.source_file} {chunk.locator}):\n{chunk.text}"


def extract_from_chunk(
    settings: Settings, chunk: Chunk
) -> tuple[list[dict], list[dict]]:
    parsed, _recovered = chat(
        settings,
        settings.llm_extraction,
        SYSTEM,
        build_prompt(chunk),
        schema=EXTRACTION_SCHEMA,
        temperature=0.0,
    )
    entities = [
        {"name": (e.get("name") or "").strip(), "label": e.get("label")}
        for e in parsed.get("entities") or []
        if (e.get("name") or "").strip()
    ]
    relations = [
        {
            "source": (r.get("source") or "").strip(),
            "relation": r.get("relation"),
            "target": (r.get("target") or "").strip(),
            "component": (r.get("component") or "").strip() or None,
        }
        for r in parsed.get("relations") or []
        if (r.get("source") or "").strip() and (r.get("target") or "").strip()
    ]
    return entities, relations


def plan_writes(
    chunk: Chunk, entities: list[dict], relations: list[dict]
) -> tuple[list[NodeWrite], list[EdgeWrite], dict[str, int]]:
    stats = {
        "invalid_label": 0, "invalid_relation": 0, "orphan_relation": 0,
        "component_as_entity": 0, "wrong_entity_type": 0,
    }

    component_ids = {
        slugify(r["component"])
        for r in relations
        if r.get("relation") == "SUPPLIES" and r.get("component")
    }

    nodes: list[NodeWrite] = []
    label_of: dict[str, str] = {}
    for entity in entities:
        label = entity.get("label")
        if label not in LABELS:
            stats["invalid_label"] += 1
            continue
        name = entity["name"]
        node_id = slugify(name)
        if node_id in component_ids:
            stats["component_as_entity"] += 1
            continue
        nodes.append(NodeWrite(label=label, node_id=node_id, name=name))
        label_of[node_id] = label

    edges: list[EdgeWrite] = []
    for relation in relations:
        rel_type = relation.get("relation")
        if rel_type not in RELATIONS:
            stats["invalid_relation"] += 1
            continue
        source_id = slugify(relation["source"])
        target_id = slugify(relation["target"])
        if source_id not in label_of or target_id not in label_of:
            stats["orphan_relation"] += 1
            continue
        expected_source, expected_targets = RELATION_ROLES[rel_type]
        if label_of[source_id] != expected_source or label_of[target_id] not in expected_targets:
            stats["wrong_entity_type"] += 1
            continue
        edges.append(
            EdgeWrite(
                relation=rel_type,
                source_id=source_id,
                target_id=target_id,
                component=relation.get("component") if rel_type == "SUPPLIES" else None,
                source_file=chunk.source_file,
                locator=chunk.locator,
                cid=chunk.cid,
                doc_hash=chunk.doc_hash,
            )
        )

    return nodes, edges, stats


def extract_graph(settings: Settings, chunks: list[Chunk], store, reset: bool = True) -> dict:
    if reset:
        store.reset()
    store.ensure_constraints()

    totals = {
        "chunks": 0, "entities": 0, "relations": 0,
        "invalid_label": 0, "invalid_relation": 0, "orphan_relation": 0,
        "component_as_entity": 0, "wrong_entity_type": 0,
    }
    for chunk in chunks:
        entities, relations = extract_from_chunk(settings, chunk)
        nodes, edges, stats = plan_writes(chunk, entities, relations)
        for node in nodes:
            store.upsert_node(node.label, node.node_id, {"name": node.name})
        for edge in edges:
            props = {
                "source_file": edge.source_file,
                "locator": edge.locator,
                "cid": edge.cid,
                "doc_hash": edge.doc_hash,
            }
            if edge.component:
                props["component"] = edge.component
            store.upsert_edge(edge.relation, edge.source_id, edge.target_id, props)

        totals["chunks"] += 1
        totals["entities"] += len(nodes)
        totals["relations"] += len(edges)
        for key in (
            "invalid_label", "invalid_relation", "orphan_relation",
            "component_as_entity", "wrong_entity_type",
        ):
            totals[key] += stats[key]

    return totals
