from dataclasses import dataclass, field

import yaml

RELATIONS = (
    "owns",
    "member_of",
    "reports_to",
    "supports",
    "supplies",
    "depends_on",
    "affects",
    "caused_by",
)

ENTITY_SECTIONS = ("projects", "people", "teams", "vendors", "risks")


class GraphError(Exception):
    """The ground truth is malformed."""


@dataclass
class Graph:
    projects: dict[str, dict] = field(default_factory=dict)
    people: dict[str, dict] = field(default_factory=dict)
    teams: dict[str, dict] = field(default_factory=dict)
    vendors: dict[str, dict] = field(default_factory=dict)
    risks: dict[str, dict] = field(default_factory=dict)
    relations: dict[str, list[tuple]] = field(default_factory=dict)

    def entities(self) -> dict[str, dict]:
        merged: dict[str, dict] = {}
        for section in ENTITY_SECTIONS:
            merged.update(getattr(self, section))
        return merged

    def name_of(self, entity_id: str) -> str:
        entity = self.entities().get(entity_id)
        if entity is None:
            raise GraphError(f"unknown entity: {entity_id}")
        # Risks are identified by their description rather than a short name.
        return entity.get("name") or entity.get("description", entity_id)

    def related(self, relation: str, source: str) -> list[str]:
        return [e[1] for e in self.relations[relation] if e[0] == source]

    def inverse(self, relation: str, target: str) -> list[str]:
        return [e[0] for e in self.relations[relation] if e[1] == target]


def load_graph(path: str) -> Graph:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    graph = Graph(**{s: raw.get(s) or {} for s in ENTITY_SECTIONS})

    relations = raw.get("relations") or {}
    for name in RELATIONS:
        if name not in relations:
            raise GraphError(f"relation '{name}' is missing from the graph")
        graph.relations[name] = [tuple(edge) for edge in relations[name] or []]

    known = set(graph.entities())
    for name, edges in graph.relations.items():
        for edge in edges:
            for endpoint in edge[:2]:
                if endpoint not in known:
                    raise GraphError(
                        f"relation '{name}' references unknown entity "
                        f"'{endpoint}'"
                    )
    return graph
