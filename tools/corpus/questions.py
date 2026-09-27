from tools.corpus.documents import RELATION_SOURCE, Document
from tools.corpus.model import Graph

FAMILY_FILE = {
    "status_report": "portfolio_status_report.pdf",
    "roster": "org_roster.pdf",
    "team_structure": "team_structure.pdf",
    "vendor_register": "vendor_register.pdf",
    "dependency_map": "dependency_map.pdf",
    "risk_register": "risk_register.pdf",
}


class CorpusError(Exception):
    """The generated corpus violates a property the evaluation depends on."""


def _files_for(*relations: str) -> list[str]:
    families = {RELATION_SOURCE[r] for r in relations}
    return sorted(FAMILY_FILE[f] for f in families)


class _Sequence:

    def __init__(self) -> None:
        self.counter = 0

    def next_id(self) -> str:
        self.counter += 1
        return f"Q{self.counter:02d}"


def _item(seq: _Sequence, qtype: str, question: str, requires: list[str],
          entities: list[str], relations: tuple[str, ...]) -> dict:
    return {
        "id": seq.next_id(),
        "type": qtype,
        "question": question,
        "requires_all": requires,
        "query_entities": entities,
        "expected_files": _files_for(*relations),
        "expected_sources": ["document"],
    }


def _risk_team_projects(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for rid, risk in g.risks.items():
        teams = [c for c in g.related("caused_by", rid) if c in g.teams]
        for tid in teams:
            projects = g.related("supports", tid)
            if len(projects) < 2:
                continue
            out.append(_item(
                seq,
                "multi_hop_2",
                f"Which projects does the team responsible for the "
                f"{risk['description']} support?",
                [g.projects[p]["name"] for p in projects],
                [risk["description"]],
                ("caused_by", "supports"),
            ))
    return out


def _vendor_project_owner(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for vid, vendor in g.vendors.items():
        projects = [e[1] for e in g.relations["supplies"] if e[0] == vid]
        owners = [g.inverse("owns", p) for p in projects]
        names = [g.name_of(o[0]) for o in owners if o]
        if not names:
            continue
        out.append(_item(
            seq,
            "multi_hop_2",
            f"If {vendor['name']} misses a deadline, which project owners "
            f"need to be told?",
            names,
            [vendor["name"]],
            ("supplies", "owns"),
        ))
    return out


def _dependency_sources(g: Graph) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for source, target in g.relations["depends_on"]:
        grouped.setdefault(source, []).append(target)
    return list(grouped.items())


def _dependency_owner(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for source, targets in _dependency_sources(g):
        owners = [g.name_of(o[0]) for t in targets
                  for o in [g.inverse("owns", t)] if o]
        if not owners:
            continue
        out.append(_item(
            seq,
            "multi_hop_2",
            f"Who owns the projects that {g.projects[source]['name']} "
            f"depends on?",
            owners,
            [g.projects[source]["name"]],
            ("depends_on", "owns"),
        ))
    return out


def _team_member_projects(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for tid, team in g.teams.items():
        members = g.inverse("member_of", tid)
        owned = [p for m in members for p in g.related("owns", m)]
        if not owned:
            continue
        out.append(_item(
            seq,
            "multi_hop_2",
            f"Which projects are owned by members of the {team['name']}?",
            [g.projects[p]["name"] for p in owned],
            [team["name"]],
            ("member_of", "owns"),
        ))
    return out


def _risk_team_project_owner(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for rid, risk in g.risks.items():
        for tid in [c for c in g.related("caused_by", rid) if c in g.teams]:
            owners = [
                g.name_of(o[0])
                for p in g.related("supports", tid)
                for o in [g.inverse("owns", p)]
                if o
            ]
            if not owners:
                continue
            out.append(_item(
                seq,
                "multi_hop_3",
                f"Who owns the projects supported by the team responsible "
                f"for the {risk['description']}?",
                owners,
                [risk["description"]],
                ("caused_by", "supports", "owns"),
            ))
    return out


def _dependency_owner_manager(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for source, targets in _dependency_sources(g):
        managers = [
            g.name_of(m[0])
            for t in targets
            for o in [g.inverse("owns", t)] if o
            for m in [g.related("reports_to", o[0])] if m
        ]
        if not managers:
            continue
        out.append(_item(
            seq,
            "multi_hop_3",
            f"Who do the owners of the projects that "
            f"{g.projects[source]['name']} depends on report to?",
            managers,
            [g.projects[source]["name"]],
            ("depends_on", "owns", "reports_to"),
        ))
    return out


def _team_member_vendors(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for tid, team in g.teams.items():
        owned = [
            p for m in g.inverse("member_of", tid) for p in g.related("owns", m)
        ]
        vendors = sorted({
            g.vendors[e[0]]["name"]
            for e in g.relations["supplies"]
            if e[1] in owned
        })
        if not vendors:
            continue
        out.append(_item(
            seq,
            "multi_hop_3",
            f"Which vendors supply the projects owned by members of the "
            f"{team['name']}?",
            vendors,
            [team["name"]],
            ("member_of", "owns", "supplies"),
        ))
    return out


def _superseded(project: dict) -> list[str]:
    return project.get("superseded") or []


def _single_hop_budget(g: Graph, seq: _Sequence) -> list[dict]:
    return [
        _item(
            seq, "single_hop",
            f"What is {p['name']}'s budget, and how much remains?",
            [f"{p['budget']:,}", f"{p['remaining']:,}"],
            [p["name"]], ("owns",),
        )
        for p in g.projects.values()
    ]


def _single_hop_owner(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for pid, project in g.projects.items():
        owner = g.inverse("owns", pid)
        if not owner or "owner" in _superseded(project):
            continue
        out.append(_item(
            seq, "single_hop",
            f"Who owns {project['name']}?",
            [g.name_of(owner[0])], [project["name"]], ("owns",),
        ))
    return out


def _single_hop_schedule(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for project in g.projects.values():
        if "launch" in _superseded(project):
            continue
        out.append(_item(
            seq, "single_hop",
            f"When is {project['name']} planned to launch, and what is its "
            f"reported status?",
            [str(project["launch"]), project["status"]],
            [project["name"]], ("owns",),
        ))
    return out


def _single_hop_team(g: Graph, seq: _Sequence) -> list[dict]:
    return [
        _item(
            seq, "single_hop",
            f"Which projects does the {team['name']} support?",
            [g.projects[p]["name"] for p in g.related("supports", tid)],
            [team["name"]], ("supports",),
        )
        for tid, team in g.teams.items()
    ]


def _single_hop_vendor(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for vid, vendor in g.vendors.items():
        supplied = [g.projects[e[1]]["name"]
                    for e in g.relations["supplies"] if e[0] == vid]
        out.append(_item(
            seq, "single_hop",
            f"What service does {vendor['name']} provide, and which projects "
            f"does it supply?",
            [vendor["service"], *supplied], [vendor["name"]], ("supplies",),
        ))
    return out


def _single_hop_risk(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for rid, risk in g.risks.items():
        causes = g.related("caused_by", rid)
        if not causes:
            continue
        out.append(_item(
            seq, "single_hop",
            f"What severity is recorded for the {risk['description']}, and "
            f"who is it attributed to?",
            [risk["severity"], g.name_of(causes[0])],
            [risk["description"]], ("caused_by",),
        ))
    return out


def _single_hop_person(g: Graph, seq: _Sequence) -> list[dict]:
    out = []
    for pid, person in g.people.items():
        manager = g.related("reports_to", pid)
        if not manager:
            continue
        out.append(_item(
            seq, "single_hop",
            f"What is {person['name']}'s role, and who do they report to?",
            [person["role"], g.name_of(manager[0])],
            [person["name"]], ("reports_to",),
        ))
    return out


def _single_hop_dependency(g: Graph, seq: _Sequence) -> list[dict]:
    return [
        _item(
            seq, "single_hop",
            f"Which projects does {g.projects[source]['name']} depend on?",
            [g.projects[t]["name"] for t in targets],
            [g.projects[source]["name"]], ("depends_on",),
        )
        for source, targets in _dependency_sources(g)
    ]


def _traps(g: Graph, seq: _Sequence) -> list[dict]:
    atlas = g.projects["atlas"]
    out = [
        _item(seq, "trap", "What is Project Atlas's marketing budget?", [], [],
              ("owns",)),
        _item(seq, "trap", "Who is the project manager for Project Borealis?",
              [], [], ("owns",)),
        _item(seq, "trap", "What is the Data Team's annual budget?", [], [],
              ("supports",)),
        _item(seq, "trap", "When was Meridian Systems founded?", [], [],
              ("supplies",)),
    ]
    forbidden = [
        [f"{atlas['budget']:,}", f"{atlas['remaining']:,}", f"{atlas['spent']:,}"],
        ["Sam Patel", "Maria Chen", "Lena Ortiz"],
        [f"{g.projects['cirrus']['budget']:,}",
         f"{g.projects['borealis']['budget']:,}"],
        ["2026", "2027"],
    ]
    for item, values in zip(out, forbidden):
        item["unanswerable"] = True
        item["must_not_contain"] = values
        item["requires_all"] = []
        item["expected_files"] = []
        item["expected_sources"] = []
    return out


def _cross_source_and_temporal(g: Graph, seq: _Sequence) -> list[dict]:
    items = []
    for qtype, question, requires in [
        ("cross_source",
         "Who owns Project Atlas, and when was its launch moved to?",
         ["Maria", "September"]),
        ("cross_source",
         "How much of Project Atlas's budget remains, and what is it being "
         "reallocated to?",
         ["35,000", "quality assurance"]),
        ("cross_source",
         "What is Project Fathom's remaining budget, and what risk was raised "
         "about it in the meeting?",
         ["4,000", "rollback"]),
        ("temporal",
         "Who is currently taking over as owner of Project Atlas?",
         ["David"]),
        ("temporal",
         "Has Project Atlas's launch date changed from what the report says?",
         ["September"]),
        ("temporal",
         "Which project has had its delivery date brought forward since the "
         "report?",
         ["Borealis"]),
    ]:
        items.append({
            "id": seq.next_id(),
            "type": qtype,
            "question": question,
            "requires_all": requires,
            "query_entities": [],
            "expected_files": [],
            "expected_sources": ["document", "audio"] if qtype == "cross_source"
                                else ["audio"],
        })
    return items


def derive_questions(graph: Graph) -> list[dict]:
    seq = _Sequence()
    items: list[dict] = []
    for builder in (
        _risk_team_projects,
        _vendor_project_owner,
        _dependency_owner,
        _team_member_projects,
        _risk_team_project_owner,
        _dependency_owner_manager,
        _team_member_vendors,
        _single_hop_budget,
        _single_hop_owner,
        _single_hop_schedule,
        _single_hop_team,
        _single_hop_vendor,
        _single_hop_risk,
        _single_hop_person,
        _single_hop_dependency,
        _traps,
        _cross_source_and_temporal,
    ):
        items.extend(builder(graph, seq))
    return items


def assert_single_hop(items: list[dict], documents: list[Document]) -> None:
    for item in items:
        if item["type"] != "single_hop":
            continue
        entities = [e.lower() for e in item["query_entities"]]
        answers = [a.lower() for a in item["requires_all"]]
        if any(
            all(e in text for e in entities) and all(a in text for a in answers)
            for doc in documents
            for text in (" ".join(g).lower() for g in doc.page.groups())
        ):
            continue
        raise CorpusError(
            f"{item['id']} is labelled single_hop but no single passage "
            f"contains both {item['query_entities']!r} and "
            f"{item['requires_all']!r}, so it is not a control question"
        )


def assert_multi_hop(items: list[dict], documents: list[Document]) -> None:
    for item in items:
        if not item["type"].startswith("multi_hop"):
            continue
        entities = [e.lower() for e in item["query_entities"]]
        answers = [a.lower() for a in item["requires_all"]]
        for doc in documents:
            for group in doc.page.groups():
                text = " ".join(group).lower()
                if (all(e in text for e in entities)
                        and all(a in text for a in answers)):
                    raise CorpusError(
                        f"{item['id']} is labelled {item['type']} but a single "
                        f"passage of {doc.filename} contains both its query "
                        f"entities and its answer, so it is answerable without "
                        f"traversal: {group!r}"
                    )
