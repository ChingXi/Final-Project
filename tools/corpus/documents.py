from dataclasses import dataclass

from tools.corpus.model import Graph
from tools.corpus.render import Page

RELATION_SOURCE = {
    "owns": "status_report",
    "member_of": "roster",
    "reports_to": "roster",
    "supports": "team_structure",
    "supplies": "vendor_register",
    "depends_on": "dependency_map",
    "affects": "risk_register",
    "caused_by": "risk_register",
}


@dataclass
class Document:
    filename: str
    family: str
    page: Page


def _status_report(g: Graph) -> Document:
    page = Page("Portfolio Status Report", "Q1 2026  |  3 March 2026")
    for pid, project in g.projects.items():
        owner = g.inverse("owns", pid)
        page.heading(project["name"])
        if owner:
            page.field(f"Owner: {g.name_of(owner[0])}")
        page.field(f"Budget: ${project['budget']:,}")
        page.field(f"Spent to date: ${project['spent']:,}")
        page.field(f"Remaining: ${project['remaining']:,}")
        page.field(f"Planned launch: {project['launch']}")
        page.field(f"Status: {project['status']}")
        page.gap()
    return Document("portfolio_status_report.pdf", "status_report", page)


def _roster(g: Graph) -> Document:
    page = Page("Organisation Roster", "People, teams and reporting lines")
    for tid, team in g.teams.items():
        page.heading(team["name"])
        for pid in g.inverse("member_of", tid):
            person = g.people[pid]
            manager = g.related("reports_to", pid)
            suffix = f", reports to {g.name_of(manager[0])}" if manager else ""
            page.item(f"{person['name']} - {person['role']}{suffix}")
        page.gap()
    return Document("org_roster.pdf", "roster", page)


def _team_structure(g: Graph) -> Document:
    page = Page("Team Structure", "Which teams support which projects")
    for tid, team in g.teams.items():
        page.heading(team["name"])
        for pid in g.related("supports", tid):
            page.item(f"Supports {g.projects[pid]['name']}")
        page.gap()
    return Document("team_structure.pdf", "team_structure", page)


def _vendor_register(g: Graph) -> Document:
    page = Page("Vendor Register", "FY26 supplier commitments")
    for vid, vendor in g.vendors.items():
        page.heading(vendor["name"])
        page.gap()
        page.item(f"Service: {vendor['service']}")
        for edge in g.relations["supplies"]:
            if edge[0] == vid:
                page.item(f"Supplies {g.projects[edge[1]]['name']}: {edge[2]}")
        page.gap()
    return Document("vendor_register.pdf", "vendor_register", page)


def _dependency_map(g: Graph) -> Document:
    page = Page("Project Dependency Map", "Delivery sequencing, FY26")
    for source, target in g.relations["depends_on"]:
        page.item(
            f"{g.projects[source]['name']} depends on "
            f"{g.projects[target]['name']}"
        )
    return Document("dependency_map.pdf", "dependency_map", page)


def _risk_register(g: Graph) -> Document:
    page = Page("Risk Register", "Open risks, Q1 2026")
    for rid, risk in g.risks.items():
        page.heading(f"Risk {rid.replace('r_', '').replace('_', ' ').title()}")
        page.gap()
        page.item(f"Description: {risk['description']}")
        page.item(f"Severity: {risk['severity']}")
        for pid in g.related("affects", rid):
            page.item(f"Affects {g.projects[pid]['name']}")
        for cause in g.related("caused_by", rid):
            page.item(f"Attributed to {g.name_of(cause)}")
        page.gap()
    return Document("risk_register.pdf", "risk_register", page)


MINUTES = [
    (
        "steering_committee_january.pdf",
        "Steering Committee Minutes",
        "14 January 2026",
        [
            "Attendees noted the quarter opened on plan.",
            "Budget tracking was reviewed and no escalations were raised.",
            "A follow-up on procurement timelines was deferred to February.",
            "The committee agreed to keep the reporting cadence unchanged.",
        ],
    ),
    (
        "steering_committee_february.pdf",
        "Steering Committee Minutes",
        "11 February 2026",
        [
            "Procurement timelines were discussed at length.",
            "The committee reviewed spend against forecast for the quarter.",
            "No changes to delivery sequencing were agreed at this meeting.",
            "An action was raised to refresh the risk register before Q2.",
        ],
    ),
    (
        "delivery_review_february.pdf",
        "Delivery Review Notes",
        "25 February 2026",
        [
            "Delivery leads walked through progress across the portfolio.",
            "Testing capacity was raised as a general concern for the quarter.",
            "The review closed without changes to reported status.",
            "Notes were circulated to attendees for comment.",
        ],
    ),
    (
        "budget_review_march.pdf",
        "Budget Review Notes",
        "2 March 2026",
        [
            "Finance presented spend to date across all active projects.",
            "Forecast variance was within tolerance at portfolio level.",
            "No reallocations were approved at this review.",
            "The next review was scheduled for the start of Q2.",
        ],
    ),
]


def _minutes(g: Graph) -> list[Document]:
    docs = []
    for filename, title, date, lines in MINUTES:
        page = Page(title, date)
        for text in lines:
            page.line(text)
        docs.append(Document(filename, "minutes", page))
    return docs


def build_documents(graph: Graph) -> list[Document]:
    return [
        _status_report(graph),
        _roster(graph),
        _team_structure(graph),
        _vendor_register(graph),
        _dependency_map(graph),
        _risk_register(graph),
        *_minutes(graph),
    ]
