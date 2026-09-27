from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    cid: str
    text: str
    source_file: str
    modality: str
    locator: str
    doc_hash: str
    scope: str = "shared"
    vector: Any = field(default=None, repr=False)


@dataclass
class Claim:
    text: str
    source_id: str


@dataclass
class CheckedClaim:
    text: str
    source_id: str
    similarity: float
    status: str
    entailment: str | None = None
    entail_p: float | None = None


@dataclass
class VerifiedAnswer:
    question: str
    answer: str
    claims: list[CheckedClaim] = field(default_factory=list)
    contexts: list[Chunk] = field(default_factory=list)
    faithfulness: float = 0.0
    recovered_json: bool = False
    error: str | None = None

    @property
    def supported(self) -> int:
        return sum(1 for c in self.claims if c.status == "supported")

    @property
    def disputed(self) -> int:
        return sum(1 for c in self.claims if c.status == "DISPUTED")
