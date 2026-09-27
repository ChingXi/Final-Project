from collections.abc import Callable

import numpy as np

from akg.models import CheckedClaim, Chunk, Claim
from akg.verify.entailment import EntailmentChecker


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def verify_claims(
    embed: Callable[[str], np.ndarray],
    contexts: list[Chunk],
    claims: list[Claim],
    threshold: float,
    entail: EntailmentChecker | None = None,
) -> tuple[list[CheckedClaim], float]:
    sources = {c.cid: c for c in contexts}
    checked: list[CheckedClaim] = []

    for claim in claims:
        chunk = sources.get(claim.source_id)
        if chunk is None or chunk.vector is None:
            checked.append(
                CheckedClaim(claim.text, claim.source_id, 0.0, "INVALID_CITATION")
            )
            continue
        similarity = cosine(embed(claim.text), chunk.vector)
        checked.append(
            CheckedClaim(
                text=claim.text,
                source_id=claim.source_id,
                similarity=round(similarity, 3),
                status="supported" if similarity >= threshold else "UNSUPPORTED",
            )
        )

    if entail is not None:
        apply_entailment(entail, sources, checked)

    supported = sum(1 for c in checked if c.status == "supported")
    return checked, (supported / len(checked) if checked else 0.0)


def faithfulness_of(checked: list[CheckedClaim]) -> float:
    supported = sum(1 for c in checked if c.status == "supported")
    return supported / len(checked) if checked else 0.0


def apply_entailment(
    entail: EntailmentChecker,
    sources: dict[str, Chunk],
    checked: list[CheckedClaim],
) -> None:
    indices = [i for i, c in enumerate(checked) if c.status == "supported"]
    pairs = [(sources[checked[i].source_id].text, checked[i].text) for i in indices]
    for i, verdict in zip(indices, entail.check(pairs)):
        claim = checked[i]
        claim.entailment = verdict.label
        claim.entail_p = verdict.probability
        if verdict.contradicted:
            claim.status = "DISPUTED"
