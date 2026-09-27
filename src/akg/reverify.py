from dataclasses import dataclass
from akg.models import CheckedClaim, Chunk
from akg.verify.citations import apply_entailment


@dataclass
class Change:
    qid: str
    claim: str
    source_id: str
    modality: str
    similarity: float
    entailment: str
    entail_p: float
    was: str
    now: str
    question_correct: bool | None


def _texts(index) -> dict[tuple[str, str], str]:
    return {
        (payload["source_file"], payload["locator"]): payload["text"]
        for payload in index.iter_payloads()
    }


def reverify_run(run: dict, index, entail) -> dict:
    texts = _texts(index)
    changes: list[Change] = []
    labels: dict[str, int] = {}
    unresolved = 0
    total = 0

    for question in run.get("questions", []):
        located = {
            source["cid"]: (
                source.get("source_file"),
                source.get("locator"),
                source.get("modality"),
            )
            for source in (question.get("retrieved") or [])
        }

        sources: dict[str, Chunk] = {}
        checked: list[CheckedClaim] = []
        modality: dict[int, str] = {}
        for claim in question.get("claims") or []:
            total += 1
            where = located.get(claim["source_id"])
            text = texts.get((where[0], where[1])) if where else None
            if text is None:
                unresolved += 1
                continue
            sources[claim["source_id"]] = Chunk(
                cid=claim["source_id"],
                text=text,
                source_file=where[0],
                modality=where[2],
                locator=where[1],
                doc_hash="",
            )
            modality[len(checked)] = where[2]
            checked.append(
                CheckedClaim(
                    text=claim["text"],
                    source_id=claim["source_id"],
                    similarity=claim["similarity"],
                    status=claim["status"],
                )
            )

        before = [c.status for c in checked]
        apply_entailment(entail, sources, checked)
        for i, claim in enumerate(checked):
            labels[claim.entailment or "not checked"] = (
                labels.get(claim.entailment or "not checked", 0) + 1
            )
            if claim.status != before[i]:
                changes.append(
                    Change(
                        qid=question["id"],
                        claim=claim.text,
                        source_id=claim.source_id,
                        modality=modality.get(i, "?"),
                        similarity=claim.similarity,
                        entailment=claim.entailment or "",
                        entail_p=claim.entail_p or 0.0,
                        was=before[i],
                        now=claim.status,
                        question_correct=question.get("correct"),
                    )
                )

    return {
        "claims": total,
        "unresolved": unresolved,
        "labels": labels,
        "changes": [c.__dict__ for c in changes],
    }


def format_report(summary: dict) -> str:
    lines = [
        f"claims re-verified: {summary['claims'] - summary['unresolved']}"
        f" of {summary['claims']}"
    ]
    if summary["unresolved"]:
        lines.append(
            f"unresolved: {summary['unresolved']} "
            "(cited a source with no chunk in the collection)"
        )

    lines.append("\nentailment labels:")
    for label, count in sorted(summary["labels"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {label:14} {count:4}")

    changes = summary["changes"]
    lines.append(f"\nclaims demoted to DISPUTED: {len(changes)}")
    if changes:
        lines.append(
            "\nEvery one of these passed the similarity threshold. The answer's"
            "\ncorrectness is shown alongside because a wrong citation under a"
            "\nright answer is precisely what faithfulness exists to expose."
        )
        for c in sorted(changes, key=lambda c: -c["entail_p"]):
            lines.append(
                f"\n  {c['qid']:5} {c['modality']:8} sim={c['similarity']:<6} "
                f"{c['entailment']} p={c['entail_p']:<6} "
                f"answer_correct={c['question_correct']}"
            )
            lines.append(f"        {c['claim'][:100]}")
    return "\n".join(lines)
