from akg.models import VerifiedAnswer


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def score_answer(item: dict, result: VerifiedAnswer) -> bool:
    answer = result.answer or ""

    if item.get("unanswerable"):
        return not any(
            _contains(answer, forbidden)
            for forbidden in item.get("must_not_contain", [])
        )

    required = item.get("requires_all", [])
    return all(_contains(answer, fact) for fact in required)


def retrieval_hit(item: dict, result: VerifiedAnswer) -> bool | None:
    expected = item.get("expected_sources", [])
    if item.get("unanswerable") or not expected:
        return None
    retrieved = {c.modality for c in result.contexts}
    return all(source in retrieved for source in expected)


def document_retrieval_hit(item: dict, result: VerifiedAnswer) -> bool | None:
    expected = item.get("expected_files") or []
    if not expected:
        return None
    retrieved = {c.source_file for c in result.contexts}
    return all(name in retrieved for name in expected)


def graph_retrieval_hit(item: dict, result: VerifiedAnswer) -> bool | None:
    graph_text = " ".join(
        c.text for c in result.contexts if c.modality == "graph"
    )
    required = item.get("requires_all") or []
    if not graph_text or not required or item.get("unanswerable"):
        return None
    return all(_contains(graph_text, fact) for fact in required)
