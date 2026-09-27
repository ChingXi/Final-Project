import json

from akg.config import Settings
from akg.index.vector import VectorIndex
from akg.models import VerifiedAnswer
from akg.pipeline import answer
from akg.scoring import (
    document_retrieval_hit,
    graph_retrieval_hit,
    retrieval_hit,
    score_answer,
)
from akg.synthesis.llm import LLMError, chat_text

VERDICTS = ("not_in_sources", "wrong_question", "answers")

JUDGE_SYSTEM = (
    "You judge whether an answer addresses the question using only the given "
    "sources. Reply with exactly one word and nothing else:\n"
    "  answers          - the answer addresses the question\n"
    "  not_in_sources   - the sources do not contain what the question asks\n"
    "  wrong_question   - the answer is about something other than what was "
    "asked"
)


def parse_verdict(raw: str) -> str | None:
    lowered = (raw or "").lower()
    for verdict in VERDICTS:
        if verdict in lowered:
            return verdict
    return None


def judge_relevance(settings: Settings, result: VerifiedAnswer) -> dict:
    sources = "\n".join(f"[{c.cid}] {c.text}" for c in result.contexts)
    user = (
        f"QUESTION:\n{result.question}\n\nANSWER:\n{result.answer}\n\n"
        f"SOURCES:\n{sources}\n\nOne word only."
    )
    try:
        raw = chat_text(settings, settings.llm_judge, JUDGE_SYSTEM, user,
                        temperature=0.0)
    except LLMError as e:
        return {"verdict": "error", "reason": str(e)}

    verdict = parse_verdict(raw)
    if verdict is None:
        return {"verdict": "unparsed", "reason": raw[:200]}
    return {"verdict": verdict, "reason": raw.strip()[:200]}


def aggregate(
    results: list[VerifiedAnswer], gold: list[dict] | None = None
) -> dict:
    scored = [r for r in results if r.error is None]

    claimed = [r for r in scored if r.claims]
    abstained = len(scored) - len(claimed)

    stats = {
        "total": len(results),
        "scored": len(scored),
        "errored": len(results) - len(scored),
        "abstained": abstained,
        "recovered_json": sum(1 for r in scored if r.recovered_json),
        "mean_faithfulness": (
            sum(r.faithfulness for r in claimed) / len(claimed) if claimed else None
        ),
    }

    if gold is None:
        return stats

    pairs = [(i, r) for i, r in zip(gold, results) if r.error is None]
    correct = [score_answer(i, r) for i, r in pairs]
    hits = [h for h in (retrieval_hit(i, r) for i, r in pairs) if h is not None]

    stats["correct"] = sum(correct)
    stats["accuracy"] = (sum(correct) / len(correct)) if correct else None
    stats["retrieval_scored"] = len(hits)
    stats["retrieval_accuracy"] = (sum(hits) / len(hits)) if hits else None

    doc_hits = [
        h for h in (document_retrieval_hit(i, r) for i, r in pairs)
        if h is not None
    ]
    stats["document_retrieval_scored"] = len(doc_hits)
    stats["document_retrieval_accuracy"] = (
        (sum(doc_hits) / len(doc_hits)) if doc_hits else None
    )

    graph_hits = [
        h for h in (graph_retrieval_hit(i, r) for i, r in pairs) if h is not None
    ]
    stats["graph_retrieval_scored"] = len(graph_hits)
    stats["graph_retrieval_accuracy"] = (
        (sum(graph_hits) / len(graph_hits)) if graph_hits else None
    )

    by_source: dict[str, dict[str, int]] = {}
    for _item, result in pairs:
        kinds = {c.cid: c.modality for c in result.contexts}
        for claim in result.claims:
            bucket = by_source.setdefault(
                kinds.get(claim.source_id, "unknown"),
                {"claims": 0, "supported": 0},
            )
            bucket["claims"] += 1
            bucket["supported"] += int(claim.status == "supported")
    stats["faithfulness_by_source"] = by_source

    by_type: dict[str, dict[str, int]] = {}
    for item, result in pairs:
        bucket = by_type.setdefault(
            item.get("type", "untyped"),
            {"n": 0, "correct": 0, "doc_retrieval_n": 0, "doc_retrieval_hit": 0},
        )
        bucket["n"] += 1
        bucket["correct"] += int(score_answer(item, result))
        hit = document_retrieval_hit(item, result)
        if hit is not None:
            bucket["doc_retrieval_n"] += 1
            bucket["doc_retrieval_hit"] += int(hit)
    stats["by_type"] = by_type

    return stats


def run_gold_set(settings: Settings, gold_path: str) -> dict:
    settings.require_independent_judge()

    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    index = VectorIndex(settings)

    from akg.pipeline import open_entailment, open_graph

    store = open_graph(settings, required=True)
    entail = open_entailment(settings)

    arm = "hybrid" if store is not None else "vector-only"
    if entail is not None:
        arm += ", entailment"
    print(f"pass 1/2: answering {len(gold)} question(s) with "
          f"{settings.llm_synthesis} ({arm})")
    results: list[VerifiedAnswer] = []
    for item in gold:
        result = answer(settings, index, item["question"], store, entail)
        results.append(result)
        if result.error:
            print(f"  {item.get('id', '?'):<4} ERROR  {item['question'][:50]}")
            continue
        correct = "ok " if score_answer(item, result) else "WRONG"
        print(
            f"  {item.get('id', '?'):<4} {correct}  faith={result.faithfulness:>4.0%}"
            f"  {item['question'][:46]}"
        )

    print(f"\npass 2/2: judging with {settings.llm_judge}")
    verdicts: dict[str, int] = {}
    per_question = []
    for item, result in zip(gold, results):
        verdict = "skipped"
        if not result.error:
            verdict = judge_relevance(settings, result).get("verdict", "?")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        print(f"  {verdict:>14}  {result.question[:56]}")

        per_question.append({
            "id": item.get("id"),
            "type": item.get("type"),
            "question": result.question,
            "answer": result.answer,
            "correct": None if result.error else score_answer(item, result),
            "retrieval_hit": None if result.error else retrieval_hit(item, result),
            "document_retrieval_hit": (
                None if result.error else document_retrieval_hit(item, result)
            ),
            "requires_all": item.get("requires_all"),
            "graph_retrieval_hit": (
                None if result.error else graph_retrieval_hit(item, result)
            ),
            "expected_files": item.get("expected_files"),
            "must_not_contain": item.get("must_not_contain"),
            "expected_sources": item.get("expected_sources"),
            "faithfulness": result.faithfulness,
            "verdict": verdict,
            "error": result.error,
            "claims": [
                {
                    "text": c.text,
                    "source_id": c.source_id,
                    "similarity": c.similarity,
                    "status": c.status,
                    "entailment": c.entailment,
                    "entail_p": c.entail_p,
                }
                for c in result.claims
            ],
            "retrieved": [
                {"cid": c.cid, "modality": c.modality, "source_file": c.source_file,
                 "locator": c.locator}
                for c in result.contexts
            ],
        })

    if store is not None:
        store.close()

    stats = aggregate(results, gold)
    stats["questions"] = per_question
    stats["verdicts"] = verdicts
    stats["graph_enabled"] = settings.graph_enabled
    stats["graph_max_depth"] = settings.graph_max_depth
    stats["graph_max_edges"] = settings.graph_max_edges
    stats["synthesis_model"] = settings.llm_synthesis
    stats["judge_model"] = settings.llm_judge
    stats["threshold"] = settings.citation_sim_threshold
    stats["entailment_enabled"] = settings.entailment_enabled
    stats["entailment_model"] = (
        settings.entailment_model if settings.entailment_enabled else None
    )
    stats["top_k"] = settings.top_k
    return stats
