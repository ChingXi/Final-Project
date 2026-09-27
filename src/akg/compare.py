
from akg.stats import mcnemar_exact


def _index(run: dict) -> dict[str, dict]:
    return {q["id"]: q for q in run.get("questions", []) if q.get("id")}


def _abstained(record: dict) -> bool:
    return not record.get("claims")


def compare_runs(a: dict, b: dict) -> dict:
    ai, bi = _index(a), _index(b)
    shared = sorted(set(ai) & set(bi))
    if not shared:
        raise ValueError(
            f"no questions in common: {len(ai)} and {len(bi)} question(s) with "
            f"no overlapping ids. Are these runs of the same gold set?"
        )

    excluded = [
        qid for qid in shared
        if ai[qid].get("error") or bi[qid].get("error")
    ]
    paired = [qid for qid in shared if qid not in set(excluded)]

    only_a, only_b, concordant = [], [], 0
    by_type: dict[str, dict[str, int]] = {}

    for qid in paired:
        ra, rb = ai[qid], bi[qid]
        ca, cb = bool(ra.get("correct")), bool(rb.get("correct"))

        bucket = by_type.setdefault(
            ra.get("type") or "untyped",
            {"n": 0, "a_correct": 0, "b_correct": 0, "b": 0, "c": 0},
        )
        bucket["n"] += 1
        bucket["a_correct"] += int(ca)
        bucket["b_correct"] += int(cb)

        if ca and not cb:
            only_a.append(qid)
            bucket["b"] += 1
        elif cb and not ca:
            only_b.append(qid)
            bucket["c"] += 1
        else:
            concordant += 1

    return {
        "a_model": a.get("synthesis_model"),
        "b_model": b.get("synthesis_model"),
        "paired": len(paired),
        "unpaired_ids": sorted(set(ai) ^ set(bi)),
        "excluded_errored": excluded,
        "a_correct": sum(1 for q in paired if ai[q].get("correct")),
        "b_correct": sum(1 for q in paired if bi[q].get("correct")),
        "a_abstained": sum(1 for q in paired if _abstained(ai[q])),
        "b_abstained": sum(1 for q in paired if _abstained(bi[q])),
        "b": len(only_a),
        "c": len(only_b),
        "concordant": concordant,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "p_value": mcnemar_exact(len(only_a), len(only_b)),
        "by_type": by_type,
    }


def format_report(result: dict) -> str:
    a, b = result["a_model"], result["b_model"]
    lines = [
        f"A: {a}",
        f"B: {b}",
        "",
        f"paired:     {result['paired']}"
        + (
            f"  (excluded {len(result['excluded_errored'])} errored: "
            f"{', '.join(result['excluded_errored'])})"
            if result["excluded_errored"] else ""
        ),
        f"correct:    A {result['a_correct']}   B {result['b_correct']}",
        f"abstained:  A {result['a_abstained']}   B {result['b_abstained']}",
        "",
        f"discordant: b={result['b']} (A only)   c={result['c']} (B only)"
        f"   concordant={result['concordant']}",
        f"McNemar exact p = {result['p_value']:.4f}",
        "",
        f"{'type':<14} {'n':>3} {'A':>4} {'B':>4} {'b':>3} {'c':>3}",
    ]
    for name, t in sorted(result["by_type"].items()):
        lines.append(
            f"{name:<14} {t['n']:>3} {t['a_correct']:>4} {t['b_correct']:>4} "
            f"{t['b']:>3} {t['c']:>3}"
        )

    if result["only_b_correct"]:
        lines += ["", "B fixed: " + ", ".join(result["only_b_correct"])]
    if result["only_a_correct"]:
        lines += ["B broke: " + ", ".join(result["only_a_correct"])]
    if result["unpaired_ids"]:
        lines += ["", "unpaired: " + ", ".join(result["unpaired_ids"])]
    return "\n".join(lines)
