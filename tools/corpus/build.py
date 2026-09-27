import argparse
import json
import os

from tools.corpus.documents import build_documents
from tools.corpus.model import load_graph
from tools.corpus.questions import (
    CorpusError,
    assert_multi_hop,
    assert_single_hop,
    derive_questions,
)
from tools.corpus.speech import VOICE_NOTES

GRAPH = "corpus/graph.yaml"
DOC_DIR = "data/documents"
AUDIO_DIR = "data/audio"
GOLD = "eval/gold.json"


def _audio_items(items: list[dict]) -> list[dict]:
    return [
        item for item in items
        if not item.get("unanswerable")
        and "audio" in item.get("expected_sources", [])
    ]


def _audio_failures(items: list[dict], spoken: str, where: str) -> list[str]:
    failures = []
    for item in items:
        facts = item["requires_all"]
        present = [f for f in facts if f.lower() in spoken]
        if set(item["expected_sources"]) == {"audio"}:
            for fact in facts:
                if fact not in present:
                    failures.append(f"{item['id']}: {fact!r} {where}")
        elif not present:
            failures.append(
                f"{item['id']}: none of {facts!r} {where}, so the question is "
                f"not cross-source"
            )
    return failures


def validate_scripts(items: list[dict]) -> list[str]:
    spoken = " ".join(s for _f, _v, s in VOICE_NOTES).lower()
    return _audio_failures(_audio_items(items), spoken, "is in no voice note script")


def validate_parse(documents, items, doc_dir: str = DOC_DIR) -> list[str]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    corpus_text = []
    for doc in documents:
        path = os.path.join(doc_dir, doc.filename)
        parsed = converter.convert(path).document.export_to_markdown()
        corpus_text.append(parsed.lower())
    combined = " ".join(corpus_text)

    failures = []
    for item in items:
        if item.get("unanswerable") or "audio" in item.get("expected_sources", []):
            continue
        for fact in item.get("requires_all", []):
            if fact.lower() not in combined:
                failures.append(f"{item['id']}: {fact!r} not found in parsed text")
    return failures


def validate_transcripts(items: list[dict], audio_dir: str = AUDIO_DIR) -> list[str]:
    from akg.config import load_settings
    from akg.ingest.audio import transcribe

    settings = load_settings()
    heard = []
    for filename, _voice, _script in VOICE_NOTES:
        path = os.path.join(audio_dir, filename)
        if not os.path.exists(path):
            return [f"{path} is missing; run the build with audio enabled"]
        chunks = transcribe(path, settings, doc_hash="validation")
        heard.append(" ".join(c.text for c in chunks).lower())

    return _audio_failures(
        _audio_items(items), " ".join(heard), "did not survive transcription"
    )


def build(
    doc_dir: str = DOC_DIR,
    audio_dir: str = AUDIO_DIR,
    gold_path: str = GOLD,
    with_audio: bool = True,
    graph_path: str = GRAPH,
) -> dict:
    graph = load_graph(graph_path)
    documents = build_documents(graph)
    items = derive_questions(graph)

    assert_multi_hop(items, documents)
    assert_single_hop(items, documents)

    script_failures = validate_scripts(items)
    if script_failures:
        raise CorpusError("; ".join(script_failures))

    os.makedirs(doc_dir, exist_ok=True)
    for doc in documents:
        doc.page.save(os.path.join(doc_dir, doc.filename))

    if with_audio:
        from tools.corpus.speech import synthesise

        synthesise(audio_dir)

    os.makedirs(os.path.dirname(gold_path) or ".", exist_ok=True)
    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
        f.write("\n")

    counts: dict[str, int] = {}
    for item in items:
        counts[item["type"]] = counts.get(item["type"], 0) + 1

    return {
        "documents": len(documents),
        "questions": len(items),
        "multi_hop": sum(v for k, v in counts.items() if k.startswith("multi_hop")),
        "by_type": counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate the evaluation corpus")
    ap.add_argument("--no-audio", action="store_true",
                    help="skip speech synthesis")
    ap.add_argument("--skip-parse-check", action="store_true",
                    help="skip the Docling and Whisper round-trip checks (slow)")
    args = ap.parse_args()

    try:
        stats = build(with_audio=not args.no_audio)
    except CorpusError as e:
        raise SystemExit(f"corpus invalid: {e}") from e

    print(json.dumps(stats, indent=2))

    if args.skip_parse_check:
        return

    graph = load_graph(GRAPH)
    documents = build_documents(graph)
    items = derive_questions(graph)

    print("\nchecking every gold fact survives Docling...")
    failures = validate_parse(documents, items)
    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        raise SystemExit(f"{len(failures)} gold fact(s) unreadable after OCR")
    print("  all gold facts readable after OCR")

    if not args.no_audio:
        print("\nchecking every audio gold fact survives transcription...")
        failures = validate_transcripts(items)
        if failures:
            for failure in failures:
                print(f"  FAIL {failure}")
            raise SystemExit(
                f"{len(failures)} gold fact(s) lost in transcription"
            )
        print("  all audio gold facts survive transcription")


if __name__ == "__main__":
    main()
