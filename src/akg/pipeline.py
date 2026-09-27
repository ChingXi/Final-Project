"""Ingest and query pipelines, wiring the stages together."""

import os
from dataclasses import replace

from akg.config import Settings
from akg.index.vector import VectorIndex, file_hash
from akg.ingest import audio, document
from akg.models import Chunk, VerifiedAnswer
from akg.retrieve.search import retrieve
from akg.synthesis.answer import synthesize
from akg.synthesis.llm import LLMError
from akg.verify.citations import verify_claims

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".flac"}
DOC_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}


def _files_under(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    return sorted(
        os.path.join(root, name)
        for root, _dirs, names in os.walk(path)
        for name in names
    )


def ingest_path(
    settings: Settings,
    path: str,
    force: bool = False,
    index: VectorIndex | None = None,
    scope: str = "shared",
) -> dict[str, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such file or directory: {path}")

    index = index or VectorIndex(settings)
    stats = {"ingested": 0, "skipped": 0, "chunks": 0}

    for file_path in _files_under(path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in AUDIO_EXTS and ext not in DOC_EXTS:
            continue
        if os.path.getsize(file_path) == 0:
            print(f"  skip  {os.path.basename(file_path)} (empty file)")
            stats["skipped"] += 1
            continue

        digest = file_hash(file_path)
        if not force and index.already_ingested(digest):
            print(f"  skip  {os.path.basename(file_path)} (already ingested)")
            stats["skipped"] += 1
            continue

        print(f"  read  {os.path.basename(file_path)}")
        chunks = (
            audio.transcribe(file_path, settings, digest)
            if ext in AUDIO_EXTS
            else document.parse(file_path, settings, digest)
        )
        if scope != "shared":
            chunks = [replace(c, scope=scope) for c in chunks]
        stats["chunks"] += index.add(chunks)
        stats["ingested"] += 1

    return stats


def _relabel(contexts: list[Chunk]) -> list[Chunk]:
    counters: dict[str, int] = {}
    out = []
    for chunk in contexts:
        prefix = {"audio": "A", "graph": "G"}.get(chunk.modality, "D")
        counters[prefix] = counters.get(prefix, 0) + 1
        out.append(replace(chunk, cid=f"{prefix}{counters[prefix]}"))
    return out


def open_graph(settings: Settings, required: bool):
    if not settings.graph_enabled:
        return None
    from akg.index.graph import GraphStore

    try:
        return GraphStore(settings)
    except RuntimeError:
        if required:
            raise
        return None


def open_entailment(settings: Settings):
    if not settings.entailment_enabled:
        return None
    from akg.verify.entailment import CrossEncoderChecker

    return CrossEncoderChecker(settings.entailment_model)


def answer(
    settings: Settings,
    index: VectorIndex,
    question: str,
    store=None,
    entail=None,
    allowed_scopes: list[str] | None = None,
) -> VerifiedAnswer:
    contexts = retrieve(index, question, allowed_scopes=allowed_scopes)
    if settings.graph_enabled and store is not None:
        from akg.retrieve.graph import graph_sources

        contexts = contexts + graph_sources(
            store, index, question, settings, allowed_scopes
        )
    contexts = _relabel(contexts)
    try:
        text, claims, recovered = synthesize(settings, question, contexts)
    except LLMError as e:
        return VerifiedAnswer(
            question=question, answer="", contexts=contexts, error=str(e)
        )

    checked, faithfulness = verify_claims(
        index.embed, contexts, claims, settings.citation_sim_threshold, entail
    )
    return VerifiedAnswer(
        question=question,
        answer=text,
        claims=checked,
        contexts=contexts,
        faithfulness=faithfulness,
        recovered_json=recovered,
    )
