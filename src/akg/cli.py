"""Command line interface."""

import json
import time

import typer

from akg.config import load_settings
from akg.evaluation import run_gold_set
from akg.index.vector import VectorIndex
from akg.pipeline import answer, ingest_path

app = typer.Typer(add_completion=False, help="Multimodal knowledge gateway")


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    force: bool = typer.Option(False, "--force", help="Re-ingest unchanged files"),
    scope: str = typer.Option(
        "shared",
        "--scope",
        help=(
            "Access scope to tag these chunks with. Anything other than "
            "'shared' is retrievable only by a chat assigned that scope in "
            "TELEGRAM_CHAT_SCOPES."
        ),
    ),
) -> None:
    settings = load_settings()
    try:
        stats = ingest_path(settings, path, force=force, scope=scope)
    except (FileNotFoundError, RuntimeError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    typer.echo(
        f"\ningested {stats['ingested']} file(s), skipped {stats['skipped']}, "
        f"{stats['chunks']} chunk(s) indexed"
    )


@app.command()
def extract(
    reset: bool = typer.Option(
    ),
) -> None:
    from akg.extract.graph import extract_graph
    from akg.index.graph import GraphStore

    settings = load_settings()
    try:
        index = VectorIndex(settings)
        store = GraphStore(settings)
    except RuntimeError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    from akg.index.vector import _to_chunk

    found, _ = index.client.scroll(
        index.collection, limit=10000, with_payload=True, with_vectors=False
    )
    chunks = [_to_chunk(p.payload) for p in found]
    if not chunks:
        typer.secho(
            "error: the vector index is empty; run 'akg ingest data/' first",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"extracting from {len(chunks)} chunk(s) with {settings.llm_extraction}...")
    try:
        stats = extract_graph(settings, chunks, store, reset=reset)
    finally:
        store.close()

    nodes, edges = stats["entities"], stats["relations"]
    typer.echo(
        f"\nextracted {nodes} entity mention(s), {edges} relation(s) "
        f"from {stats['chunks']} chunk(s)"
    )
    dropped = (
        stats["invalid_label"] + stats["invalid_relation"]
        + stats["orphan_relation"] + stats["component_as_entity"]
        + stats["wrong_entity_type"]
    )
    if dropped:
        typer.secho(
            f"dropped {dropped}: {stats['invalid_label']} invalid label(s), "
            f"{stats['invalid_relation']} invalid relation(s), "
            f"{stats['orphan_relation']} orphan relation(s), "
            f"{stats['component_as_entity']} component(s) also listed as an entity, "
            f"{stats['wrong_entity_type']} relation(s) with the wrong entity type "
            f"on an end",
            fg=typer.colors.YELLOW,
        )


@app.command()
def ask(question: str = typer.Argument(..., help="Question to answer")) -> None:
    settings = load_settings()
    try:
        index = VectorIndex(settings)
    except RuntimeError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    from akg.pipeline import open_entailment, open_graph

    store = open_graph(settings, required=False)
    if settings.graph_enabled and store is None:
        typer.secho(
            "warning: Neo4j unavailable, answering from the vector index only",
            fg=typer.colors.YELLOW,
            err=True,
        )

    started = time.perf_counter()
    try:
        result = answer(settings, index, question, store, open_entailment(settings))
    finally:
        if store is not None:
            store.close()
    elapsed = time.perf_counter() - started

    if result.error:
        typer.secho(f"error: {result.error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo(f"\n{result.answer}\n")
    typer.echo("Citations:")
    for claim in result.claims:
        colour = (
            typer.colors.GREEN if claim.status == "supported" else typer.colors.YELLOW
        )
        typer.secho(
            f"  [{claim.source_id:>3}] sim={claim.similarity:<6} "
            f"{claim.status:<17} {claim.text[:58]}",
            fg=colour,
        )
    typer.echo(
        f"\nfaithfulness: {result.supported}/{len(result.claims)} "
        f"= {result.faithfulness:.0%}"
    )
    typer.echo(
        "retrieved: "
        + ", ".join(f"{c.cid}({c.modality})" for c in result.contexts)
    )
    typer.echo(f"elapsed: {elapsed:.1f}s")


@app.command()
def eval(
    gold: str = typer.Argument("eval/gold.json", help="Path to the gold set"),
    out: str = typer.Option("", "--out", help="Write results JSON here"),
) -> None:
    settings = load_settings()
    try:
        stats = run_gold_set(settings, gold)
    except ValueError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    except (FileNotFoundError, RuntimeError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    typer.echo("\n" + json.dumps(stats, indent=2))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        typer.echo(f"\nwrote {out}")


@app.command()
def reverify(
    run: str = typer.Argument(..., help="Results JSON from a previous eval run"),
    out: str = typer.Option("", "--out", help="Write the summary JSON here"),
) -> None:
    from akg.reverify import format_report, reverify_run
    from akg.verify.entailment import CrossEncoderChecker

    settings = load_settings()
    try:
        with open(run, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    index = VectorIndex(settings)
    typer.echo(f"collection: {settings.qdrant_collection}")
    typer.echo(f"entailment: {settings.entailment_model}\n")

    summary = reverify_run(data, index, CrossEncoderChecker(settings.entailment_model))
    typer.echo(format_report(summary))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        typer.echo(f"\nwrote {out}")


@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Results JSON for arm A"),
    candidate: str = typer.Argument(..., help="Results JSON for arm B"),
    out: str = typer.Option("", "--out", help="Write the comparison JSON here"),
) -> None:
    from akg.compare import compare_runs, format_report

    try:
        with open(baseline, encoding="utf-8") as f:
            a = json.load(f)
        with open(candidate, encoding="utf-8") as f:
            b = json.load(f)
    except FileNotFoundError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    try:
        result = compare_runs(a, b)
    except ValueError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    typer.echo("\n" + format_report(result))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        typer.echo(f"\nwrote {out}")


@app.command()
def bot() -> None:
    from akg.messaging.bot import Deps, run
    from akg.messaging.telegram import HttpTransport

    import dataclasses

    base = load_settings()
    settings = dataclasses.replace(base, qdrant_collection=base.telegram_collection)
    try:
        settings.require_telegram_token()
        transport = HttpTransport(settings)
        index = VectorIndex(settings)
    except (ValueError, RuntimeError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    if not settings.telegram_allowed_chats:
        typer.secho(
            "warning: TELEGRAM_ALLOWED_CHATS is empty, so every chat is "
            "denied. Message the bot once and copy the id printed here "
            "into .env, then restart.",
            fg=typer.colors.YELLOW,
        )

    from akg.pipeline import open_entailment, open_graph

    store = open_graph(settings, required=False)
    if settings.graph_enabled and store is None:
        typer.secho(
            "warning: Neo4j unavailable, answering from the vector index only",
            fg=typer.colors.YELLOW,
            err=True,
        )

    entail = open_entailment(settings)

    typer.echo(f"collection: {settings.qdrant_collection}")
    typer.echo(f"synthesis:  {settings.llm_synthesis}")
    typer.echo(f"entailment: {settings.entailment_model if entail else 'off'}")
    typer.echo(f"incoming:   {settings.telegram_incoming_dir}")

    def _allowed_scopes(chat_id: int) -> list[str]:
        scope = settings.telegram_chat_scopes.get(chat_id)
        return [scope] if scope else []

    deps = Deps(
        transport=transport,
        answer=lambda q, chat_id: answer(
            settings, index, q, store, entail, _allowed_scopes(chat_id)
        ),
        ingest=lambda p: ingest_path(settings, p, index=index),
        incoming_dir=settings.telegram_incoming_dir,
    )
    try:
        run(settings, deps)
    except KeyboardInterrupt:
        typer.echo("\nstopped")
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    app()
