import os
import time
from dataclasses import dataclass, field
from typing import Callable

from akg.config import Settings
from akg.messaging.telegram import Transport

KIND_EXTS = {"voice": ".ogg", "video_note": ".mp4", "photo": ".jpg"}

HELP = (
    "Send me a voice note or a photo of a document and I will add it to the "
    "knowledge base.\n"
    "Send me a question as text and I will answer it with citations.\n\n"
    "/start or /help - this message\n"
    "/whoami - show this chat's id"
)


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class Deps:

    transport: Transport
    answer: Callable[[str, int], object]
    ingest: Callable[[str], dict]
    incoming_dir: str = "var/telegram"
    seen_updates: set = field(default_factory=set)


def authorised(chat_id: int, settings: Settings) -> bool:
    return chat_id in set(settings.telegram_allowed_chats)


def describe(message: dict) -> tuple[str, str, str] | None:
    for kind in ("voice", "audio", "video_note", "document"):
        payload = message.get(kind)
        if payload:
            name = payload.get("file_name") or ""
            ext = os.path.splitext(name)[1].lower() or KIND_EXTS.get(kind, "")
            return kind, payload["file_id"], f"{payload['file_unique_id']}{ext}"

    photos = message.get("photo")
    if photos:
        largest = max(photos, key=lambda p: p.get("file_size") or 0)
        return "photo", largest["file_id"], f"{largest['file_unique_id']}.jpg"
    return None


def format_answer(result) -> str:
    if getattr(result, "error", None):
        return f"Something went wrong: {result.error}"

    lines = [result.answer or "(no answer)"]
    if result.claims:
        lines.append("\nCitations:")
        for claim in result.claims:
            mark = "OK " if claim.status == "supported" else "!! "
            detail = f"sim {claim.similarity}"
            if claim.entailment and claim.status == "DISPUTED":
                detail += f", {claim.entailment} {claim.entail_p}"
            lines.append(
                f"  {mark}[{claim.source_id}] {claim.status} "
                f"({detail})\n      {claim.text}"
            )
        lines.append(
            f"\nFaithfulness: {result.supported}/{len(result.claims)} "
            f"= {result.faithfulness:.0%}"
        )
        if result.disputed:
            lines.append(
                f"{result.disputed} claim(s) contradicted by the passage cited. "
                "Treat this answer with caution."
            )
    else:
        lines.append("\n(No claims were made, so nothing was cited.)")

    if result.contexts:
        seen = sorted({c.source_file for c in result.contexts})
        lines.append("Sources consulted: " + ", ".join(seen))
    return "\n".join(lines)


def handle(update: dict, settings: Settings, deps: Deps) -> None:
    update_id = update.get("update_id")
    message = update.get("message") or update.get("edited_message")
    if not message:
        return

    chat_id = (message.get("chat") or {}).get("id")
    if chat_id is None:
        return

    if update_id in deps.seen_updates:
        return
    deps.seen_updates.add(update_id)

    if not authorised(chat_id, settings):
        log(f"  denied  chat_id={chat_id} (add it to TELEGRAM_ALLOWED_CHATS)")
        return

    text = (message.get("text") or "").strip()
    if text in ("/start", "/help"):
        deps.transport.send_message(chat_id, HELP)
        return
    if text == "/whoami":
        deps.transport.send_message(chat_id, f"This chat's id is {chat_id}")
        return

    attachment = describe(message)
    try:
        if attachment:
            _ingest(attachment, chat_id, deps)
        elif text:
            _answer(text, chat_id, deps)
        else:
            deps.transport.send_message(
                chat_id, "I can read text, voice notes and document photos."
            )
    except Exception as e:  # noqa: BLE001 - the loop must outlive any one message
        log(f"  error   chat_id={chat_id}: {e}")
        deps.transport.send_message(chat_id, f"Sorry, that failed: {e}")


def _answer(question: str, chat_id: int, deps: Deps) -> None:
    deps.transport.send_action(chat_id)
    started = time.perf_counter()
    result = deps.answer(question, chat_id)
    elapsed = time.perf_counter() - started

    claims = len(getattr(result, "claims", []) or [])
    faith = getattr(result, "faithfulness", 0.0)
    log(
        f"  query   chat={chat_id} {elapsed:.1f}s claims={claims} "
        f"faith={faith:.0%} q={question[:48]!r}"
    )
    deps.transport.send_message(chat_id, format_answer(result))


def _ingest(attachment: tuple[str, str, str], chat_id: int, deps: Deps) -> None:
    kind, file_id, filename = attachment
    deps.transport.send_action(chat_id, "upload_document")
    dest = os.path.join(deps.incoming_dir, filename)
    started = time.perf_counter()
    deps.transport.download(file_id, dest)

    stats = deps.ingest(dest)
    elapsed = time.perf_counter() - started
    log(
        f"  ingest  chat={chat_id} {kind} {elapsed:.1f}s "
        f"chunks={stats.get('chunks', 0)} file={filename}"
    )

    if stats.get("ingested"):
        deps.transport.send_message(
            chat_id,
            f"Added your {kind} to the knowledge base: "
            f"{stats['chunks']} passage(s) indexed. Ask me about it.",
        )
    elif stats.get("skipped"):
        deps.transport.send_message(
            chat_id, f"I already have that {kind}. Nothing changed."
        )
    else:
        deps.transport.send_message(
            chat_id, f"I could not read anything usable from that {kind}."
        )


def batches(updates: list[dict]) -> list[list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for update in updates:
        message = update.get("message") or update.get("edited_message") or {}
        group_id = message.get("media_group_id")
        key = ("group", group_id) if group_id else ("single", id(update))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(update)
    return [grouped[key] for key in order]


def handle_batch(updates: list[dict], settings: Settings, deps: Deps) -> None:
    if len(updates) == 1:
        handle(updates[0], settings, deps)
        return

    first = updates[0].get("message") or {}
    chat_id = (first.get("chat") or {}).get("id")
    if chat_id is None or not authorised(chat_id, settings):
        if chat_id is not None:
            log(f"  denied  chat_id={chat_id} (add it to TELEGRAM_ALLOWED_CHATS)")
        return

    deps.transport.send_action(chat_id, "upload_document")
    indexed = chunks = failed = 0
    for update in updates:
        update_id = update.get("update_id")
        if update_id in deps.seen_updates:
            continue
        deps.seen_updates.add(update_id)

        attachment = describe(update.get("message") or {})
        if not attachment:
            continue
        kind, file_id, filename = attachment
        dest = os.path.join(deps.incoming_dir, filename)
        try:
            started = time.perf_counter()
            deps.transport.download(file_id, dest)
            stats = deps.ingest(dest)
            log(
                f"  ingest  chat={chat_id} {kind} "
                f"{time.perf_counter() - started:.1f}s "
                f"chunks={stats.get('chunks', 0)} file={filename} (group)"
            )
            indexed += bool(stats.get("ingested"))
            chunks += stats.get("chunks", 0)
        except Exception as e:  # noqa: BLE001
            # One unreadable page must not lose the pages that did work.
            log(f"  error   chat={chat_id} {filename}: {e}")
            failed += 1

    parts = [f"Added {indexed} file(s): {chunks} passage(s) indexed."]
    if failed:
        parts.append(f"{failed} could not be read.")
    deps.transport.send_message(chat_id, " ".join(parts))


def run(settings: Settings, deps: Deps, poll_timeout: int = 30) -> None:
    transport = deps.transport
    offset = 0
    backoff = 1
    log(f"listening (allowlist: {list(settings.telegram_allowed_chats) or 'EMPTY'})")
    while True:
        try:
            updates = transport.get_updates(offset, poll_timeout)
            backoff = 1
        except Exception as e:  # noqa: BLE001
            log(f"  poll failed ({e}); retrying in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue

        for update in updates:
            offset = max(offset, update.get("update_id", 0) + 1)
        for batch in batches(updates):
            handle_batch(batch, settings, deps)
