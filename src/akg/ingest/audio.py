import os

from akg.config import Settings
from akg.models import Chunk

# (start_seconds, end_seconds, text)
Segment = tuple[float, float, str]


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "??:??"
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def segments_to_chunks(
    segments: list[Segment],
    source_file: str,
    doc_hash: str,
    max_chars: int,
) -> list[Chunk]:
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    length = 0

    for start, end, raw in segments:
        text = raw.strip()
        if not text:
            continue
        if current and length + len(text) + 1 > max_chars:
            groups.append(current)
            current, length = [], 0
        current.append((start, end, text))
        length += len(text) + 1

    if current:
        groups.append(current)

    return [
        Chunk(
            cid=f"A{i + 1}",
            text=" ".join(t for _s, _e, t in group),
            source_file=source_file,
            modality="audio",
            locator=f"{format_timestamp(group[0][0])}-{format_timestamp(group[-1][1])}",
            doc_hash=doc_hash,
        )
        for i, group in enumerate(groups)
    ]


def transcribe(path: str, settings: Settings, doc_hash: str) -> list[Chunk]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        settings.asr_model,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
    )
    raw, _info = model.transcribe(path, vad_filter=True)
    segments: list[Segment] = [(s.start, s.end, s.text) for s in raw]
    return segments_to_chunks(
        segments, os.path.basename(path), doc_hash, settings.chunk_chars
    )
