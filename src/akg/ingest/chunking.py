import re

_HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def _normalise_block(block: str) -> str:
    lines = (" ".join(line.split()) for line in block.splitlines())
    return "\n".join(line for line in lines if line)


def _blocks(text: str) -> list[tuple[str, bool]]:
    marked = _HEADING.sub(lambda m: "\n\n" + m.group(0), text)
    out: list[tuple[str, bool]] = []
    for raw in re.split(r"\n\s*\n", marked):
        block = raw.strip()
        if block:
            out.append((_normalise_block(block), bool(_HEADING.match(block))))
    return out


def _pack_sentences(text: str, max_chars: int) -> list[str]:
    packed, buf = [], ""
    for sentence in split_sentences(text):
        if buf and len(buf) + len(sentence) + 1 > max_chars:
            packed.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        packed.append(buf)
    return packed


def split_structured(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    buf = ""
    under_heading = False

    def flush() -> None:
        nonlocal buf
        if buf:
            chunks.extend(_pack_sentences(buf, max_chars) if len(buf) > max_chars
                          else [buf])
            buf = ""

    for block, is_heading in _blocks(text):
        if is_heading:
            flush()
            buf, under_heading = block, True
        elif under_heading and buf and len(buf) + len(block) + 1 <= max_chars:
            buf = f"{buf} {block}"
        else:
            flush()
            buf, under_heading = block, False
    flush()
    return chunks
