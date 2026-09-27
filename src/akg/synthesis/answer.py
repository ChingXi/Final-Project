from akg.config import Settings
from akg.models import Chunk, Claim
from akg.synthesis.llm import chat

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_id": {"type": "string"},
                },
                "required": ["text", "source_id"],
            },
        },
    },
    "required": ["answer", "claims"],
}

SYSTEM = """
You answer questions strictly from the provided sources. Do not use outside knowledge.
Sources often have to be combined. If the answer follows from linking facts across two or more sources, give that answer: combining the sources you were given is not outside knowledge. Say the sources do not contain the answer only when a fact you need is absent from all of them, and in that case return an empty claims list.
Answer the question that was asked, in full. If more than one thing satisfies the question, name every one of them.
Every claim must be a single fact that its one cited source id supports on its own. The answer field may state a conclusion that follows from several claims together; the claims list holds the individual facts it rests on.
"""


def build_prompt(question: str, contexts: list[Chunk]) -> str:
    sources = "\n".join(
        f"[{c.cid}] ({c.modality}, {c.source_file} {c.locator}): {c.text}"
        for c in contexts
    )
    ids = ", ".join(c.cid for c in contexts)
    return (
        f"QUESTION:\n{question}\n\n"
        f"SOURCES (cite only these ids: {ids}):\n{sources}"
    )


def synthesize(
    settings: Settings, question: str, contexts: list[Chunk]
) -> tuple[str, list[Claim], bool]:
    parsed, recovered = chat(
        settings,
        settings.llm_synthesis,
        SYSTEM,
        build_prompt(question, contexts),
        schema=ANSWER_SCHEMA,
    )
    claims = [
        Claim(
            text=(c.get("text") or "").strip(),
            source_id=(c.get("source_id") or "").strip(),
        )
        for c in parsed.get("claims") or []
        if (c.get("text") or "").strip()
    ]
    return (parsed.get("answer") or "").strip(), claims, recovered
