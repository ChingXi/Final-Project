from akg.index.vector import VectorIndex
from akg.models import Chunk


def retrieve(
    index: VectorIndex,
    question: str,
    k: int | None = None,
    allowed_scopes: list[str] | None = None,
) -> list[Chunk]:
    return index.search(
        question, k if k is not None else index.settings.top_k, allowed_scopes
    )
