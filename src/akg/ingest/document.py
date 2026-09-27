import os

from akg.config import Settings
from akg.ingest.chunking import split_structured
from akg.models import Chunk


def markdown_to_chunks(
    markdown: str, source_file: str, doc_hash: str, max_chars: int
) -> list[Chunk]:
    return [
        Chunk(
            cid=f"D{i + 1}",
            text=piece,
            source_file=source_file,
            modality="document",
            locator=f"section {i + 1}",
            doc_hash=doc_hash,
        )
        for i, piece in enumerate(split_structured(markdown, max_chars))
    ]


def parse(path: str, settings: Settings, doc_hash: str) -> list[Chunk]:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(path)
    markdown = result.document.export_to_markdown()
    return markdown_to_chunks(
        markdown, os.path.basename(path), doc_hash, settings.chunk_chars
    )
