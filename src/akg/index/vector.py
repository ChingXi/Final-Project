
import hashlib
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from akg.config import Settings
from akg.models import Chunk

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _point_id(doc_hash: str, cid: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_hash}:{cid}"))


def _to_chunk(payload: dict) -> Chunk:
    return Chunk(
        cid=payload["cid"],
        text=payload["text"],
        source_file=payload["source_file"],
        modality=payload["modality"],
        locator=payload["locator"],
        doc_hash=payload["doc_hash"],
        scope=payload.get("scope") or "shared",
    )


class VectorIndex:
    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer

        self.settings = settings
        self.collection = settings.qdrant_collection
        try:
            self.client = QdrantClient(url=settings.qdrant_url)
            self.client.get_collections()
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Qdrant at {settings.qdrant_url}: {e}. "
                f"Start it with: docker compose up -d"
            ) from e
        self.embedder = SentenceTransformer(settings.embed_model)
        get_dim = getattr(
            self.embedder,
            "get_embedding_dimension",
            self.embedder.get_sentence_embedding_dimension,
        )
        self.dim = get_dim()
        self.ensure_collection(self.dim)

    def ensure_collection(self, dim: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.ensure_collection(self.dim)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def iter_payloads(self):

        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                yield point.payload
            if offset is None:
                return

    def embed(self, text: str) -> np.ndarray:
        v = self.embedder.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed(BGE_QUERY_PREFIX + text)

    def already_ingested(self, doc_hash: str) -> bool:
        found, _ = self.client.scroll(
            self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_hash", match=MatchValue(value=doc_hash))]
            ),
            limit=1,
        )
        return bool(found)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.encode(
            [c.text for c in chunks], normalize_embeddings=True
        )
        self.client.upsert(
            self.collection,
            points=[
                PointStruct(
                    id=_point_id(c.doc_hash, c.cid),
                    vector=np.asarray(v, dtype=np.float32).tolist(),
                    payload={
                        "cid": c.cid,
                        "text": c.text,
                        "source_file": c.source_file,
                        "modality": c.modality,
                        "locator": c.locator,
                        "doc_hash": c.doc_hash,
                        "scope": c.scope,
                    },
                )
                for c, v in zip(chunks, vectors)
            ],
        )
        return len(chunks)

    def search(
        self, query: str, k: int, allowed_scopes: list[str] | None = None
    ) -> list[Chunk]:
        query_filter = (
            Filter(must=[FieldCondition(
                key="scope", match=MatchAny(any=[*allowed_scopes, "shared"])
            )])
            if allowed_scopes is not None
            else None
        )
        resp = self.client.query_points(
            self.collection,
            query=self.embed_query(query).tolist(),
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            with_vectors=True,
        )
        out = []
        for p in resp.points:
            chunk = _to_chunk(p.payload)
            chunk.vector = np.asarray(p.vector, dtype=np.float32)
            out.append(chunk)
        return out

    def get_with_vector(self, cid: str) -> tuple[Chunk, np.ndarray] | None:
        found, _ = self.client.scroll(
            self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="cid", match=MatchValue(value=cid))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
        if not found:
            return None
        point = found[0]
        return _to_chunk(point.payload), np.asarray(point.vector, dtype=np.float32)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def get_chunk(self, doc_hash: str, cid: str) -> tuple[Chunk, np.ndarray] | None:
        found = self.client.retrieve(
            self.collection,
            ids=[_point_id(doc_hash, cid)],
            with_payload=True,
            with_vectors=True,
        )
        if not found:
            return None
        point = found[0]
        return _to_chunk(point.payload), np.asarray(point.vector, dtype=np.float32)
