
import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    ollama_url: str = "http://localhost:11434"
    llm_synthesis: str = "qwen3.5:9b"
    llm_extraction: str = "gemma4"
    llm_judge: str = "gemma4"

    asr_model: str = "base"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    embed_model: str = "BAAI/bge-small-en-v1.5"

    top_k: int = 5
    chunk_chars: int = 400
    citation_sim_threshold: float = 0.55
    entailment_enabled: bool = False
    entailment_model: str = "cross-encoder/nli-deberta-v3-base"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    graph_enabled: bool = True
    graph_max_depth: int = 3
    graph_max_edges: int = 12

    telegram_bot_token: str = ""
    telegram_allowed_chats: tuple[int, ...] = ()
    telegram_chat_scopes: dict[int, str] = field(default_factory=dict)
    telegram_incoming_dir: str = "var/telegram"
    telegram_collection: str = "chunks_live"

    def require_telegram_token(self) -> None:
        if not self.telegram_bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather "
                "and put the token in .env."
            )

    def require_independent_judge(self) -> None:
        if self.llm_judge == self.llm_synthesis:
            raise ValueError(
                f"LLM_JUDGE and LLM_SYNTHESIS must differ, both are "
                f"'{self.llm_synthesis}'. A model cannot independently judge "
                f"its own output; set LLM_JUDGE to a different model."
            )


def _chat_ids(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    entries = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        return tuple(int(entry) for entry in entries)
    except ValueError as e:
        raise ValueError(
            f"{name} must be a comma-separated list of numeric chat ids, "
            f"got {raw!r}"
        ) from e


def _chat_scopes(name: str, default: dict[int, str]) -> dict[int, str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    scopes: dict[int, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        chat_id, _, scope = part.partition(":")
        if not scope:
            raise ValueError(
                f"{name} must be a comma-separated list of chat_id:scope "
                f"pairs, got {raw!r}"
            )
        try:
            scopes[int(chat_id)] = scope
        except ValueError as e:
            raise ValueError(
                f"{name} must be a comma-separated list of chat_id:scope "
                f"pairs with a numeric chat id, got {raw!r}"
            ) from e
    return scopes


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(use_dotenv: bool = True) -> Settings:
    if use_dotenv:
        from dotenv import load_dotenv

        load_dotenv()
    d = Settings()
    return Settings(
        qdrant_url=os.getenv("QDRANT_URL", d.qdrant_url),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", d.qdrant_collection),
        ollama_url=os.getenv("OLLAMA_URL", d.ollama_url),
        llm_synthesis=os.getenv("LLM_SYNTHESIS", d.llm_synthesis),
        llm_extraction=os.getenv("LLM_EXTRACTION", d.llm_extraction),
        llm_judge=os.getenv("LLM_JUDGE", d.llm_judge),
        asr_model=os.getenv("ASR_MODEL", d.asr_model),
        asr_device=os.getenv("ASR_DEVICE", d.asr_device),
        asr_compute_type=os.getenv("ASR_COMPUTE_TYPE", d.asr_compute_type),
        embed_model=os.getenv("EMBED_MODEL", d.embed_model),
        top_k=int(os.getenv("TOP_K", d.top_k)),
        chunk_chars=int(os.getenv("CHUNK_CHARS", d.chunk_chars)),
        citation_sim_threshold=float(
            os.getenv("CITATION_SIM_THRESHOLD", d.citation_sim_threshold)
        ),
        entailment_enabled=_flag("ENTAILMENT_ENABLED", d.entailment_enabled),
        entailment_model=os.getenv("ENTAILMENT_MODEL", d.entailment_model),
        neo4j_uri=os.getenv("NEO4J_URI", d.neo4j_uri),
        neo4j_user=os.getenv("NEO4J_USER", d.neo4j_user),
        neo4j_password=os.getenv("NEO4J_PASSWORD", d.neo4j_password),
        graph_enabled=_flag("GRAPH_ENABLED", d.graph_enabled),
        graph_max_depth=int(os.getenv("GRAPH_MAX_DEPTH", d.graph_max_depth)),
        graph_max_edges=int(os.getenv("GRAPH_MAX_EDGES", d.graph_max_edges)),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", d.telegram_bot_token),
        telegram_allowed_chats=_chat_ids(
            "TELEGRAM_ALLOWED_CHATS", d.telegram_allowed_chats
        ),
        telegram_chat_scopes=_chat_scopes(
            "TELEGRAM_CHAT_SCOPES", d.telegram_chat_scopes
        ),
        telegram_incoming_dir=os.getenv(
            "TELEGRAM_INCOMING_DIR", d.telegram_incoming_dir
        ),
        telegram_collection=os.getenv(
            "TELEGRAM_COLLECTION", d.telegram_collection
        ),
    )
