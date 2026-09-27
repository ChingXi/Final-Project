# Autonomous Enterprise Knowledge Gateway

Multimodal retrieval over voice notes and scanned documents, answering natural
language questions with citations that are **verified programmatically** rather
than trusted from the language model.

Enterprise knowledge is fragmented across formats conventional search cannot
read: recorded meetings, voice notes, photographed reports. This system ingests
that material, indexes it, and answers questions with every claim traced back to
an identifiable source passage. A decision made on an unattributable answer is
close to a guess, so verifiable citation is treated as a functional requirement
rather than a nicety.

## Architecture

Three pre-trained models operate across three data types:

| Modality | Model | Role |
| --- | --- | --- |
| Audio | Whisper (`faster-whisper`) | Transcribe voice notes, timestamped |
| Image | Docling | Recover layout, tables, reading order from scans |
| Text | Local LLM via Ollama | Synthesize the cited answer |


## Requirements

- Python 3.12
- Docker
- [Ollama](https://ollama.com) running on the host

Ollama is deliberately **not** containerised: keeping it native preserves direct
GPU access and avoids a second copy of downloaded models.

## Setup

```bash
cp .env.example .env          # then set NEO4J_PASSWORD
docker compose up -d          # Qdrant

python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"

ollama pull qwen3.5:9b        # synthesis model
ollama pull gemma4            # judge (must differ from synthesis)
```


```bash
docker compose --profile graph up -d
```

## Usage

```bash
akg ingest data/                                  # ingest the corpus
akg ask "When does Project Atlas launch?"         # answer with citations
akg eval eval/gold.json                           # run the gold set
```

Ingestion is idempotent: files are content-hashed, so re-running skips anything
unchanged rather than re-transcribing it.

## Models

Three roles, three settings, because they have different requirements:

- `LLM_SYNTHESIS` is on the interactive path and needs speed and reliable
  structured output.
- `LLM_EXTRACTION` runs only at ingest time, so it can be large and slow.
- `LLM_JUDGE` runs only under `akg eval` and **must differ** from
  `LLM_SYNTHESIS` — a model cannot independently grade its own output.
  Evaluation refuses to run if they match.

## Tests

```bash
pytest -m "not integration and not llm"   # fast, no services needed
pytest -m integration                     # requires Qdrant
pytest -m llm                             # requires Ollama
```