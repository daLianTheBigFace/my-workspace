# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI inference service that aggregates five independent capability packages behind a **single entry point** (`src/intent_recognition/api/app.py`). Each package is self-contained under `src/` and exposes a FastAPI `router` + `warm()` via its own `api/app.py`. Per-endpoint request/response details and usage examples live in `README.md` — don't duplicate them here.

| Package | Capability | Prefix |
|---|---|---|
| `intent_recognition` | Pluggable multi-scheme intent recognition (BERT / TFIDF / LLM stub) | `/predict` `/models` `/health` `/similarity` |
| `sentence_bert` | Sentence similarity (bge-small-zh-v1.5) | `/similarity` (also imported by other packages) |
| `rag_retrieval` | RAG multi-path recall + RRF fusion + rerank + optional LLM answer | `/rag/*` |
| `es_search` | Local Elasticsearch full-text / filter / vector search | `/es/*` |
| `pageindex_svc` | Document retrieval (VectifyAI PageIndex local engine, DeepSeek) | `/pageindex/*` |

## Commands

Environment is managed with [uv](https://docs.astral.sh/uv/) (Python 3.13; torch pinned to the PyTorch `cu130` CUDA index, see `pyproject.toml` `[tool.uv]`).

```bash
uv sync                                   # install deps (editable-installs the src/ packages)

# Run the main service (all 5 packages behind one app)
uv run uvicorn intent_recognition.api.app:app --reload --port 8000

# Run pageindex standalone (it can also be served alone)
uv run uvicorn pageindex_svc.serve:app --port 8010

# Tests
uv run pytest                              # full (loads real models / touches real ES)
uv run pytest -m "not slow"                # fast path only (no real model / no ES)
uv run pytest tests/test_predictors.py -m "not slow"          # single file
uv run pytest tests/test_predictors.py::test_something -m "not slow"   # single test

# Build indexes (one-time; outputs git-ignored under data/)
uv run python -m rag_retrieval.build_index              # chunks.json + pages.json + embeddings.npy
PYTHONPATH=src uv run python -m es_search.build_index   # reuses data/rag_index/ (needs ES running)

# Train (optional; artifacts already present under models/)
uv run python -m intent_recognition.train.bert           # needs GPU
uv run python -m intent_recognition.train.tfidf
```

External dependency: local Elasticsearch 8.17.8 at `http://localhost:9200` (no auth), started via `E:\elasticsearch-8.17.8\bin\elasticsearch.bat`. `/es/*` and `test_es_search.py` need it; everything else runs without it.

## Architecture

### Capability-package skeleton (the core pattern)

Every package has the same shape. `api/__init__.py` re-exports `router` and `warm`, and that is all the main app imports from a package:

```
src/<name>/
├── config.py          # dataclasses for paths + hyperparams (optional if no config needed)
├── <business>.py      # pure logic, no FastAPI import (e.g. search.py / service.py)
└── api/
    ├── __init__.py    # from .app import router, warm
    └── app.py         # router = APIRouter(prefix=...) + endpoints + def warm()
```

Adding a new capability package is a documented, repeatable workflow — invoke the `add-api-package` skill, which walks through the exact steps (skeleton → `api/app.py` → `api/__init__.py` → wire into main `app.py` → register in `pyproject.toml` `packages` → deps → tests).

### Single entry point + lifespan pre-warm

`src/intent_recognition/api/app.py` is the **only** process entry. It `include_router`s each package's router and, in a `lifespan` context manager, calls each package's `warm()`. `warm()` must **never raise** — ES down / no LLM key / missing index are normal states, so it only `logger.warning`s (or the main app wraps it in `try/except`). Otherwise one missing dependency takes down the whole service.

### Pluggable predictor registry (intent_recognition)

Inference schemes implement `BasePredictor.predict(text, top_k) -> list[Prediction]` and self-register via the `@register_predictor` decorator. `get_predictor(name)` returns a lazy singleton (constructed once, cached). `predictors/__init__.py` imports `bert` and `tfidf` for their registration side effects; `llm.py` is a reserved template and deliberately *not* imported. Multi-intent output uses `split_intents(ranked, threshold)` (all intents ≥ threshold, independent confidences) vs the legacy `split_main_sub` (primary/sub, kept for compatibility).

### Config pattern

Each package's `config.py` is self-contained (copyable to another project) and follows the same conventions:

- `PROJECT_ROOT = Path(__file__).resolve().parents[2]` — resolves the repo root from anywhere, so paths work regardless of CWD.
- `os.environ.setdefault("HF_HOME", PROJECT_ROOT / "models" / ".hf_cache")` — pinned to E: before any transformers/huggingface import (default would write to C:).
- `load_dotenv(PROJECT_ROOT / ".env")` — reads `DEEPSEEK_API_KEY` / `RAG_LLM_*` / `PAGEINDEX_*` from the repo root; existing env vars take precedence.
- `device` defaults to `"cuda"` (sentence_bert, rag_retrieval) — set to `"cpu"` on a GPU-less machine.

Heavy resources use a **module-level lazy singleton** (not loaded at import): `sentence_bert.get_model()`, `es_search.get_client()`, `pageindex_svc.get_client()`.

## Conventions & gotchas

- **FastAPI 0.141 / Starlette 1.6 removed `add_event_handler` / `on_event`.** Startup pre-warm must go in a `lifespan` context manager (see `intent_recognition/api/app.py`). Never use the removed APIs.
- **All endpoints declare a Pydantic `response_model`** (keeps Swagger consistent); request `query`/`text` fields use `field_validator` to strip whitespace.
- **Dependency-not-ready → `HTTPException` (503/502) with a Chinese hint**, not a bare exception. See `es_search`'s `_require_es()` and `pageindex_svc`'s `_require_llm()`.
- **`elasticsearch-py` client must be `< 9`** (`elasticsearch>=8.17,<9`): 9.x clients send `compatible-with=9` and are rejected by the 8.17 server. IK plugin version must match ES exactly.
- **Top-level package names must not collide with pip/git deps.** `pageindex` collided with the PyPI package, so the wrapper is `pageindex_svc` (the git dep `pageindex` is installed via `[tool.uv.sources]`).
- **Large files / caches stay off C:** uv cache is set to `E:\uv-cache`, models/indexes/data live under `assets/models`, `models/`, `data/` (all git-ignored). `data/rag_index/` is shared between `rag_retrieval` and `es_search`.
- **RAG behavior on no-match:** if the reranked top1 score < `min_rerank_score` (0.3), the search returns `matched: false` and does *not* call the LLM (out-of-manual questions aren't hard-fabricated).
- **MCP server** (`mcp_servers/intent_server.py`) lets Claude call `predict_intent` / `list_models` directly without starting uvicorn; registered via `.mcp.json` (mcp 2.x `MCPServer`, stdio). A `PostToolUse` hook (`mcp_servers/log_prediction.py`) appends every `predict_intent` call to `logs/intent_predictions.csv` as an eval set.
