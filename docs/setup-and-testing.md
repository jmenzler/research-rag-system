# Setup and testing

## Credential-free checks

Use a source checkout with Python 3.11, uv, and Node.js 22.12 or newer. Keep `bin/`, `scripts/`, and `prompts/` alongside `src/`; installing only the built Python wheel is not a supported deployment method.

```sh
uv sync --frozen --python 3.11 --extra dev
uv run ruff check .
uv run mypy src
uv run pytest -m 'not slow and not e2e' -q
cd frontend
npm ci
npm test -- --run
npm run build
```

Do not populate `.env` for these checks. The suite mocks external providers and data stores. It includes local concurrency tests that need permission to create loopback sockets. `slow` and `e2e` retain their existing opt-in behavior; frontend Playwright tests also require a running application and browser installation.

The backend development extra includes PDF inspection libraries used by tests. It does not install the optional GPU parser stack. First-time package installation can download large native dependencies, including PyTorch; model weights are separate.

## Running against your own documents

Review `.env.example`, then set only the variables needed for your chosen path. Keep credentials out of version control. Model defaults are recorded implementation choices, not a guarantee of current provider availability or pricing.

You need a compatible Milvus instance at `MILVUS_URI`, a configured embedding provider, and documents you may process. Ingestion and retrieval must agree on embedding dimensions and collection schema. Generation, rewriting, and evaluation can additionally require Gemini, DeepSeek, or OpenRouter credentials, depending on model selection. Do not assume setting one key configures every stage.

The default reranker calls a local llama.cpp-compatible endpoint at `http://127.0.0.1:8090`. Alternatively, `RERANK_BACKEND=local` loads the configured sentence-transformers cross-encoder and downloads its weights on first use. Other remote backends have their own endpoints, credentials, and model requirements.

Run commands from the checkout root:

```sh
uv run python bin/ragctl --help
uv run python -m src.server
```

The API listens on `127.0.0.1:8766` by default. Startup reconciles persisted jobs and starts discovery/monitoring workers; use a fresh data directory and configuration you understand. This is not a read-only document viewer.

For browser development, run `npm run dev` inside `frontend/`. Its proxy targets the loopback API by default; `frontend/.env.example` shows the override. To use FastAPI's static serving instead, build with `npm run build -- --outDir ../src/server/static`, then restart the API. Build output is excluded from Git.

For MCP over stdio:

```sh
uv run python -m src.mcp_server
```

`RAG_API_URL` selects its backend. Example retrieval request, once a corpus and services are configured:

```sh
curl --fail http://127.0.0.1:8766/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which methods compare retrieval results?","collection":"system","synthesize":false}'
```

## Optional paths

- PDF tools: the `pdf` extra declares optional parser dependencies. The managed MinerU daemon specifically looks for `.venv-mineru/bin/mineru-api`, so installing the extra into `.venv` alone does not configure that daemon. It also requires the separately supplied `gpu-broker` integration and daemon, which are not distributed here. Missing broker support fails explicitly; these managed GPU paths are not turnkey in this snapshot.
- Reranker daemon: `scripts/reranker_daemon.py` likewise requires the separate broker integration, a `llama-server` binary, and a GGUF file supplied through `RERANKER_MODEL_PATH`. Both its proxy and backend bind loopback. You can instead run a compatible independent reranking service.
- Citation graph: supply your own compatible metadata/graph stores under `CITATIONS_DATA_DIR`. Import tools are under `src/citations/`; database snapshots and papers are not included.
- Additional retrievers: optional dependency groups are declared in `pyproject.toml`. They are outside default CI coverage; inspect the adapter's availability/error behavior before relying on it.
- Fetching and deep research: some flows need separately installed browser tooling or authenticated NotebookLM tooling. They are not exercised by credential-free tests.

No default check downloads a research corpus or starts the full application stack.
