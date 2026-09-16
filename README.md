# InterroAI

[![CI](https://github.com/Alex110506/InterroAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Alex110506/InterroAI/actions/workflows/ci.yml)

InterroAI is a desktop AI coding agent. Open a project folder in the app and it indexes the codebase, then answers questions about it or runs a Plan → Code → Verify cycle to change it.

Every request is classified before anything runs: a question gets a read-only, grounded answer; anything that would change the codebase goes through the full Plan → Code → Verify cycle, informed by your actual file tree, git state and a semantic index of your codebase.

## Architecture

The agent's tools always run on your machine, so edits land on your disk directly. Model access and the semantic index sit behind two ports, and the local runtime runs in one of two modes:

| | `local` | `cloud` |
| --- | --- | --- |
| **Models** (`ModelGateway`) | OpenAI directly, with your key from the OS keychain | The Cloud API's LLM gateway: platform key, model allowlist, rate limits and daily quotas |
| **Semantic index** (`SemanticIndex`) | ChromaDB on disk, with an in-process worker | Blob upload → Service Bus → Embed Worker → pgvector |
| **Sign-in** | None | GitHub, in the system browser, with PKCE |

`core/providers.py` is the one place that picks implementations, from `INTERROAI_MODE`. The agents never import OpenAI, the vector store or the cloud clients, and `tests/test_boundaries.py` fails the build if they start to.

![InterroAI on Azure: the Electron app and a local Python runtime on the user's machine, where the agent loop and its file tools run; in Azure, a Web API container app for GitHub PKCE sign-in, projects and sync, the LLM gateway and search, feeding index jobs through Service Bus to an Embed Worker that scales from zero, with PostgreSQL and pgvector holding vectors and line ranges but no code, and Blob Storage holding uploads](docs/azure-architecture.svg)

The reasoning behind the diagram's main choices is recorded in [`docs/adr/`](docs/adr/README.md).

- **Electron app** (`frontend/`) — React UI. Its main process starts the local runtime on a free `127.0.0.1` port with a random launch token, and runs the browser half of GitHub sign-in.
- **Local runtime** (`backend/`: `api/`, `agents/`, `core/`) — FastAPI over the agent pipeline. `core/` is grouped by where each part runs: `workspace/` (your project on disk), `index/` (the indexing service), `models/` (OpenAI access), `local/` (cache and keychain) and `remote/` (the Cloud API clients). Payloads that cross service boundaries live in `contracts/`.
- **Cloud API** (`backend/cloud/api/`) — FastAPI for Azure Container Apps: GitHub sign-in, projects, sync, upload URLs, index jobs with live progress over server-sent events, search, and the LLM gateway.
- **Embed Worker** (`backend/cloud/worker/`) — takes index jobs off Service Bus, embeds the chunks, and changes the index in one transaction.
- **Data** — PostgreSQL with pgvector and row-level security per user; Blob Storage for uploads; Service Bus for jobs.

## Install

Python 3.11+ (3.12 in the cloud images) and Node 18+. **OpenAI-only** by design.

```bash
cd backend
uv pip sync requirements.lock      # the desktop runtime, fully pinned and hash-verified
pip install -e ".[dev]"            # plus the cloud packages, ruff and pytest

cd ../frontend
npm install
```

The backend's dependencies come in three sets, so the cloud images stay small: the base every process needs, `runtime` (ChromaDB, the chunker and Redis, for the desktop only) and `cloud` (SQLAlchemy, asyncpg, Alembic and the Azure SDKs). `requirements.lock` pins the base and `runtime`; `requirements-cloud.lock` pins the base and `cloud` for the images.

## Run in local mode

```bash
cd frontend
npm run dev      # Vite on :5173 and the Electron window, which starts the runtime itself
```

Open **Settings** to add your OpenAI API key (stored in the OS keychain), then **New project** to pick a folder. Pick the model for each request from the picker under the chat input.

The app starts the runtime with `python3` from your `PATH`. If the backend's packages live in a virtualenv, point `INTERROAI_PYTHON` at its interpreter:

```bash
INTERROAI_PYTHON="$PWD/../backend/.venv/bin/python" npm run dev
```

### Conversations are fresh, not amnesiac

Each project's conversation lives in the app's memory, so follow-ups work — ask "what is this project about", then "sure", and the agent knows what it offered. Every request re-sends the conversation, and older turns are dropped once it exceeds a character budget, so a long session doesn't quietly inflate the cost of every request.

Nothing is stored: there is no transcript on disk and nothing to resume after closing the app. The only state that outlives it is the index and your settings.

## Run in cloud mode, on the local stack

Docker stands in for the Azure services: Postgres with pgvector, Azurite for Blob Storage, and the Service Bus emulator. The emulator needs SQL Server, which has no arm64 image, so on Apple Silicon turn on Docker Desktop's Rosetta setting.

1. Copy `.env.example` to `.env` at the repo root and fill it in; its comments explain each value. You need a GitHub OAuth App with the callback URL `http://localhost:8080/auth/github/callback`, your GitHub login in `INTERROAI_ALLOWED_GITHUB_LOGINS`, and `INTERROAI_MODE=cloud`.
2. Start the stack and create the schema:
   ```bash
   docker compose up -d
   cd backend
   alembic -c cloud/alembic.ini upgrade head
   ```
3. From `backend/`, start the Cloud API and the Embed Worker, each in its own terminal:
   ```bash
   uvicorn cloud.api.main:create_app --factory --port 8080
   python -m cloud.worker
   ```
4. Run `npm run dev` in `frontend/` and choose **Sign in with GitHub**.

### Container images

```bash
docker build --target api    -t interroai-api    backend
docker build --target worker -t interroai-worker backend
```

Both install `requirements-cloud.lock` and nothing else, run as a non-root user, and log one JSON object per line. Migrations run from the API image as a one-off: `alembic -c cloud/alembic.ini upgrade head`.

## Tests

```bash
cd backend
pytest                      # unit tests, offline
pytest -m integration       # against the docker compose stack

cd ../frontend
npm test                    # the Electron main process: sign-in flow and runtime launcher
```

CI runs all of them, builds both images and checks that the API image starts. Its integration job brings up Postgres and Azurite. The Service Bus tests join once the repository variable `ACCEPT_SERVICEBUS_EMULATOR_EULA` is set to `Y`: starting the emulator means accepting Microsoft's licence terms for it and for SQL Server, which is the repository owner's decision.

## Features

1. **Two-phase project indexing, incremental by default**
   - *Phase 1:* file tree + git context, respecting `.gitignore`.
   - *Phase 2:* the runtime hashes every file and syncs with the index; only changed files are chunked (AST- and language-aware) and handed to the indexing worker as a job, which embeds them with `text-embedding-3-small`, stores them and prunes whatever they replaced. An untouched project queues no job at all.
   - Deleted, renamed, emptied and shrunken files lose their stale vectors. A job changes the index in one step, and embeddings already paid for are cached, so a failure part-way through costs nothing on the retry. A chunk the provider refuses is skipped and *reported*, never hidden.
2. **No source code in the index** — the index keeps vectors, paths, line ranges and file hashes. Search results are read back from your disk, and a hit from a file edited since indexing is labelled stale so the agent re-reads it before trusting a line number.
3. **Intent classification** — each request is routed to `answer` or `implement`, grounded in your file tree and git state.
4. **Coding Agent: Plan → Code → Verify**
   - *Plan:* a Markdown plan of attack, streamed as it's generated.
   - *Code:* a sandboxed tool loop (`read_file`, `write_file`, `patch_file`, `search_grep`, `search_semantic`). Every path is resolved against the project root and anything escaping it is rejected.
   - *Verify:* `ruff` and `pytest` run against your project, with up to 3 autonomous self-correction rounds. A check that could not run reports as **skipped**, never as passed.
5. **Secure credential storage** — your API key lives in the OS keychain via `keyring`, never in a config file, and is never echoed back to the UI. In cloud mode the runtime keeps the session's refresh token the same way, and the Electron app never sees a token.
6. **Cloud protections** — row-level security keeps each user's projects, chunks and jobs apart in Postgres; refresh tokens rotate, and a replayed one ends every session of that user; the runtime answers only the app that started it; request size limits, per-minute rate limits and exact daily quotas guard the platform key.
7. **Optional Redis cache** (local mode) — caches the AST repo map and chunk embeddings, keyed by content so a renamed file costs nothing to re-index. Strictly an accelerator: without a `redis-server` the app logs one line and carries on.

## Roadmap

- **Deploy to Azure with Terraform:** Container Apps for the API and the worker (the worker scaled by queue length, down to zero), Postgres Flexible Server, Service Bus Basic, Blob Storage with a lifecycle rule for abandoned uploads, and secrets as Key Vault references.
- **Security Guardrail Agent:** scan generated code for vulnerabilities before it is written to disk.

## State outside the repo

| Path | Holds |
| --- | --- |
| `~/.interroai/config.json` | Username and other non-sensitive config |
| `~/.interroai/chroma/` | The local-mode index, one collection per project — vectors, paths, line ranges and file hashes, no source text |
| `~/.interroai/cloud_projects.json` | Cloud mode: which server project each local folder is. The server is never told the path |
| OS keychain (`interroai` / `openai_api_key`) | Your API key, for local mode |
| OS keychain (`interroai` / `cloud_refresh_token:<api url>`) | Your cloud session's refresh token |
| Redis, if running (`INTERROAI_REDIS_URL`) | Repo-map and embedding caches. Derived data — safe to flush |
