# InterroAI

[![CI](https://github.com/Alex110506/InterroAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Alex110506/InterroAI/actions/workflows/ci.yml)

InterroAI is a desktop AI coding agent. Open a project folder in the app and it indexes the codebase, then answers questions about it or runs a Plan → Code → Verify cycle to change it.

Every request is classified before anything runs: a question gets a read-only, grounded answer; anything that would change the codebase goes through the full Plan → Code → Verify cycle, informed by your actual file tree, git state and a semantic index of your codebase.

## Architecture & Stack

Two local processes today, already shaped for the planned move to Azure — the target architecture is [`docs/azure-architecture.svg`](docs/azure-architecture.svg).

- **Electron app (`frontend/`)** — React UI. Talks to the local runtime over HTTP + WebSocket on `127.0.0.1:8000`.
- **Local runtime (`backend/`)** — FastAPI adapters (`api/`) over the agent pipeline (`agents/`) and `core/`, which is grouped by where each part will run: `workspace/` (your project on disk), `index/` (the indexing service), `models/` (OpenAI access) and `local/` (cache and keychain). The payloads that cross service boundaries live in `contracts/`.

The agent's tools always run on your machine, so edits land on your disk directly. Everything that is headed for the cloud already sits behind a port with an in-process implementation:

| Port | Today (local) | Cloud build (planned) |
| --- | --- | --- |
| `ModelGateway` | OpenAI directly, key in the OS keychain | Web API LLM gateway with a platform key and per-user quotas |
| `SemanticIndex` | ChromaDB on disk, embedded by an in-process worker | Blob upload → Service Bus → embedding worker → pgvector |
| `JobQueue` | `asyncio.Queue` | Azure Service Bus |

`core/providers.py` is the one place that picks implementations. The agents never import OpenAI or the vector store, and `tests/test_boundaries.py` fails the build if they start to.

Python 3.11+, Node 18+ and an OpenAI API key are the requirements. A local `redis-server` is optional. **OpenAI-only** by design.

## Install

```bash
cd backend
uv pip sync requirements.lock      # reproducible: fully pinned + hash-verified
pip install -e ".[dev]"

cd ../frontend
npm install
```

## Run

```bash
cd frontend
npm run dev      # starts the runtime on :8000, Vite on :5173, and the Electron window
```

Open **Settings** to add your OpenAI API key (stored in the OS keychain), then **New project** to pick a folder. Pick the model for each request from the picker under the chat input.

### Conversations are fresh, not amnesiac

Each project's conversation lives in the app's memory, so follow-ups work — ask "what is this project about", then "sure", and the agent knows what it offered. Every request re-sends the conversation, and older turns are dropped once it exceeds a character budget, so a long session doesn't quietly inflate the cost of every request.

Nothing is stored: there is no transcript on disk and nothing to resume after closing the app. The only state that outlives it is the vector index and your settings.

## Features

1. **Two-phase project indexing, incremental by default**
   - *Phase 1:* file tree + git context, respecting `.gitignore`.
   - *Phase 2:* the runtime hashes every file and syncs with the index; only changed files are chunked (AST- and language-aware) and handed to the indexing worker as a job, which embeds them with `text-embedding-3-small`, stores them and prunes whatever they replaced. An untouched project queues no job at all.
   - Deleted, renamed, emptied and shrunken files lose their stale vectors. Embeddings are stored batch by batch, so a failure part-way through keeps everything already paid for. A chunk the provider refuses is skipped and *reported*, never hidden.
2. **No source code in the index** — the index keeps vectors, paths, line ranges and file hashes. Search results are read back from your disk, and a hit from a file edited since indexing is labelled stale so the agent re-reads it before trusting a line number.
3. **Intent classification** — each request is routed to `answer` or `implement`, grounded in your file tree and git state.
4. **Manual model selection** — no auto-routing; an unknown model ID is rejected with the list of valid ones rather than silently substituted.
5. **Coding Agent: Plan → Code → Verify**
   - *Plan:* a Markdown plan of attack, streamed as it's generated.
   - *Code:* a sandboxed tool loop (`read_file`, `write_file`, `patch_file`, `search_grep`, `search_semantic`). Every path is resolved against the project root and anything escaping it is rejected.
   - *Verify:* `ruff` and `pytest` run against your project, with up to 3 autonomous self-correction rounds. A check that could not run reports as **skipped**, never as passed.
6. **Secure credential storage** — the API key lives in the OS keychain via `keyring`, never in a config file, and is never echoed back to the UI.
7. **Optional Redis cache** — caches the AST repo map and chunk embeddings, keyed by content so a renamed file costs nothing to re-index. Strictly an accelerator: without a `redis-server` the app logs one line and carries on.

## Roadmap

- **Azure cloud build** ([`docs/azure-architecture.svg`](docs/azure-architecture.svg)): GitHub sign-in, a platform API key behind an LLM gateway, and a cloud index with a queue-backed embedding worker on pgvector — provisioned with Terraform.
- **Security Guardrail Agent:** scan generated code for vulnerabilities before it is written to disk.
- **Time-Travel Rollback:** automatic local snapshots to undo AI-generated modifications.

## State outside the repo

| Path | Holds |
| --- | --- |
| `~/.interroai/config.json` | Username and other non-sensitive config |
| `~/.interroai/chroma/` | Vector index, one collection per project — vectors, paths, line ranges and file hashes, no source text |
| OS keychain (`interroai` / `openai_api_key`) | Your API key |
| Redis, if running (`INTERROAI_REDIS_URL`) | Repo-map and embedding caches. Derived data — safe to flush |

---

For deeper technical detail see `CLAUDE.md`, `agent_platform_architecture_v2.md` and `docs/azure-architecture.svg`.
