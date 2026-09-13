# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

InterroAI is a local, terminal-based AI coding agent: a Python CLI that indexes a project, classifies each request as a question or an implementation task, then runs a Plan → Code → Verify cycle against implementation tasks. Architecture spec is in `agent_platform_architecture_v2.md` (treat it as the source of truth for design intent). **OpenAI-only**: do not introduce other LLM providers.

Everything lives under `backend/`. The directory name is a leftover from when there was an Electron/React frontend beside it; there is no other process now, and `backend/cli/` *is* the frontend.

## Common commands

```bash
cd backend

uv pip sync requirements.lock      # reproducible: fully pinned + hash-verified
pip install -e ".[dev]"            # editable install + ruff/pytest, puts `interroai` on PATH

interroai                          # index and open the current directory
interroai ~/code/myapp             # index and open that path
interroai --model gpt-5.4-mini     # start on a specific model
interroai --reindex                # discard the index and rebuild it from scratch
interroai --verbose                # agent logging to stderr

interroai-backend                  # optional: serve the same pipeline over HTTP on :8000
```

A local `redis-server` is **optional**: it caches the repo map and chunk embeddings (see *The cache*, below). Without one the app runs normally, just slower on repeat work. `brew services start redis` if you want it.

`pyproject.toml` holds the abstract dependencies (lower bounds) and is the input to the lockfile. After changing a dependency there, regenerate the lock:

```bash
uv pip compile pyproject.toml --generate-hashes --python-version 3.11 -o requirements.lock
```

### Tests

```bash
cd backend
pytest                              # 477 tests
pytest --cov --cov-report=term-missing
```

`backend/tests/` mirrors the source layout, one module per source module. Three rules make the suite safe to run anywhere:

- **`tests/conftest.py` redirects `HOME` to a temp directory before importing any app module.** This is load-bearing — `config.py` performs filesystem writes at *import* time, so the redirect must stay above those imports or a test run will write into the developer's real `~/.interroai`.
- **No test makes a network call, touches the OS keychain, or reaches a real Redis.** `keyring` is replaced with an in-memory double (the shared `fake_keyring` fixture), OpenAI clients are stubbed via `core.llm`'s seams (`get_client` / `chat_completion` / `chat_stream` / `embed_batch`), and `conftest.py`'s `fake_redis` fixture is **autouse** — a developer running a real `redis-server` must not get different results from CI, and a vector cached by one test must not satisfy the next one's assertions about how many API calls it made.
- **No test needs a tty.** `test_cli.py` constructs the real `InterroCLI` but never reads from its `PromptSession`; it either calls commands directly or replaces `InterroCLI.ask` with scripted answers, and renders into a `Console` backed by `StringIO`.

Note the Coder Agent invokes `ruff` and `pytest` against *user* projects via `core/sandbox.py`; those tools are not required for InterroAI itself to run. When absent, the check returns `CheckStatus.SKIPPED` and the CLI reports it as *skipped* — deliberately distinct from *passed*, so a missing linter is never mistaken for a clean result.

Ruff is configured but the repo is **not** lint-clean (~60 pre-existing findings, mostly `E501` on embedded prompt text where rewrapping would change what the model sees). New files are expected to pass `ruff check`; do not treat the existing baseline as a regression.

## Architecture

One process. The layering rule is that **transport never leaks into the pipeline**:

```
cli/            api/            ← two frontends, interchangeable
    \          /
     agents/  core/             ← the pipeline; knows nothing about either
```

This is why `agents/session.py` and `core/project_index.py` exist as separate modules from `api/chat.py` and `api/projects.py`. The pipeline previously lived *inside* the WebSocket handlers and could not be driven any other way; extracting it is what made a CLI possible. Do not reintroduce transport objects (a `WebSocket`, a `Console`) below the `agents/` or `core/` line.

### CLI (`backend/cli/`) — the frontend

- `main.py` — argparse entry (`interroai`), logging policy, then hands off to `InterroCLI`.
- `app.py` — the REPL. Owns the terminal, the current project and the current model. `rich` renders output; `prompt_toolkit` reads input.
- `commands.py` — the slash-command registry. `COMMANDS` is the single source for both dispatch and `/help`, so the listing cannot drift from what runs.
- `render.py` — `EventRenderer` turns pipeline events into terminal output. It consumes exactly the dicts `agents/session.py` emits.
- `indexing.py` — startup indexing and its progress display.
- `settings.py` — the `/user` data layer (name → config file, key → keychain).
- `theme.py` — palette, glyphs, and `model_label()`.

Two details worth not rediscovering:

- **`prompt_toolkit` remembers the arguments of the last `prompt()` call.** After the masked API-key prompt, every later prompt stayed masked until `is_password=False` was passed explicitly in `_read_request`. Always pass it.
- **Model output is printed with `markup=False`.** A bracket in a traceback or a `list[str]` in a plan is not a rich style tag. Styling for those lines is passed as `style=`, never embedded in the string.

### HTTP API (`backend/api/`) — the optional second frontend

`main.py` mounts three routers. Each is a thin adapter; none contains pipeline logic.

- `api/settings.py` — user name → `~/.interroai/config.json`; OpenAI API key → OS keychain via `keyring` (`core/security.py`). The key is never echoed to the client.
- `api/projects.py` — `POST /api/projects/index` (Phase 1) and `WS /api/projects/ws/embed` (Phase 2), both delegating to `core/project_index.py`.
- `api/chat.py` — `WS /api/chat/ws`. Turns each frame into a `ChatSession` call and relays the events back as JSON.

### The pipeline (`backend/agents/session.py`)

`ChatSession` is one user request, from classification to finished implementation. It is single-use but **not amnesiac**: the caller passes the conversation so far as `history` (OpenAI message shape, oldest first) and the session threads it into both places that need it — the classifier and the Coder agent.

This split matters and was got wrong once. "Conversations are always fresh" means *nothing is loaded from disk when you open a project* — it does **not** mean the agent forgets what it said thirty seconds ago. Dropping history entirely made `sure` (in reply to "shall I diagram that?") classify and execute as a brand-new request. The transcript is owned by the frontend, held in memory, trimmed to `_MAX_HISTORY_CHARS` at the point of use, and never persisted.

1. **Intent classification** — `classify_intent` (gpt-5.4-mini, JSON-mode) labels the message `answer | implement`, reading it against the prior turns so a bare "sure" classifies as what it agrees to. Unusable *model output* falls back to `implement`; an unusable *environment* (no API key, provider down) propagates rather than misrouting.
2. **Model validation** — there is no automatic routing. The caller names a display ID, validated against `coder._MODEL_MAP`. An unknown ID yields an `error` event but deliberately leaves the session usable, so the user can pick a valid one and carry on. Omitting the model applies `_DEFAULT_MODEL`.
3. **Supervisor** (`agents/supervisor.py`) — `stream()` is the transport-agnostic form the session uses; `run()` wraps it for a WebSocket.
4. **Coder Agent** (`agents/coder.py`) — three phases (`_plan` → `_tool_loop` → `_validate`) yielding typed events (`plan_chunk`, `plan`, `tool_call`, `tool_result`, `impl_done`, `validation_result`, `correction`, `done`, `error`). Up to `_MAX_TOOL_ROUNDS=20` tool calls and `_MAX_CORRECTIONS=3` self-fix attempts.

`ChatSession._guard` converts a failure into a final `error` event and sets `finished`. Both frontends have to render errors anyway, so the conversion happens once, in the session.

### Project indexing (`backend/core/project_index.py`)

- **Phase 1** (`build_index`): file-tree + git context scan, returns `ProjectIndex`. Honors `.gitignore` via `pathspec` and a hardcoded `_EXCLUDE_DIRS`.
- **Phase 2** (`embed_project`): an async generator of `{step: A|B|C|D, status}` events for traversal → AST/language-aware chunking (`core/chunker.py` uses LangChain `RecursiveCharacterTextSplitter`) → `text-embedding-3-small` embeddings (`core/embeddings.py`) → ChromaDB persistent collection at `~/.interroai/chroma/`, keyed by sanitized project path.

`core/embeddings.py::embed_batches` is an async generator, so `embed_project` re-yields its progress directly and persists each batch as it lands. (An earlier version bridged an embedding progress *callback* to the generator with an `asyncio.Queue`; that machinery is gone.) Errors are *yielded* as `{"step": "error"}`, not raised, since every caller would only have to catch and render them anyway — but vectors from batches that already completed are already stored, and the next run reconciles and finishes the job.

**Phase 2 is incremental, and that is what makes it safe to run on every startup.** Each stored chunk carries `file_hash` in its ChromaDB metadata, so a run starts by diffing the hashes on disk against `vector_store.stored_manifest()` and splits the project into unchanged (free), changed/new (re-chunked and re-embedded), and gone (pruned). An untouched project costs one local hashing pass and zero API calls. The CLI therefore no longer skips Phase 2 when vectors exist — skipping is what let the index drift out of date the moment you edited anything outside the tool.

Two rules worth not rediscovering:

- **Upserting is not enough.** Chunk ids are `file_path:start_line`, so a deleted or renamed file — and the tail of a file that *shrank* — leaves chunks behind that keep answering searches with content the file no longer has. `delete_ids` is called for those every run; pruning happens *after* storing, so a run that dies mid-way leaves the old vectors rather than a hole.
- **The manifest lives in Chroma, not Redis.** It has to stay consistent with the vectors it describes; a manifest that could be evicted independently would eventually disagree with them, and reconciling against a wrong manifest is exactly how orphans come back.

`--reindex` / `/index` still forces a full rebuild via `vector_store.reset_collection()` — the reason to ask for one is that the store itself is suspect (corrupt, or vectors from a different embedding model), which is precisely the state reconciliation would preserve.

A chunk the provider refuses (`core.llm.ITEM_ERRORS` — over the token limit, say) is isolated by bisecting the batch, skipped, and *reported* in the `done` event's `skipped`/`skipped_files`; the CLI prints it as a warning. Anything else propagates, because skipping items one at a time through an outage would quietly produce an empty index and call it success.

### The cache (`backend/core/cache.py`)

Redis holds two recomputable things: the AST repo map (keyed by `ast_map.fingerprint_sources`, a hash of the source files' paths/mtimes/sizes) and chunk embeddings (keyed by a hash of the chunk text plus the model name). Content addressing is what makes a renamed or moved file free to re-index.

It is an **accelerator, never a dependency**: with no `redis-server` running, the first call logs one warning, the module marks itself unavailable, and every later call short-circuits so nothing pays the connect timeout twice. Override the location with `INTERROAI_REDIS_URL` (default `redis://localhost:6379/0`).

### Display IDs vs OpenAI API IDs

The CLI exposes fictional model names (`gpt-5.4-*`, `gpt-5.5-*`). `backend/agents/coder.py::_MODEL_MAP` is the **single source of truth** mapping them to real OpenAI IDs. `agents/session.py::AVAILABLE_MODELS` is the display order used by `/model` and `--model`; `test_session.py` asserts the two sets are identical, so a new model added to one and forgotten in the other fails the suite instead of shipping a picker that offers an ID the backend rejects.

Labels are *derived* from the ID by `cli/theme.py::model_label`, not tabulated — one less place to forget a new model. Reasoning-model handling (no `temperature`, no streaming for Phase 1) is gated by `coder._REASONING_MODELS`.

### Coder Agent tools

The five function-calling tools (`_TOOLS` in `coder.py`) are sandboxed by `CoderAgent._resolve`, which rejects any path that escapes the project root. `patch_file` requires verbatim search blocks (no fuzzy matching); the system prompt instructs the model to `read_file` first. `write_file` is for new files only. All paths are relative to the active project root.

The **Knowledge Tree** (`_build_knowledge_tree`) is what the agent sees as context — a `<repo_map>` of class/function signatures from `core/ast_map.py` plus system instructions. Per-task RAG happens via the `search_semantic` tool (vector-store lookup against the per-project ChromaDB collection), not upfront in the tree.

Building that map parses every source file, and the agent needs it on *every* request, so it is cached in Redis behind `ast_map.fingerprint_sources` — stat-ing files is orders of magnitude cheaper than parsing them. The agent's own writes move an mtime, so the fingerprint invalidates itself; there is no manual invalidation to forget.

### State that lives outside the repo

- `~/.interroai/config.json` — user name and other non-sensitive config.
- `~/.interroai/chroma/` — persistent ChromaDB store (one collection per indexed project), including the `file_hash` manifest that drives incremental indexing.
- OS keychain (service `interroai`, account `openai_api_key`) — API key.
- Redis, if one is running (`INTERROAI_REDIS_URL`, default `redis://localhost:6379/0`) — the repo-map and embedding caches under the `interroai:v1:` prefix, 30-day TTL. Purely derived data: flushing it costs time, never correctness.

There is deliberately **no** conversation store — the transcript lives in `InterroCLI._history` and dies with the process. A `~/.interroai/messages.db` from an older build is dead and can be deleted.

When debugging "no API key" errors, check the keychain via `core/security.py::retrieve_openai_key`, not env vars or config files.
