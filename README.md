# InterroAI

InterroAI is a local, terminal-based AI coding agent. It indexes the project you point it at, then runs a Plan → Code → Verify cycle against your workspace.

Every request is classified before anything runs: a question gets a read-only, grounded answer; anything that would change the codebase goes through the full Plan → Code → Verify cycle, informed by your actual file tree, git state and a semantic index of your codebase.

## Architecture & Stack

A single Python process. No server to start, no browser, no Electron window.

- **CLI (`backend/cli/`)** — `rich` for rendering, `prompt_toolkit` for input. This is the frontend.
- **Agents (`backend/agents/`)** — the chat session, the Coding agent, and the supervisor that drives them.
- **Core (`backend/core/`)** — project indexing, chunking, embeddings, the vector store, the sandbox, and the OpenAI client policy.
- **HTTP API (`backend/api/`)** — the same pipeline exposed over FastAPI + WebSocket, for anything that wants to drive it remotely. Optional; the CLI does not use it.

Python 3.11+ and an OpenAI API key are the only requirements. A local `redis-server` is optional — it caches the repo map and embeddings, and the app runs fine (just slower on repeat work) without one. **OpenAI-only** by design.

## Install

```bash
cd backend
uv pip sync requirements.lock      # reproducible: fully pinned + hash-verified
pip install -e "."                 # puts `interroai` on your PATH
```

## Use

```bash
interroai                    # index and open the current directory
interroai ~/code/myapp       # index and open that path
interroai --model gpt-5.4-mini
interroai --reindex          # force a fresh embedding pass
```

On first run it asks for your OpenAI API key and stores it in the OS keychain.

### Commands

| Command | Does |
| --- | --- |
| `/model` | Pick the model for the next request, from a numbered list |
| `/user` | Set your username (1) or OpenAI API key (2) |
| `/index` | Re-index the project for semantic search |
| `/clear` | Clear the screen and forget the conversation |
| `/help` | List the commands |
| `/exit` | Quit (`Ctrl-D` also works) |

Anything that isn't a slash command is sent to the agent. `Ctrl-C` interrupts a request in flight.

### Conversations are fresh, not amnesiac

The conversation lives in memory for exactly as long as the process does, so follow-ups work — ask "what is this project about", then "sure", and the agent knows what it offered. Older turns are dropped once the transcript exceeds a character budget, so a long session doesn't quietly inflate the cost of every request.

Nothing is loaded at startup and nothing is written at exit. Opening a project always gives you an empty conversation, there is no transcript to resume, and `/clear` empties it again mid-run. The only state that outlives a run is the vector index and your settings.

## Features

1. **Two-phase project indexing, incremental by default**
   - *Phase 1:* file tree + git context, respecting `.gitignore`.
   - *Phase 2:* AST- and language-aware chunking, embedded with `text-embedding-3-small` into a persistent local ChromaDB at `~/.interroai/chroma/`.
   - Every run hashes the files and embeds only what changed, prunes the vectors of files you deleted or renamed, and leaves the rest alone — so an untouched project re-indexes for zero API calls and the index never silently goes stale. `--reindex` or `/index` discards it and rebuilds from scratch.
   - Embeddings are stored batch by batch, so a failure part-way through keeps everything already paid for; the next run finishes the job. A chunk the provider refuses is skipped and *reported*, never hidden.
2. **Intent classification** — each request is routed to `answer` or `implement`, grounded in your file tree, git state and a semantic index of the codebase.
3. **Manual model selection** — no auto-routing. You pick the model with `/model`, and an unknown ID is rejected with the list of valid ones rather than silently substituted.
4. **Coding Agent: Plan → Code → Verify**
   - *Plan:* a Markdown plan of attack, streamed as it's generated.
   - *Code:* a sandboxed tool loop (`read_file`, `write_file`, `patch_file`, `search_grep`, `search_semantic`). Every path is resolved against the project root and anything escaping it is rejected.
   - *Verify:* `ruff` and `pytest` run against your project, with up to 3 autonomous self-correction rounds. A check that could not run reports as **skipped**, never as passed.
5. **Secure credential storage** — the API key lives in the OS keychain via `keyring`, never in a config file, and is masked on input and on display.
6. **Optional Redis cache** — caches the AST repo map (rebuilt otherwise on every single request) and chunk embeddings, keyed by content so a renamed file costs nothing to re-index. Strictly an accelerator: with no `redis-server` running the app logs one line and carries on.

## Roadmap

- **Security Guardrail Agent:** scan generated code for vulnerabilities before it is written to disk.
- **Time-Travel Rollback:** automatic local snapshots to undo AI-generated modifications.

## State outside the repo

| Path | Holds |
| --- | --- |
| `~/.interroai/config.json` | Username and other non-sensitive config |
| `~/.interroai/chroma/` | Vector store, one collection per indexed project |
| OS keychain (`interroai` / `openai_api_key`) | Your API key |
| Redis, if running (`INTERROAI_REDIS_URL`) | Repo-map and embedding caches. Derived data — safe to flush |

---

For deeper technical detail see `agent_platform_architecture_v2.md` and `CLAUDE.md`.
