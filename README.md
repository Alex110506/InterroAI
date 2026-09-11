# InterroAI

InterroAI is a local, terminal-based AI coding agent. It indexes the project you point it at, **interrogates you** about vague requests before writing any code, then runs a Plan → Code → Verify cycle against your workspace.

The distinguishing idea is the interrogation step: rather than guessing at an under-specified request and producing plausible-looking wrong code, the agent asks targeted clarifying questions first, grounded in your actual file tree, git state and a semantic index of your codebase.

## Architecture & Stack

A single Python process. No server to start, no browser, no Electron window.

- **CLI (`backend/cli/`)** — `rich` for rendering, `prompt_toolkit` for input. This is the frontend.
- **Agents (`backend/agents/`)** — the chat session, the Grill agent, the Coding agent, and the supervisor that drives them.
- **Core (`backend/core/`)** — project indexing, chunking, embeddings, the vector store, the sandbox, and the OpenAI client policy.
- **HTTP API (`backend/api/`)** — the same pipeline exposed over FastAPI + WebSocket, for anything that wants to drive it remotely. Optional; the CLI does not use it.

Python 3.11+ and an OpenAI API key are the only requirements. **OpenAI-only** by design.

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
| `/skip` | Stop the clarifying questions and implement from what's been gathered |
| `/clear` | Clear the screen and forget the conversation |
| `/help` | List the commands |
| `/exit` | Quit (`Ctrl-D` also works) |

Anything that isn't a slash command is sent to the agent. `Ctrl-C` interrupts a request in flight.

### Conversations are fresh, not amnesiac

The conversation lives in memory for exactly as long as the process does, so follow-ups work — ask "what is this project about", then "sure", and the agent knows what it offered. Older turns are dropped once the transcript exceeds a character budget, so a long session doesn't quietly inflate the cost of every request.

Nothing is loaded at startup and nothing is written at exit. Opening a project always gives you an empty conversation, there is no transcript to resume, and `/clear` empties it again mid-run. The only state that outlives a run is the vector index and your settings.

## Features

1. **Two-phase project indexing**
   - *Phase 1:* file tree + git context, respecting `.gitignore`.
   - *Phase 2:* AST- and language-aware chunking, embedded with `text-embedding-3-small` into a persistent local ChromaDB at `~/.interroai/chroma/`. Skipped on startup when the project already has vectors — pass `--reindex` or run `/index` to refresh.
2. **Interrogation (Grill Agent)** — clarifying questions grounded in a hybrid context of file tree, git state and RAG, up to 5 turns. `/skip` ends it early and implements from what's known so far.
3. **Intent classification** — each request is routed to `answer`, `interrogate`, or `implement`.
4. **Manual model selection** — no auto-routing. You pick the model with `/model`, and an unknown ID is rejected with the list of valid ones rather than silently substituted.
5. **Coding Agent: Plan → Code → Verify**
   - *Plan:* a Markdown plan of attack, streamed as it's generated.
   - *Code:* a sandboxed tool loop (`read_file`, `write_file`, `patch_file`, `search_grep`, `search_semantic`). Every path is resolved against the project root and anything escaping it is rejected.
   - *Verify:* `ruff` and `pytest` run against your project, with up to 3 autonomous self-correction rounds. A check that could not run reports as **skipped**, never as passed.
6. **Secure credential storage** — the API key lives in the OS keychain via `keyring`, never in a config file, and is masked on input and on display.

## Roadmap

- **Security Guardrail Agent:** scan generated code for vulnerabilities before it is written to disk.
- **Time-Travel Rollback:** automatic local snapshots to undo AI-generated modifications.

## State outside the repo

| Path | Holds |
| --- | --- |
| `~/.interroai/config.json` | Username and other non-sensitive config |
| `~/.interroai/chroma/` | Vector store, one collection per indexed project |
| OS keychain (`interroai` / `openai_api_key`) | Your API key |

---

For deeper technical detail see `agent_platform_architecture_v2.md` and `CLAUDE.md`.
