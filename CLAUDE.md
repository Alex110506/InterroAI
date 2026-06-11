# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

InterroAI is a local desktop AI multi-agent orchestration platform — a FastAPI Python backend + Electron/React frontend, modeled on Codex/Antigravity. Architecture spec is in `agent_platform_architecture_v2.md` (treat it as the source of truth for design intent). **OpenAI-only**: do not introduce other LLM providers.

## Common commands

All commands run from `frontend/`:

```bash
# Full dev stack (backend + Vite + Electron in one terminal)
npm run dev

# Just the FastAPI backend (autoreload on :8000)
npm run backend

# Just the Electron window (assumes Vite already running on :5173)
npm run electron

# Production renderer build
npm run build
```

Backend dependencies: `cd backend && pip install -r requirements.txt`. Python 3.11+ assumed (PEP 604 unions, `Path` patterns).

There is no test suite in this repo yet. The Coder Agent invokes `ruff` and `pytest` against *user* projects via `core/sandbox.py`; those tools are not required for InterroAI itself to run (they're skipped silently when absent).

## Architecture

Two processes communicate over HTTP + WebSocket on `localhost:8000`.

### Backend (`backend/`) — FastAPI orchestrator

Entry point `main.py` mounts three routers:

- `api/settings.py` — user name → `~/.interroai/config.json`; OpenAI API key → OS keychain via `keyring` (`core/security.py`). The key is never echoed to the client.
- `api/projects.py` — two-phase project indexing:
  - **Phase 1** (`POST /api/projects/index`): synchronous file-tree + git context scan, returns `ProjectIndex`. Honors `.gitignore` via `pathspec` and a hardcoded `_EXCLUDE_DIRS`.
  - **Phase 2** (`WS /api/projects/ws/embed`): streams `{step: A|B|C|D, status}` events for traversal → AST/language-aware chunking (`core/chunker.py` uses LangChain `RecursiveCharacterTextSplitter`) → `text-embedding-3-small` embeddings (`core/embeddings.py`) → ChromaDB persistent collection at `~/.interroai/chroma/`, keyed by sanitized project path.
- `api/chat.py` — the single WebSocket (`/api/chat/ws`) that drives the whole agent pipeline.

### Agent pipeline (`backend/agents/`)

Every chat message flows through `api/chat.py`'s WebSocket handler:

1. **Intent classification** — `_classify_intent` (gpt-4o-mini, JSON-mode) labels the message `answer | interrogate | implement`. `answer` returns Markdown directly; the other two continue.
2. **Grill Agent** (`agents/grill.py`) — only for `interrogate`. Stateful, up to `_MAX_TURNS=3` clarifying questions. Each turn pulls hybrid context: file tree + git context (from the cached `ProjectIndex`) + ChromaDB RAG. Returns strict JSON `{is_prompt_ready, question | refined_prompt}`. A `force_ready` WebSocket message lets the user bail early.
3. **Model Router** (`agents/router.py`) — gpt-4o-mini, JSON-mode, only invoked when the user has `model="auto"`. Returns one of the five **display IDs** (`gpt-5.4-mini`, `gpt-5.4-low-effort`, `gpt-5.4-high-effort`, `gpt-5.5-low-effort`, `gpt-5.5-high-effort`).
4. **Supervisor** (`agents/supervisor.py`) — thin wrapper that streams Coder Agent events back over the WebSocket.
5. **Coder Agent** (`agents/coder.py`) — three phases (`_plan` → `_tool_loop` → `_validate`) yielding typed events (`plan_chunk`, `tool_call`, `tool_result`, `impl_done`, `validation_result`, `correction`, `done`, `error`). Up to `_MAX_TOOL_ROUNDS=20` tool calls and `_MAX_CORRECTIONS=3` self-fix attempts.

### Display IDs vs OpenAI API IDs

The UI exposes fictional model names (`gpt-5.4-*`, `gpt-5.5-*`). These are mapped to real OpenAI IDs in **two places** that must stay in sync:

- `backend/agents/coder.py::_MODEL_MAP` — coder execution
- `backend/api/chat.py::_QA_MODEL_MAP` — direct Q&A answers

When adding or renaming a display ID, update **both** maps and `agents/router.py::_VALID_MODELS`. Reasoning-model handling (no `temperature`, no streaming for Phase 1) is gated by `coder._REASONING_MODELS`.

### Coder Agent tools

The five function-calling tools (`_TOOLS` in `coder.py`) are sandboxed by `CoderAgent._resolve`, which rejects any path that escapes the project root. `patch_file` requires verbatim search blocks (no fuzzy matching); the system prompt instructs the model to `read_file` first. `write_file` is for new files only. All paths are relative to the active project root.

The **Knowledge Tree** (`_build_knowledge_tree`) is what the agent sees as context — a `<repo_map>` of class/function signatures from `core/ast_map.py` plus system instructions. Per-task RAG happens via the `search_semantic` tool (vector-store lookup against the per-project ChromaDB collection), not upfront in the tree.

### Frontend (`frontend/`)

- Electron shell (`main.js`) with `hiddenInset` titlebar + macOS `under-window` vibrancy. The only IPC channel is `select-folder` (native folder picker).
- React 18 + Vite renderer in `src/`. `App.jsx` owns the project list + active selection; `Sidebar`, `ChatPanel`, `ThoughtPanel`, `SettingsModal` are the panels. CSS Modules per component.
- `src/lib/api.js` is the only network layer: REST via `fetch`, two WebSocket helpers (`openChatSocket` for chat, `embedProject` for the indexing stream). Override the backend URL with `VITE_BACKEND_URL`.

### State that lives outside the repo

- `~/.interroai/config.json` — user name and other non-sensitive config.
- `~/.interroai/chroma/` — persistent ChromaDB store (one collection per indexed project).
- OS keychain (service `interroai`, account `openai_api_key`) — API key.

When debugging "no API key" errors, check the keychain via `core/security.py::retrieve_openai_key`, not env vars or config files.
