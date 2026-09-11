# InterroAI — Technical Improvement Roadmap

A grounded review of the current architecture and a prioritized backlog covering the three
things you asked about: **(1) shipping a CLI**, **(2) multi-model + local/open-source model
support**, and **(3) overhauling the RAG + indexing pipeline** — plus correctness fixes and
new features.

> **Scope note / design decision to make first:** the current `CLAUDE.md` and
> `agent_platform_architecture_v2.md` both mandate **OpenAI-only** ("do not introduce other
> LLM providers", "THIS PROJECT WILL ONLY USE MODELS FROM OPENAI"). Your goal of running
> **local open-source models** directly overrides that founding constraint. That's a fine
> call to make — it's your project — but it should be a *deliberate* amendment: update both
> docs so they stop being the "source of truth" for a rule you no longer follow. Every task
> in §4 assumes this amendment is accepted.

---

## 0. Priority snapshot

| # | Item | Impact | Effort | Why now |
|---|------|--------|--------|---------|
| 3.1 | Fix stale/orphaned chunks on re-index | High | S | Silent RAG-correctness bug — retrieval returns deleted code |
| 4.1 | Provider abstraction (`base_url`) + model registry | High | M | Unlocks local mo dels *and* kills the dual-map hazard |
| 4.2 | Local embeddings (offline RAG) | High | M | RAG can't be "local" while embeddings hit OpenAI cloud |
| 5.1 | Extract framework-agnostic orchestrator | High | M | Prerequisite for the CLI; also de-risks the WS handler |
| 5.2 | `interroai` CLI (Typer + Rich) | High | M | Your headline ask |
| 6.1 | Incremental indexing (hash manifest) | High | M | Kills full re-embed cost; also fixes 3.1 cleanly |
| 6.2 | Reranking + hybrid (BM25 + vector) retrieval | High | M | Single biggest retrieval-precision lever |
| 6.3 | True AST chunking (tree-sitter) | Med | M | Spec claims AST; today it's char-splitting |
| 7.1 | Server-side ProjectIndex cache / in-proc index | Med | S | Client currently resends a possibly-stale index every turn |
| 8.x | Diff-preview gate, rollback, guardrail, graph | High | M–L | Trust + safety + the unbuilt spec §8 features |
| 2.x | Doc ↔ code reconciliation | Med | S | `CLAUDE.md` describes constants/maps that no longer exist |

Legend: effort S = <1 day, M = a few days, L = 1–2 weeks.

---

## 1. Current state — honest assessment

**What's genuinely good and worth keeping:**

- Clean separation: `agents/` (grill, router, supervisor, coder) and `core/` (chunker,
  embeddings, vector_store, ast_map, sandbox, patcher, security) are already
  framework-agnostic. Agents are **async generators yielding typed dict events** — the
  FastAPI WebSocket is just one consumer. This is the single most important property for the
  CLI: the core doesn't know about HTTP.
- Path-escape sandboxing in `CoderAgent._resolve` (`coder.py:425`) is correct
  (`is_relative_to`, symlink-aware) rather than the naive `startswith`.
- Keychain-based secret storage (`core/security.py`) is the right call.
- Local TF-IDF router (`agents/router.py`) is a nice touch — sub-ms, offline complexity
  classification.

**The gap between the spec and the code** (worth knowing before you build on top):

| Spec / `CLAUDE.md` says | Code actually does | File |
|---|---|---|
| "AST / language-aware **semantic chunking**" | `RecursiveCharacterTextSplitter` — separator heuristics, **not** an AST | `core/chunker.py` |
| "LangChain **Semantic Router** over prompt vectors" (§5) | TF-IDF + LogisticRegression on raw text; intent is a *separate* OpenAI call | `agents/router.py`, `api/chat.py` |
| "from the **cached** `ProjectIndex`" | No server-side cache — client resends `project_index` on every `start` | `api/chat.py:111` |
| `_QA_MODEL_MAP` in `api/chat.py` | Doesn't exist — replaced by `_CODER_MODEL_MAP` reuse | `api/chat.py` |
| Grill `_MAX_TURNS = 3` | Actually `5`, and grill uses `gpt-4o-mini` while intent uses `gpt-5.4-mini` | `agents/grill.py:28-29` |
| Knowledge Tree includes target files + git + RAG (§6A) | Only `system_instructions` + `repo_map` — coder starts "blind" | `coder.py:491` |
| §8: guardrail agent, agent graph, time-travel rollback | None implemented | — |

None of these are fatal, but they mean the docs can't be trusted as-is. Fix them (§2) so
future-you (and any agent working in this repo) isn't misled.

---

## 2. Documentation ↔ code reconciliation (quick, do alongside everything)

- [ ] Update `CLAUDE.md`: remove `_QA_MODEL_MAP`; correct `_MAX_TURNS` (5, in `grill.py`);
      correct the intent model (`gpt-5.4-mini`, not `gpt-4o-mini`); stop calling the chunker
      "AST".
- [ ] Update `agent_platform_architecture_v2.md`: either implement the LangChain Semantic
      Router as specced **or** rewrite §5 to describe the actual TF-IDF classifier + separate
      OpenAI intent step. Same for the OpenAI-only clause (see scope note).
- [ ] Rename references to `interrogator.py` → `grill.py` in the spec's monorepo diagram.

---

## 3. Correctness bugs (small, high-value — fix first)

### 3.1 Stale/orphaned chunks survive a re-index  ⚠️ *silent RAG corruption*
`store_chunks` (`core/vector_store.py:47`) **upserts** by `id = f"{file_path}:{start_line}"`
and never deletes. Consequences:
- Edit a file so a function moves or the file shrinks → the old chunks at their old start
  lines **stay in the DB** and keep getting retrieved. The agent gets code that no longer
  exists.
- Delete a file → **all** its chunks remain forever.

Fix (also folds into §6.1): before writing a file's new chunks, purge its old ones.
```python
# delete this file's previous chunks, then re-add
col.delete(where={"file_path": rel})
col.upsert(ids=..., embeddings=..., documents=..., metadatas=...)
# and for files that vanished entirely:
stored = set(col.get(include=[])["ids"])
col.delete(ids=[i for i in stored if i.split(":")[0] not in current_files])
```

### 3.2 `search_chunks` can create a mismatched collection
`search_chunks` (`vector_store.py:69`) calls `get_or_create_collection` **without** the
`hnsw:space="cosine"` metadata that `store_chunks` sets. If a search ever runs before an
index exists, the collection is created with the default L2 space; subsequent stores then
disagree with reads. Pass the same metadata in both places (or a single `_get_collection`
helper).

### 3.3 `patch_file` is verbatim + first-occurrence only
`core/patcher.py:26` does `content.replace(search_block, replace_block, 1)`. If the search
block appears twice it silently patches the wrong one. Recommend: count occurrences, **fail
loudly if >1** (force the model to add context), or move to line-anchored / `git apply`
unified-diff application.

### 3.4 Intent classifier swallows every error into `"implement"`
`_classify_intent` (`api/chat.py:88`) returns `"implement"` on *any* exception — including
"no API key" (line 64) — which then routes to the coder, which fails again on the missing
key. Surface "no key" as a first-class UI error instead of a misroute.

### 3.5 Repo map truncates **alphabetically** at 80 files
`build_repo_map` (`core/ast_map.py:94`) does `sorted(root.rglob("*"))[:80]` — on a large repo
it keeps `a…` and drops `z…`, which may be exactly the files the task touches. Prioritize by
git-recency / proximity to the target, or raise/removes the cap with token budgeting.

### 3.6 No task cancellation
The chat WS `break`s out of its loop right after dispatching to the supervisor
(`api/chat.py:152`), so a long coder run can't be stopped (the `force_ready` escape only
exists during the grill phase). Add a cancel message + `asyncio.Task` you can `.cancel()`.

---

## 4. Multi-model + local / open-source models  *(your headline ask)*

The whole stack already speaks the **OpenAI SDK**. The good news: llama.cpp, **Ollama**, LM
Studio, and vLLM all expose an **OpenAI-compatible `/v1` endpoint**, so "support local models"
is mostly "make `base_url` and the model list configurable" — not a rewrite.

### 4.1 Provider abstraction + a real model registry
Today display IDs are mapped to API IDs in **two hand-synced dicts** (`coder._MODEL_MAP`,
plus a stale copy in `prompt_classifier/test.py`), and reasoning behavior is a hardcoded set
(`coder._REASONING_MODELS`). Replace all of it with one declarative registry.

```yaml
# backend/models.yaml  (single source of truth)
providers:
  openai:   { kind: openai, base_url: https://api.openai.com/v1, key: keychain:openai_api_key }
  ollama:   { kind: openai, base_url: http://localhost:11434/v1, key: "ollama" }   # dummy key
  lmstudio: { kind: openai, base_url: http://localhost:1234/v1,  key: "lm-studio" }

models:
  - id: gpt-5.5-high-effort            # display id (keep your fictional names if you like)
    provider: openai
    api_model: gpt-5.5
    context_window: 200000
    supports_tools: true
    supports_temperature: false        # replaces the _REASONING_MODELS set
    supports_json_mode: true
  - id: qwen2.5-coder-32b
    provider: ollama
    api_model: qwen2.5-coder:32b
    context_window: 32768
    supports_tools: true
    supports_json_mode: false          # drives the fallback in 4.3
```

```python
# backend/core/providers.py
def get_client(model_id: str) -> tuple[AsyncOpenAI, ModelSpec]:
    spec = REGISTRY[model_id]
    prov = PROVIDERS[spec.provider]
    return AsyncOpenAI(base_url=prov.base_url, api_key=resolve_key(prov.key)), spec
```
- [ ] Introduce `ModelSpec` (capabilities: tools, temperature, json_mode, context window).
- [ ] `CoderAgent`, `GrillAgent`, `_classify_intent`, `embeddings` all resolve their client
      via `get_client(...)` instead of `AsyncOpenAI(api_key=key)`.
- [ ] Delete `coder._MODEL_MAP`, `_REASONING_MODELS`, and the inlined map in `test.py`;
      derive everything from the registry. Router's `_VALID_MODELS` becomes "registry keys".

### 4.2 Local embeddings (so RAG is *actually* local)
`core/embeddings.py` hardcodes `text-embedding-3-small` against OpenAI. For an offline RAG:
- [ ] Add an embedding-provider switch: OpenAI **or** a local backend —
      `nomic-embed-text` / `bge-m3` via Ollama's `/v1/embeddings`, or in-process
      `sentence-transformers`.
- [ ] **Critical:** ChromaDB fixes a collection's vector dimension on first insert.
      `text-embedding-3-small` is 1536-dim; `nomic-embed-text` is 768. Switching models on the
      same collection **silently corrupts search**. Namespace the collection by embedding model
      (`{project}__{embed_model}`) and store `embed_model` in collection metadata; warn/rebuild
      on mismatch.
- [ ] Add retry/backoff (tenacity) around embedding + completion calls — there is none today.

### 4.3 Tool-calling & JSON-mode fallbacks for local models
Local models vary: many don't support OpenAI `tools` function-calling or
`response_format={"type":"json_object"}` well. Grill and intent both *require* JSON mode
(`grill.py:163`, `chat.py:79`) and the coder *requires* tool-calling — these will break on a
model that lacks them.
- [ ] Gate on `spec.supports_json_mode` / `spec.supports_tools`; when false, fall back to
      prompt-engineered JSON + a tolerant extractor (or a lib like `instructor` / `outlines`).
- [ ] Default local coding to models with strong tool support (Qwen2.5-Coder, Llama 3.3,
      Devstral, etc.) and say so in the model picker.
- [ ] Expose a per-provider "detected models" endpoint (Ollama `/api/tags`) so the UI/CLI can
      list what's actually installed instead of a hardcoded dropdown.

---

## 5. Ship a CLI  *(your headline ask)*

The backend logic is reusable as-is; the only thing tying it to Electron/HTTP is the WS
handler in `api/chat.py`. Extract the pipeline, then wrap it in a terminal UI.

### 5.1 Extract a transport-agnostic orchestrator
Move the pipeline in `chat_ws` (`api/chat.py:95-224`) into a plain async generator:
```python
# backend/core/orchestrator.py
async def run_turn(message, project_path, model, index, *, on_question) -> AsyncIterator[Event]:
    action = await classify_intent(message, index)
    ...  # grill loop, router, coder — yields the same typed events today's WS emits
```
- The WS handler becomes a ~15-line adapter (`async for ev in run_turn(...): ws.send_json(ev)`).
- The CLI becomes the same adapter for a terminal.
- `on_question` is a callback so grill can ask interactively in either transport.

### 5.2 `interroai` CLI (Typer + Rich)
- [ ] `pyproject.toml` with a console entry point: `interroai = interroai.cli:app`; make the
      backend an installable package (`pip install -e .`).
- [ ] Commands:
  - `interroai index <path>` — runs Phase 1 + 2 **in-process** (no HTTP), Rich progress bar
    driven by the same step A–D events.
  - `interroai chat [--model … | auto] [--project .]` — interactive REPL; streams the plan
    live (`rich.live`), shows tool calls as a spinner/tree, renders the summary as Markdown.
  - `interroai run "<task>" --project .` — one-shot, non-interactive (auto-answers grill via
    `--yes`/`force_ready`), good for scripting/CI.
  - `interroai models` — list registry + locally-detected Ollama models.
  - `interroai config set-key` / provider config.
- [ ] **Headless key access:** `retrieve_openai_key()` is keychain-only. Add an env-var
      fallback (`OPENAI_API_KEY`, per-provider vars) so the CLI works over SSH / in CI where no
      keychain/GUI exists.
- [ ] **Diff-preview gate in the terminal** (see 8.1): before `patch_file`/`write_file` hits
      disk, print the diff and require confirmation unless `--auto-approve`.

This also cleanly decouples you from Electron long-term — the desktop app and CLI become two
frontends over one core.

---

## 6. RAG pipeline overhaul  *(your explicit focus)*

Current pipeline: walk → `RecursiveCharacterTextSplitter` (800/200) → OpenAI embed →
Chroma cosine → single-vector top-k (k=8 grill, k=5 coder). It works but leaves a lot of
precision on the table and re-embeds everything on every run.

### 6.1 Incremental indexing with a content-hash manifest  *(also fixes 3.1)*
- [ ] Persist `.interroai/manifest.json`: `{ rel_path: {sha256, mtime, chunk_ids[]} }`.
- [ ] On re-index: hash each candidate file; skip unchanged; for changed files
      `col.delete(where={"file_path": rel})` then re-embed; for files missing from the current
      walk, delete their chunk ids. Result: **only the diff is embedded** (big cost/latency
      win) and stale chunks can't linger.
- [ ] Report **file-level** progress ("47/120 files") — the spec promises this; today Step C
      only reports chunk counts.

### 6.2 Hybrid retrieval + reranking  *(biggest precision lever)*
Pure dense top-k misses exact-symbol matches; reranking fixes ordering.
- [ ] Add a lexical index (BM25 via `rank_bm25`, or Chroma's full-text) alongside the vector
      search; fuse with **Reciprocal Rank Fusion**.
- [ ] Add a **reranker** over the fused candidates: a local cross-encoder
      (`bge-reranker-v2-m3`) keeps it offline, or an LLM-rerank pass. Retrieve ~30, rerank to
      top 5–8.
- [ ] Add **MMR / dedup** so overlapping chunks (the 200-char overlap causes near-duplicates)
      don't crowd out diversity.

### 6.3 True AST chunking (tree-sitter)
- [ ] Replace/augment `core/chunker.py` with **tree-sitter** to split on real function/class
      boundaries across languages, keeping each chunk a whole logical unit; fall back to the
      current recursive splitter for unsupported types. This makes the "language-aware
      chunking" claim actually true and improves every downstream retrieval.

### 6.4 Contextual chunk augmentation (cheap recall win)
- [ ] Before embedding, prepend each chunk with its `file_path` + enclosing symbol (and
      optionally a one-line LLM-generated summary — Anthropic's "contextual retrieval").
      Embedding a bare 800-char slice loses the "where am I" signal; prepending path+symbol is
      free and measurably improves recall.
- [ ] Store richer metadata per chunk (language, symbol name, kind) to enable **filtered**
      search (e.g. "only `.py`", "only under `backend/agents/`").

### 6.5 Keep the index live
- [ ] Optional **watch mode** (`watchdog`) that re-indexes changed files on save, so RAG never
      goes stale mid-session. Pairs naturally with 6.1.

---

## 7. Indexing / project-context improvements

### 7.1 Cache the ProjectIndex server-side (or compute it in-proc for the CLI)
Right now the renderer holds the index and **resends it on every chat `start`**
(`api/chat.py:111`); if files changed since indexing, the grill/intent context is stale, and
there's no cache to invalidate. Add an in-memory (or `~/.interroai`) cache keyed by project
path, with a cheap "is this still fresh?" check (git HEAD + dir mtime).

### 7.2 Honor nested & global `.gitignore`
`_walk_indexable_files` (`api/projects.py:207`) only reads the **root** `.gitignore`. Nested
`.gitignore` files and the user's global ignore are ignored, so build artifacts deep in the
tree get embedded. Either walk a pathspec stack per directory or batch through
`git check-ignore --stdin`.

### 7.3 Parallelize + harden the embed pipeline
- [ ] Chunking is a serial Python loop (`api/projects.py:312`); move to a process pool for
      large repos.
- [ ] Embedding batches are sequential; add bounded concurrency + retry/backoff + explicit
      rate-limit handling.
- [ ] Consider a hard budget / file-count guard for very large repos (prioritize by
      git-recency) so a huge monorepo doesn't blow up cost/time on first open.

### 7.4 Seed the coder's Knowledge Tree
`_build_knowledge_tree` (`coder.py:491`) ships only system rules + repo map. Per spec §6A,
add the **file tree**, **git context**, and the **top-k RAG hits for the task** up front so
the agent doesn't burn tool-rounds rediscovering structure it could have been handed.

---

## 8. New features worth adding

### 8.1 Diff-preview + human approval gate  *(highest-trust, do early)*
Today `write_file`/`patch_file` hit disk immediately (`coder.py:449-462`). Buffer edits,
present a unified diff (UI panel / CLI `rich` diff), and apply only on approval (with an
`--auto-approve` / "YOLO" toggle). This is the single biggest trust improvement and a
prerequisite for people running it on real repos.

### 8.2 Time-travel rollback (spec §8.3, unbuilt)
Before the first edit of a task, create a restore point (`git stash create` / a shadow commit
/ a `.interroai/snapshots` copy). Expose one-click / `interroai undo`.

### 8.3 Security guardrail (spec §8.1, unbuilt)
Scan the *proposed diff* before it's written: `gitleaks` (secrets), `semgrep`/`bandit`
(vulns). Block or warn. Cheap to add given the sandbox subprocess pattern in
`core/sandbox.py`.

### 8.4 Plan-approval checkpoint (spec §6.1, unbuilt)
The spec promises a checkpoint after Phase 1 planning; today the plan streams and Phase 2
starts automatically. Add an optional "approve plan?" gate.

### 8.5 Observability
- [ ] Persist full **run transcripts** (plan, every tool call/result, validations,
      corrections) — you already have `messages.db`; add a `runs` table.
- [ ] **Token + cost telemetry** per run (per provider) surfaced in the UI/CLI.
- [ ] Agent-graph visualization (spec §8.2) via React Flow in the renderer.

### 8.6 MCP tool support
Let the coder call **MCP servers** as additional tools (filesystem, web, DB, etc.) — cheap
extensibility and increasingly a standard expectation.

---

## 9. Testing & project hygiene

- [ ] **There is no test suite for InterroAI itself.** Add `pytest` covering the highest-risk
      pure logic: `patcher.apply_patch`, `CoderAgent._resolve` (path-escape), `chunker`
      line-mapping, `vector_store` id/delete semantics, router classification, and the new
      orchestrator (mock the LLM client). These are all deterministic and easy to test.
- [ ] Add CI (GitHub Actions): ruff + pytest on push.
- [ ] `pyproject.toml` for the backend (packaging + entry point for the CLI) instead of a bare
      `requirements.txt`.
- [ ] Pin/track the fictional↔real model mapping in one place (the registry) so
      demos don't break when an underlying API model is renamed.

---

## 10. Recommended sequencing

1. **Foundation (small, unblock everything):** 3.1 stale chunks, 3.2 collection metadata,
   2.x doc fixes, add retry/backoff. → correctness baseline.
2. **Decouple:** 5.1 orchestrator extraction + 9 first tests. → makes the CLI and future work
   safe.
3. **Providers:** 4.1 registry + 4.2 local embeddings + 4.3 fallbacks. → local models work
   end-to-end (this is also what makes the RAG "local").
4. **CLI:** 5.2 on top of 5.1. → your headline deliverable.
5. **RAG quality:** 6.1 incremental → 6.2 rerank/hybrid → 6.4 contextual → 6.3 tree-sitter.
6. **Trust & features:** 8.1 diff gate → 8.2 rollback → 8.3 guardrail → observability.

---

*Generated from a full read of the backend (`agents/`, `core/`, `api/`), the frontend network
layer, and `agent_platform_architecture_v2.md`. File:line references point at the exact code
each item concerns.*
