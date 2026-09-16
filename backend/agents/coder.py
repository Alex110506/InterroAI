"""
Coding Agent — Plan → Code → Verify cycle (Section 6 of the architecture spec).

Phase 1  Planning        The agent generates a Markdown plan of attack.
Phase 2  Implementation  Tool-calling loop: read_file / write_file / patch_file / search_grep.
Phase 3  Validation      Linter + tests; up to MAX_CORRECTIONS autonomous self-fixes.

Events yielded (for WebSocket streaming):
  {"type": "plan",              "content": str}
  {"type": "tool_call",         "tool": str, "args": dict}
  {"type": "tool_result",       "tool": str, "output": str}
  {"type": "impl_done",         "content": str}
  {"type": "validation_result", "phase": "lint"|"test", "passed": bool, "output": str}
  {"type": "correction",        "attempt": int, "errors": str}
  {"type": "done",              "summary": str}
  {"type": "error",             "message": str}
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from contracts.indexing import SearchHit, SearchRequest
from core import providers
from core.errors import (
    FileNotFoundInProjectError,
    InterroAIError,
    PathEscapeError,
    ToolError,
)
from core.index.semantic_index import SemanticIndex
from core.local import cache
from core.models.gateway import LONG_TIMEOUT, ModelGateway
from core.workspace.ast_map import build_repo_map, fingerprint_sources
from core.workspace.hashing import file_hash
from core.workspace.patcher import apply_patch
from core.workspace.sandbox import run_linter, run_tests

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 20
_MAX_CORRECTIONS = 3

# Map display-facing model IDs to real OpenAI API model IDs. They are identical
# for the GPT-5.6 family; the indirection stays because it is also what decides
# whether a caller's choice is valid (`agents/session.py`) and where an alias
# for a renamed model would go. Effort is chosen separately, per request.
_MODEL_MAP: dict[str, str] = {
    "gpt-5.6-sol":   "gpt-5.6-sol",
    "gpt-5.6-terra": "gpt-5.6-terra",
    "gpt-5.6-luna":  "gpt-5.6-luna",
}

# Reasoning models reject `temperature` and take `reasoning_effort` instead.
# Every GPT-5.6 model reasons; the o-series stays listed because a raw OpenAI
# id passes through `_MODEL_MAP` unchanged and must still be handled correctly.
_REASONING_MODELS = {
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini",
}

# Models whose streaming is unreliable, so their plan is fetched in one call.
# Only the o1 family. This used to be gated on "is it a reasoning model", which
# was the same set in practice — but now every model the app offers reasons, and
# that would silence the streamed plan the right-hand panel exists to show.
_NO_STREAM_MODELS = {"o1", "o1-mini", "o1-preview"}

# ── Tool schemas ──────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's content from the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to project root."},
                    "line_start": {"type": "integer", "description": "First line to read (1-indexed). Omit to start from line 1."},  # noqa: E501
                    "line_end": {"type": "integer", "description": "Last line to read (inclusive). Omit to read to end of file."},  # noqa: E501
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file. Use only for brand-new files; use patch_file to edit existing ones.",  # noqa: E501
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Apply a search-and-replace edit to an existing file. "
                "The search_block must match the file verbatim (correct indentation, no approximations). "  # noqa: E501
                "Always read_file first to confirm the exact text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "search_block": {"type": "string", "description": "Exact text to find."},
                    "replace_block": {"type": "string", "description": "Text to replace it with."},
                },
                "required": ["path", "search_block", "replace_block"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_grep",
            "description": "Search the project for a string or regex pattern. Returns matching lines with file:line context. Use when you know the exact name or string.",  # noqa: E501
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Substring or regex to search for."},  # noqa: E501
                    "file_glob": {"type": "string", "description": "Optional glob to limit scope, e.g. '*.py'."},  # noqa: E501
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic",
            "description": (
                "Search the codebase by concept using vector similarity. "
                "Use when you don't know the exact name — e.g. 'authentication middleware', "
                "'database session handling', 'error response formatting'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language description of the code you're looking for."},  # noqa: E501
                    "n": {"type": "integer", "description": "Number of results (default 5, max 10)."},  # noqa: E501
                },
                "required": ["query"],
            },
        },
    },
]

# ── System prompts ────────────────────────────────────────────────────────────

_PLAN_SYSTEM = """\
You are an expert software engineer. You have been given a fully-specified implementation task \
and a Knowledge Tree with project structure and relevant code context.

Your job RIGHT NOW is to produce only a PLAN OF ATTACK — no code yet.

The plan must:
- List every file that will be created or modified.
- Describe what logic will be added or changed in each file.
- Note any new dependencies, imports, or data structures needed.

Format: numbered Markdown list with `code spans` for file paths and symbols. \
Be concise and technical. Do NOT write any code blocks.
"""

_IMPL_SYSTEM = """\
You are an expert software engineer implementing a plan against a real codebase.

You have five tools:
  • read_file       — inspect any file before editing it
  • search_grep     — find code by exact string or regex (use when you know the name)
  • search_semantic — find code by concept using vector search (use when you don't know the exact name)
  • patch_file      — surgically modify an existing file (preferred for all edits)
  • write_file      — create a new file from scratch (only for files that do not yet exist)

Rules:
  1. Always read_file a file before patching it to confirm the exact content.
  2. Use search_semantic when you need to find code by concept or behaviour.
  3. Use search_grep when you know the exact symbol, import, or string.
  4. Use patch_file with verbatim search blocks — correct indentation, no paraphrasing.
  5. Never rewrite an entire existing file with write_file; use patch_file instead.
  6. Implement the plan fully, then stop calling tools and write a **Markdown summary** of \
what was changed: which files were modified, what logic was added, and any caveats. \
Use `code spans` for file paths and symbol names.
"""

_QA_IMPL_SYSTEM = """\
You are an expert software engineer analyzing a codebase.

You have three tools:
  • read_file       — inspect any file
  • search_grep     — find code by exact string or regex (use when you know the name)
  • search_semantic — find code by concept using vector search (use when you don't know the exact name)

Rules:
  1. Use search_semantic when you need to find code by concept or behaviour.
  2. Use search_grep when you know the exact symbol, import, or string.
  3. Explore the codebase using the tools until you have enough information to confidently answer the user's question.
  4. Once you are done investigating, stop calling tools and write a **Markdown summary** that completely answers the user's query.
  5. Format your response cleanly and use `code spans` for file paths and symbol names.
"""

# ── Agent ─────────────────────────────────────────────────────────────────────

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


class CoderAgent:
    def __init__(
        self,
        project_path: str,
        model: str,
        intent: str = "implement",
        history: list[dict] | None = None,
        gateway: ModelGateway | None = None,
        index: SemanticIndex | None = None,
        effort: str | None = None,
    ) -> None:
        self._intent = intent
        # Prior turns of this conversation, spliced in ahead of the current
        # request so a follow-up ("now the other one") has a referent.
        self._history: list[dict] = list(history or [])
        self._path = Path(project_path).resolve()
        self._api_model = _MODEL_MAP.get(model, model)   # display ID → real API ID
        self._is_reasoning = self._api_model in _REASONING_MODELS
        # Only reasoning models take an effort; sending one to the others is a 400.
        self._effort = effort if self._is_reasoning else None
        # Where model calls and semantic searches go. The agent neither knows
        # nor cares whether that is OpenAI and a local index, or the cloud.
        self._gateway = gateway or providers.model_gateway()
        self._index = index or providers.semantic_index()
        self._modified: set[str] = set()   # absolute paths of files written/patched
        logger.info("CoderAgent init: requested=%r  api_model=%r  reasoning=%s  effort=%s",
                    model, self._api_model, self._is_reasoning, self._effort)

    # ── Public entry point ─────────────────────────────────────────────────

    async def execute(self, prompt: str):
        if not self._path.is_dir():
            yield {"type": "error", "message": f"Project directory not found: {self._path}"}
            return

        logger.info("CoderAgent.execute: project=%r  api_model=%r", str(self._path), self._api_model)  # noqa: E501

        try:
            knowledge_tree = await self._build_knowledge_tree()

            if self._intent == "answer":
                read_tools = [t for t in _TOOLS if t["function"]["name"] in {"read_file", "search_grep", "search_semantic"}]  # noqa: E501
                messages = [
                    {"role": "system", "content": _QA_IMPL_SYSTEM},
                    *self._history,
                    {"role": "user", "content": f"{knowledge_tree}\n\nQUESTION:\n{prompt}"},
                ]
                summary = ""
                async for event in self._tool_loop(messages, tools=read_tools):
                    yield event
                    if event["type"] == "impl_done":
                        summary = event["content"]
                yield {"type": "done", "summary": summary}
                return

            # Phase 1 — Planning
            plan = ""
            async for event in self._plan(prompt, knowledge_tree):
                yield event
                if event["type"] == "plan":
                    plan = event["content"]

            # Phase 2 — Implementation (shared message history so Phase 3 can extend it)
            messages: list[dict] = [
                {"role": "system", "content": _IMPL_SYSTEM},
                *self._history,
                {"role": "user", "content": f"{knowledge_tree}\n\nTASK:\n{prompt}\n\nPLAN:\n{plan}"},  # noqa: E501
            ]

            summary = ""
            async for event in self._tool_loop(messages):
                yield event
                if event["type"] == "impl_done":
                    summary = event["content"]

            # Phase 3 — Validation + self-correction (extends Phase 2 history)
            async for event in self._validate(messages):
                yield event

            yield {"type": "done", "summary": summary}

        except InterroAIError as exc:
            # Deliberate, already-explained failure — no stack trace needed.
            logger.info("CoderAgent stopped: %s", exc)
            yield {"type": "error", "message": str(exc)}
        except Exception as exc:
            logger.exception("CoderAgent.execute failed")
            yield {"type": "error", "message": f"Coder agent error: {exc}"}

    # ── Phase 1: Planning ──────────────────────────────────────────────────

    def _build_create_kwargs(self, messages: list[dict], temperature: float = 0.2, **extra) -> dict:
        kwargs: dict = {"model": self._api_model, "messages": messages, **extra}
        # Chat completions refuses `reasoning_effort` together with function
        # tools ("use /v1/responses, or set reasoning_effort to 'none'"). So the
        # effort a display ID asks for reaches the planning call, which is where
        # the thinking happens, and the tool rounds go without it.
        if self._effort and not kwargs.get("tools"):
            kwargs["reasoning_effort"] = self._effort
        if not self._is_reasoning:
            kwargs["temperature"] = temperature
        return kwargs

    async def _plan(self, prompt: str, knowledge_tree: str):
        messages = [
            {"role": "system", "content": _PLAN_SYSTEM},
            *self._history,
            {"role": "user", "content": f"{knowledge_tree}\n\nTASK:\n{prompt}"},
        ]

        if self._api_model in _NO_STREAM_MODELS:
            # Streaming is unreliable here, so take the whole plan in one call.
            response = await self._gateway.chat(
                timeout=LONG_TIMEOUT,
                **self._build_create_kwargs(messages=messages, temperature=0.2),
            )
            content = response.choices[0].message.content or ""
            yield {"type": "plan_chunk", "chunk": content}
            yield {"type": "plan", "content": content}
            return

        # Streaming: emit tokens as they arrive so the right panel fills live.
        # Only the handshake is retried; see ModelGateway.chat_stream.
        full = ""
        stream = await self._gateway.chat_stream(
            timeout=LONG_TIMEOUT,
            **self._build_create_kwargs(messages=messages, temperature=0.2),
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full += token
                yield {"type": "plan_chunk", "chunk": token}
        yield {"type": "plan", "content": full}

    # ── Phase 2: Tool-calling loop ─────────────────────────────────────────

    async def _tool_loop(self, messages: list[dict], tools: list[dict] | None = None):
        if tools is None:
            tools = _TOOLS
        for _ in range(_MAX_TOOL_ROUNDS):
            response = await self._gateway.chat(
                timeout=LONG_TIMEOUT,
                **self._build_create_kwargs(
                    messages=messages,
                    temperature=0.1,
                    tools=tools,
                    tool_choice="auto",
                ),
            )
            msg = response.choices[0].message

            a_msg: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                a_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(a_msg)

            if not msg.tool_calls:
                yield {"type": "impl_done", "content": msg.content or ""}
                return

            for tc in msg.tool_calls:
                fn = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                yield {"type": "tool_call", "tool": fn, "args": args}
                result = await self._run_tool(fn, args)
                yield {"type": "tool_result", "tool": fn, "output": result[:1500]}

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        yield {"type": "impl_done", "content": "Implementation complete (tool round limit reached)."}  # noqa: E501

    # ── Phase 3: Validation + self-correction ──────────────────────────────

    async def _validate(self, messages: list[dict]):
        modified = sorted(self._modified)

        if not modified:
            yield {
                "type": "validation_result", "phase": "lint",
                "passed": True, "status": "skipped", "output": "No files modified.",
            }
            return

        for attempt in range(_MAX_CORRECTIONS):
            lint = await run_linter(modified, str(self._path))
            yield {
                "type": "validation_result", "phase": "lint",
                "passed": lint.ok, "status": lint.status.value, "output": lint.output,
            }

            test = await run_tests(str(self._path))
            yield {
                "type": "validation_result", "phase": "test",
                "passed": test.ok, "status": test.status.value, "output": test.output,
            }

            if lint.ok and test.ok:
                return

            if attempt >= _MAX_CORRECTIONS - 1:
                break

            errors = ""
            if not lint.ok:
                errors += f"LINTER:\n{lint.output}\n\n"
            if not test.ok:
                errors += f"TESTS:\n{test.output}"

            yield {"type": "correction", "attempt": attempt + 1, "errors": errors}

            messages.append({"role": "user", "content": f"Validation failed — fix the errors:\n\n{errors}"})  # noqa: E501
            async for event in self._tool_loop(messages):
                yield event

    # ── Tool execution ─────────────────────────────────────────────────────

    async def _run_tool(self, name: str, args: dict) -> str:
        """
        Dispatch one tool call and return the result as text for the model.

        Three outcomes, deliberately kept apart:
          * `ToolError`      — the model's fault and the model's to fix. Hand the
                               message straight back so it can retry correctly.
          * `InterroAIError` — terminal for the whole run (no API key). Re-raise;
                               letting the model "retry" a missing key wastes
                               tool rounds and hides the real cause.
          * anything else    — a bug in InterroAI. Log the traceback, then tell
                               the model plainly rather than leaking internals
                               into the transcript as if they were its mistake.
        """
        try:
            if name == "read_file":
                return self._read_file(
                    args.get("path", ""),
                    args.get("line_start"),
                    args.get("line_end"),
                )
            if name == "write_file":
                return self._write_file(args.get("path", ""), args.get("content", ""))
            if name == "patch_file":
                return self._patch_file(
                    args.get("path", ""),
                    args.get("search_block", ""),
                    args.get("replace_block", ""),
                )
            if name == "search_grep":
                return self._search_grep(args.get("pattern", ""), args.get("file_glob"))
            if name == "search_semantic":
                return await self._search_semantic(args.get("query", ""), args.get("n", 5))
            return f"Unknown tool: {name}"
        except ToolError as exc:
            logger.info("Tool %r rejected the model's request: %s", name, exc)
            return f"Error: {exc}"
        except InterroAIError:
            raise
        except Exception:
            logger.exception("Unexpected failure in tool %r with args %r", name, args)
            return (
                f"Internal error while running {name!r}. This is a defect in "
                "InterroAI, not a problem with your arguments — try a different "
                "approach."
            )

    async def _search_semantic(self, query: str, n: int = 5) -> str:
        n = min(max(1, int(n)), 10)
        hits = await self._index.search(
            SearchRequest(project_id=str(self._path), query=query, n=n)
        )
        return self._format_hits(hits)

    def _format_hits(self, hits: list[SearchHit]) -> str:
        """
        Turn index hits into code the model can read — read from *this* disk.

        The index says where to look and what the file hashed to when it was
        indexed; the text comes from the working tree. When the hash no longer
        matches, the file has been edited since and the lines may have shifted.
        The hit is still shown, since it is usually close, but labelled so the
        model reads the file before trusting a line number.

        A hit whose file is gone, whose lines no longer exist, or whose path
        points outside the project is dropped: there is nothing true to show,
        and an index is not trusted to name paths the agent may read.
        """
        files: dict[Path, tuple[list[str], str]] = {}
        parts: list[str] = []

        for hit in hits:
            try:
                path = self._resolve(hit.file_path)
            except PathEscapeError:
                logger.warning("Dropping a search hit outside the project: %r", hit.file_path)
                continue

            if path not in files:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                files[path] = (text.splitlines(keepends=True), file_hash(path))
            lines, current_hash = files[path]

            excerpt = "".join(lines[hit.start_line - 1 : hit.end_line])
            if not excerpt.strip():
                continue

            header = f"// {hit.file_path}  (lines {hit.start_line}–{hit.end_line})"
            if current_hash != hit.file_hash:
                header += "  [stale: changed since indexing, lines may have moved — read_file first]"  # noqa: E501
            parts.append(f"{header}\n{excerpt}")

        return "\n\n---\n\n".join(parts) if parts else "No semantically similar code found."

    def _resolve(self, path: str) -> Path:
        resolved = (self._path / path).resolve()
        # is_relative_to is symlink-aware and exact; str.startswith would
        # accept "/Users/alex2/..." as a child of "/Users/alex" and trip on
        # trailing-slash edges.
        if resolved != self._path and not resolved.is_relative_to(self._path):
            raise PathEscapeError(f"Path '{path}' escapes the project directory.")
        return resolved

    def _read_file(self, path: str, line_start: int | None, line_end: int | None) -> str:
        p = self._resolve(path)
        if not p.exists():
            logger.info("read_file: not found | path=%r | resolved=%r", path, str(p))
            raise FileNotFoundInProjectError(f"File not found: {path}")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        if line_start is not None or line_end is not None:
            s = (line_start or 1) - 1
            e = line_end or len(lines)
            lines = lines[s:e]
        content = "".join(lines)
        if len(content) > 12_000:
            content = content[:12_000] + "\n... (truncated — use line_start/line_end to read more)"
        return content

    def _write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self._modified.add(str(p))
        return f"Written: {path}"

    def _patch_file(self, path: str, search_block: str, replace_block: str) -> str:
        p = self._resolve(path)
        # Raises PatchError / FileNotFoundInProjectError, caught by _run_tool.
        apply_patch(str(p), search_block, replace_block)
        self._modified.add(str(p))
        return f"Patched: {path}"

    def _search_grep(self, pattern: str, file_glob: str | None) -> str:
        try:
            re_pat = re.compile(pattern)
        except re.error:
            re_pat = re.compile(re.escape(pattern))

        results: list[str] = []
        for f in sorted(self._path.rglob(file_glob or "*")):
            if any(p in _SKIP_DIRS for p in f.parts) or not f.is_file():
                continue
            if f.stat().st_size > 500_000:
                continue
            try:
                for i, line in enumerate(
                    f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if re_pat.search(line):
                        results.append(f"{f.relative_to(self._path)}:{i}: {line.rstrip()}")
                        if len(results) >= 50:
                            results.append("... (capped at 50 matches)")
                            return "\n".join(results)
            except OSError:
                continue
        return "\n".join(results) if results else "No matches found."

    # ── Knowledge Tree builder ─────────────────────────────────────────────

    async def _build_knowledge_tree(self) -> str:
        import asyncio

        # The agent needs this on every request, and parsing every source file
        # again to produce the identical map is the most expensive thing it
        # does before the first token. Stat the files instead (cheap), and
        # reuse the cached map while they are untouched. The agent's own writes
        # move an mtime, so the fingerprint invalidates itself.
        path = str(self._path)
        fingerprint = await asyncio.to_thread(fingerprint_sources, path)
        repo_map = await cache.get_repo_map(path, fingerprint)

        if repo_map is None:
            # Run the synchronous AST walk in a thread so it doesn't block the event loop
            repo_map = await asyncio.to_thread(build_repo_map, path)
            await cache.set_repo_map(path, fingerprint, repo_map)

        return (
            "<knowledge_tree>\n\n"
            "<system_instructions>\n"
            "Follow the existing code style and patterns. "
            "Use async/await where the codebase does. "
            "Maintain strict type hints. "
            "Apply the minimal change that satisfies the task.\n"
            "</system_instructions>\n\n"
            f"<repo_map>\n{repo_map or '(empty)'}\n</repo_map>\n\n"
            "</knowledge_tree>"
        )
