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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

from core import cache  # noqa: E402
from core.ast_map import build_repo_map, fingerprint_sources  # noqa: E402
from core.embeddings import embed_texts  # noqa: E402
from core.errors import (  # noqa: E402
    FileNotFoundInProjectError,
    InterroAIError,
    PathEscapeError,
    ToolError,
)
from core.llm import LONG_TIMEOUT, chat_completion, chat_stream, get_client  # noqa: E402
from core.patcher import apply_patch  # noqa: E402
from core.sandbox import run_linter, run_tests  # noqa: E402
from core.vector_store import search_chunks  # noqa: E402

_MAX_TOOL_ROUNDS = 20
_MAX_CORRECTIONS = 3

# Map display-facing model IDs to real OpenAI API model IDs
_MODEL_MAP: dict[str, str] = {
    "gpt-5.4-mini":        "gpt-5.4-mini",
    "gpt-5.4-low-effort":  "gpt-5.4",
    "gpt-5.4-high-effort": "gpt-5.4",
    "gpt-5.5-low-effort":  "gpt-5.5",
    "gpt-5.5-high-effort": "gpt-5.5",
}

# Reasoning models require temperature to be omitted
# gpt-5.4 and gpt-5.5 are high-reasoning models — omit temperature to avoid API errors
_REASONING_MODELS = {"gpt-5.5", "gpt-5.4", "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"}  # noqa: E501

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
    ) -> None:
        self._intent = intent
        # Prior turns of this conversation, spliced in ahead of the current
        # request so a follow-up ("now the other one") has a referent.
        self._history: list[dict] = list(history or [])
        self._path = Path(project_path).resolve()
        self._api_model = _MODEL_MAP.get(model, model)   # resolve display ID → real API ID
        self._is_reasoning = self._api_model in _REASONING_MODELS
        self._client: AsyncOpenAI | None = None
        self._modified: set[str] = set()   # absolute paths of files written/patched
        logger.info("CoderAgent init: requested=%r  api_model=%r  reasoning=%s",
                    model, self._api_model, self._is_reasoning)

    # ── Public entry point ─────────────────────────────────────────────────

    async def execute(self, prompt: str):
        if not self._path.is_dir():
            yield {"type": "error", "message": f"Project directory not found: {self._path}"}
            return

        logger.info("CoderAgent.execute: project=%r  api_model=%r", str(self._path), self._api_model)  # noqa: E501

        try:
            self._client = get_client(LONG_TIMEOUT)
        except InterroAIError as exc:
            # Expected and actionable (e.g. no API key) — show it as-is.
            yield {"type": "error", "message": str(exc)}
            return

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
        if not self._is_reasoning:
            kwargs["temperature"] = temperature
        return kwargs

    async def _plan(self, prompt: str, knowledge_tree: str):
        messages = [
            {"role": "system", "content": _PLAN_SYSTEM},
            *self._history,
            {"role": "user", "content": f"{knowledge_tree}\n\nTASK:\n{prompt}"},
        ]

        if self._is_reasoning:
            # o1-family doesn't stream reliably — single-shot call
            response = await chat_completion(
                self._client,
                **self._build_create_kwargs(messages=messages, temperature=0.2),
            )
            content = response.choices[0].message.content or ""
            yield {"type": "plan_chunk", "chunk": content}
            yield {"type": "plan", "content": content}
            return

        # Streaming: emit tokens as they arrive so the right panel fills live.
        # Only the handshake is retried; see core.llm.chat_stream.
        full = ""
        stream = await chat_stream(
            self._client,
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
            response = await chat_completion(
                self._client,
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
        embeddings = await embed_texts([query])
        chunks = search_chunks(str(self._path), embeddings[0], n=n)
        return _fmt_chunks(chunks) if chunks else "No semantically similar code found."

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


def _fmt_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant code found in the vector store."
    parts = []
    for c in chunks:
        header = f"// {c['file_path']}  (lines {c['start_line']}–{c['end_line']})"
        parts.append(f"{header}\n{c['content']}")
    return "\n\n---\n\n".join(parts)
