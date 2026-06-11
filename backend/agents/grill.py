"""
Grill Agent — Proactive requirement clarification (Section 4 of architecture spec).

Drives an iterative interrogation session using gpt-4o-mini with a three-level
hybrid context:
  1. Project file tree   — structural understanding of the workspace.
  2. Git context         — current branch and recently modified files.
  3. Local RAG           — relevant code chunks retrieved from ChromaDB.

Each turn returns strict JSON:
  { "is_prompt_ready": false, "question": "..." }
  { "is_prompt_ready": true,  "refined_prompt": "..." }

Stopping criteria (whichever comes first):
  - The model sets is_prompt_ready = true.
  - _MAX_TURNS clarifying questions have been asked (then we force-summarise).
"""
from __future__ import annotations

import json

from openai import AsyncOpenAI

from core.embeddings import embed_texts
from core.security import retrieve_openai_key
from core.vector_store import search_chunks

_MODEL = "gpt-4o-mini"
_MAX_TURNS = 3          # cap on clarifying questions before forcing ready
_RAG_RESULTS = 8        # chunks to retrieve per query
_TREE_MAX_DEPTH = 4     # how many levels of the file tree to render

_SYSTEM_PROMPT = """\
You are an expert software engineer clarifying the exact requirements for an \
implementation task. The user has already been confirmed to want to build, \
modify, fix, or create something — your only job is to fill in the missing \
technical details so the task can be implemented precisely.

You are given three layers of project context:
  1. PROJECT STRUCTURE — file tree of the workspace.
  2. GIT CONTEXT — current branch and recently modified files.
  3. RELEVANT CODE — semantically related code chunks from a local vector DB.

━━━ STOPPING CRITERIA (set is_prompt_ready = true) ━━━
  • You know exactly which files / modules need to change.
  • You understand the desired behaviour and expected output.
  • You know the relevant dependencies, APIs, or data structures involved.
  • The task is already fully specified — nothing important is ambiguous.
  • You have asked """ + str(_MAX_TURNS) + """ questions (MUST stop regardless).

━━━ QUESTION RULES ━━━
  • ONE short, direct question per response — no preamble, no bullet lists.
  • Ask about WHAT or WHY, not HOW (you infer the HOW from the code context).
  • Never ask about anything already visible in the file tree or code context.

━━━ refined_prompt ━━━
  A complete technical specification: which files, what behaviour, what \
  constraints, what expected output — enough for a developer to implement \
  without any further clarification.

━━━ RESPONSE FORMAT — always valid JSON, nothing else ━━━

While clarifying:
{"is_prompt_ready": false, "question": "<single focused question>"}

When ready:
{"is_prompt_ready": true, "refined_prompt": "<complete technical specification>"}
"""


# ── Context formatters ────────────────────────────────────────────────────────

def _fmt_tree(node: dict, depth: int = 0) -> str:
    if depth > _TREE_MAX_DEPTH or not node:
        return ""
    indent = "  " * depth
    if node.get("kind") == "file":
        return f"{indent}{node['name']}\n"
    lines = [f"{indent}{node.get('name', '?')}/\n"]
    for child in node.get("children") or []:
        lines.append(_fmt_tree(child, depth + 1))
    return "".join(lines)


def _fmt_git(git: dict) -> str:
    if not git or not git.get("is_git_repo"):
        return "Not a git repository."
    parts: list[str] = []
    if git.get("branch"):
        parts.append(f"Branch: {git['branch']}")
    if git.get("modified_files"):
        parts.append("Modified:\n" + "\n".join(f"  {f}" for f in git["modified_files"]))
    if git.get("recent_commits"):
        parts.append("Recent commits:\n" + "\n".join(f"  {c}" for c in git["recent_commits"]))
    return "\n".join(parts) or "Clean working tree."


def _fmt_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant code found in the vector store."
    parts = []
    for c in chunks:
        header = f"// {c['file_path']}  (lines {c['start_line']}–{c['end_line']})"
        parts.append(f"{header}\n{c['content']}")
    return "\n\n---\n\n".join(parts)


# ── Agent class ───────────────────────────────────────────────────────────────

class GrillAgent:
    """
    Stateful interrogation session for a single user request.

    Usage:
        agent = GrillAgent(project_path, project_index)
        result = await agent.start(initial_prompt)
        # result: {"is_prompt_ready": bool, "question"/"refined_prompt": ...}

        while not result["is_prompt_ready"]:
            result = await agent.answer(user_answer)
    """

    def __init__(self, project_path: str, project_index: dict) -> None:
        self._path = project_path
        self._index = project_index or {}
        self._history: list[dict] = []
        self._turns = 0

    # ── Context builder ───────────────────────────────────────────────────────

    async def _build_context(self, query: str) -> str:
        tree_str = _fmt_tree(self._index.get("file_tree") or {})
        git_str = _fmt_git(self._index.get("git_context") or {})

        try:
            embeddings = await embed_texts([query])
            chunks = search_chunks(self._path, embeddings[0], n=_RAG_RESULTS)
            rag_str = _fmt_chunks(chunks)
        except Exception:
            rag_str = "Vector search unavailable (project may not be fully indexed yet)."

        return (
            f"PROJECT STRUCTURE:\n{tree_str or '(empty)'}\n\n"
            f"GIT CONTEXT:\n{git_str}\n\n"
            f"RELEVANT CODE:\n{rag_str}"
        )

    # ── OpenAI call ───────────────────────────────────────────────────────────

    async def _call(self) -> dict:
        key = retrieve_openai_key()
        if not key:
            raise ValueError("No OpenAI API key configured — add yours in Settings.")

        client = AsyncOpenAI(api_key=key)
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                *self._history,
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        raw = response.choices[0].message.content
        self._history.append({"role": "assistant", "content": raw})

        result = json.loads(raw)
        if not result.get("is_prompt_ready", False):
            self._turns += 1
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self, user_prompt: str) -> dict:
        """Begin the session with the user's initial prompt."""
        context = await self._build_context(user_prompt)
        self._history = [
            {"role": "user", "content": f"{context}\n\nUSER REQUEST:\n{user_prompt}"}
        ]
        return await self._call()

    async def answer(self, user_answer: str) -> dict:
        """Feed the user's answer to the last clarifying question."""
        if self._turns >= _MAX_TURNS:
            return await self._force_ready()
        self._history.append({"role": "user", "content": user_answer})
        return await self._call()

    async def _force_ready(self) -> dict:
        """Force the model to consolidate everything into a refined_prompt."""
        self._history.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of clarifying questions. "
                "Consolidate all the information gathered so far and produce the "
                "final refined_prompt now. "
                'Respond with {"is_prompt_ready": true, "refined_prompt": "..."}'
            ),
        })
        return await self._call()
