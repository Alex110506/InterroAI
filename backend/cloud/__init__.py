"""
The cloud side of InterroAI: the Web API and the indexing worker.

Nothing in this package runs on the user's machine. It shares `contracts/` and
two port-side modules, `core.errors` and `core.models.gateway`, with the
runtime, and never imports the runtime's own modules (`agents/`, `api/`,
`core/workspace/`, `core/remote/`); `tests/test_boundaries.py` enforces both
directions.

`api/` is the app shell, `db/` the shared schema and scopes, and each other
folder one feature: `auth/`, `projects/`, `llm/`, `indexing/` and `worker/`.
"""
