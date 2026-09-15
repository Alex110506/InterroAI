"""
The cloud side of InterroAI: the Web API and the indexing worker.

Nothing in this package runs on the user's machine. It shares `contracts/` and
the index logic in `core/index/` with the runtime, and never imports the
runtime's own modules (`agents/`, `api/`, `core/workspace/`, `core/remote/`);
`tests/test_boundaries.py` enforces both directions.
"""
