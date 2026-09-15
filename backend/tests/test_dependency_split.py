"""
The cloud processes run without the desktop runtime's dependencies.

The images install `.[cloud]` only (backend/Dockerfile). If cloud code started
importing the local index, the chunker or the Redis cache, the image would fail
at startup in Azure. Here it fails first: the cloud entry points are imported in
a fresh interpreter in which those packages cannot be found.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
RUNTIME_ONLY = ("chromadb", "langchain_text_splitters", "pathspec", "redis")


def _import_without_runtime_packages(statement: str) -> subprocess.CompletedProcess:
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        class NotInTheCloudImage(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] in {RUNTIME_ONLY!r}:
                    message = f"No module named {{name!r}} (not in the cloud image)"
                    raise ModuleNotFoundError(message)
                return None

        sys.meta_path.insert(0, NotInTheCloudImage())
        {statement}
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script], cwd=BACKEND, capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize(
    "statement",
    [
        "import cloud.api.main; cloud.api.main.create_app()",
        "import cloud.worker.main",
        "import cloud.api.services",
    ],
)
def test_a_cloud_entry_point_needs_nothing_from_the_runtime_extra(statement):
    result = _import_without_runtime_packages(statement)
    assert result.returncode == 0, result.stderr


def test_the_guard_really_hides_the_runtime_packages():
    """Without this, a blocker that blocked nothing would pass the test above vacuously."""
    result = _import_without_runtime_packages("import core.index.adapters.chroma")

    assert result.returncode != 0
    assert "not in the cloud image" in result.stderr
