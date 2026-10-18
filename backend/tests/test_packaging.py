"""
Every package on disk is declared in `pyproject.toml`.

The layout is flat, so setuptools is handed an explicit `packages` list rather
than discovering one. A package missing from that list is not an error: an
editable install and the test suite still find it, and a non-editable install
silently leaves it out. This test is where that omission fails first.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def _packages_on_disk() -> set[str]:
    """The dotted name of every package under a top-level package of `backend/`."""
    roots = [
        path for path in BACKEND.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "__init__.py").exists()
    ]
    return {
        ".".join(init.parent.relative_to(BACKEND).parts)
        for root in roots
        for init in root.rglob("__init__.py")
    }


def test_every_package_is_declared():
    """A package missing here is silently left out of a non-editable install."""
    pyproject = tomllib.loads((BACKEND / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(pyproject["tool"]["setuptools"]["packages"])
    assert _packages_on_disk() == declared
