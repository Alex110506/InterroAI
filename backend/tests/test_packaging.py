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


#: Folders of Python files that are not packages: Alembic loads its scripts by path.
_NOT_PACKAGES = (("cloud", "migrations"),)


def _packages_on_disk() -> set[str]:
    """
    The dotted name of every folder of Python files under a top-level package.

    Counted by its modules rather than its `__init__.py`, so a folder created
    without one, which Python still imports as a namespace package, shows up
    here, and fails the comparison until it is declared.
    """
    roots = [
        path for path in BACKEND.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "__init__.py").exists()
    ]
    folders = {
        module.parent.relative_to(BACKEND).parts
        for root in roots
        for module in root.rglob("*.py")
    }
    return {
        ".".join(parts)
        for parts in folders
        if not any(parts[: len(excluded)] == excluded for excluded in _NOT_PACKAGES)
    }


def test_every_package_is_declared():
    """A package missing here is silently left out of a non-editable install."""
    pyproject = tomllib.loads((BACKEND / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(pyproject["tool"]["setuptools"]["packages"])
    assert _packages_on_disk() == declared
