"""
The decoupling, enforced.

Each rule below is a sentence from the architecture, checked against the real
imports (parsed with `ast`, never executed):

  * the runtime — `agents/`, `api/`, `core/workspace/` — reaches models and the
    index only through their ports, and never names a concrete implementation;
  * the indexing logic the cloud runs — `ports`, `indexer`, `embeddings` and
    `manifest_diff` in `cloud/indexing/` — is written against its ports and
    names no adapter or infrastructure SDK, and the worker's half never
    touches the filesystem at all;
  * `core/models/` and `core/local/` are plumbing that no feature leaks into;
  * `contracts/` depends on nothing else in the backend;
  * the runtime and the cloud side never import each other;
  * only the app shell (`cloud/api/`) and each feature's `routes.py` speak
    HTTP: nothing else in `cloud/` imports the web framework or the shell;
  * only the app factory imports a feature's routes, so features never reach
    into each other's endpoints.

The cloud rules are stated per file, over every module under `cloud/` at any
depth, so a folder added or moved later is covered without editing a list.

They are tests rather than code-review conventions, so breaking one fails CI
instead of waiting for a reviewer to notice.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent


def _modules(*folders: str) -> list[str]:
    """Every module directly inside *folders*, as a backend-relative path."""
    return sorted(
        str(path.relative_to(BACKEND))
        for folder in folders
        for path in (BACKEND / folder).glob("*.py")
        if path.name != "__init__.py"
    )


def _cloud_modules() -> list[str]:
    """Every module under `cloud/`, at any depth, except the Alembic migrations."""
    return sorted(
        str(path.relative_to(BACKEND))
        for path in (BACKEND / "cloud").rglob("*.py")
        if path.name != "__init__.py" and "migrations" not in path.parts
    )


def _is_http(relative: str) -> bool:
    """The app shell, and each feature's routes: the only cloud modules that speak HTTP."""
    return relative.startswith("cloud/api/") or relative.endswith("/routes.py")


def _imports(relative: str) -> set[str]:
    """Every module *relative* imports, including `from package import name`."""
    tree = ast.parse((BACKEND / relative).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _violations(relative: str, forbidden: tuple[str, ...]) -> list[str]:
    return sorted(
        name
        for name in _imports(relative)
        if any(name == banned or name.startswith(f"{banned}.") for banned in forbidden)
    )


#: What the ports exist to hide on this side: the model SDK, and the cloud
#: clients that implement the ports, which only `core/providers.py` may name.
#: Everything further behind them is in `cloud/`, which the runtime never imports.
_BEHIND_THE_PORTS = ("openai", "core.remote")

_FEATURES = ("core.workspace", "core.index", "agents", "api")


@pytest.mark.parametrize("module", _modules("agents", "api", "core/workspace"))
def test_the_runtime_reaches_models_and_the_index_only_through_ports(module):
    assert _violations(module, _BEHIND_THE_PORTS) == []


#: The indexing logic the worker runs. It is written against `ports.py` alone;
#: the adapters and repositories beside it are handed in by its callers.
_INDEXING_LOGIC = ("ports.py", "indexer.py", "embeddings.py", "manifest_diff.py")
_INDEXING_LOGIC_MAY_USE = (
    "cloud.indexing.ports", "cloud.indexing.embeddings", "cloud.llm.openai_client",
)
_INFRASTRUCTURE = ("sqlalchemy", "asyncpg", "azure", "fastapi", "starlette")


@pytest.mark.parametrize("module", [f"cloud/indexing/{name}" for name in _INDEXING_LOGIC])
def test_the_indexing_logic_names_no_adapter(module):
    stray = sorted(
        name for name in _imports(module)
        if name.startswith("cloud.") and not name.startswith(_INDEXING_LOGIC_MAY_USE)
    )
    assert stray == []
    assert _violations(module, _INFRASTRUCTURE) == []


def test_the_worker_never_touches_the_filesystem():
    """It has to run on a machine that has never seen the repository."""
    assert _violations("cloud/indexing/indexer.py", ("os", "pathlib")) == []


@pytest.mark.parametrize("module", _modules("core/models", "core/local"))
def test_plumbing_never_depends_on_a_feature(module):
    assert _violations(module, _FEATURES) == []


@pytest.mark.parametrize("module", _modules("core/remote"))
def test_the_cloud_clients_stay_clients(module):
    """They implement the ports: features call them, never the reverse."""
    assert _violations(module, ("core.workspace", "agents", "api")) == []


@pytest.mark.parametrize("module", _modules("contracts"))
def test_the_contracts_depend_on_nothing_else_in_the_backend(module):
    assert _violations(module, ("core", "agents", "api", "cloud", "config", "main")) == []


@pytest.mark.parametrize(
    "module",
    _modules(
        "agents", "api", "core", "core/models", "core/workspace",
        "core/index", "core/local", "core/remote",
    ),
)
def test_the_runtime_never_imports_the_cloud_side(module):
    """The runtime ships to users' machines; the cloud code and its SDKs do not."""
    assert _violations(module, ("cloud",)) == []


@pytest.mark.parametrize("module", _cloud_modules())
def test_the_cloud_side_never_imports_the_runtime(module):
    """There is no workspace, agent or local transport on a server to reach for."""
    assert _violations(module, ("agents", "api", "core.workspace", "core.remote")) == []


@pytest.mark.parametrize("module", [m for m in _cloud_modules() if not _is_http(m)])
def test_only_the_shell_and_routes_speak_http(module):
    """A feature's logic and storage serve its routes, never the reverse."""
    assert _violations(module, ("fastapi", "starlette", "cloud.api")) == []


@pytest.mark.parametrize("module", [m for m in _cloud_modules() if m != "cloud/api/main.py"])
def test_only_the_app_factory_imports_a_features_routes(module):
    """Features meet through `Services` and their own modules, never each other's endpoints."""
    routes = sorted(
        name for name in _imports(module)
        if name.startswith("cloud.") and "routes" in name.split(".")
    )
    assert routes == []


def test_the_guard_can_see_imports_at_all():
    """A parser — or a glob — that found nothing would pass every rule vacuously."""
    assert "core.index.semantic_index" in _imports("core/workspace/project_index.py")
    assert "core.providers" in _imports("agents/coder.py")

    folders = (
        "agents", "api", "contracts", "core/workspace",
        "core/index", "core/models", "core/local",
        "cloud/indexing/adapters", "cloud/api", "cloud/auth", "cloud/db",
        "cloud/indexing", "cloud/llm", "cloud/projects", "cloud/worker",
    )
    assert all(_modules(folder) for folder in folders)

    # The cloud rules walk every depth, and never into the migrations.
    assert "cloud/indexing/indexer.py" in _cloud_modules()
    assert not any(module.startswith("cloud/migrations/") for module in _cloud_modules())
