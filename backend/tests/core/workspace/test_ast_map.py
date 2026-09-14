"""
The repo map is the Coder Agent's entire up-front view of the codebase, so the
signatures it emits have to be faithful and the file cap has to hold.
"""
from __future__ import annotations

import core.workspace.ast_map as ast_map
from core.workspace.ast_map import build_repo_map, fingerprint_sources

# ── Python extraction ────────────────────────────────────────────────────────


def test_top_level_functions_and_classes_are_listed(tmp_project):
    out = build_repo_map(str(tmp_project))
    assert "class Greeter:" in out
    assert "def greet(self, name: str) -> str: ..." in out
    assert "async def run(count: int) -> None: ..." in out


def test_default_values_are_omitted_from_signatures(tmp_path):
    """
    Known limitation, pinned so a future change is a deliberate one: `_py_args`
    renders names and annotations but drops defaults, so `count: int = 1`
    appears as `count: int`.
    """
    (tmp_path / "a.py").write_text("def f(limit: int = 10): pass\n", encoding="utf-8")
    out = build_repo_map(str(tmp_path))
    assert "def f(limit: int): ..." in out
    assert "= 10" not in out


def test_methods_are_indented_under_their_class(tmp_project):
    out = build_repo_map(str(tmp_project))
    assert "    def greet(" in out


def test_async_functions_keep_the_async_keyword(tmp_path):
    (tmp_path / "a.py").write_text("async def go(): pass\n", encoding="utf-8")
    assert "async def go()" in build_repo_map(str(tmp_path))


def test_argument_forms_are_rendered(tmp_path):
    (tmp_path / "a.py").write_text(
        "def f(pos, /, normal: int, *args, kw: str = 'x', **kwargs) -> bool:\n    return True\n",
        encoding="utf-8",
    )
    out = build_repo_map(str(tmp_path))
    for fragment in ["pos", "/", "normal: int", "*args", "kw: str", "**kwargs", "-> bool"]:
        assert fragment in out


def test_bodies_are_omitted(tmp_path):
    """The map is signatures only — including bodies would blow the context."""
    (tmp_path / "a.py").write_text(
        "def f():\n    secret_token = 'do-not-include'\n    return 1\n", encoding="utf-8"
    )
    assert "secret_token" not in build_repo_map(str(tmp_path))


def test_nested_functions_are_not_hoisted(tmp_path):
    (tmp_path / "a.py").write_text(
        "def outer():\n    def inner():\n        pass\n", encoding="utf-8"
    )
    out = build_repo_map(str(tmp_path))
    assert "def outer()" in out
    assert "inner" not in out


def test_a_syntax_error_skips_only_that_file(tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n", encoding="utf-8")
    (tmp_path / "fine.py").write_text("def good(): pass\n", encoding="utf-8")
    out = build_repo_map(str(tmp_path))
    assert "def good()" in out
    assert "broken.py" not in out


def test_a_file_with_no_definitions_is_omitted(tmp_path):
    (tmp_path / "consts.py").write_text("A = 1\nB = 2\n", encoding="utf-8")
    assert build_repo_map(str(tmp_path)) == ""


# ── JavaScript / TypeScript extraction ───────────────────────────────────────


def test_js_classes_and_functions_are_detected(tmp_project):
    out = build_repo_map(str(tmp_project))
    assert "class Widget" in out
    assert "function mount()" in out


def test_js_arrow_consts_are_detected(tmp_project):
    assert "function handler()" in build_repo_map(str(tmp_project))


def test_typescript_files_are_mapped(tmp_path):
    (tmp_path / "a.ts").write_text("export function typed(): void {}\n", encoding="utf-8")
    assert "function typed()" in build_repo_map(str(tmp_path))


# ── Traversal rules ──────────────────────────────────────────────────────────


def test_vendor_directories_are_skipped(tmp_project):
    """node_modules would swamp the map and the file budget."""
    assert "node_modules" not in build_repo_map(str(tmp_project))


def test_non_source_files_are_ignored(tmp_project):
    out = build_repo_map(str(tmp_project))
    assert "README.md" not in out


def test_paths_are_relative_to_the_project_root(tmp_project):
    assert "## utils/helpers.py" in build_repo_map(str(tmp_project))


def test_missing_directory_returns_empty(tmp_path):
    assert build_repo_map(str(tmp_path / "nope")) == ""


def test_a_file_path_instead_of_a_directory_returns_empty(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("def f(): pass\n", encoding="utf-8")
    assert build_repo_map(str(path)) == ""


def test_file_budget_is_enforced_and_announced(tmp_path, monkeypatch):
    """
    Known limitation worth pinning: the cap applies to an alphabetically sorted
    walk, so it keeps `a*` and drops `z*` rather than choosing by relevance.
    """
    monkeypatch.setattr(ast_map, "_MAX_FILES", 3)
    for i in range(10):
        (tmp_path / f"mod_{i:02d}.py").write_text(f"def f{i}(): pass\n", encoding="utf-8")

    out = build_repo_map(str(tmp_path))
    assert "truncated" in out
    assert "def f0()" in out
    assert "def f9()" not in out


# ── Fingerprint (what makes the map cacheable) ───────────────────────────────


def test_the_fingerprint_is_stable_for_an_untouched_project(tmp_project):
    assert fingerprint_sources(str(tmp_project)) == fingerprint_sources(str(tmp_project))


def test_editing_a_source_file_changes_the_fingerprint(tmp_project):
    before = fingerprint_sources(str(tmp_project))
    (tmp_project / "main.py").write_text("def different(): pass\n", encoding="utf-8")
    assert fingerprint_sources(str(tmp_project)) != before


def test_adding_a_source_file_changes_the_fingerprint(tmp_project):
    before = fingerprint_sources(str(tmp_project))
    (tmp_project / "extra.py").write_text("def added(): pass\n", encoding="utf-8")
    assert fingerprint_sources(str(tmp_project)) != before


def test_deleting_a_source_file_changes_the_fingerprint(tmp_project):
    before = fingerprint_sources(str(tmp_project))
    (tmp_project / "utils" / "helpers.py").unlink()
    assert fingerprint_sources(str(tmp_project)) != before


def test_an_unparsed_file_type_does_not_change_the_fingerprint(tmp_project):
    """
    It must track exactly what the map reads. A README cannot change the map,
    so it must not invalidate a cached one either.
    """
    before = fingerprint_sources(str(tmp_project))
    (tmp_project / "NOTES.md").write_text("# prose nobody parses\n", encoding="utf-8")
    assert fingerprint_sources(str(tmp_project)) == before


def test_a_skipped_directory_does_not_change_the_fingerprint(tmp_project):
    (tmp_project / "node_modules" / "pkg" / "new.js").write_text("export function n(){}\n")
    before = fingerprint_sources(str(tmp_project))
    (tmp_project / "node_modules" / "pkg" / "another.js").write_text("export function a(){}\n")
    assert fingerprint_sources(str(tmp_project)) == before


def test_an_unusable_path_has_no_fingerprint(tmp_path):
    """Matches `build_repo_map`, so nothing is cached against a meaningful key."""
    assert fingerprint_sources(str(tmp_path / "nope")) == ""
