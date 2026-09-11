"""
Project indexing — the file walk, git context and the Phase 1 HTTP endpoint.

The walk rules are what keep a 200k-file `node_modules` out of the index, so
each exclusion gets its own test.
"""
from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

from core.project_index import (
    _build_tree,
    _detect_languages,
    _get_git_context,
    _walk_indexable_files,
)
from main import app


@pytest.fixture
def client():
    return TestClient(app)


def _names(paths, root):
    return {str(p.relative_to(root)) for p in paths}


# ── File tree ────────────────────────────────────────────────────────────────


def test_tree_is_rooted_at_the_project(tmp_project):
    node, _, _ = _build_tree(tmp_project, tmp_project)
    assert node.kind == "dir"
    assert node.name == tmp_project.name


def test_tree_lists_ordinary_files(tmp_project):
    node, _, _ = _build_tree(tmp_project, tmp_project)
    assert "main.py" in {c.name for c in node.children}


def test_tree_counts_files_and_directories(tmp_project):
    _, files, dirs = _build_tree(tmp_project, tmp_project)
    assert files > 0
    assert dirs > 1


def test_tree_omits_vendor_directories(tmp_project):
    node, _, _ = _build_tree(tmp_project, tmp_project)
    assert "node_modules" not in {c.name for c in node.children}


def test_tree_omits_hidden_entries(tmp_project):
    node, _, _ = _build_tree(tmp_project, tmp_project)
    names = {c.name for c in node.children}
    assert ".hidden" not in names
    assert ".gitignore" not in names


def test_tree_keeps_meaningful_dotfiles(tmp_path):
    """`.env` is configuration a developer genuinely wants to see."""
    (tmp_path / ".env").write_text("KEY=1\n", encoding="utf-8")
    node, _, _ = _build_tree(tmp_path, tmp_path)
    assert ".env" in {c.name for c in node.children}


def test_tree_skips_symlinks(tmp_path):
    """Following them risks both cycles and escaping the project root."""
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "link.py").symlink_to(tmp_path / "real.py")
    node, _, _ = _build_tree(tmp_path, tmp_path)
    assert "link.py" not in {c.name for c in node.children}


def test_tree_depth_is_capped(tmp_path):
    deep = tmp_path
    for i in range(10):
        deep = deep / f"level{i}"
    deep.mkdir(parents=True)
    (deep / "buried.py").write_text("x = 1\n", encoding="utf-8")

    _, files, _ = _build_tree(tmp_path, tmp_path)
    assert files == 0, "a file past the depth cap should not be counted"


def test_tree_paths_are_relative_to_the_root(tmp_project):
    node, _, _ = _build_tree(tmp_project, tmp_project)
    utils = next(c for c in node.children if c.name == "utils")
    assert utils.children[0].path == "utils/helpers.py"


def test_an_unreadable_directory_does_not_abort_the_walk(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "x.py").write_text("x = 1\n", encoding="utf-8")
    blocked.chmod(0o000)
    try:
        node, _, _ = _build_tree(tmp_path, tmp_path)
        assert node is not None
    finally:
        blocked.chmod(0o755)


# ── Language detection ───────────────────────────────────────────────────────


def test_languages_are_detected_from_extensions(tmp_project):
    node, _, _ = _build_tree(tmp_project, tmp_project)
    languages = _detect_languages(node)
    assert "Python" in languages
    assert "JavaScript" in languages


def test_languages_are_ordered_by_frequency(tmp_path):
    for i in range(3):
        (tmp_path / f"m{i}.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "one.go").write_text("package main\n", encoding="utf-8")

    node, _, _ = _build_tree(tmp_path, tmp_path)
    assert _detect_languages(node)[0] == "Python"


def test_unknown_extensions_are_ignored(tmp_path):
    (tmp_path / "a.xyz").write_text("?", encoding="utf-8")
    node, _, _ = _build_tree(tmp_path, tmp_path)
    assert _detect_languages(node) == []


def test_extensionless_files_are_ignored(tmp_path):
    (tmp_path / "Makefile").write_text("all:\n", encoding="utf-8")
    node, _, _ = _build_tree(tmp_path, tmp_path)
    assert _detect_languages(node) == []


# ── Git context ──────────────────────────────────────────────────────────────


def test_a_plain_directory_is_not_a_repository(tmp_path):
    assert _get_git_context(tmp_path).is_git_repo is False


@pytest.fixture
def git_repo(tmp_path):
    def run(*args):
        subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, check=True)

    run("init", "-b", "main")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "Test")
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "tracked.py")
    run("commit", "-m", "initial commit")
    return tmp_path


def test_a_repository_is_detected(git_repo):
    assert _get_git_context(git_repo).is_git_repo is True


def test_the_current_branch_is_reported(git_repo):
    assert _get_git_context(git_repo).branch == "main"


def test_recent_commits_are_reported(git_repo):
    assert any("initial commit" in c for c in _get_git_context(git_repo).recent_commits)


def test_modified_and_untracked_files_are_separated(git_repo):
    (git_repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    (git_repo / "brand_new.py").write_text("y = 1\n", encoding="utf-8")

    context = _get_git_context(git_repo)
    assert "tracked.py" in context.modified_files
    assert "brand_new.py" in context.untracked_files
    assert "brand_new.py" not in context.modified_files


def test_the_first_modified_filename_is_not_truncated(git_repo):
    """
    Regression: `_run_git` used to `.strip()` its output, which removed the
    leading space of porcelain's first line (" M tracked.py"). The column-based
    `line[3:]` parse then reported "racked.py" — a filename that does not
    exist — as context to the agent.
    """
    (git_repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    assert _get_git_context(git_repo).modified_files == ["tracked.py"]


def test_every_reported_path_actually_exists(git_repo):
    """The strongest form of the same guarantee, across both lists."""
    (git_repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    (git_repo / "another.py").write_text("z = 1\n", encoding="utf-8")

    context = _get_git_context(git_repo)
    for name in context.modified_files + context.untracked_files:
        assert (git_repo / name).exists(), f"{name!r} was reported but does not exist"


# ── Indexable walk ───────────────────────────────────────────────────────────


def test_walk_collects_source_files(tmp_project):
    found = _names(_walk_indexable_files(tmp_project), tmp_project)
    assert "main.py" in found
    assert "utils/helpers.py" in found


def test_walk_honours_a_gitignore_directory_rule(tmp_project):
    assert "ignored/secret.py" not in _names(_walk_indexable_files(tmp_project), tmp_project)


def test_walk_honours_a_gitignore_glob_rule(tmp_project):
    assert not any(p.suffix == ".log" for p in _walk_indexable_files(tmp_project))


def test_walk_skips_vendor_directories(tmp_project):
    assert not any("node_modules" in str(p) for p in _walk_indexable_files(tmp_project))


def test_walk_skips_hidden_directories(tmp_project):
    assert not any(".hidden" in str(p) for p in _walk_indexable_files(tmp_project))


def test_walk_skips_non_indexable_files(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    assert _names(_walk_indexable_files(tmp_path), tmp_path) == {"code.py"}


def test_walk_skips_symlinks(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "alias.py").symlink_to(tmp_path / "real.py")
    assert "alias.py" not in _names(_walk_indexable_files(tmp_path), tmp_path)


def test_walk_of_an_empty_project_returns_nothing(tmp_path):
    assert _walk_indexable_files(tmp_path) == []


def test_walk_without_a_gitignore_still_works(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert len(_walk_indexable_files(tmp_path)) == 1


# ── Phase 1 endpoint ─────────────────────────────────────────────────────────


def test_index_returns_the_project_shape(client, tmp_project):
    body = client.post("/api/projects/index", json={"path": str(tmp_project)}).json()
    assert body["name"] == tmp_project.name
    assert body["total_files"] > 0
    assert "Python" in body["languages"]
    assert body["file_tree"]["kind"] == "dir"
    assert "indexed_at" in body


def test_index_rejects_a_missing_path(client, tmp_path):
    response = client.post("/api/projects/index", json={"path": str(tmp_path / "nope")})
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_index_rejects_a_file(client, tmp_path):
    path = tmp_path / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")
    response = client.post("/api/projects/index", json={"path": str(path)})
    assert response.status_code == 400
    assert "not a directory" in response.json()["detail"].lower()


def test_index_requires_a_path(client):
    assert client.post("/api/projects/index", json={}).status_code == 422
