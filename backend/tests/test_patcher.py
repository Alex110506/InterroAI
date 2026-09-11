"""
`apply_patch` is the only code path that mutates a user's source file, so the
edge cases that matter are the ones where it must refuse to act.
"""
from __future__ import annotations

import pytest

from core.errors import FileNotFoundInProjectError, PatchError
from core.patcher import apply_patch


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("alpha = 1\nbeta = 2\ngamma = 3\n", encoding="utf-8")
    return path


def test_applies_a_unique_patch(source):
    apply_patch(str(source), "beta = 2", "beta = 20")
    assert source.read_text(encoding="utf-8") == "alpha = 1\nbeta = 20\ngamma = 3\n"


def test_returns_none_on_success(source):
    """Success is signalled by *not* raising; there is no boolean to ignore."""
    assert apply_patch(str(source), "beta = 2", "beta = 20") is None


def test_missing_file_raises_typed_error(tmp_path):
    with pytest.raises(FileNotFoundInProjectError, match="File not found"):
        apply_patch(str(tmp_path / "ghost.py"), "a", "b")


def test_absent_search_block_raises_and_says_why(source):
    with pytest.raises(PatchError, match="not found verbatim"):
        apply_patch(str(source), "delta = 4", "delta = 40")


def test_absent_search_block_leaves_the_file_untouched(source):
    before = source.read_text(encoding="utf-8")
    with pytest.raises(PatchError):
        apply_patch(str(source), "delta = 4", "delta = 40")
    assert source.read_text(encoding="utf-8") == before


def test_ambiguous_search_block_is_refused(tmp_path):
    """
    The regression this guards: the old implementation used
    `content.replace(block, new, 1)` and silently patched the *first* of several
    identical blocks — a coin flip that could edit the wrong function.
    """
    path = tmp_path / "dup.py"
    path.write_text("x = 1\ny = 0\nx = 1\n", encoding="utf-8")
    with pytest.raises(PatchError, match="appears 2 times"):
        apply_patch(str(path), "x = 1", "x = 2")
    assert path.read_text(encoding="utf-8") == "x = 1\ny = 0\nx = 1\n"


def test_ambiguity_message_tells_the_model_how_to_recover(tmp_path):
    path = tmp_path / "dup.py"
    path.write_text("p\np\n", encoding="utf-8")
    with pytest.raises(PatchError, match="more surrounding"):
        apply_patch(str(path), "p", "q")


def test_disambiguating_with_more_context_succeeds(tmp_path):
    """The documented escape hatch from the ambiguity error must actually work."""
    path = tmp_path / "dup.py"
    path.write_text("x = 1\ny = 0\nx = 1\n", encoding="utf-8")
    apply_patch(str(path), "y = 0\nx = 1", "y = 0\nx = 2")
    assert path.read_text(encoding="utf-8") == "x = 1\ny = 0\nx = 2\n"


def test_a_dedented_fragment_still_matches_as_a_substring(tmp_path):
    """
    Matching is plain substring search, so a fragment without the leading
    indentation still matches inside an indented line — and only the fragment
    is replaced, leaving the indentation intact. Documented because it is the
    behaviour the `patch_file` tool description relies on.
    """
    path = tmp_path / "ind.py"
    path.write_text("def f():\n    return 1\n", encoding="utf-8")
    apply_patch(str(path), "return 1", "return 2")
    assert path.read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_wrong_indentation_in_a_multiline_block_is_rejected(tmp_path):
    """Across lines, indentation is part of the text and must match exactly."""
    path = tmp_path / "ind.py"
    path.write_text("def f():\n    return 1\n", encoding="utf-8")
    with pytest.raises(PatchError, match="not found verbatim"):
        apply_patch(str(path), "def f():\nreturn 1", "def f():\nreturn 2")


def test_tabs_do_not_match_spaces(tmp_path):
    path = tmp_path / "tabs.py"
    path.write_text("def f():\n    return 1\n", encoding="utf-8")
    with pytest.raises(PatchError, match="not found verbatim"):
        apply_patch(str(path), "\treturn 1", "\treturn 2")


def test_over_indented_search_block_is_rejected(tmp_path):
    """More leading whitespace than the file has cannot match."""
    path = tmp_path / "ind.py"
    path.write_text("def f():\n  return 1\n", encoding="utf-8")
    with pytest.raises(PatchError, match="not found verbatim"):
        apply_patch(str(path), "        return 1", "        return 2")


def test_multiline_block_is_replaced(tmp_path):
    path = tmp_path / "multi.py"
    path.write_text("def f():\n    a = 1\n    b = 2\n    return a\n", encoding="utf-8")
    apply_patch(str(path), "    a = 1\n    b = 2\n", "    a = 9\n")
    assert path.read_text(encoding="utf-8") == "def f():\n    a = 9\n    return a\n"


def test_replacement_can_delete_content(tmp_path):
    path = tmp_path / "del.py"
    path.write_text("keep\ndrop\n", encoding="utf-8")
    apply_patch(str(path), "drop\n", "")
    assert path.read_text(encoding="utf-8") == "keep\n"


def test_unicode_content_survives_a_round_trip(tmp_path):
    path = tmp_path / "uni.py"
    path.write_text("msg = 'héllo — wörld'\n", encoding="utf-8")
    apply_patch(str(path), "héllo", "goodbye")
    assert path.read_text(encoding="utf-8") == "msg = 'goodbye — wörld'\n"


def test_trailing_newline_is_preserved(tmp_path):
    path = tmp_path / "nl.py"
    path.write_text("a = 1\n", encoding="utf-8")
    apply_patch(str(path), "a = 1", "a = 2")
    assert path.read_text(encoding="utf-8").endswith("\n")
