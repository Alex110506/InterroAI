"""Terminal palette, glyphs and the display-ID → human-label rule."""
from __future__ import annotations

from rich.theme import Theme

#: Named styles used across the CLI. Defining them once means a change of
#: palette is a change to this file, not a sweep through every print call.
THEME = Theme(
    {
        "brand": "bold #d97757",
        "hint": "grey50",
        "label": "bold #d97757",
        "user": "bold white",
        "ok": "green",
        "warn": "yellow",
        "fail": "red",
        "skip": "grey50",
        "tool": "cyan",
        "arg": "grey62",
        "rule": "grey35",
    }
)

# ── Glyphs ───────────────────────────────────────────────────────────────────
BANNER = "✻"
BULLET = "⏺"
PASS = "✓"
FAIL = "✗"
SKIP = "○"
PENDING = "·"
PROMPT = "› "


def model_label(display_id: str) -> str:
    """
    'gpt-5.5-high-effort' → 'GPT-5.5 High Effort'.

    Derived rather than tabulated: a hand-maintained label map is one more
    place a new model ID can be forgotten, and the ID already carries
    everything the label needs. The family and its version stay hyphenated
    ("GPT-5.5"); the qualifiers after it become words.
    """
    parts = display_id.split("-")
    if not parts:
        return display_id

    head = "GPT" if parts[0].lower() == "gpt" else parts[0].capitalize()
    rest = parts[1:]

    # A leading version number belongs to the family name, not the qualifiers.
    if rest and rest[0][:1].isdigit():
        head = f"{head}-{rest[0]}"
        rest = rest[1:]

    return " ".join([head, *(w.capitalize() for w in rest)])
