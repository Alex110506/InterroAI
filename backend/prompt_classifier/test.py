"""
Quick CLI tester for the local router classifier.

Run from the backend/ directory:

  python -m prompt_classifier.test                          # interactive REPL
  python -m prompt_classifier.test "your prompt here"       # one-shot
  python -m prompt_classifier.test -f path/to/prompt.txt    # read from file

Each prediction shows:
  - The chosen display tier and the underlying OpenAI API model
  - The confidence distribution across all three classes
  - The same `agents.router` INFO log line you'd see in the running backend
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow `python prompt_classifier/test.py ...` as a fallback to `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import router as model_router  # noqa: E402


# Display tier → real OpenAI API model. Inlined (not imported from
# agents.coder) so the test script keeps running while coder.py is edited.
_TIER_TO_API: dict[str, str] = {
    "gpt-5.4-mini":        "gpt-4o-mini",
    "gpt-5.4-low-effort":  "gpt-4o",
    "gpt-5.4-high-effort": "gpt-4o",
    "gpt-5.5-low-effort":  "gpt-4o",
    "gpt-5.5-high-effort": "o1",
}


def _print_distribution(prompt: str, chosen_tier: str) -> None:
    pipeline = model_router._load_pipeline()
    if pipeline is None:
        print("  [fallback path — classifier unavailable]")
        return
    probs = pipeline.predict_proba([prompt])[0]
    ranked = sorted(zip(pipeline.classes_, probs), key=lambda kv: -kv[1])
    print("  Confidence distribution:")
    for cls, p in ranked:
        cls_s = str(cls)
        bar = "█" * int(round(p * 30))
        marker = "  ◀ chosen" if cls_s == chosen_tier else ""
        print(f"    {cls_s:<24s} {p:6.1%}  {bar}{marker}")


async def _classify_and_show(prompt: str) -> None:
    tier = await model_router.classify(prompt)
    api = _TIER_TO_API.get(tier, "?")
    print(f"\n  Chosen tier: {tier}  (API model: {api})")
    _print_distribution(prompt, tier)


async def _repl() -> None:
    print("Local router classifier — type a prompt and press Enter.")
    print("Empty line or Ctrl-D to quit.")
    while True:
        try:
            line = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            return
        await _classify_and_show(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the local prompt-complexity classifier with arbitrary input.",
    )
    parser.add_argument("prompt", nargs="?", help="prompt to classify (omit for the REPL)")
    parser.add_argument("-f", "--file", type=Path, help="read the prompt from a UTF-8 file")
    args = parser.parse_args()

    # Surface the router's own INFO/WARN log line — same one the running backend emits.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

    if args.file:
        if not args.file.exists():
            sys.exit(f"File not found: {args.file}")
        text = args.file.read_text(encoding="utf-8").strip()
        if not text:
            sys.exit(f"File is empty: {args.file}")
        asyncio.run(_classify_and_show(text))
    elif args.prompt:
        asyncio.run(_classify_and_show(args.prompt))
    else:
        asyncio.run(_repl())


if __name__ == "__main__":
    main()
