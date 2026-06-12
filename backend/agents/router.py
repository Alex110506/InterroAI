"""
Automated Model Routing Layer — local classifier.

Replaces the previous OpenAI gpt-4o-mini call with an offline TF-IDF +
Logistic Regression model trained on `prompt_classifier/prompt_examples_dataset.csv`.

Predicted tier IDs (3 of the 5 display tiers — the other two stay reachable
via the manual dropdown):
  - "gpt-5.4-mini"        ← low-complexity prompts        → gpt-4o-mini
  - "gpt-5.4-high-effort" ← medium-complexity prompts     → gpt-4o
  - "gpt-5.5-high-effort" ← high-complexity prompts       → o1

Inference is sub-millisecond; we still expose `classify` as `async` and
run it via `asyncio.to_thread` so the call-site stays identical to the
previous OpenAI version and the event loop is never blocked on cold load.

Train the model with:
    cd backend && python -m prompt_classifier.train
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

ModelId = str

_VALID_MODELS: set[str] = {
    "gpt-5.4-mini",
    "gpt-5.4-low-effort",
    "gpt-5.4-high-effort",
    "gpt-5.5-low-effort",
    "gpt-5.5-high-effort",
}
_DEFAULT: ModelId = "gpt-5.4-high-effort"

_MODEL_PATH = Path(__file__).resolve().parent.parent / "prompt_classifier" / "router_model.joblib"

_pipeline = None          # cached sklearn pipeline
_load_attempted = False   # one-shot warning if file is missing


def _load_pipeline():
    """Lazy-load the joblib pipeline once; subsequent calls return the cache."""
    global _pipeline, _load_attempted
    if _pipeline is not None:
        return _pipeline
    if _load_attempted:
        return None  # already failed once — don't spam logs
    _load_attempted = True

    if not _MODEL_PATH.exists():
        logger.warning(
            "Router model not found at %s — falling back to %r. "
            "Run `python -m prompt_classifier.train` to build it.",
            _MODEL_PATH, _DEFAULT,
        )
        return None
    try:
        _pipeline = joblib.load(_MODEL_PATH)
        logger.info("Loaded local router classifier from %s", _MODEL_PATH)
        return _pipeline
    except Exception:
        logger.exception("Failed to load router classifier — falling back to %r", _DEFAULT)
        return None


def _snippet(prompt: str, max_len: int = 80) -> str:
    s = (prompt or "").strip().replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _classify_sync(prompt: str) -> ModelId:
    pipeline = _load_pipeline()
    snippet = _snippet(prompt)

    if pipeline is None:
        logger.warning(
            "Router classifier UNAVAILABLE (no model file) — using fallback %r | prompt=%r",
            _DEFAULT, snippet,
        )
        return _DEFAULT
    if not prompt.strip():
        logger.warning("Router got empty prompt — using fallback %r", _DEFAULT)
        return _DEFAULT

    try:
        raw_pred = pipeline.predict([prompt])[0]
        pred = str(raw_pred)  # sklearn returns numpy.str_; coerce for clean logs
        probs = pipeline.predict_proba([prompt])[0]
        confidence = float(probs[list(pipeline.classes_).index(raw_pred)])
    except Exception:
        logger.exception(
            "Router prediction FAILED — using fallback %r | prompt=%r",
            _DEFAULT, snippet,
        )
        return _DEFAULT

    if pred not in _VALID_MODELS:
        logger.warning(
            "Router predicted unknown tier %r — using fallback %r | prompt=%r",
            pred, _DEFAULT, snippet,
        )
        return _DEFAULT

    logger.info(
        "Router classifier chose %r (confidence=%.2f) | prompt=%r",
        pred, confidence, snippet,
    )
    return pred


async def classify(prompt: str) -> ModelId:
    """Return the model ID best suited for this prompt (sub-ms, runs off-thread)."""
    return await asyncio.to_thread(_classify_sync, prompt)
