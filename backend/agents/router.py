"""
Automated Model Routing Layer.

Classifies prompt complexity to decide which model tier to use if 'auto' is selected:
  - Simple (Q&A, minor fixes): gpt-5.4-mini or gpt-5.4-low-effort
  - Medium (Features, standard coding): gpt-5.4-high-effort or gpt-5.5-low-effort
  - Complex (Architecture, large refactors): gpt-5.5-high-effort
"""

from __future__ import annotations
import json
from openai import AsyncOpenAI
from core.security import retrieve_openai_key

ModelId = str

_ROUTER_SYSTEM_PROMPT = """\
You are an expert system that routes user prompts to the most appropriate coding model.
You must analyze the complexity of the prompt and return one of the following model IDs:
- "gpt-5.4-mini": for extremely trivial tasks, basic Q&A, formatting.
- "gpt-5.4-low-effort": for simple bug fixes, single-function creations.
- "gpt-5.4-high-effort": for standard feature development in a single file.
- "gpt-5.5-low-effort": for cross-file features, moderate refactoring.
- "gpt-5.5-high-effort": for complex architecture, major refactors, difficult debugging.

Respond strictly with a JSON object: {"model": "<model_id>"}
"""

_VALID_MODELS = {
    "gpt-5.4-mini",
    "gpt-5.4-low-effort",
    "gpt-5.4-high-effort",
    "gpt-5.5-low-effort",
    "gpt-5.5-high-effort",
}
_DEFAULT = "gpt-5.4-high-effort"


async def classify(prompt: str) -> ModelId:
    """Return the model ID best suited for this prompt."""
    key = retrieve_openai_key()
    if not key:
        return _DEFAULT

    client = AsyncOpenAI(api_key=key)
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        selected = data.get("model", _DEFAULT)
        if selected not in _VALID_MODELS:
            return _DEFAULT
        return selected
    except Exception:
        return _DEFAULT

