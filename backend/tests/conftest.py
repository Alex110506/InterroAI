"""
Shared fixtures.

The HOME redirect at the top of this module is load-bearing and must stay
before every application import: `config.py` instantiates `_AppConfig()` at
*import* time, which writes into `~/.interroai`. Redirecting HOME first keeps
the developer's real config and Chroma store untouched by a test run.
"""
from __future__ import annotations

import os
import tempfile

_FAKE_HOME = tempfile.mkdtemp(prefix="interroai-test-home-")
os.environ["HOME"] = _FAKE_HOME
os.environ["USERPROFILE"] = _FAKE_HOME  # Windows equivalent

# ── Application imports (must come after the HOME redirect) ──────────────────
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
from redis.exceptions import RedisError  # noqa: E402

import core.cache as cache_module  # noqa: E402
import core.llm as llm_module  # noqa: E402
import core.security as security  # noqa: E402
import core.vector_store as vector_store  # noqa: E402

# ── Fake OpenAI plumbing ─────────────────────────────────────────────────────


def make_tool_call(call_id: str, name: str, arguments: str):
    """Mimic one entry of `response.choices[0].message.tool_calls`."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_response(content: str | None = None, tool_calls: list | None = None):
    """Mimic the shape of an OpenAI chat completion response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    """Records every call and replays a queued list of results."""

    def __init__(self, results):
        self._results = list(results)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._results:
            raise AssertionError("FakeCompletions ran out of queued results")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeOpenAIClient:
    """Stands in for AsyncOpenAI: only the surface the app actually uses."""

    def __init__(self, results=()):
        self.completions = FakeCompletions(results)
        self.chat = SimpleNamespace(completions=self.completions)
        self.embeddings = self.completions

    @property
    def calls(self) -> list[dict]:
        return self.completions.calls


@pytest.fixture
def fake_client():
    """Factory: `fake_client([response, ...])`."""
    return FakeOpenAIClient


# ── API-key control ──────────────────────────────────────────────────────────


@pytest.fixture
def with_api_key(monkeypatch):
    """Make `core.llm.get_client` believe a key is stored."""
    monkeypatch.setattr(llm_module, "retrieve_openai_key", lambda: "sk-test-key")
    llm_module._clients.clear()
    yield "sk-test-key"
    llm_module._clients.clear()


@pytest.fixture
def fake_keyring(monkeypatch):
    """Replace the OS keychain with an in-memory dict — no test touches the real one."""
    store: dict[tuple[str, str], str] = {}

    class K:
        @staticmethod
        def set_password(service, account, password):
            store[(service, account)] = password

        @staticmethod
        def get_password(service, account):
            return store.get((service, account))

        @staticmethod
        def delete_password(service, account):
            store.pop((service, account), None)

    monkeypatch.setattr(security, "keyring", K())
    return store


@pytest.fixture
def without_api_key(monkeypatch):
    """Make `core.llm.get_client` believe no key is stored."""
    monkeypatch.setattr(llm_module, "retrieve_openai_key", lambda: None)
    llm_module._clients.clear()
    yield
    llm_module._clients.clear()


# ── Isolated persistence ─────────────────────────────────────────────────────


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    """Point the vector store at a throwaway Chroma directory."""
    store = tmp_path / "chroma"
    monkeypatch.setattr(vector_store, "_STORE_DIR", store)
    return store


class FakePipeline:
    """`core.cache` queues SETs and executes them in one go."""

    def __init__(self, client: FakeRedis) -> None:
        self._client = client
        self._queued: list[tuple[str, bytes]] = []

    def set(self, key, value, ex=None):
        self._queued.append((key, value))
        return self

    async def execute(self):
        self._client.fail_if_broken()
        for key, value in self._queued:
            self._client.store[key] = value
        self._client.sets += len(self._queued)
        return [True] * len(self._queued)


class FakeRedis:
    """
    The slice of `redis.asyncio.Redis` that `core.cache` actually uses.

    In-memory, so `core/cache.py`'s own keying and float32 packing still get
    exercised by every test that reaches the cache, without a server.
    """

    def __init__(self, *, broken: bool = False) -> None:
        self.store: dict[str, bytes] = {}
        self.broken = broken
        self.reads = 0
        self.sets = 0

    def fail_if_broken(self) -> None:
        if self.broken:
            raise RedisError("fake redis is unreachable")

    async def get(self, key):
        self.fail_if_broken()
        self.reads += 1
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.fail_if_broken()
        self.sets += 1
        self.store[key] = value
        return True

    async def mget(self, keys):
        self.fail_if_broken()
        self.reads += 1
        return [self.store.get(key) for key in keys]

    def pipeline(self, transaction=False):
        return FakePipeline(self)

    async def aclose(self):
        return None


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """
    Replace Redis with an in-memory double — for *every* test.

    Autouse deliberately: a developer running a real `redis-server` must not
    see different behaviour from CI, and a vector cached by one test must not
    silently satisfy the next one's assertions about how many API calls it made.
    """
    fake = FakeRedis()
    monkeypatch.setattr(cache_module, "_client", fake)
    monkeypatch.setattr(cache_module, "_unavailable", False)
    return fake


# ── Sample project tree ──────────────────────────────────────────────────────


@pytest.fixture
def tmp_project(tmp_path) -> Path:
    """
    A small but representative project:

      * source in several languages (drives chunker / ast_map / language detection)
      * a .gitignore with a directory rule and a glob rule
      * an excluded vendor directory and a hidden directory
    """
    root = tmp_path / "proj"
    (root / "utils").mkdir(parents=True)
    (root / "web").mkdir()
    (root / "ignored").mkdir()
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / ".hidden").mkdir()

    (root / ".gitignore").write_text("ignored/\n*.log\n", encoding="utf-8")
    (root / "main.py").write_text(
        "import os\n\n\n"
        "class Greeter:\n"
        "    def greet(self, name: str) -> str:\n"
        "        return f'hi {name}'\n\n\n"
        "async def run(count: int = 1) -> None:\n"
        "    print(count)\n",
        encoding="utf-8",
    )
    (root / "utils" / "helpers.py").write_text(
        "def slugify(text):\n    return text.lower()\n", encoding="utf-8"
    )
    (root / "web" / "app.js").write_text(
        "export class Widget {}\n"
        "export function mount() {}\n"
        "const handler = () => {}\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Demo\n\nSome prose.\n", encoding="utf-8")
    (root / "ignored" / "secret.py").write_text("TOKEN = 'x'\n", encoding="utf-8")
    (root / "debug.log").write_text("noise\n", encoding="utf-8")
    (root / "node_modules" / "pkg" / "index.js").write_text("export function v() {}\n", encoding="utf-8")  # noqa: E501
    (root / ".hidden" / "x.py").write_text("HIDDEN = 1\n", encoding="utf-8")
    return root
