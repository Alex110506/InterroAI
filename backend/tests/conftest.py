"""
Shared fixtures.

The HOME redirect at the top of this module is load-bearing and must stay
before every application import: `config.py` instantiates `_AppConfig()` at
*import* time, which writes into `~/.interroai`. Redirecting HOME first keeps
the developer's real config and Chroma store untouched by a test run.

Model calls are faked at the `ModelGateway` port (`FakeGateway`, handed to the
code under test), not by patching `core.models.llm`. Only the tests of `core.models.llm`
itself and of the embedding client still stub the OpenAI SDK directly.
"""
from __future__ import annotations

import os
import tempfile

_FAKE_HOME = tempfile.mkdtemp(prefix="interroai-test-home-")
os.environ["HOME"] = _FAKE_HOME
os.environ["USERPROFILE"] = _FAKE_HOME  # Windows equivalent

# The suite always exercises the local build. An environment variable beats a
# `.env` file in pydantic-settings, so a developer's `.env` saying
# INTERROAI_MODE=cloud cannot quietly route tests through the network.
os.environ["INTERROAI_MODE"] = "local"

# ── Application imports (must come after the HOME redirect) ──────────────────
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
from redis.exceptions import RedisError  # noqa: E402

import core.index.adapters.chroma as vector_store  # noqa: E402
import core.local.cache as cache_module  # noqa: E402
import core.local.security as security  # noqa: E402
import core.models.llm as llm_module  # noqa: E402

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


class FakeStream:
    """Async-iterates OpenAI-shaped streaming deltas, one per token."""

    def __init__(self, tokens):
        self._tokens = list(tokens)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._tokens:
            raise StopAsyncIteration
        delta = SimpleNamespace(content=self._tokens.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeGateway:
    """
    A `ModelGateway` that replays queued replies and records every request.

    `replies` answer `chat` in order; `streams` (each a list of tokens) answer
    `chat_stream`. An exception in either queue is raised instead of returned,
    which is how a test makes the provider fail. Calling with nothing queued is
    an assertion failure, so an unexpected model call cannot pass unnoticed.
    """

    def __init__(self, replies=(), *, streams=()):
        self._replies = list(replies)
        self._streams = list(streams)
        self.requests: list[dict] = []
        self.timeouts: list = []

    async def chat(self, *, timeout=None, **request):
        return self._next(self._replies, "chat", timeout, request)

    async def chat_stream(self, *, timeout=None, **request):
        return FakeStream(self._next(self._streams, "chat_stream", timeout, request))

    def _next(self, queue, name, timeout, request):
        # Snapshot `messages`: the agent appends the reply to the same list
        # after the call, which would otherwise rewrite what was recorded.
        self.requests.append(
            {key: list(value) if key == "messages" else value for key, value in request.items()}
        )
        self.timeouts.append(timeout)
        if not queue:
            raise AssertionError(f"FakeGateway.{name} was called with nothing queued")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


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


@pytest.fixture(autouse=True)
def no_configured_platform_key(monkeypatch):
    """A platform key set by one test (`llm.use_api_key`) must not leak into the next."""
    monkeypatch.setattr(llm_module, "_configured_key", None)


@pytest.fixture
def with_api_key(monkeypatch):
    """Make `core.models.llm.get_client` believe a key is stored."""
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
    """Make `core.models.llm.get_client` believe no key is stored."""
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
    """`core.local.cache` queues SETs and executes them in one go."""

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
    The slice of `redis.asyncio.Redis` that `core.local.cache` actually uses.

    In-memory, so `core/local/cache.py`'s own keying and float32 packing still get
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


# ── The local stack (integration tests only) ─────────────────────────────────
#
# Used by tests marked `integration`, which the default run deselects. They take
# the stack's coordinates from the same `.env` it was started with, point both
# database roles at the dedicated `interroai_test` database (created by
# infra/local/postgres/init/01-roles.sh), rebuild its schema from the migrations
# once per run, and empty every table before each test. Imports are local so an
# ordinary run never pays for SQLAlchemy or Alembic.

_TEST_DATABASE = "interroai_test"
_CLOUD_TABLES = (
    "users, refresh_tokens, login_codes, projects, chunks, embedding_cache, index_jobs, usage"
)


@pytest.fixture(scope="session")
def stack_settings():
    """Whatever of the stack's coordinates `.env` provides; each fixture checks its own."""
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class StackSettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix="INTERROAI_",
            env_file=(Path(__file__).resolve().parents[2] / ".env", ".env"),
            extra="ignore",
        )
        database_url: str | None = None
        migrations_database_url: str | None = None
        blob_connection_string: str | None = None
        servicebus_connection_string: str | None = None
        servicebus_queue: str = "index-jobs"

    return StackSettings()


@pytest.fixture(scope="session")
def database_urls(stack_settings):
    """The app-role and owner URLs, both moved onto the test database."""
    from sqlalchemy.engine import make_url

    if not (stack_settings.database_url and stack_settings.migrations_database_url):
        pytest.skip("INTERROAI_DATABASE_URL / INTERROAI_MIGRATIONS_DATABASE_URL are not set")

    def on_test_database(url: str) -> str:
        return make_url(url).set(database=_TEST_DATABASE).render_as_string(hide_password=False)

    return SimpleNamespace(
        app=on_test_database(stack_settings.database_url),
        owner=on_test_database(stack_settings.migrations_database_url),
    )


@pytest.fixture
async def blob_uploads(stack_settings):
    """A BlobUploadStore on a container of its own, so tests never touch dev uploads."""
    from azure.core.exceptions import ServiceRequestError

    from cloud.adapters.blob_uploads import BlobUploadStore

    if not stack_settings.blob_connection_string:
        pytest.skip("INTERROAI_BLOB_CONNECTION_STRING is not set")
    store = BlobUploadStore.from_connection_string(
        stack_settings.blob_connection_string, "interroai-test-uploads"
    )
    try:
        await store.ensure_container()
    except ServiceRequestError as exc:
        await store.close()
        pytest.skip(f"Azurite is not reachable ({exc}); run `docker compose up -d`.")
    yield store
    await store.close()


@pytest.fixture
async def service_bus_queue(stack_settings):
    """The emulator's queue, drained first so no earlier test's message leaks in."""
    from azure.servicebus.aio import ServiceBusClient
    from azure.servicebus.exceptions import ServiceBusError

    from cloud.adapters.service_bus import ServiceBusJobQueue

    if not stack_settings.servicebus_connection_string:
        pytest.skip("INTERROAI_SERVICEBUS_CONNECTION_STRING is not set")
    connection_string = stack_settings.servicebus_connection_string
    queue_name = stack_settings.servicebus_queue

    try:
        async with ServiceBusClient.from_connection_string(connection_string) as client:
            async with client.get_queue_receiver(queue_name) as receiver:
                while messages := await receiver.receive_messages(
                    max_message_count=50, max_wait_time=1
                ):
                    for message in messages:
                        await receiver.complete_message(message)
    except (ServiceBusError, OSError) as exc:
        pytest.skip(f"The Service Bus emulator is not reachable ({exc}).")

    queue = ServiceBusJobQueue.from_connection_string(connection_string, queue_name)
    yield queue
    await queue.close()


@pytest.fixture(scope="session")
def migrated_database(database_urls):
    """The test database's schema, rebuilt from the migrations once per run."""
    from alembic import command
    from alembic.config import Config

    config = Config(str(Path(__file__).resolve().parents[1] / "cloud" / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_urls.owner)
    config.attributes["configure_logger"] = False
    try:
        command.downgrade(config, "base")
    except OSError as exc:
        pytest.skip(f"The local stack is not reachable ({exc}); run `docker compose up -d`.")
    command.upgrade(config, "head")


@pytest.fixture
async def owner_engine(migrated_database, database_urls):
    """The owner role — bypasses row-level security, so only for arranging data."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_urls.owner)
    async with engine.begin() as connection:
        await connection.execute(text(f"TRUNCATE {_CLOUD_TABLES} CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def app_sessions(owner_engine, database_urls):
    """Sessions as the app role, which row-level security applies to."""
    from cloud.db.session import create_engine, create_session_factory

    engine = create_engine(database_urls.app)
    yield create_session_factory(engine)
    await engine.dispose()


@pytest.fixture
def pg_new_user(owner_engine):
    from sqlalchemy import text

    async def make(login: str = "someone") -> str:
        async with owner_engine.begin() as connection:
            result = await connection.execute(
                text(
                    "INSERT INTO users (id, github_id, login) "
                    "VALUES (gen_random_uuid(), floor(random() * 1e12)::bigint, :login) "
                    "RETURNING id"
                ),
                {"login": login},
            )
            return str(result.scalar_one())

    return make


@pytest.fixture
def pg_new_project(owner_engine, pg_new_user):
    from sqlalchemy import text

    async def make(owner_id: str | None = None) -> str:
        owner_id = owner_id or await pg_new_user()
        async with owner_engine.begin() as connection:
            result = await connection.execute(
                text(
                    "INSERT INTO projects (id, owner_id, name) "
                    "VALUES (gen_random_uuid(), :owner, 'test-project') RETURNING id"
                ),
                {"owner": owner_id},
            )
            return str(result.scalar_one())

    return make
