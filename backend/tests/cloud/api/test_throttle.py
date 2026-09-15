"""Rate limits: the token buckets, and the routes that spend from them."""
from __future__ import annotations

from datetime import timedelta

import pytest
from fakes.accounts import InMemoryAccounts
from fakes.github import FakeGitHub
from fakes.models import MODEL, FakeGateway
from fakes.usage import InMemoryUsageMeter
from fastapi.testclient import TestClient

from cloud.api.main import create_app
from cloud.api.services import Limits, Services
from cloud.api.signin import SignInService
from cloud.api.throttle import RateLimiter, RateLimits
from cloud.api.tokens import TokenSigner


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def limiter(clock):
    return RateLimiter(clock=clock)


# ── Buckets ──────────────────────────────────────────────────────────────────


def test_a_burst_up_to_the_limit_goes_through_and_then_has_to_wait(limiter):
    assert [limiter.hit("chat", "alice", 3) for _ in range(3)] == [0, 0, 0]
    assert limiter.hit("chat", "alice", 3) == pytest.approx(20.0)


def test_a_bucket_refills_steadily(limiter, clock):
    for _ in range(3):
        limiter.hit("chat", "alice", 3)

    clock.now += 20

    assert limiter.hit("chat", "alice", 3) == 0
    assert limiter.hit("chat", "alice", 3) > 0


def test_callers_and_limits_are_counted_apart(limiter):
    for _ in range(3):
        limiter.hit("chat", "alice", 3)

    assert limiter.hit("chat", "bob", 3) == 0
    assert limiter.hit("search", "alice", 3) == 0


def test_a_zero_limit_admits_nothing(limiter):
    assert limiter.hit("chat", "alice", 0) > 0


def test_memory_stays_bounded(clock):
    limiter = RateLimiter(clock=clock, max_buckets=100)

    for caller in range(1_000):
        limiter.hit("sign_in", f"10.0.{caller // 250}.{caller % 250}", 30)
        clock.now += 0.01

    assert len(limiter) <= 100


def test_idle_buckets_are_dropped_first_and_busy_ones_keep_their_count(clock):
    limiter = RateLimiter(clock=clock, max_buckets=2)
    limiter.hit("chat", "idle", 3)
    clock.now += 61
    for _ in range(3):
        limiter.hit("chat", "busy", 3)

    limiter.hit("chat", "newcomer", 3)

    assert limiter.hit("chat", "busy", 3) > 0, "a busy caller's count survives eviction"


# ── Routes ───────────────────────────────────────────────────────────────────


def _signer() -> TokenSigner:
    return TokenSigner("t" * 32, issuer="http://testserver", access_ttl=timedelta(minutes=15))


def test_chat_past_its_rate_is_429_rate_limited_per_user():
    signer = _signer()
    services = Services(
        signin=None,
        signer=signer,
        accounts=None,
        projects=None,
        usage=InMemoryUsageMeter(),
        gateway=FakeGateway(),
        limits=Limits(rates=RateLimits(chat=2)),
    )
    client = TestClient(create_app(services))
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

    def chat_as(user: str):
        headers = {"Authorization": f"Bearer {signer.access_token(user, user)}"}
        return client.post("/llm/chat", json=body, headers=headers)

    assert [chat_as("alice").status_code for _ in range(2)] == [200, 200]
    refused = chat_as("alice")

    assert refused.status_code == 429
    assert refused.json()["detail"]["code"] == "rate_limited"
    assert int(refused.headers["retry-after"]) >= 1
    assert chat_as("bob").status_code == 200, "each user has their own bucket"


def test_sign_in_is_rate_limited_per_client_address():
    signer, accounts = _signer(), InMemoryAccounts()
    signin = SignInService(
        accounts=accounts,
        github=FakeGitHub(),
        signer=signer,
        allowlist=frozenset({"alex110506"}),
        refresh_ttl=timedelta(days=30),
    )
    services = Services(
        signin=signin,
        signer=signer,
        accounts=accounts,
        projects=None,
        limits=Limits(rates=RateLimits(sign_in=2)),
    )
    client = TestClient(create_app(services))
    guess = {"grant_type": "refresh_token", "refresh_token": "a-guess"}

    statuses = [client.post("/auth/token", json=guess).status_code for _ in range(3)]

    assert statuses == [400, 400, 429]
