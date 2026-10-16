"""
The sign-in flow as a service: in-memory accounts, a fake GitHub, a controllable clock.

The HTTP layer over it is covered in `test_auth_routes.py`; the Postgres
accounts in `tests/cloud/db/test_accounts.py`.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fakes.accounts import InMemoryAccounts
from fakes.github import FakeGitHub

from cloud.api.github import GitHubError, GitHubIdentity
from cloud.api.signin import SIGNIN_TTL, SignInError, SignInService
from cloud.api.tokens import SignInState, TokenSigner, digest

SECRET = "k" * 32
ISSUER = "http://localhost:8080"
VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
CHALLENGE = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
REDIRECT = "http://127.0.0.1:53682/callback"


class FakeClock:
    """Starts at the real time, so JWTs signed alongside it are not born expired."""

    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


def _service(world, *, allowlist=frozenset({"alex110506"})) -> SignInService:
    return SignInService(
        accounts=world.accounts,
        github=world.github,
        signer=world.signer,
        allowlist=allowlist,
        refresh_ttl=timedelta(days=30),
        clock=world.clock,
    )


@pytest.fixture
def world():
    world = SimpleNamespace(
        accounts=InMemoryAccounts(),
        github=FakeGitHub(),
        signer=TokenSigner(SECRET, issuer=ISSUER, access_ttl=timedelta(minutes=15)),
        clock=FakeClock(),
    )
    world.service = _service(world)
    return world


def _start(world, **overrides):
    params = {
        "redirect_uri": REDIRECT,
        "code_challenge": CHALLENGE,
        "code_challenge_method": "S256",
        "app_state": "app-state-1",
    }
    return world.service.start(**(params | overrides))


def _split(url: str) -> tuple[str, dict[str, str]]:
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}{parts.path}"
    return base, {key: values[0] for key, values in parse_qs(parts.query).items()}


async def _finish(world, started, **overrides) -> tuple[str, dict[str, str]]:
    params = {
        "state": _split(started.github_url)[1]["state"],
        "browser_nonce": started.browser_nonce,
        "code": "gh-code",
        "error": None,
    }
    return _split(await world.service.finish(**(params | overrides)))


async def _login_code(world) -> str:
    _, params = await _finish(world, _start(world))
    return params["code"]


async def _signed_in(world):
    return await world.service.redeem(code=await _login_code(world), code_verifier=VERIFIER)


async def _refused(awaitable, code: str) -> SignInError:
    with pytest.raises(SignInError) as raised:
        await awaitable
    assert raised.value.code == code
    return raised.value


# ── Start ────────────────────────────────────────────────────────────────────


def test_start_sends_the_browser_to_github_carrying_the_apps_details(world):
    started = _start(world)

    state = world.signer.read_signin_state(_split(started.github_url)[1]["state"])
    assert state == SignInState(
        redirect_uri=REDIRECT,
        code_challenge=CHALLENGE,
        app_state="app-state-1",
        nonce=started.browser_nonce,
    )


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"redirect_uri": "https://evil.example/callback"}, "invalid_redirect_uri"),
        ({"code_challenge_method": "plain"}, "invalid_request"),
        ({"code_challenge": "not-a-challenge"}, "invalid_request"),
        ({"app_state": ""}, "invalid_request"),
        ({"app_state": "x" * 513}, "invalid_request"),
    ],
)
def test_start_refuses_what_the_flow_cannot_trust(world, overrides, code):
    with pytest.raises(SignInError) as raised:
        _start(world, **overrides)
    assert raised.value.code == code


# ── Finish (GitHub's redirect back) ──────────────────────────────────────────


async def test_finishing_hands_the_app_a_login_code_and_its_own_state(world):
    target, params = await _finish(world, _start(world))

    assert target == REDIRECT
    assert set(params) == {"code", "state"}
    assert params["state"] == "app-state-1"
    [user] = world.accounts.users.values()
    assert (user.github_id, user.login) == (583231, "Alex110506")


async def test_a_login_code_is_stored_only_as_its_hash(world):
    code = await _login_code(world)
    assert code not in world.accounts.login_codes
    assert digest(code) in world.accounts.login_codes


@pytest.mark.parametrize("nonce", [None, "someone-elses-nonce"])
async def test_finishing_in_another_browser_is_refused(world, nonce):
    await _refused(_finish(world, _start(world), browser_nonce=nonce), "invalid_state")
    assert world.github.codes == [], "GitHub's code must not be spent on a refused sign-in"


async def test_a_forged_state_is_refused(world):
    await _refused(_finish(world, _start(world), state="forged"), "invalid_state")


async def test_a_sign_in_left_too_long_is_refused(world):
    long_ago = TokenSigner(
        SECRET,
        issuer=ISSUER,
        access_ttl=timedelta(minutes=15),
        clock=lambda: datetime.now(UTC) - SIGNIN_TTL - timedelta(minutes=1),
    )
    stale = long_ago.signin_state(SignInState(REDIRECT, CHALLENGE, "app-state-1", "n"), SIGNIN_TTL)

    await _refused(
        world.service.finish(state=stale, browser_nonce="n", code="gh-code", error=None),
        "invalid_state",
    )


async def test_someone_off_the_allowlist_is_sent_back_refused_and_not_recorded(world):
    world.github.identity = GitHubIdentity(id=1, login="stranger", avatar_url=None)

    target, params = await _finish(world, _start(world))

    assert target == REDIRECT
    assert params == {"error": "access_denied", "state": "app-state-1"}
    assert world.accounts.users == {}


async def test_the_allowlist_ignores_case(world):
    world.github.identity = GitHubIdentity(id=1, login="ALEX110506", avatar_url=None)
    _, params = await _finish(world, _start(world))
    assert "code" in params


async def test_declining_on_github_is_reported_to_the_app(world):
    _, params = await _finish(world, _start(world), code=None, error="access_denied")

    assert params == {"error": "access_denied", "state": "app-state-1"}
    assert world.github.codes == []


async def test_github_failing_is_reported_to_the_app(world):
    world.github.failure = GitHubError("GitHub is down")
    _, params = await _finish(world, _start(world))
    assert params == {"error": "server_error", "state": "app-state-1"}


# ── Redeem ───────────────────────────────────────────────────────────────────


async def test_a_login_code_and_its_verifier_sign_the_app_in(world):
    pair = await _signed_in(world)

    claims = world.signer.read_access_token(pair.access_token)
    assert claims.login == "Alex110506"
    assert pair.expires_in == 900
    assert digest(pair.refresh_token) in world.accounts.refresh_tokens
    assert pair.refresh_token not in world.accounts.refresh_tokens


async def test_a_login_code_works_once(world):
    code = await _login_code(world)
    await world.service.redeem(code=code, code_verifier=VERIFIER)
    await _refused(world.service.redeem(code=code, code_verifier=VERIFIER), "invalid_grant")


async def test_the_wrong_verifier_is_refused_and_spends_the_code(world):
    """A code gets one guess: whoever intercepted it cannot try verifiers until one fits."""
    code = await _login_code(world)
    await _refused(world.service.redeem(code=code, code_verifier="x" * 43), "invalid_grant")
    await _refused(world.service.redeem(code=code, code_verifier=VERIFIER), "invalid_grant")


async def test_a_login_code_expires(world):
    code = await _login_code(world)
    world.clock.advance(minutes=3)
    await _refused(world.service.redeem(code=code, code_verifier=VERIFIER), "invalid_grant")


# ── Refresh and sign-out ─────────────────────────────────────────────────────


async def test_refreshing_trades_the_token_for_a_new_pair(world):
    first = await _signed_in(world)
    second = await world.service.refresh(first.refresh_token)

    assert second.refresh_token != first.refresh_token
    assert world.signer.read_access_token(second.access_token).login == "Alex110506"


async def test_reusing_a_refresh_token_ends_every_session_of_that_user(world):
    first = await _signed_in(world)
    second = await world.service.refresh(first.refresh_token)

    await _refused(world.service.refresh(first.refresh_token), "invalid_grant")
    await _refused(world.service.refresh(second.refresh_token), "invalid_grant")


async def test_a_refresh_token_expires(world):
    pair = await _signed_in(world)
    world.clock.advance(days=31)
    await _refused(world.service.refresh(pair.refresh_token), "invalid_grant")


async def test_someone_taken_off_the_allowlist_cannot_refresh(world):
    pair = await _signed_in(world)
    world.service = _service(world, allowlist=frozenset())

    error = await _refused(world.service.refresh(pair.refresh_token), "access_denied")

    assert error.status == 403


async def test_signing_out_ends_the_session(world):
    pair = await _signed_in(world)

    await world.service.sign_out(pair.refresh_token)
    await world.service.sign_out(pair.refresh_token)  # twice is harmless

    await _refused(world.service.refresh(pair.refresh_token), "invalid_grant")
