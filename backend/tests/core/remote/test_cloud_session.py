"""
The runtime's session with the Cloud API, against a scripted stand-in for the
API's token handling. The keychain is the in-memory `fake_keyring`.
"""
from __future__ import annotations

import asyncio
import itertools
import json

import httpx
import pytest

from core.errors import CloudError, CloudUnavailableError, NotSignedInError, QuotaExceededError
from core.remote.session import CloudSession

API = "https://api.example"
KEYCHAIN_ENTRY = ("interroai", "cloud_refresh_token:https://api.example")
VERIFIER = "v" * 43


def _refusal(status: int, code: str) -> httpx.Response:
    return httpx.Response(status, json={"detail": {"code": code, "message": f"Refused: {code}"}})


class FakeApi:
    """Just enough of the Cloud API's token handling to drive a session."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.valid_access: set[str] = set()
        self.live_refresh: set[str] = set()
        self.revoked: list[str] = []
        self.refreshes = 0
        self.refresh_delay = 0.0
        self.expires_in = 900
        self.unreachable = False
        self._serial = itertools.count(1)

    def issue(self) -> dict:
        serial = next(self._serial)
        access, refresh = f"access-{serial}", f"refresh-{serial}"
        self.valid_access.add(access)
        self.live_refresh.add(refresh)
        return {
            "access_token": access,
            "refresh_token": refresh,
            "token_type": "Bearer",
            "expires_in": self.expires_in,
        }

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.unreachable:
            raise httpx.ConnectError("unreachable", request=request)
        path = request.url.path
        self.calls.append(path)

        if path == "/auth/token":
            body = json.loads(request.content)
            if body["grant_type"] == "authorization_code":
                if body["code"] != "good-code":
                    return _refusal(400, "invalid_grant")
                return httpx.Response(200, json=self.issue())
            self.refreshes += 1
            await asyncio.sleep(self.refresh_delay)
            if body["refresh_token"] not in self.live_refresh:
                return _refusal(400, "invalid_grant")
            self.live_refresh.discard(body["refresh_token"])
            return httpx.Response(200, json=self.issue())

        if path == "/auth/logout":
            self.revoked.append(json.loads(request.content)["refresh_token"])
            return httpx.Response(204)

        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if token not in self.valid_access:
            return _refusal(401, "invalid_token")
        if path == "/me":
            return httpx.Response(
                200, json={"id": "user-1", "login": "octocat", "avatar_url": None}
            )
        if path == "/quota":
            return _refusal(429, "quota_exceeded")
        if path == "/broken":
            return _refusal(503, "llm_unavailable")
        if path == "/refused":
            return _refusal(400, "model_not_allowed")
        if path == "/burst":
            return _refusal(429, "rate_limited")
        return httpx.Response(200, json={"ok": True})


@pytest.fixture
def api():
    return FakeApi()


def _session(api) -> CloudSession:
    return CloudSession(API, transport=httpx.MockTransport(api))


@pytest.fixture
def session(api, fake_keyring):
    return _session(api)


async def _sign_in(session: CloudSession) -> CloudSession:
    await session.sign_in(code="good-code", code_verifier=VERIFIER)
    return session


# ── Signing in and out ───────────────────────────────────────────────────────


async def test_signing_in_keeps_the_refresh_token_in_the_keychain(session, fake_keyring):
    identity = await session.sign_in(code="good-code", code_verifier=VERIFIER)

    assert identity.login == "octocat"
    assert fake_keyring[KEYCHAIN_ENTRY] == "refresh-1"
    assert session.signed_in


async def test_a_refused_login_code_signs_nobody_in(session, fake_keyring):
    with pytest.raises(CloudError) as raised:
        await session.sign_in(code="bad-code", code_verifier=VERIFIER)

    assert raised.value.code == "invalid_grant"
    assert KEYCHAIN_ENTRY not in fake_keyring
    assert not session.signed_in


async def test_signing_out_revokes_the_session_and_forgets_it(session, api, fake_keyring):
    await _sign_in(session)

    await session.sign_out()

    assert api.revoked == ["refresh-1"]
    assert KEYCHAIN_ENTRY not in fake_keyring
    with pytest.raises(NotSignedInError):
        await session.request("GET", "/things")


async def test_signing_out_while_offline_still_forgets_the_session(session, api, fake_keyring):
    await _sign_in(session)
    api.unreachable = True

    await session.sign_out()

    assert KEYCHAIN_ENTRY not in fake_keyring


# ── Tokens ───────────────────────────────────────────────────────────────────


async def test_requests_carry_the_access_token(session, api):
    await _sign_in(session)

    assert (await session.request("GET", "/things")).json() == {"ok": True}
    assert api.refreshes == 0


async def test_a_rejected_access_token_is_refreshed_once_and_the_request_retried(
    session, api, fake_keyring
):
    await _sign_in(session)
    api.valid_access.clear()

    assert (await session.request("GET", "/things")).status_code == 200

    assert api.refreshes == 1
    assert fake_keyring[KEYCHAIN_ENTRY] == "refresh-2", "the rotated token replaces the spent one"


async def test_requests_meeting_an_expired_token_together_share_one_refresh(session, api):
    """Refresh tokens rotate: refreshing twice with the same one would end the session."""
    await _sign_in(session)
    api.valid_access.clear()
    api.refresh_delay = 0.05

    responses = await asyncio.gather(*(session.request("GET", "/things") for _ in range(5)))

    assert {response.status_code for response in responses} == {200}
    assert api.refreshes == 1


async def test_a_token_about_to_expire_is_refreshed_before_it_is_sent(session, api):
    api.expires_in = 10  # inside the safety margin
    await _sign_in(session)
    api.calls.clear()

    await session.request("GET", "/things")

    assert api.calls == ["/auth/token", "/things"]


async def test_a_restarted_runtime_picks_the_session_up_from_the_keychain(api, fake_keyring):
    await _sign_in(_session(api))
    restarted = _session(api)

    assert restarted.signed_in
    assert (await restarted.request("GET", "/things")).status_code == 200
    assert api.refreshes == 1


async def test_a_session_ended_on_the_server_ends_here_too(session, api, fake_keyring):
    await _sign_in(session)
    api.valid_access.clear()
    api.live_refresh.clear()

    with pytest.raises(NotSignedInError):
        await session.request("GET", "/things")

    assert KEYCHAIN_ENTRY not in fake_keyring
    assert not session.signed_in


async def test_without_a_session_nothing_is_sent(session, api):
    with pytest.raises(NotSignedInError):
        await session.request("GET", "/things")
    assert api.calls == []


# ── Refusals and failures ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "error", "code"),
    [
        ("/quota", QuotaExceededError, "quota_exceeded"),
        ("/broken", CloudUnavailableError, "llm_unavailable"),
        ("/refused", CloudError, "model_not_allowed"),
    ],
)
async def test_refusals_become_errors_carrying_the_apis_code(session, path, error, code):
    await _sign_in(session)

    with pytest.raises(error) as raised:
        await session.request("GET", path)

    assert raised.value.code == code


async def test_an_unreachable_api_is_cloud_unavailable(session, api):
    await _sign_in(session)
    api.unreachable = True

    with pytest.raises(CloudUnavailableError):
        await session.request("GET", "/things")


async def test_a_rate_limit_is_not_mistaken_for_the_daily_quota(session):
    await _sign_in(session)

    with pytest.raises(CloudError) as raised:
        await session.request("GET", "/burst")

    assert raised.value.code == "rate_limited"
    assert not isinstance(raised.value, QuotaExceededError)


async def test_a_stream_is_handed_over_only_once_its_status_is_good(session, api):
    await _sign_in(session)
    api.valid_access.clear()  # the stream goes through the same refresh

    async with session.stream("GET", "/things") as response:
        assert response.status_code == 200
        assert json.loads(await response.aread()) == {"ok": True}

    assert api.refreshes == 1
