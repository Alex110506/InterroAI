"""The GitHub client, against httpx's mock transport. No request leaves the test."""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from cloud.api.github import TOKEN_URL, USER_URL, GitHubError, GitHubIdentity, GitHubOAuth

CALLBACK = "http://localhost:8080/auth/github/callback"


def _github(handler) -> GitHubOAuth:
    return GitHubOAuth(
        client_id="the-client-id",
        client_secret="the-client-secret",
        callback_url=CALLBACK,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _profile_after_token(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url == httpx.URL(TOKEN_URL):
            return httpx.Response(200, json={"access_token": "gho_x", "scope": ""})
        return httpx.Response(
            200, json={"id": 583231, "login": "octocat", "avatar_url": "https://avatars/583231"}
        )

    return handler


def test_the_authorize_url_asks_for_no_scopes():
    url = _github(lambda request: httpx.Response(500)).authorize_url("the-state")
    parts = urlsplit(url)

    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "https://github.com/login/oauth/authorize"
    assert parse_qs(parts.query) == {
        "client_id": ["the-client-id"],
        "redirect_uri": [CALLBACK],
        "state": ["the-state"],
        "allow_signup": ["false"],
    }


async def test_identify_trades_the_code_for_the_profile():
    requests: list[httpx.Request] = []

    identity = await _github(_profile_after_token(requests)).identify("gh-code")

    assert identity == GitHubIdentity(
        id=583231, login="octocat", avatar_url="https://avatars/583231"
    )
    exchange, profile = requests
    assert (exchange.method, exchange.url) == ("POST", httpx.URL(TOKEN_URL))
    form = parse_qs(exchange.content.decode())
    assert form["code"] == ["gh-code"]
    assert form["client_secret"] == ["the-client-secret"]
    assert form["redirect_uri"] == [CALLBACK]
    assert exchange.headers["Accept"] == "application/json"
    assert (profile.method, profile.url) == ("GET", httpx.URL(USER_URL))
    assert profile.headers["Authorization"] == "Bearer gho_x"


async def test_a_refused_code_is_a_github_error():
    """GitHub reports a bad code with a 200 and an `error` field, not an error status."""

    def handler(request):
        return httpx.Response(200, json={"error": "bad_verification_code"})

    with pytest.raises(GitHubError, match="bad_verification_code"):
        await _github(handler).identify("gh-code")


async def test_a_failing_profile_request_is_a_github_error():
    def handler(request):
        if request.url == httpx.URL(TOKEN_URL):
            return httpx.Response(200, json={"access_token": "gho_x"})
        return httpx.Response(401, json={"message": "Bad credentials"})

    with pytest.raises(GitHubError):
        await _github(handler).identify("gh-code")


async def test_github_being_unreachable_is_a_github_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(GitHubError):
        await _github(handler).identify("gh-code")


async def test_a_profile_without_an_id_is_a_github_error():
    def handler(request):
        if request.url == httpx.URL(TOKEN_URL):
            return httpx.Response(200, json={"access_token": "gho_x"})
        return httpx.Response(200, json={"login": "octocat"})

    with pytest.raises(GitHubError):
        await _github(handler).identify("gh-code")
