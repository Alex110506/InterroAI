"""
Checking Auth0's access tokens, and reading a GitHub profile from `/userinfo`.

The fake tenant signs real RS256 tokens, so every check here runs the same code
the API runs in production; only the key set and the HTTP endpoint are local.
"""
from __future__ import annotations

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fakes.auth0 import AUDIENCE, FakeAuth0

from cloud.auth.auth0 import Auth0Unavailable, GitHubProfile, InvalidTokenError, github_id


@pytest.fixture
def tenant() -> FakeAuth0:
    return FakeAuth0().add_user(583231, "Alex110506", "https://avatars.example/583231")


async def test_a_token_auth0_signed_for_this_api_names_its_subject(tenant):
    assert await tenant.client().verify(tenant.token(583231)) == "github|583231"


async def test_a_token_whose_only_audience_is_this_api_is_accepted(tenant):
    assert await tenant.client().verify(tenant.token(583231, audience=AUDIENCE)) == "github|583231"


@pytest.mark.parametrize(
    "wrong",
    [
        {"audience": "https://some-other-api.example"},
        {"issuer": "https://someone-else.auth0.example/"},
        {"expires_in": -60},
    ],
    ids=["audience", "issuer", "expired"],
)
async def test_a_token_meant_for_something_else_is_refused(tenant, wrong):
    with pytest.raises(InvalidTokenError):
        await tenant.client().verify(tenant.token(583231, **wrong))


async def test_a_token_signed_by_another_key_is_refused(tenant):
    stranger = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(InvalidTokenError):
        await tenant.client().verify(tenant.token(583231, key=stranger))


async def test_a_symmetric_token_is_refused_whatever_its_header_says(tenant):
    # The algorithm comes from a fixed list, never from the token itself.
    forged = tenant.token(583231, algorithm="HS256", key="x" * 32)
    with pytest.raises(InvalidTokenError):
        await tenant.client().verify(forged)


async def test_garbage_is_an_invalid_token(tenant):
    with pytest.raises(InvalidTokenError):
        await tenant.client().verify("not-a-jwt")


async def test_a_key_auth0_does_not_publish_is_an_invalid_token(tenant):
    tenant.jwks.failure = jwt.PyJWKClientError("Unable to find a signing key")
    with pytest.raises(InvalidTokenError):
        await tenant.client().verify(tenant.token(583231))


async def test_an_unreachable_key_set_is_auth0_unavailable_not_a_bad_token(tenant):
    tenant.jwks.failure = jwt.PyJWKClientConnectionError("connection refused")
    with pytest.raises(Auth0Unavailable):
        await tenant.client().verify(tenant.token(583231))


async def test_userinfo_gives_the_github_login_and_avatar(tenant):
    assert await tenant.client().profile(tenant.token(583231)) == GitHubProfile(
        github_id=583231, login="Alex110506", avatar_url="https://avatars.example/583231"
    )


async def test_userinfo_refusing_the_token_is_an_invalid_token(tenant):
    with pytest.raises(InvalidTokenError):
        await tenant.client().profile(tenant.token(999))


async def test_userinfo_down_is_auth0_unavailable(tenant):
    tenant.down = True
    with pytest.raises(Auth0Unavailable):
        await tenant.client().profile(tenant.token(583231))


def test_the_issuer_is_the_tenant_url_with_its_trailing_slash(tenant):
    assert tenant.config.issuer == "https://interroai.auth0.example/"


@pytest.mark.parametrize(
    ("subject", "expected"),
    [("github|583231", 583231), ("google-oauth2|583231", None), ("github|abc", None), ("", None)],
)
def test_only_github_subjects_name_a_github_account(subject, expected):
    assert github_id(subject) == expected
