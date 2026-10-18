"""
From an Auth0 access token to the user a request runs as.

The first request a GitHub account makes reads its profile from Auth0 and
creates its user row; every later one finds the row. The allowlist is checked
on every request, so taking someone off it takes effect at once.
"""
from __future__ import annotations

import pytest
from fakes.accounts import InMemoryAccounts
from fakes.auth0 import FakeAuth0

from cloud.auth.auth0 import Auth0Unavailable, InvalidTokenError
from cloud.auth.identity import AccessClaims, AccessDeniedError


@pytest.fixture
def tenant() -> FakeAuth0:
    return (
        FakeAuth0()
        .add_user(583231, "Alex110506", "https://avatars.example/583231")
        .add_user(777, "stranger")
    )


@pytest.fixture
def accounts() -> InMemoryAccounts:
    return InMemoryAccounts()


async def test_a_first_request_creates_the_user_from_its_github_profile(tenant, accounts):
    claims = await tenant.identity(accounts).resolve(tenant.token(583231))

    user = await accounts.get_by_github_id(583231)
    assert claims == AccessClaims(user_id=user.id, login="Alex110506")
    assert user.avatar_url == "https://avatars.example/583231"


async def test_a_known_user_is_found_without_asking_auth0_again(tenant, accounts):
    identity = tenant.identity(accounts)
    first = await identity.resolve(tenant.token(583231))
    second = await identity.resolve(tenant.token(583231))

    assert first == second
    assert tenant.userinfo_calls == 1


async def test_the_allowlist_ignores_case(tenant, accounts):
    identity = tenant.identity(accounts, allowlist=("ALEX110506",))
    assert (await identity.resolve(tenant.token(583231))).login == "Alex110506"


async def test_someone_off_the_allowlist_is_refused_and_gets_no_user_row(tenant, accounts):
    with pytest.raises(AccessDeniedError):
        await tenant.identity(accounts).resolve(tenant.token(777))
    assert await accounts.get_by_github_id(777) is None


async def test_a_refused_account_is_not_looked_up_on_every_request(tenant, accounts):
    identity = tenant.identity(accounts)
    for _ in range(3):
        with pytest.raises(AccessDeniedError):
            await identity.resolve(tenant.token(777))
    assert tenant.userinfo_calls == 1


async def test_taking_someone_off_the_allowlist_ends_access_at_once(tenant, accounts):
    await tenant.identity(accounts).resolve(tenant.token(583231))

    with pytest.raises(AccessDeniedError):
        await tenant.identity(accounts, allowlist=()).resolve(tenant.token(583231))


async def test_an_empty_allowlist_admits_nobody(tenant, accounts):
    with pytest.raises(AccessDeniedError):
        await tenant.identity(accounts, allowlist=()).resolve(tenant.token(583231))


async def test_a_login_through_anything_but_github_is_refused(tenant, accounts):
    with pytest.raises(AccessDeniedError):
        await tenant.identity(accounts).resolve(tenant.token(subject="google-oauth2|583231"))
    assert tenant.userinfo_calls == 0


async def test_an_invalid_token_is_refused_before_anything_is_looked_up(tenant, accounts):
    with pytest.raises(InvalidTokenError):
        await tenant.identity(accounts).resolve(tenant.token(583231, expires_in=-60))
    assert tenant.userinfo_calls == 0


async def test_auth0_down_on_a_first_request_is_unavailable(tenant, accounts):
    tenant.down = True
    with pytest.raises(Auth0Unavailable):
        await tenant.identity(accounts).resolve(tenant.token(583231))


async def test_auth0_down_does_not_stop_a_known_user(tenant, accounts):
    identity = tenant.identity(accounts)
    claims = await identity.resolve(tenant.token(583231))
    tenant.down = True
    assert await identity.resolve(tenant.token(583231)) == claims


async def test_refreshing_the_profile_follows_a_github_rename(tenant, accounts):
    identity = tenant.identity(accounts, allowlist=("alex110506", "alex-renamed"))
    claims = await identity.resolve(tenant.token(583231))
    tenant.add_user(583231, "alex-renamed", "https://avatars.example/new")

    user = await identity.refresh_profile(tenant.token(583231), claims)

    assert (user.id, user.login, user.avatar_url) == (
        claims.user_id,
        "alex-renamed",
        "https://avatars.example/new",
    )


async def test_a_rename_off_the_allowlist_is_refused(tenant, accounts):
    identity = tenant.identity(accounts)
    claims = await identity.resolve(tenant.token(583231))
    tenant.add_user(583231, "alex-renamed")

    with pytest.raises(AccessDeniedError):
        await identity.refresh_profile(tenant.token(583231), claims)


async def test_refreshing_the_profile_while_auth0_is_down_keeps_the_stored_one(tenant, accounts):
    identity = tenant.identity(accounts)
    claims = await identity.resolve(tenant.token(583231))
    tenant.down = True

    user = await identity.refresh_profile(tenant.token(583231), claims)
    assert (user.id, user.login) == (claims.user_id, "Alex110506")
