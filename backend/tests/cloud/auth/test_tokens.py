"""Access tokens, sign-in state, one-time secrets, PKCE and loopback redirects."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from cloud.api.tokens import (
    ACCESS_AUDIENCE,
    AccessClaims,
    InvalidTokenError,
    SignInState,
    TokenSigner,
    digest,
    is_loopback_redirect,
    is_valid_challenge,
    new_secret,
    s256_challenge,
    utc_now,
    verifier_matches,
)

SECRET = "s" * 32
ISSUER = "http://localhost:8080"
#: RFC 7636, Appendix B.
RFC_VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
RFC_CHALLENGE = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
STATE = SignInState(
    redirect_uri="http://127.0.0.1:53682/callback",
    code_challenge=RFC_CHALLENGE,
    app_state="app-state",
    nonce="nonce",
)


def _signer(*, secret: str = SECRET, clock=utc_now) -> TokenSigner:
    return TokenSigner(secret, issuer=ISSUER, access_ttl=timedelta(minutes=15), clock=clock)


# ── Access tokens ────────────────────────────────────────────────────────────


def test_an_access_token_reads_back_as_its_user():
    token = _signer().access_token("user-1", "octocat")
    assert _signer().read_access_token(token) == AccessClaims(user_id="user-1", login="octocat")


def test_an_access_token_lasts_its_ttl():
    assert _signer().access_ttl_seconds == 900


def test_an_expired_access_token_is_refused():
    an_hour_ago = _signer(clock=lambda: datetime.now(UTC) - timedelta(hours=1))
    with pytest.raises(InvalidTokenError):
        _signer().read_access_token(an_hour_ago.access_token("user-1", "octocat"))


def test_a_token_signed_with_another_key_is_refused():
    forged = _signer(secret="o" * 32).access_token("user-1", "octocat")
    with pytest.raises(InvalidTokenError):
        _signer().read_access_token(forged)


def test_an_unsigned_token_is_refused():
    now = datetime.now(UTC)
    unsigned = jwt.encode(
        {
            "sub": "user-1",
            "login": "octocat",
            "iss": ISSUER,
            "aud": ACCESS_AUDIENCE,
            "iat": now,
            "exp": now + timedelta(minutes=5),
        },
        "",
        algorithm="none",
    )
    with pytest.raises(InvalidTokenError):
        _signer().read_access_token(unsigned)


def test_a_token_from_another_issuer_is_refused():
    other = TokenSigner(SECRET, issuer="https://elsewhere.example", access_ttl=timedelta(minutes=5))
    with pytest.raises(InvalidTokenError):
        _signer().read_access_token(other.access_token("user-1", "octocat"))


@pytest.mark.parametrize("garbage", ["", "not.a.jwt", "a.b.c"])
def test_garbage_is_refused(garbage):
    with pytest.raises(InvalidTokenError):
        _signer().read_access_token(garbage)


# ── Sign-in state ────────────────────────────────────────────────────────────


def test_sign_in_state_reads_back_unchanged():
    signer = _signer()
    assert signer.read_signin_state(signer.signin_state(STATE, timedelta(minutes=10))) == STATE


def test_sign_in_state_and_access_tokens_cannot_stand_in_for_each_other():
    signer = _signer()
    with pytest.raises(InvalidTokenError):
        signer.read_access_token(signer.signin_state(STATE, timedelta(minutes=10)))
    with pytest.raises(InvalidTokenError):
        signer.read_signin_state(signer.access_token("user-1", "octocat"))


def test_expired_sign_in_state_is_refused():
    started_long_ago = _signer(clock=lambda: datetime.now(UTC) - timedelta(minutes=11))
    with pytest.raises(InvalidTokenError):
        _signer().read_signin_state(started_long_ago.signin_state(STATE, timedelta(minutes=10)))


# ── Secrets ──────────────────────────────────────────────────────────────────


def test_new_secrets_are_long_and_never_repeat():
    secrets = {new_secret() for _ in range(100)}
    assert len(secrets) == 100
    assert min(len(secret) for secret in secrets) >= 43


def test_the_stored_form_of_a_secret_is_its_sha256():
    assert digest("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# ── PKCE ─────────────────────────────────────────────────────────────────────


def test_the_s256_challenge_matches_rfc_7636():
    assert s256_challenge(RFC_VERIFIER) == RFC_CHALLENGE


def test_a_verifier_matches_only_its_own_challenge():
    assert verifier_matches(RFC_VERIFIER, RFC_CHALLENGE)
    assert not verifier_matches("x" * 43, RFC_CHALLENGE)


@pytest.mark.parametrize("verifier", ["too-short", "a" * 129, "a" * 42 + " ", "ä" * 43])
def test_a_malformed_verifier_never_matches(verifier):
    assert not verifier_matches(verifier, s256_challenge("a" * 43))


@pytest.mark.parametrize(
    ("challenge", "valid"),
    [(RFC_CHALLENGE, True), ("short", False), (RFC_CHALLENGE + "A", False), ("=" * 43, False)],
)
def test_challenges_must_look_like_s256(challenge, valid):
    assert is_valid_challenge(challenge) is valid


# ── Loopback redirects ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "uri",
    ["http://127.0.0.1:53682/callback", "http://127.0.0.1:1/", "http://[::1]:53682/callback"],
)
def test_loopback_addresses_are_accepted(uri):
    assert is_loopback_redirect(uri)


@pytest.mark.parametrize(
    "uri",
    [
        "https://127.0.0.1:53682/callback",  # the app's listener speaks plain http
        "http://127.0.0.1/callback",  # no port: not the app's listener
        "http://127.0.0.1:0/callback",
        "http://127.0.0.1:99999/callback",
        "http://localhost:53682/callback",  # a name can be re-pointed; an address cannot
        "http://127.0.0.1.evil.example:53682/callback",
        "http://evil.example:53682/callback",
        "http://10.0.0.5:53682/callback",
        "http://user@127.0.0.1:53682/callback",
        "http://127.0.0.1:53682/callback?next=https://evil.example",
        "http://127.0.0.1:53682/callback#fragment",
        "javascript:alert(1)",
        "",
    ],
)
def test_anything_else_is_refused(uri):
    assert not is_loopback_redirect(uri)
