"""
Signing in to the platform, from the desktop app's button to its first tokens.

  1. The app starts a listener on 127.0.0.1, makes a PKCE verifier, and opens
     `/auth/github/start` in the browser with the listener's address and the
     verifier's challenge.
  2. The API sends the browser on to GitHub. The listener address and the
     challenge travel in GitHub's `state`, signed; a cookie ties the sign-in to
     this browser.
  3. GitHub sends the browser back to `/auth/github/callback`. The API reads the
     profile, checks the allowlist, and redirects to the app's listener with a
     login code: one use, two minutes, bound to the challenge.
  4. The app redeems the code at `/auth/token` with its verifier and gets an
     access token and a refresh token.

Nothing a browser, a redirect or a log line carries is enough on its own. The
code is useless without the verifier, which never leaves the app, and the
verifier is useless without a code, which only reaches the app's own listener.

Refresh tokens rotate: each one is spent when used and replaced by a new one. A
token presented a second time means two parties hold it, and the API cannot tell
which one is the thief, so every refresh token that user has is revoked and both
must sign in again.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol
from urllib.parse import urlencode

from cloud.api.github import GitHubError, GitHubIdentity
from cloud.api.tokens import (
    Clock,
    InvalidTokenError,
    SignInState,
    TokenSigner,
    digest,
    is_loopback_redirect,
    is_valid_challenge,
    new_secret,
    utc_now,
    verifier_matches,
)
from cloud.db.accounts import Accounts, UserRecord

logger = logging.getLogger(__name__)

LOGIN_CODE_TTL = timedelta(minutes=2)
#: How long a person may take on GitHub's pages.
SIGNIN_TTL = timedelta(minutes=10)
_MAX_APP_STATE = 512


class IdentityProvider(Protocol):
    def authorize_url(self, state: str) -> str: ...

    async def identify(self, code: str) -> GitHubIdentity: ...


class SignInError(Exception):
    """A refused step. `code` is stable and meant for the app; `message` is for a person."""

    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True)
class SignInStart:
    github_url: str
    #: Set as a cookie on the browser; the callback must present it back.
    browser_nonce: str


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    expires_in: int


class SignInService:
    def __init__(
        self,
        *,
        accounts: Accounts,
        github: IdentityProvider,
        signer: TokenSigner,
        allowlist: frozenset[str],
        refresh_ttl: timedelta,
        clock: Clock = utc_now,
    ) -> None:
        self._accounts = accounts
        self._github = github
        self._signer = signer
        #: Lower-cased: GitHub logins are case-insensitive.
        self._allowlist = allowlist
        self._refresh_ttl = refresh_ttl
        self._clock = clock

    def start(
        self,
        *,
        redirect_uri: str,
        code_challenge: str,
        code_challenge_method: str,
        app_state: str,
    ) -> SignInStart:
        if not is_loopback_redirect(redirect_uri):
            raise SignInError(
                "invalid_redirect_uri",
                "Sign-in must return to the InterroAI app on this computer.",
            )
        if code_challenge_method != "S256" or not is_valid_challenge(code_challenge):
            raise SignInError("invalid_request", "Sign-in needs a PKCE S256 code challenge.")
        if not app_state or len(app_state) > _MAX_APP_STATE:
            raise SignInError("invalid_request", "Sign-in needs a state value from the app.")

        nonce = new_secret()
        state = self._signer.signin_state(
            SignInState(
                redirect_uri=redirect_uri,
                code_challenge=code_challenge,
                app_state=app_state,
                nonce=nonce,
            ),
            SIGNIN_TTL,
        )
        return SignInStart(github_url=self._github.authorize_url(state), browser_nonce=nonce)

    async def finish(
        self,
        *,
        state: str,
        browser_nonce: str | None,
        code: str | None,
        error: str | None,
    ) -> str:
        """
        Handle GitHub's redirect and return where to send the browser next.

        Once the state checks out, every outcome, refusals included, goes back
        to the app's listener so the app can tell the user what happened. Before
        that, nothing does: an unverified `redirect_uri` is not somewhere to
        send anything.
        """
        try:
            flow = self._signer.read_signin_state(state)
        except InvalidTokenError:
            raise SignInError(
                "invalid_state",
                "This sign-in link has expired or did not come from InterroAI. "
                "Start again from the app.",
            ) from None
        if browser_nonce is None or not secrets.compare_digest(browser_nonce, flow.nonce):
            raise SignInError(
                "invalid_state",
                "This sign-in was started in a different browser. Start again from the app.",
            )

        if error or not code:
            # Most often: the user pressed Cancel on GitHub's consent page.
            declined = error == "access_denied"
            return _to_app(flow, error="access_denied" if declined else "server_error")

        try:
            identity = await self._github.identify(code)
        except GitHubError:
            logger.warning("GitHub sign-in failed", exc_info=True)
            return _to_app(flow, error="server_error")

        if identity.login.lower() not in self._allowlist:
            # No user row is created for someone who is refused.
            logger.info("Refused sign-in for GitHub user %s: not on the allowlist", identity.login)
            return _to_app(flow, error="access_denied")

        user = await self._accounts.upsert_github_user(
            identity.id, identity.login, identity.avatar_url
        )
        login_code = new_secret()
        now = self._clock()
        await self._accounts.add_login_code(
            digest(login_code),
            user_id=user.id,
            code_challenge=flow.code_challenge,
            expires_at=now + LOGIN_CODE_TTL,
        )
        # Where the two credential tables get their housekeeping: sign-ins are
        # rare, and a lapsed row is read by nothing. Failing at it is not this
        # user's problem — they are in the middle of signing in.
        try:
            await self._accounts.purge_expired(now=now)
        except Exception:  # noqa: BLE001
            logger.warning("Could not clear out expired sign-in credentials", exc_info=True)
        return _to_app(flow, code=login_code)

    async def redeem(self, *, code: str, code_verifier: str) -> TokenPair:
        """Trade a login code and the verifier it is bound to for the first tokens."""
        # The code is spent before the verifier is checked, so each code gets
        # exactly one guess and cannot be brute-forced.
        redeemed = await self._accounts.redeem_login_code(digest(code), now=self._clock())
        if redeemed is None or not verifier_matches(code_verifier, redeemed.code_challenge):
            raise SignInError("invalid_grant", "The sign-in code is invalid or has expired.")
        user = await self._accounts.get_user(redeemed.user_id)
        if user is None:
            raise SignInError("invalid_grant", "The sign-in code is invalid or has expired.")
        return await self._issue(user)

    async def refresh(self, refresh_token: str) -> TokenPair:
        now = self._clock()
        outcome = await self._accounts.use_refresh_token(digest(refresh_token), now=now)

        if outcome.status == "reused" and outcome.user_id is not None:
            logger.warning(
                "A spent refresh token was presented again; revoking every session of user %s",
                outcome.user_id,
            )
            await self._accounts.revoke_all_refresh_tokens(outcome.user_id, now=now)
        if outcome.status != "ok" or outcome.user_id is None:
            raise SignInError("invalid_grant", "The session has ended. Sign in again.")

        user = await self._accounts.get_user(outcome.user_id)
        if user is None:
            raise SignInError("invalid_grant", "The session has ended. Sign in again.")
        if user.login.lower() not in self._allowlist:
            # Taken off the allowlist since signing in: that ends it within one
            # access token's lifetime.
            await self._accounts.revoke_all_refresh_tokens(user.id, now=now)
            raise SignInError(
                "access_denied",
                "This account is no longer allowed to use InterroAI.",
                status=403,
            )
        return await self._issue(user)

    async def sign_out(self, refresh_token: str) -> None:
        await self._accounts.revoke_refresh_token(digest(refresh_token), now=self._clock())

    async def _issue(self, user: UserRecord) -> TokenPair:
        refresh_token = new_secret()
        await self._accounts.add_refresh_token(
            digest(refresh_token),
            user_id=user.id,
            expires_at=self._clock() + self._refresh_ttl,
        )
        return TokenPair(
            access_token=self._signer.access_token(user.id, user.login),
            refresh_token=refresh_token,
            expires_in=self._signer.access_ttl_seconds,
        )


def _to_app(flow: SignInState, **params: str) -> str:
    return f"{flow.redirect_uri}?{urlencode({**params, 'state': flow.app_state})}"
