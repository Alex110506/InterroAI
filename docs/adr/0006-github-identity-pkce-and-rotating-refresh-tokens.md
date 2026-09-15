# ADR 0006: GitHub for identity, PKCE through a loopback redirect, rotating refresh tokens

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

The desktop app has to sign users in, and the cloud has to know who is spending
its OpenAI key. The app is a public client: anything shipped inside it can be
extracted, so it cannot hold a client secret.

## Decision

- **GitHub is used for identity only.** The API requests no scopes and discards
  GitHub's token once it has read the profile. Access is by an allowlist of
  GitHub logins, checked at sign-in and again on every refresh.
- **The API issues its own credentials, through PKCE.** The Electron main process
  makes a code verifier, listens on a free `127.0.0.1` port, and opens the API's
  start URL in the system browser. The challenge and the loopback address travel
  through GitHub's `state`, signed. After GitHub, the API redirects to the loopback
  address with a one-time, two-minute login code bound to that challenge. Only
  loopback IP redirect URIs are accepted, and a cookie ties the callback to the
  browser that started the sign-in.
- **The runtime holds the session.** It redeems the code with the verifier for a
  15-minute access token (an HS256 JWT) and a 30-day refresh token, which it keeps
  in the OS keychain. The Electron app never sees either token.
- **Refresh tokens rotate.** Each is spent on use and replaced. Presenting a spent
  one revokes every session of that user, since the API cannot tell the thief from
  the owner. Only sha256 digests of login codes and refresh tokens are stored.

## Alternatives considered

- **GitHub's device flow.** No redirect to handle, but typing a code into a
  browser is a worse experience when the app can simply open one.
- **Sign-in inside an embedded webview.** It lets the app observe what the user
  types into GitHub, which is exactly what OAuth for native apps (RFC 8252) warns
  against.
- **Long-lived access tokens.** Nothing to refresh, but no way to end a session
  short of rotating the signing key.

## Consequences

- An intercepted login code is useless without the verifier, and the verifier
  never leaves the user's machine except to go to the runtime.
- An access token cannot be revoked before it expires; fifteen minutes bounds that.
- Requests meeting an expired token together share one refresh in the runtime.
  Two refreshes with the same token would look like theft and sign the user out.
