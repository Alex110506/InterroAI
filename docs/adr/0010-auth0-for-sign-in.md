# ADR 0010: Auth0 runs sign-in; the API only checks its tokens

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** [ADR 0006](0006-github-identity-pkce-and-rotating-refresh-tokens.md)

## Context

ADR 0006 had the Cloud API act as its own authorization server. It ran
GitHub's OAuth flow, issued login codes bound to a PKCE challenge, signed HS256
access tokens, and stored and rotated refresh tokens with reuse detection. That
was about 900 lines across `cloud/auth/`, two tables, and the matching refresh
logic in the runtime, all of it security-critical code to maintain. A managed
identity provider does the same job as a standard product.

## Decision

- **Auth0 issues every credential.** The Electron app runs the authorization
  code flow with PKCE against Auth0's `/authorize`, with `connection=github` so
  the browser goes straight to GitHub. GitHub is the only connection enabled for
  the application. The runtime redeems the code at Auth0's `/oauth/token`,
  refreshes there, and revokes there on sign-out. Auth0's refresh-token rotation
  with reuse detection replaces the API's own.
- **The API only verifies.** `current_user` checks each access token's RS256
  signature against the tenant's published keys, its issuer, its audience (this
  API's identifier) and its expiry. The API stores no codes and no tokens;
  migration 0002 drops both tables.
- **Users are created on first sight.** A token names its user only as
  `github|<numeric id>`, the same GitHub id the `users` table already keyed on,
  so existing rows carry over. On a user's first request the API reads the login
  and avatar from Auth0's `/userinfo`, checks the allowlist, and inserts the
  row. `/me`, called once per app start, re-reads them, so a GitHub rename is
  followed.
- **The allowlist stays in the API**, and is now checked on every request, not
  at sign-in and refresh. Auth0 will sign in any GitHub account; the API answers
  `403 access_denied` to one it does not admit, and the runtime forgets that
  session.
- **Where to sign in comes from the API.** `GET /auth/config` returns the
  tenant, the app's client id and the audience, so the desktop build carries
  only the API's address.
- **Fixed loopback ports.** Auth0 matches callback URLs exactly, port included,
  and we found no sign that it accepts any port on a loopback URL as RFC 8252
  recommends. The app therefore listens on the first free port of three
  (47213–47215), each registered as an allowed callback.

## Alternatives considered

- **Keep ADR 0006's own implementation.** Standards-based and tested, and it
  needs no third party in the sign-in path, but it is code whose only purpose
  is to re-implement what an identity provider sells.
- **Auth0 Actions for the allowlist and profile claims.** They would let the
  API skip `/userinfo` and check the allowlist without a database read. They
  would also put the allowlist and part of the logic in the Auth0 dashboard,
  outside the repository and its tests.
- **Microsoft Entra External ID**, to stay within Azure. It has no built-in
  GitHub connection, and GitHub's OAuth is not OpenID Connect, so it would need
  a custom bridge.

## Consequences

- If Auth0 is down, nobody can sign in, refresh, or make a first request.
  Requests from known users keep working while their access tokens last, as long
  as the cached signing keys are still valid.
- Running the cloud side now needs an Auth0 tenant, configured as
  `.env.example` describes.
- An access token still cannot be revoked before it expires. Taking someone off
  the allowlist takes effect at their next request.
- One port among three must be free for sign-in to complete.
