# ADR 0007: A launch token for the local runtime

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

The runtime listens on `127.0.0.1`. Other machines cannot reach it, but other
programs on the same machine can, and so can any web page open in a browser,
which may send requests to localhost. The runtime can edit files and run tests,
and in cloud mode it holds the user's session.

## Decision

- The Electron main process starts the runtime itself, on a free port, with a
  random 256-bit launch token in its environment.
- The runtime refuses every request that does not carry the token: in the
  `X-Interroai-Token` header, or as `?token=` on WebSocket URLs, since a browser
  cannot put headers on those. A query-string token is not accepted for plain
  HTTP, because URLs end up in logs.
- CORS is locked to the app's own origin.
- `/health` stays open. It tells a caller only that the runtime is up.
- Uvicorn runs without an access log, so a token in a WebSocket URL is never
  written down.
- The renderer gets the port and token from the main process over IPC.

## Alternatives considered

- **A Unix domain socket or named pipe.** Out of reach for web pages by nature,
  but the renderer's `fetch` and `WebSocket` cannot use one, so every call would
  have to be relayed through IPC.
- **Relying on CORS alone.** CORS decides whether a page may read a response, not
  whether the request is sent, and simple requests go through without a preflight.

## Consequences

- A web page cannot drive the agent through the user's browser, and another local
  program would need access to the app's own process to find the token.
- The UI still talks to the runtime directly, over HTTP and WebSockets.
- Started by hand for development (`uvicorn main:app`), the runtime has no token
  and answers anything that can reach the port, as it did before.
