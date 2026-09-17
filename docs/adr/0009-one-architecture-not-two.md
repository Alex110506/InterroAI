# ADR 0009: One architecture, not two

- **Status:** Accepted
- **Date:** 2026-09-16

## Context

The runtime shipped two implementations of every port, chosen by `INTERROAI_MODE`:
`local` (OpenAI directly with a key from the OS keychain, ChromaDB on disk, an
in-process embedding worker) and `cloud` (everything through the Cloud API).
[ADR 0001](0001-local-execution-cloud-services.md) keeps the agent loop on the
user's machine either way, so the pipeline never knew which one it had.

The cost was not the ports — those earn their keep — but everything that had to
exist twice: two `ChunkStore` adapters, two credential paths, two settings
shapes, two branches through the frontend's session handling, and `chromadb` in
the desktop dependency set for an index the shipped product does not use.

The local path also exercised a *different failure surface* from the one users
meet: "no API key" rather than "signed out", and a store without transactions
rather than one with them. A defect found in one was not found in the other —
chunk ids are `file_path:start_line`, and a collision was silently absorbed by
Chroma's upsert while Postgres refused the statement outright.

## Decision

Delete local mode. `core/providers.py` always returns `RemoteModelGateway` and
`RemoteSemanticIndex`; `INTERROAI_MODE` is gone, and `INTERROAI_API_URL` alone
says where the services are. The Chroma chunk store, the Redis embedding cache
and the OpenAI key in the keychain go with it.

## Alternatives considered

- **Keep local mode as an offline fallback.** It was never one: the models are
  remote in both modes, so nothing works without a network either way.
- **Keep the Chroma adapter for tests only.** The in-memory adapters already
  satisfy the same port contract suites, run in-process and need no files on
  disk, so the second real store bought nothing a fake did not.

## Consequences

- The app cannot be used without a Cloud API to sign in to. For development that
  is the docker compose stack; `INTERROAI_API_URL` points at it.
- The ports stay, and still earn their keep: they are what stops the agents from
  importing OpenAI or the index, and what lets the tests hand in fakes.
- `chromadb` leaves the `runtime` extra. `redis` stays, for the AST repo map
  alone — still an accelerator, never a dependency.
- The in-memory adapters are now the test substrate for indexing, and
  `tests/fakes/index.py::InProcessIndex` runs both halves of Phase 2 in one
  process, over the same `run_job` the Embed Worker runs in Azure.
- `MissingAPIKeyError` now means the **platform's** key is unset — an operator's
  misconfiguration. A user with nothing configured is *signed out* instead.
