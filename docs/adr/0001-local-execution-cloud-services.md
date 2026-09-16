# ADR 0001: The agent runs locally; the cloud provides identity, models and the index

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

InterroAI edits a developer's project: it reads files, patches them and runs the
project's linter and tests. Moving the product to Azure could mean moving the
agent too, into a sandbox that clones the repository, or keeping the agent where
the files already are.

## Decision

The agent's tool loop stays in the local runtime, and edits land on the user's
disk. The cloud provides what one machine cannot or should not hold:

- sign-in, and knowing who is spending the platform's resources;
- the platform's OpenAI key, behind a gateway with a model allowlist and quotas;
- the semantic index, with a queue-backed embedding worker.

This is the split desktop coding agents such as Cursor use.

## Alternatives considered

- **A cloud sandbox working from GitHub clones.** Edits would come back as pull
  requests to pull, the agent could not run the user's local toolchain, and every
  task would need repository credentials.
- **Uploading project snapshots for a cloud agent.** A snapshot drifts from the
  working tree the moment the user edits a file, and it puts whole files, secrets
  included, on a server.

## Consequences

- Only chunks of changed files leave the machine, and only to be embedded
  ([ADR 0002](0002-no-source-text-in-the-index.md)).
- The runtime still works without the cloud. Both modes sit behind the same ports
  (`ModelGateway`, `SemanticIndex`), and `core/providers.py` picks one from
  `INTERROAI_MODE`.
- A local process now holds a cloud session, so it has to be protected from
  other local processes ([ADR 0007](0007-a-launch-token-for-the-local-runtime.md)).
