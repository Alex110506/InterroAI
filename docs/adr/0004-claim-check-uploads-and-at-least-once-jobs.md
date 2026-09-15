# ADR 0004: Claim-check uploads, and index jobs that run at least once

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

Indexing a large repository produces megabytes of chunks, and embedding them can
take minutes. The work has to survive a worker being scaled in or crashing, and
the user wants to watch it progress.

## Decision

- The runtime asks the API for a short-lived SAS URL that can create and write one
  blob, and uploads the chunks straight to Blob Storage. Chunk data never passes
  through the API.
- The API creates a job row, then puts a small message on Service Bus: the job id,
  the project and a reference to the blob. This is the claim-check pattern, since
  queue messages are size-limited. A project has at most one active job, enforced
  by a partial unique index.
- The Embed Worker receives messages in peek-lock mode:
  - a job that reached an outcome, success or failure, completes its message;
  - a failure around the job (database, storage) abandons it for redelivery;
  - on the last allowed delivery the job is marked failed and the message
    dead-lettered, so no client waits on a job nobody will finish.
- Each job embeds everything first, banking vectors in the embedding cache as
  batches finish, then changes the index in one transaction.
- Progress is appended to the job row. Clients follow it as server-sent events
  that resume with `Last-Event-ID`, woken by Postgres `NOTIFY` and falling back to
  polling.

## Alternatives considered

- **Chunks in the queue message.** Service Bus messages top out far below a large
  repository's chunks.
- **Uploading through the API.** It would tie up API replicas streaming megabytes
  and count against their memory, for no benefit.
- **Service Bus sessions for ordering.** Basic tier has none, and the job row's
  uniqueness already stops two jobs racing on one project.

## Consequences

- Delivery is at least once, so the worker is idempotent. A redelivered job that
  already settled completes without running, and a re-run pays only for embeddings
  the cache does not already hold.
- The blob is written by the user, so it is untrusted input. The worker checks its
  size and validates it, and takes the project from the job row, never from the
  blob: the worker writes outside row-level security
  ([ADR 0005](0005-tenant-isolation-with-row-level-security.md)).
- A blob left behind by a failed run needs a lifecycle rule on the container to
  remove it.
