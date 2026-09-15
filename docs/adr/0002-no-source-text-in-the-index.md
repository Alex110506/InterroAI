# ADR 0002: The index stores no source text

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

A semantic index usually keeps each chunk's text beside its vector, so a search
can return the code directly. For a private codebase, that makes the index
database a second copy of the source.

## Decision

For each chunk the index keeps the vector, the file path, the line range and the
hash of the file it was cut from, and nothing else. Chunk text passes through the
Embed Worker to be embedded and is then discarded.

A search returns locations. The runtime reads those lines from its own disk, and
compares the stored file hash with the file as it is now: a hit from a file
edited since indexing is labelled stale, so the agent re-reads the file before
trusting a line number.

## Alternatives considered

- **Store the text, encrypted per user.** Search results could come straight
  from the database, but the key management would be the hard part, and the
  runtime has the files anyway ([ADR 0001](0001-local-execution-cloud-services.md)).

## Consequences

- A leaked database exposes paths and vectors, not code.
- Search is only useful where the project is checked out, which is where the
  agent runs.
- Uploads are transient: the worker deletes each blob once its job has run.
- The embedding cache, keyed by a hash of the chunk text, holds vectors only and
  is read only by the worker, so sharing it across users saves money without
  exposing anyone's code.
