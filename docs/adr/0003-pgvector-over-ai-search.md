# ADR 0003: pgvector on Postgres Flexible Server, not Azure AI Search

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

The project runs on about $80 of Azure credit, which has to cover development
and a demo. It needs a relational database for users, projects, jobs and usage,
and a vector index.

## Decision

One Azure Database for PostgreSQL Flexible Server (Burstable B1ms) with the
pgvector extension holds both. Chunk vectors (`vector(1536)`, from
`text-embedding-3-small`) sit in an HNSW index with cosine distance.

## Alternatives considered

- **Azure AI Search.** Its Basic tier costs more per month than the whole budget,
  and it would be a second store to keep consistent with the job and project rows.
- **pgvector or Qdrant in a container.** Cheaper still, but storage, backups and
  upgrades become the project's problem, and Container Apps storage is not meant
  for databases.

## Consequences

- A job's whole index change (upserts, stale deletions, a forced reset) commits in
  one transaction, so a search sees the index from before the job or after it,
  never half of each ([ADR 0004](0004-claim-check-uploads-and-at-least-once-jobs.md)).
- Row-level security covers vectors like any other row
  ([ADR 0005](0005-tenant-isolation-with-row-level-security.md)).
- Searches filter by project, so they use pgvector's iterative index scans
  (`hnsw.iterative_scan`); a project with few chunks still gets its full result
  count from a shared HNSW index.
- A B1ms instance suits indexes of a few hundred thousand chunks. Past that the
  server tier is the limit, not the design.
