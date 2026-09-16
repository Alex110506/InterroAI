# ADR 0005: Tenant isolation with Postgres row-level security

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

Projects, chunks and index jobs all belong to a user. If isolation depends on
every query carrying `WHERE owner_id = …`, one forgotten filter hands a user
someone else's project or index.

## Decision

Row-level security in Postgres enforces ownership. Every unit of database work
runs in a transaction that first tells Postgres who is asking, through
transaction-local settings that the policies read:

- `user_scope`: the API, on a user's behalf. A project, and the chunks and jobs
  under it, are visible only when the project is that user's.
- `service_scope`: the worker, which acts for whoever owns the job it is running
  and addresses rows by that job's project.
- `anonymous_scope`: sign-in, which happens before anyone is known and sees no
  project at all.

The API and the worker connect as `interroai_app`, a role that owns no tables and
cannot bypass row-level security. Migrations run as the owner.

## Alternatives considered

- **Filters in the repositories only.** Simpler, but correct only as long as
  every future query remembers them.
- **A schema or database per user.** Strong isolation, but migrations, pooling
  and a shared HNSW index all get harder, for a product with an allowlist of users.

## Consequences

- Isolation holds however a query is written. Integration tests check it from the
  database's side: another user's rows are invisible and unwritable, and the app
  role is neither superuser nor `BYPASSRLS`.
- The settings are transaction-local, so a pooled connection never carries one
  request's identity into the next.
- The worker's scope is broad by necessity, which is why it never trusts
  user-written data to name the project it writes to
  ([ADR 0004](0004-claim-check-uploads-and-at-least-once-jobs.md)).
- A route still looks the project up first, so the answer is a 404 rather than an
  empty result; row-level security is what makes a missing check harmless.
