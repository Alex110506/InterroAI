# ADR 0008: Exact quotas in Postgres, approximate rate limits in memory, no Redis

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

The platform's OpenAI key needs protecting in two ways: a user's total spend over
a day, and bursts, such as a runaway client, that could spend it in a minute. The
sign-in endpoints also need slowing down against guessing. The usual shared
counter, Azure Cache for Redis, costs more per month than a large part of this
project's budget.

## Decision

- **Daily quotas are exact and shared.** Requests and tokens per user per day are
  counted in Postgres. A request is admitted by one conditional upsert, so
  concurrent requests cannot overshoot; tokens are added when a response finishes.
- **Rate limits are token buckets in each API process's memory:** per user for
  chat, search, sync and uploads, and per client address for sign-in, where nobody
  is known yet. A bucket idle for a minute has refilled completely and is dropped,
  which keeps memory bounded without losing anything.
- **Request bodies over 8 MB are refused before they are read.** Chunk uploads go
  to Blob Storage and are unaffected.

## Alternatives considered

- **Azure Cache for Redis.** Exact limits across replicas, at a cost the budget
  cannot carry for a feature that only needs to be roughly right.
- **Rate limits in Postgres.** Exact, but a write on every request to the busiest
  endpoints, on the smallest server tier.

## Consequences

- With N API replicas a caller can get up to N times a rate limit, and a restart
  resets the buckets. The API runs one or two replicas, and the exact quotas are
  what bound the spend.
- Per-address limits need the real client address. The API image trusts
  `X-Forwarded-For`, which is safe only because the Container Apps ingress is its
  only way in.
- If the API ever runs many replicas, the buckets can move to a shared store
  without changing the routes: they reach the limiter through one dependency.
