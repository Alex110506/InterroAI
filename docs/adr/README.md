# Architecture decision records

Why the cloud build of InterroAI looks the way it does. Each record gives the
context, the decision, what else was considered, and what the decision costs.
The architecture itself is drawn in [`../azure-architecture.svg`](../azure-architecture.svg).

| ADR | Decision |
| --- | --- |
| [0001](0001-local-execution-cloud-services.md) | The agent runs on the user's machine; the cloud provides identity, models and the index |
| [0002](0002-no-source-text-in-the-index.md) | The index stores vectors, paths, line ranges and file hashes, never source text |
| [0003](0003-pgvector-over-ai-search.md) | One Postgres Flexible Server with pgvector holds both the app data and the vectors |
| [0004](0004-claim-check-uploads-and-at-least-once-jobs.md) | Chunks go to Blob Storage by SAS URL; jobs run at least once off Service Bus |
| [0005](0005-tenant-isolation-with-row-level-security.md) | Postgres row-level security keeps each user's data apart |
| [0006](0006-github-identity-pkce-and-rotating-refresh-tokens.md) | GitHub for identity only, PKCE through a loopback redirect, rotating refresh tokens |
| [0007](0007-a-launch-token-for-the-local-runtime.md) | The local runtime answers only the app that started it |
| [0008](0008-in-process-rate-limits-without-redis.md) | Exact daily quotas in Postgres; approximate rate limits in memory, no Redis |
