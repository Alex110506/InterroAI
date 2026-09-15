"""An in-memory `UsageMeter`, held to the same contract as the Postgres one."""
from __future__ import annotations

from datetime import date

from cloud.db.usage import Quota, UsageTotals


class InMemoryUsageMeter:
    def __init__(self) -> None:
        #: (user_id, day) → [requests, prompt_tokens, completion_tokens]
        self.rows: dict[tuple[str, date], list[int]] = {}

    async def admit(self, user_id: str, day: date, quota: Quota) -> bool:
        if quota.requests <= 0 or quota.tokens <= 0:
            return False
        row = self.rows.setdefault((user_id, day), [0, 0, 0])
        if row[0] >= quota.requests or row[1] + row[2] >= quota.tokens:
            return False
        row[0] += 1
        return True

    async def add_tokens(self, user_id: str, day: date, *, prompt: int, completion: int) -> None:
        row = self.rows.setdefault((user_id, day), [0, 0, 0])
        row[1] += prompt
        row[2] += completion

    async def totals(self, user_id: str, day: date) -> UsageTotals:
        requests, prompt, completion = self.rows.get((user_id, day), [0, 0, 0])
        return UsageTotals(requests=requests, prompt_tokens=prompt, completion_tokens=completion)
