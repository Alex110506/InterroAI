"""
The Embed Worker's loop: one queue message at a time, from delivery to settlement.

For each delivery it looks the job up, fetches the upload the message points at,
runs `core.index.indexer.run_job` with the cloud store and cache, records every
event on the job row, and settles the message. How it settles is the point:

  * **The job ran to an end** — successfully or with an `error` event — so the
    message is completed. A job that failed because of its data fails the same
    way again; redelivering it would only spend money.
  * **Something around the job broke** — the database, blob storage — so the
    message is abandoned and Service Bus redelivers it. On the queue's last
    allowed delivery the job is marked failed and the message dead-lettered, so
    no client is left waiting on a job no worker will ever finish.
  * **The job cannot run at all** — no such job, no usable upload, or an
    upload for a different project than its job — so the message is
    dead-lettered straight away.

A job that is already done or failed is completed without running: delivery is
at least once, so a message can arrive again after its job has settled.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Protocol

from cloud import observability
from contracts.indexing import IndexEvent
from core.index.indexer import run_job
from core.index.ports import (
    ChunkStore,
    Delivery,
    EmbeddingCache,
    JobQueue,
    UnusableUploadError,
    UploadNotFoundError,
    UploadStore,
)

logger = logging.getLogger(__name__)


class JobLedger(Protocol):
    """What the runner needs from job rows. `cloud.db.jobs.JobRepository` provides it."""

    async def get(self, job_id: str): ...

    async def mark_running(self, job_id: str) -> None: ...

    async def append_event(self, job_id: str, event: IndexEvent) -> int: ...

    async def finish(self, job_id: str, *, error: str | None = None) -> None: ...


class JobRunner:
    def __init__(
        self,
        *,
        queue: JobQueue,
        uploads: UploadStore,
        store: ChunkStore,
        cache: EmbeddingCache | None,
        jobs: JobLedger,
        max_delivery_count: int,
    ) -> None:
        self._queue = queue
        self._uploads = uploads
        self._store = store
        self._cache = cache
        self._jobs = jobs
        #: Must match the queue's own MaxDeliveryCount, or the worker gives up
        #: on a job either too early or never.
        self._max_delivery_count = max_delivery_count

    async def run(self, stop: asyncio.Event) -> None:
        """Handle deliveries until *stop* is set. A job already started is always finished."""
        while not stop.is_set():
            receiving = asyncio.ensure_future(self._queue.receive())
            stopping = asyncio.ensure_future(stop.wait())
            done, _ = await asyncio.wait(
                {receiving, stopping}, return_when=asyncio.FIRST_COMPLETED
            )

            if receiving not in done:
                receiving.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiving
                break

            stopping.cancel()
            try:
                await self.handle(receiving.result())
            except Exception:  # noqa: BLE001
                # Settling itself failed (a lost lock, say). Service Bus will
                # redeliver; the loop must survive to take the next message.
                logger.exception("Could not settle a delivery")

    async def handle(self, delivery: Delivery) -> None:
        job_id = delivery.message.job_id
        # Every log line written while this job runs carries its id.
        context = observability.job_id.set(job_id)
        try:
            await self._process(delivery)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Job %s broke outside the indexing itself (delivery %d of %d)",
                job_id,
                delivery.delivery_count,
                self._max_delivery_count,
            )
            if delivery.delivery_count >= self._max_delivery_count:
                await self._give_up(delivery, f"Indexing kept failing: {exc}")
            else:
                await delivery.abandon()
        finally:
            observability.job_id.reset(context)

    async def _process(self, delivery: Delivery) -> None:
        message = delivery.message
        job = await self._jobs.get(message.job_id)

        if job is None:
            logger.warning("No job %s; dead-lettering its message", message.job_id)
            await delivery.dead_letter("unknown job")
            return

        if job.settled:
            # A redelivery of a job that already reached its end.
            await delivery.complete()
            return

        try:
            upload = await self._uploads.get(message.upload_ref)
        except UploadNotFoundError:
            await self._give_up(delivery, "The job's upload is missing; index the project again.")
            return
        except UnusableUploadError as exc:
            await self._give_up(delivery, str(exc))
            return

        if upload.project_id != job.project_id:
            # The upload is written by the user; the job row by the API, which
            # checked ownership. The worker writes outside row-level security,
            # so only the row may say which project changes. Otherwise an upload
            # naming someone else's project would index into it.
            await self._give_up(delivery, "The upload is for a different project than its job.")
            return

        await self._jobs.mark_running(message.job_id)

        final: IndexEvent | None = None
        async for event in run_job(upload, store=self._store, cache=self._cache):
            await self._jobs.append_event(message.job_id, event)
            final = event

        error = final.message if final is not None and final.step == "error" else None
        await self._jobs.finish(message.job_id, error=error)
        await self._uploads.delete(message.upload_ref)
        await delivery.complete()

    async def _give_up(self, delivery: Delivery, reason: str) -> None:
        """Tell the job's client it is over, then set the message aside for inspection."""
        job_id = delivery.message.job_id
        try:
            await self._jobs.append_event(job_id, IndexEvent(step="error", message=reason))
        except Exception:  # noqa: BLE001
            logger.exception("Could not record the final event of job %s", job_id)
        try:
            await self._jobs.finish(job_id, error=reason)
        except Exception:  # noqa: BLE001
            logger.exception("Could not mark job %s failed", job_id)
        await delivery.dead_letter(reason)
