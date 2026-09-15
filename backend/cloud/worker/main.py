"""
The Embed Worker process. From `backend/`:

    python -m cloud.worker        (or the `interroai-worker` script)

It wires the cloud adapters to `JobRunner` and runs until SIGTERM or SIGINT —
what Container Apps sends when it scales the worker in. The job in progress is
finished before the process exits; anything cut shorter than that is
redelivered by Service Bus once its lock lapses.
"""
from __future__ import annotations

import asyncio
import logging
import signal

from cloud.adapters.blob_uploads import BlobUploadStore
from cloud.adapters.pg_cache import PostgresEmbeddingCache
from cloud.adapters.postgres_store import PostgresChunkStore
from cloud.adapters.service_bus import ServiceBusJobQueue
from cloud.db.jobs import JobRepository
from cloud.db.session import create_engine, create_session_factory, service_scope
from cloud.settings import WorkerSettings
from cloud.worker.runner import JobRunner
from core.models import llm

logger = logging.getLogger("cloud.worker")


async def run_worker(settings: WorkerSettings) -> None:
    # The worker has no keychain; it embeds with the platform key.
    llm.use_api_key(settings.openai_api_key.get_secret_value())

    engine = create_engine(settings.database_url)
    sessions = create_session_factory(engine)

    def scope():
        return service_scope(sessions)

    uploads = BlobUploadStore.from_connection_string(
        settings.blob_connection_string.get_secret_value(), settings.blob_container
    )
    queue = ServiceBusJobQueue.from_connection_string(
        settings.servicebus_connection_string.get_secret_value(), settings.servicebus_queue
    )
    runner = JobRunner(
        queue=queue,
        uploads=uploads,
        store=PostgresChunkStore(scope),
        cache=PostgresEmbeddingCache(scope),
        jobs=JobRepository(scope),
        max_delivery_count=settings.max_delivery_count,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)

    logger.info("Worker listening on queue %r", settings.servicebus_queue)
    try:
        await runner.run(stop)
    finally:
        await queue.close()
        await uploads.close()
        await engine.dispose()
    logger.info("Worker stopped")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    asyncio.run(run_worker(WorkerSettings()))
