"""
The cloud `JobQueue`: Azure Service Bus (the emulator during development).

Messages are received in peek-lock mode: a delivered message is locked to this
worker rather than removed, and only settling it — complete, abandon or
dead-letter — decides its fate. A worker that crashes mid-job never settles, the
lock lapses, and Service Bus hands the job to the next worker. Locks are short
(a minute on the local queue), so `AutoLockRenewer` keeps a long job's lock
alive until it is settled.
"""
from __future__ import annotations

from azure.servicebus import ServiceBusMessage, ServiceBusReceivedMessage
from azure.servicebus.aio import AutoLockRenewer, ServiceBusClient, ServiceBusReceiver

from contracts.indexing import IndexJobMessage

#: The longest one job may keep its message locked before it is assumed lost.
_MAX_LOCK_RENEWAL_SECONDS = 30 * 60
#: How long one idle receive call waits. Short, so a stopping worker is not stuck in it.
_RECEIVE_WAIT_SECONDS = 5
#: Service Bus caps the dead-letter reason and description properties.
_MAX_REASON = 200
_MAX_DESCRIPTION = 1000


class _ServiceBusDelivery:
    def __init__(self, receiver: ServiceBusReceiver, raw: ServiceBusReceivedMessage) -> None:
        body = raw.body
        payload = body if isinstance(body, bytes | bytearray) else b"".join(body)
        self.message = IndexJobMessage.model_validate_json(payload)
        # The SDK passes on the AMQP header's count, which counts *earlier*
        # attempts — 0 on a first delivery. The port counts this one too, as
        # Service Bus's own MaxDeliveryCount does; off by one here, the worker
        # would give up a delivery after the broker had already dead-lettered it.
        self.delivery_count = (raw.delivery_count or 0) + 1
        self._receiver = receiver
        self._raw = raw

    async def complete(self) -> None:
        await self._receiver.complete_message(self._raw)

    async def abandon(self) -> None:
        await self._receiver.abandon_message(self._raw)

    async def dead_letter(self, reason: str) -> None:
        await self._receiver.dead_letter_message(
            self._raw, reason=reason[:_MAX_REASON], error_description=reason[:_MAX_DESCRIPTION]
        )


class ServiceBusJobQueue:
    def __init__(self, client: ServiceBusClient, queue_name: str) -> None:
        self._client = client
        self._queue_name = queue_name
        self._receiver: ServiceBusReceiver | None = None
        self._renewer = AutoLockRenewer(max_lock_renewal_duration=_MAX_LOCK_RENEWAL_SECONDS)

    @classmethod
    def from_connection_string(cls, connection_string: str, queue_name: str) -> ServiceBusJobQueue:
        return cls(ServiceBusClient.from_connection_string(connection_string), queue_name)

    async def enqueue(self, message: IndexJobMessage) -> None:
        async with self._client.get_queue_sender(self._queue_name) as sender:
            await sender.send_messages(
                ServiceBusMessage(
                    message.model_dump_json(),
                    message_id=message.job_id,
                    content_type="application/json",
                )
            )

    async def receive(self) -> _ServiceBusDelivery:
        if self._receiver is None:
            self._receiver = self._client.get_queue_receiver(self._queue_name)
        while True:
            messages = await self._receiver.receive_messages(
                max_message_count=1, max_wait_time=_RECEIVE_WAIT_SECONDS
            )
            if messages:
                self._renewer.register(self._receiver, messages[0])
                return _ServiceBusDelivery(self._receiver, messages[0])

    async def close(self) -> None:
        await self._renewer.close()
        if self._receiver is not None:
            await self._receiver.close()
        await self._client.close()
