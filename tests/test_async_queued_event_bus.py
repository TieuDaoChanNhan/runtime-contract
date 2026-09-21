import asyncio

from codeact_runtime.codeact.events import AsyncQueuedEventBus, FinishEvent, StartEvent


class RecordingListener:
    def __init__(self) -> None:
        self.events = []

    async def handle(self, event) -> None:
        self.events.append(event)


def test_async_queued_event_bus_emits_events_in_order():
    listener = RecordingListener()

    async def runner() -> None:
        async with AsyncQueuedEventBus([listener], concurrent=False) as bus:
            await bus.emit(StartEvent(task="alpha"))
            await bus.emit(FinishEvent(reason="finish_tool"))

    asyncio.run(runner())

    assert [type(event) for event in listener.events] == [StartEvent, FinishEvent]


def test_async_queued_event_bus_stop_flushes_queue():
    listener = RecordingListener()

    async def runner() -> None:
        bus = AsyncQueuedEventBus([listener], concurrent=False)
        await bus.__aenter__()
        try:
            await bus.emit(StartEvent(task="beta"))
            await bus.stop()
        finally:
            await bus.__aexit__(None, None, None)

    asyncio.run(runner())

    assert len(listener.events) == 1
    assert isinstance(listener.events[0], StartEvent)
