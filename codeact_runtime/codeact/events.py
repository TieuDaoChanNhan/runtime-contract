import asyncio
import traceback
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol

from codeact_runtime.llm import TokenUsage


@dataclass(frozen=True)
class StartEvent:
    task: str


@dataclass(frozen=True)
class SystemPromptEvent:
    prompts: list[str]


@dataclass(frozen=True)
class StepEvent:
    turn: int
    assistant_text: str
    code: Optional[str]
    interpreter_result: Optional[dict[str, Any]]
    token_usage: Optional[TokenUsage] = None


@dataclass(frozen=True)
class FinishEvent:
    reason: Literal["finish_tool", "max_turns", "error"]
    token_usage: Optional[TokenUsage] = None


@dataclass(frozen=True)
class ErrorEvent:
    error: str


@dataclass(frozen=True)
class ModelCallEvent:
    timestamp: str
    prompt_tokens: int


@dataclass(frozen=True)
class ModelResponseEvent:
    timestamp: str
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class UserObservationEvent:
    turn: int
    content: str


Event = (
    StartEvent
    | SystemPromptEvent
    | StepEvent
    | FinishEvent
    | ErrorEvent
    | ModelCallEvent
    | ModelResponseEvent
    | UserObservationEvent
)


class Listener(Protocol):
    async def handle(self, event: Event) -> None: ...


class EventBus:
    def __init__(self, listeners: list[Listener], *, concurrent: bool = True):
        self._listeners = listeners
        self._concurrent = concurrent
        self._stack = AsyncExitStack()

    async def __aenter__(self):
        # If listeners expose __aenter__/__aexit__, we manage them
        for listener in self._listeners:
            enter = getattr(listener, "__aenter__", None)
            if enter is not None:
                await self._stack.enter_async_context(listener)  # type: ignore
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._stack.aclose()

    async def emit(self, event: Event) -> None:
        if self._concurrent:
            # [FIX] return_exceptions=True ensures one bad listener doesn't crash the others
            results = await asyncio.gather(
                *(listener.handle(event) for listener in self._listeners),
                return_exceptions=True,
            )
            # Log errors if any occurred in the concurrent batch
            for res in results:
                if isinstance(res, BaseException):
                    print(f"Listener Error in gather: {res}")
                    traceback.print_tb(res.__traceback__)
        else:
            for listener in self._listeners:
                try:
                    await listener.handle(event)
                except Exception as e:
                    print(f"Listener Error: {e}")
                    traceback.print_tb(e.__traceback__)


class AsyncQueuedEventBus(EventBus):
    def __init__(
        self, listeners: list[Listener], *, maxsize: int = 0, concurrent: bool = True
    ):
        super().__init__(listeners, concurrent=concurrent)
        self._q: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=maxsize)
        self._task: asyncio.Task | None = None

    async def __aenter__(self):
        await super().__aenter__()
        self._task = asyncio.create_task(self._pump())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()
        await super().__aexit__(exc_type, exc, tb)

    async def emit(self, event: Event) -> None:
        await self._q.put(event)

    async def stop(self):
        await self._q.put(None)
        if self._task is not None:
            await self._task

    async def _pump(self):
        while True:
            item = await self._q.get()

            if item is None:
                self._q.task_done()
                return

            try:
                # [FIX] Wrap processing in try/except.
                # If this line crashes, the while loop would break and the bus would die.
                await super().emit(item)
            except Exception as e:
                # This catches errors in `emit` itself (though emit now handles listeners safely)
                print(
                    f"FATAL: EventBus pump encountered error processing event {type(item).__name__}: {e}"
                )
                traceback.print_exc()
            finally:
                self._q.task_done()


class MessageLogger:
    async def handle(self, event: Event) -> None:
        print(f"Logger: {event}")


class ModelTokenLogger:
    async def handle(self, event: Event) -> None:
        if isinstance(event, ModelCallEvent):
            print(f"Model call @ {event.timestamp} prompt_tokens={event.prompt_tokens}")
            return
        if isinstance(event, ModelResponseEvent):
            print(
                "Model response "
                f"@ {event.timestamp} completion_tokens={event.completion_tokens} "
                f"total_tokens={event.total_tokens}"
            )
