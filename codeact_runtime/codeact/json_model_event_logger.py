import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiofiles

from codeact_runtime.codeact.events import Event, ModelCallEvent, ModelResponseEvent


class JsonModelEventLogger:
    def __init__(self, out_path: str | Path, *, flush_every: int = 5) -> None:
        self.out_path = Path(out_path)
        self.flush_every = flush_every
        self._events: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await asyncio.to_thread(self.out_path.parent.mkdir, parents=True, exist_ok=True)
        # Write initial state (empty list)
        await self._flush()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Ensure final state is saved
        await self._flush()

    async def handle(self, event: Event) -> None:
        if not isinstance(event, (ModelCallEvent, ModelResponseEvent)):
            return

        should_flush = False

        # Acquire lock only to modify the list and check count
        async with self._lock:
            entry = {
                "type": event.__class__.__name__,
                "data": asdict(event),
            }
            self._events.append(entry)

            if self.flush_every > 0 and (len(self._events) % self.flush_every == 0):
                should_flush = True

        # Perform the flush outside the lock if needed (flush acquires the lock internally)
        if should_flush:
            await self._flush()

    async def _flush(self) -> None:
        """
        Thread-safe method to write batched events to disk in JSONL format.
        """
        events_to_write: list[dict[str, Any]] = []
        # 1. Atomically grab the current batch and clear the buffer
        async with self._lock:
            if not self._events:
                return
            events_to_write = self._events
            self._events = []

        # 2. Write to disk outside the lock (I/O is slow)
        lines = [
            json.dumps(e, ensure_ascii=False, default=str) for e in events_to_write
        ]
        async with aiofiles.open(self.out_path, mode="a") as f:
            await f.write("\n".join(lines) + "\n")
