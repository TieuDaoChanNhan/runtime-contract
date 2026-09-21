import asyncio
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiofiles

from codeact_runtime.llm import TokenUsage


class JsonTraceLogger:
    def __init__(
        self,
        out_path: str | Path,
        *,
        flush_every: int = 5,
        include_timestamps: bool = True,
    ):
        self.out_path = Path(out_path)
        self.flush_every = flush_every
        self.include_timestamps = include_timestamps

        self._started_at: Optional[datetime] = None
        self._finished_at: Optional[datetime] = None
        self._task: Optional[str] = None
        self._system_prompts: list[str] = []
        self._num_steps = 0
        self._finish_reason: Optional[str] = None
        self._errors: list[str] = []
        self._events: list[dict[str, Any]] = []
        self._token_usage: TokenUsage | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await asyncio.to_thread(self.out_path.parent.mkdir, parents=True, exist_ok=True)
        self._started_at = datetime.now()
        await self._write_trace()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._finished_at is None:
            self._finished_at = datetime.now()
        await self._write_trace()

    async def handle(self, event) -> None:
        async with self._lock:
            event_type = event.__class__.__name__
            payload = self._serialize_event(event)
            entry: dict[str, Any] = {"type": event_type, "data": payload}
            if self.include_timestamps:
                entry["timestamp"] = datetime.now().isoformat()
            self._events.append(entry)

            if event_type == "StartEvent":
                self._task = getattr(event, "task", None)
            elif event_type == "SystemPromptEvent":
                prompts = getattr(event, "prompts", None)
                if prompts:
                    self._system_prompts = [str(prompt) for prompt in prompts]
            elif event_type == "StepEvent":
                self._num_steps += 1
                step_usage = getattr(event, "token_usage", None)
                if step_usage:
                    self._token_usage = (
                        step_usage
                        if self._token_usage is None
                        else self._token_usage + step_usage
                    )
            elif event_type == "FinishEvent":
                self._finish_reason = getattr(event, "reason", None)
                finish_usage = getattr(event, "token_usage", None)
                if finish_usage:
                    self._token_usage = finish_usage
                self._finished_at = datetime.now()
            elif event_type == "ErrorEvent":
                error = getattr(event, "error", "")
                if error:
                    self._errors.append(str(error))

            if self.flush_every > 0 and (len(self._events) % self.flush_every == 0):
                await self._write_trace()

    def _serialize_event(self, event) -> dict[str, Any] | str:
        if is_dataclass(event):
            return asdict(event)  # type: ignore
        if hasattr(event, "__dict__"):
            return dict(event.__dict__)
        return str(event)

    async def _write_trace(self) -> None:
        await asyncio.to_thread(self.out_path.parent.mkdir, parents=True, exist_ok=True)
        trace = {
            "started_at": self._format_time(self._started_at),
            "finished_at": self._format_time(self._finished_at),
            "summary": {
                "task": self._task,
                "system_prompts": list(self._system_prompts),
                "num_steps": self._num_steps,
                "finish_reason": self._finish_reason,
                "errors": list(self._errors),
                "token_usage": (
                    self._token_usage.__dict__ if self._token_usage else None
                ),
            },
            "events": list(self._events),
        }

        async with aiofiles.open(self.out_path, mode="w") as f:
            await f.write(json.dumps(trace, indent=2, ensure_ascii=False, default=str))

    @staticmethod
    def _format_time(value: Optional[datetime]) -> Optional[str]:
        if value is None:
            return None
        return value.isoformat()
