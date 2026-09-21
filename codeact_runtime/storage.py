from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any


class TaskStore:
    """File-based store: one JSON file per task.

    This makes the generator trivially resumable: if a task file already exists,
    it is skipped. Files are written atomically (tmp + rename) to avoid partial
    writes if the process is killed mid-write.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.tasks_dir = self.out_dir / "tasks"
        self.cache_dir = self.out_dir / "cache"
        self.logs_dir = self.out_dir / "logs"

        for d in (self.tasks_dir, self.cache_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def task_path(self, family: str, task_id: str) -> Path:
        return self.tasks_dir / family / f"{task_id}.json"

    def exists(self, family: str, task_id: str) -> bool:
        return self.task_path(family, task_id).exists()

    async def write_task(self, family: str, task_id: str, obj: dict[str, Any]) -> Path:
        path = self.task_path(family, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")

        def _write_sync() -> None:
            data = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        await asyncio.to_thread(_write_sync)
        return path

    async def append_error(self, record: dict[str, Any]) -> None:
        path = self.logs_dir / "errors.jsonl"

        def _append_sync() -> None:
            line = json.dumps(record, ensure_ascii=False)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

        await asyncio.to_thread(_append_sync)

    async def write_manifest(self, obj: dict[str, Any]) -> Path:
        path = self.out_dir / "manifest.json"
        tmp = path.with_suffix(path.suffix + ".tmp")

        def _write_sync() -> None:
            data = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        await asyncio.to_thread(_write_sync)
        return path
