from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional


class SQLiteLLMCache:
    """A tiny persistent cache for LLM requests.

    Stores request payloads hashed by a stable key and the corresponding raw
    response JSON. This is intentionally minimal (no third-party deps), durable,
    and safe for resumable dataset generation.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now')),
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    async def get(self, key: str) -> Optional[dict[str, Any]]:
        def _get_sync() -> Optional[dict[str, Any]]:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT response_json FROM llm_cache WHERE key = ? LIMIT 1;",
                    (key,),
                )
                row = cur.fetchone()
            if not row:
                return None
            return json.loads(row[0])

        return await asyncio.to_thread(_get_sync)

    async def set(
        self,
        key: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
    ) -> None:
        req_s = json.dumps(request_payload, sort_keys=True, ensure_ascii=False)
        resp_s = json.dumps(response_payload, sort_keys=True, ensure_ascii=False)

        def _set_sync() -> None:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO llm_cache(key, request_json, response_json) VALUES(?,?,?);",
                    (key, req_s, resp_s),
                )
                self._conn.commit()

        await asyncio.to_thread(_set_sync)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
