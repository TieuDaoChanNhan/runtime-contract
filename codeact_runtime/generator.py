from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from codeact_runtime.cache import SQLiteLLMCache
from codeact_runtime.config import Settings
from codeact_runtime.families.factory import TaskFamilySpec, build_task_families
from codeact_runtime.llm import LiteLlmProxy, LlmProxy
from codeact_runtime.nl import generate_nl_wrapper
from codeact_runtime.storage import TaskStore


@dataclass(frozen=True)
class Job:
    family: str
    index: int
    seed: int
    task_id: str


class TaskGenerator:
    """Generate tasks for knapsack, navigation, and rule diagnosis.

    Design goals:
    - Async concurrency for throughput
    - LLM calls are throttled (semaphore), retried w/ backoff, and cached
    - Task files are written atomically; already existing tasks are skipped
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = TaskStore(settings.out_dir)
        self.cache = SQLiteLLMCache(settings.effective_cache_path())
        self.family_specs = build_task_families(settings)
        self.family_by_name = {spec.name: spec for spec in self.family_specs}
        self.llm: Optional[LlmProxy] = None
        if settings.llm_enabled:
            self.llm = LiteLlmProxy(
                **settings.llm.model_dump(), cache=self.cache, cache_enabled=True
            )

    def _family_seed_offset(self, family: str) -> int:
        # Stable offset derived from family name.
        h = hashlib.sha256(family.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def _make_job(self, family: str, index: int) -> Job:
        seed = int(self.settings.seed_start) + self._family_seed_offset(family) + index
        task_id = f"{family}-{index:010d}"
        return Job(family=family, index=index, seed=seed, task_id=task_id)

    def _iter_jobs(self) -> list[Job]:
        jobs: list[Job] = []
        for spec in self.family_specs:
            for i in range(spec.num_tasks):
                jobs.append(self._make_job(spec.name, i))
        return jobs

    async def generate_all(self) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        await self.store.write_manifest(
            {
                "schema_version": "codeact_runtime.taskgen.manifest.v1",
                "run_name": self.settings.run_name,
                "created_at": created_at,
                "settings": self.settings.model_dump(mode="json"),
            }
        )

        jobs = self._iter_jobs()
        q: asyncio.Queue[Optional[Job]] = asyncio.Queue()
        for j in jobs:
            # Resume: skip existing tasks.
            if self.store.exists(j.family, j.task_id):
                continue
            q.put_nowait(j)

        # Sentinel termination
        for _ in range(self.settings.max_workers):
            q.put_nowait(None)

        async def worker(worker_id: int) -> None:
            while True:
                job = await q.get()
                if job is None:
                    return
                try:
                    task = await self._generate_one(job)
                    await self.store.write_task(job.family, job.task_id, task)
                except BaseException as e:
                    await self.store.append_error(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "worker": worker_id,
                            "family": job.family,
                            "task_id": job.task_id,
                            "seed": job.seed,
                            "error": repr(e),
                        }
                    )

        workers = [
            asyncio.create_task(worker(i)) for i in range(self.settings.max_workers)
        ]
        await asyncio.gather(*workers)

    async def _generate_one(self, job: Job) -> Dict[str, Any]:
        # 1) Sample the instance (deterministic given seed).
        family_spec: TaskFamilySpec | None = self.family_by_name.get(job.family)
        if family_spec is None:
            raise ValueError(f"Unknown family: {job.family}")
        task = family_spec.sample(job.seed)

        # 2) Add stable identifiers / metadata.
        task.schema_version = "codeact_runtime.task.v1"
        task.task_id = job.task_id
        task.run_name = self.settings.run_name
        task.generated_at = datetime.now(timezone.utc)

        # 3) Optional: LLM-generated natural language wrapper.
        # If LLM is disabled, generate_nl_wrapper() returns a deterministic fallback.
        task.nl = await generate_nl_wrapper(task, self.llm)

        return task.model_dump(
            mode="json", by_alias=True
        )  # TODO: hack, better to pass actual object
