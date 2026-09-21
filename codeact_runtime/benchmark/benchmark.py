import argparse
import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn

from codeact_runtime.benchmark.config import (
    FAMILY_ENVS,
    AgentConfig,
    BenchmarkConfig,
)
from codeact_runtime.cache import SQLiteLLMCache
from codeact_runtime.codeact.agent import CodeAct
from codeact_runtime.codeact.events import AsyncQueuedEventBus
from codeact_runtime.codeact.html_trace_logger import HtmlTraceListener
from codeact_runtime.codeact.json_trace_logger import JsonTraceLogger
from codeact_runtime.llm import LiteLlmProxy
from codeact_runtime.react.agent import SimpleReAct

load_dotenv()


@dataclass(frozen=True)
class RunSpec:
    agent: AgentConfig
    family: str
    task_id: str
    task_path: Path
    output_path: Path


def _task_decode_seed(base: int | None, family: str, task_id: str) -> int | None:
    """Per-task decoding seed: fixed by the config's base seed and the task id alone.

    Two cells that differ only in runtime therefore hand the server the same seed for the
    same task, which pairs the sampling as far as the trajectories agree."""
    if base is None:
        return None
    digest = hashlib.sha256(f"{family}/{task_id}".encode("utf-8")).hexdigest()[:8]
    return (base + int(digest, 16)) % (2**31 - 1)


@dataclass(frozen=True)
class RunSummary:
    average_is_solved: float
    average_runtime: float
    average_steps: float
    average_total_tokens: float
    metrics: dict[str, dict[str, float]]
    num_tasks_executed: int
    num_tasks_solved: int


def _load_benchmark_config(path: Path) -> BenchmarkConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark config must be a YAML mapping.")

    return BenchmarkConfig.model_validate(payload)


async def _run_single_task(
    spec: RunSpec,
    llm_proxy: LiteLlmProxy,
    progress: Progress,
    progress_task: TaskID,
    progress_lock: asyncio.Lock,
) -> str:
    result_path = spec.output_path / f"{spec.family}-{spec.task_id}.json"
    result_trace_path = spec.output_path / f"{spec.family}-{spec.task_id}.trace.json"
    result_trace_html_path = (
        spec.output_path / f"{spec.family}-{spec.task_id}.trace.html"
    )

    if result_path.exists():
        async with progress_lock:
            progress.advance(progress_task)
        return "skipped"

    task = json.loads(spec.task_path.read_text(encoding="utf-8"))
    env_cls = FAMILY_ENVS[spec.family]
    env = env_cls.from_task(task)
    tools = env.get_tools()
    prompt = env.get_goal_prompt()

    bus = AsyncQueuedEventBus(
        listeners=[
            HtmlTraceListener(result_trace_html_path),
            JsonTraceLogger(result_trace_path),
        ]
    )

    status = "completed"
    error_message = None
    started_at = asyncio.get_running_loop().time()
    try:
        async with bus:
            if spec.agent.agent_type == "react":
                agent = SimpleReAct(
                    llm_proxy=llm_proxy,
                    max_num_turns=spec.agent.max_turns,
                    tools=tools,
                    bus=bus,
                    max_tool_calls=spec.agent.max_tool_calls,
                )
            else:
                agent = CodeAct(
                    llm_proxy=llm_proxy,
                    max_num_turns=spec.agent.max_turns,
                    tools=tools,
                    bus=bus,
                    persistent_state=spec.agent.persistent_state,
                    max_tool_calls=spec.agent.max_tool_calls,
                    state_carryover_on_cap=spec.agent.state_carryover_on_cap,
                    announce_carryover=spec.agent.announce_carryover,
                    llm_seed=_task_decode_seed(
                        spec.agent.llm.seed, spec.family, spec.task_id
                    ),
                )
            await asyncio.wait_for(agent.run(prompt), timeout=spec.agent.timeout_s)
    except asyncio.TimeoutError as exc:
        status = "timeout"
        error_message = str(exc)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error_message = str(exc)
    finished_at = asyncio.get_running_loop().time()

    result = env.evaluate()

    output = {
        "status": status,
        "error": error_message,
        "result": asdict(result),
        "finished_at": finished_at,
        "started_at": started_at,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    async with progress_lock:
        progress.advance(progress_task)

    return status


async def _run_specs(
    specs: list[RunSpec],
    llm_proxy: LiteLlmProxy,
    progress: Progress,
    progress_task: TaskID,
    max_concurrent: int,
) -> list[str]:
    semaphore = asyncio.Semaphore(max_concurrent)
    progress_lock = asyncio.Lock()

    async def runner(spec: RunSpec) -> str:
        async with semaphore:
            return await _run_single_task(
                spec, llm_proxy, progress, progress_task, progress_lock
            )

    return await asyncio.gather(*(runner(spec) for spec in specs))


def _summarize_results(
    output_path: Path,
    family: str,
    task_ids: list[str],
) -> RunSummary:
    executed = 0
    solved = 0
    runtimes: list[float] = []
    steps: list[int] = []
    total_tokens: list[int] = []
    metric_values: dict[str, list[float]] = {}

    for task_id in task_ids:
        result_path = output_path / f"{family}-{task_id}.json"
        if not result_path.exists():
            continue

        executed += 1
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        result_data = result_payload.get("result") or {}
        if result_data.get("is_solved") is True:
            solved += 1
        all_metrics = result_data.get("metrics") or {}
        if not isinstance(all_metrics, dict):
            all_metrics = {}

        score = result_data.get("score")
        if isinstance(score, (int, float)):
            # The top-level score has precedence over a score within the metrics dict.
            all_metrics["score"] = score

        for key, value in all_metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metric_values.setdefault(key, []).append(float(value))

        started_at = result_payload["started_at"]
        finished_at = result_payload["finished_at"]
        runtimes.append(finished_at - started_at)

        trace_path = output_path / f"{family}-{task_id}.trace.json"
        if trace_path.exists():
            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
            summary = trace_payload.get("summary") or {}
            token_usage = summary.get("token_usage") or {}
            token_total = token_usage.get("total_tokens")
            if isinstance(token_total, int):
                total_tokens.append(token_total)
            num_steps = summary.get("num_steps")
            if isinstance(num_steps, int):
                steps.append(num_steps)

    avg_is_solved = solved / executed if executed else 0.0
    avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0
    avg_steps = sum(steps) / len(steps) if steps else 0.0
    avg_total_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else 0.0
    aggregated_metrics: dict[str, dict[str, float]] = {}
    for key, values in metric_values.items():
        if not values:
            continue
        aggregated_metrics[key] = {
            "mean": sum(values) / len(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }

    return RunSummary(
        average_is_solved=avg_is_solved,
        average_runtime=avg_runtime,
        average_steps=avg_steps,
        average_total_tokens=avg_total_tokens,
        metrics=aggregated_metrics,
        num_tasks_executed=executed,
        num_tasks_solved=solved,
    )


def _parse_shard(spec: str | None) -> tuple[int, int] | None:
    """`i/n` -> (i, n): this process runs task indices i, i+n, i+2n, ... of each family.

    Shards of one cell write into the SAME output directory, one file per task, so they
    merge with no post-processing. Striding rather than slicing keeps the per-shard mix of
    instances comparable; running every cell of a task in the same shard keeps a paired
    comparison within one serving instance even when shards sit on different pods."""
    if spec is None:
        return None
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise SystemExit(f"--shard expects i/n, got {spec!r}")
    if n < 1 or not (0 <= i < n):
        raise SystemExit(f"--shard out of range: {spec!r} (need 0 <= i < n, n >= 1)")
    return i, n


async def run_benchmark(
    config_path: Path,
    tasks_root: Path,
    max_examples: int | None,
    split: str | None,
    shard: tuple[int, int] | None = None,
) -> None:
    config = _load_benchmark_config(config_path)
    tag = f".shard{shard[0]}of{shard[1]}" if shard else ""
    output_root = config.output_dir / config.run_name
    print(f"Output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "config.yaml").write_text(
        yaml.dump(config.model_dump(mode="json")), encoding="utf-8"
    )

    console = Console()
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    )

    results_dir = output_root / "results"

    families_to_run = config.task_families or list(FAMILY_ENVS.keys())

    with progress:
        for agent in config.agents:
            # a per-shard cache file: concurrent shards must not contend for one sqlite db
            cache = SQLiteLLMCache(
                config.cache_db_path.with_name(config.cache_db_path.name + tag)
                if tag
                else config.cache_db_path
            )
            llm_proxy = LiteLlmProxy(cache=cache, **agent.llm.model_dump())

            for family in families_to_run:
                tasks_dir = tasks_root / "tasks" / family
                if split is not None:
                    tasks_dir = tasks_dir / split

                task_paths = sorted(tasks_dir.glob("*.json"))
                if max_examples is not None:
                    task_paths = task_paths[:max_examples]
                if shard is not None:
                    task_paths = task_paths[shard[0] :: shard[1]]
                total_tasks = len(task_paths)
                if total_tasks == 0:
                    continue

                progress_task = progress.add_task(
                    f"{agent.name} {family}{tag}", total=total_tasks
                )
                specs = []
                for task_path in task_paths:
                    task_id = task_path.stem
                    cur_base_path = results_dir / agent.name / family

                    specs.append(
                        RunSpec(
                            agent=agent,
                            family=family,
                            task_id=task_id,
                            task_path=task_path,
                            output_path=cur_base_path,
                        )
                    )

                task_statuses = await _run_specs(
                    specs,
                    llm_proxy,
                    progress,
                    progress_task,
                    config.max_concurrent_runs,
                )

                task_status_counter = Counter(task_statuses)
                (output_root / f"task_status_{agent.name}-{family}{tag}.json").write_text(
                    json.dumps(task_status_counter, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                summary = _summarize_results(
                    results_dir / agent.name / family,
                    family,
                    [task_path.stem for task_path in task_paths],
                )
                (output_root / f"summary_{agent.name}-{family}{tag}.json").write_text(
                    json.dumps(asdict(summary), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CodeAct agents across task families and difficulty buckets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmark.example.yaml"),
        help="Path to YAML benchmark config.",
    )
    parser.add_argument(
        "--tasks_root",
        type=Path,
        default=Path("outs"),
        help="Path to YAML benchmark config.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit the number of benchmark tasks per family.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="i/n: run only task indices i, i+n, i+2n, ... Shards of one cell merge in "
        "the output directory, so they can run against separate serving pods.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            args.config,
            args.tasks_root,
            args.max_examples,
            args.split,
            _parse_shard(args.shard),
        )
    )


if __name__ == "__main__":
    main()
