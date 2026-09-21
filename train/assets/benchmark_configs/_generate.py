#!/usr/bin/env python3
"""Generate benchmark + teacher configs from _matrix.yaml + _template.yaml.j2."""

from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

HERE = Path(__file__).resolve().parent


def main() -> None:
    matrix = yaml.safe_load((HERE / "_matrix.yaml").read_text())
    env = Environment(loader=FileSystemLoader(str(HERE)), keep_trailing_newline=True)
    template = env.get_template("_template.yaml.j2")
    defaults = matrix["defaults"]

    # Benchmark configs
    generated = []
    bench = matrix["benchmark"]
    for runtime, runtime_cfg in bench["runtimes"].items():
        for model, model_cfg in bench["models"].items():
            for difficulty in bench["difficulties"]:
                rendered = template.render(
                    output_dir=f"output/benchmarks/{runtime}_{model}_{difficulty}",
                    run_name=f"{runtime}_{model}_{difficulty}",
                    seed_start=bench["seed_start"],
                    max_concurrent_runs=bench["max_concurrent_runs"],
                    cache_db_path=f"./cache/llm_cache_{runtime}_{model}_{difficulty}.sqlite",
                    task_families=defaults["task_families"],
                    agent_name=f"qwen3-8b-{runtime}-{model}-{difficulty}",
                    agent_type=defaults["agent"]["agent_type"],
                    persistent_state=runtime_cfg["persistent_state"],
                    defaults=defaults,
                    llm={"model": model_cfg["model"], **bench["llm"]},
                )
                fname = f"{runtime}_{model}_{difficulty}.yaml"
                (HERE / fname).write_text(rendered)
                generated.append(fname)

    # Teacher configs (generate_traces.*.yaml at project root)
    teacher_generated = []
    teacher_cfg = matrix["teacher"]
    project_root = HERE.parent.parent.parent
    for name, cfg in teacher_cfg["configs"].items():
        rendered = template.render(
            output_dir=teacher_cfg["output_dir"],
            run_name=cfg["run_name"],
            seed_start=teacher_cfg["seed_start"],
            cache_db_path=teacher_cfg["cache_db_path"],
            task_families=defaults["task_families"],
            agent_name=cfg["agent_name"],
            persistent_state=cfg.get("persistent_state"),
            defaults=defaults,
            llm=teacher_cfg["llm"],
        )
        fname = f"generate_traces.{name}.yaml"
        (project_root / fname).write_text(rendered)
        teacher_generated.append(fname)

    print(f"Generated {len(generated)} benchmark configs:")
    for f in sorted(generated):
        print(f"  {f}")
    print(f"Generated {len(teacher_generated)} teacher configs:")
    for f in sorted(teacher_generated):
        print(f"  {f}")


if __name__ == "__main__":
    main()
