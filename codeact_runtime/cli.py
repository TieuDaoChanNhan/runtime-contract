from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from codeact_runtime.config import Settings
from codeact_runtime.generator import TaskGenerator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate CodeAct-Runtime-style tasks (knapsack, navigation, rule diagnosis)."
    )
    p.add_argument(
        "--config",
        type=str,
        default="./codeact_runtime/task_configs/medium.json",
        required=False,
        help="Path to JSON settings file.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="output/tasks_20_v2/medium",
        help="Override output directory (optional).",
    )
    p.add_argument(
        "--run-name", type=str, default=None, help="Override run name (optional)."
    )
    p.add_argument(
        "--enable-llm",
        action="store_true",
        help="Force-enable LLM calls even if config disables them.",
    )
    p.add_argument(
        "--disable-llm",
        action="store_true",
        help="Force-disable LLM calls even if config enables them.",
    )
    return p


def main() -> None:
    load_dotenv()  # secrets for LiteLLM providers

    args = build_parser().parse_args()
    settings = Settings.load(args.config)

    if args.out is not None:
        settings.out_dir = Path(args.out)
    if args.run_name is not None:
        settings.run_name = args.run_name

    if args.enable_llm and args.disable_llm:
        raise SystemExit("Cannot set both --enable-llm and --disable-llm")
    if args.enable_llm:
        settings.llm_enabled = True
    if args.disable_llm:
        settings.llm_enabled = False

    gen = TaskGenerator(settings)
    asyncio.run(gen.generate_all())


if __name__ == "__main__":
    main()
