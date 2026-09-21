#!/usr/bin/env python3
"""
Runs Axolotl training using pre-prepared data.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import yaml


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> dict:
    """Capture deterministic git state, or empty dict if not in a repo."""
    info = {}
    try:
        for key, cmd in [
            ("commit", ["git", "rev-parse", "HEAD"]),
            ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            ("dirty", ["git", "status", "--porcelain"]),
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                val = result.stdout.strip()
                info[key] = bool(val) if key == "dirty" else val
    except Exception:
        pass
    return info


def _gpu_info() -> list[dict]:
    """Query nvidia-smi for GPU model, count, and VRAM."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            idx, name, vram_mib = [s.strip() for s in line.split(",")]
            gpus.append({"index": int(idx), "name": name, "vram_mib": int(vram_mib)})
        return gpus
    except Exception:
        return []


def _line_count(path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def build_manifest(
    args: argparse.Namespace, resolved_config: dict, config_path: Path
) -> dict:
    """Build a deterministic manifest capturing everything needed to reproduce a run."""
    data_path = Path(args.data_file).resolve()
    base_config_path = Path(args.base_config)

    # Determine training target from resolved config
    roles = resolved_config.get("datasets", [{}])[0].get("roles_to_train")
    if roles and roles == ["assistant"]:
        train_on = "assistant-only"
    else:
        train_on = "all"

    manifest = {
        "schema_version": 1,
        "train_on": train_on,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_info(),
        "cli_args": vars(args),
        "axolotl_config": resolved_config,
        "axolotl_config_path": str(config_path.resolve()),
        "data": {
            "path": str(data_path),
            "sha256": _sha256_file(data_path),
            "num_samples": _line_count(data_path),
            "size_bytes": data_path.stat().st_size,
        },
        "base_config": {
            "path": str(base_config_path.resolve()),
            "sha256": _sha256_file(base_config_path),
        },
        "seeds": {
            "axolotl_seed": resolved_config.get("seed"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "hardware": {
            "gpus": _gpu_info(),
            "hostname": platform.node(),
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }

    # Data lineage — link to prepare_data manifest if it exists
    data_dir = data_path.parent
    prep_manifest = data_dir / "manifest.json"
    if prep_manifest.exists():
        manifest["data"]["prepare_data_manifest"] = {
            "path": str(prep_manifest),
            "sha256": _sha256_file(prep_manifest),
        }

    # Frozen package list
    try:
        result = subprocess.run(
            ["uv", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            manifest["environment"]["packages"] = {
                p["name"]: p["version"] for p in json.loads(result.stdout)
            }
    except Exception:
        pass

    return manifest


def generate_axolotl_config(
    base_config: Path,
    data_path: Path,
    output_dir: str,
    max_seq_length: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
    lr: float,
    model: str,
    wandb_project: str | None = None,
) -> dict:
    with open(base_config) as f:
        config = yaml.safe_load(f)

    config["base_model"] = model
    # Point to the PREPARED jsonl file, preserving extra dataset-level fields
    # from the base config (e.g. train_on_inputs, roles_to_train)
    base_dataset_extra = {}
    if config.get("datasets") and len(config["datasets"]) > 0:
        base_ds = config["datasets"][0]
        # Carry over any keys beyond the standard chat_template ones
        standard_keys = {
            "path",
            "type",
            "field_messages",
            "message_field_role",
            "message_field_content",
        }
        base_dataset_extra = {
            k: v for k, v in base_ds.items() if k not in standard_keys
        }

    dataset_entry = {
        "path": str(data_path),
        "type": "chat_template",
        "field_messages": "messages",
        "message_field_role": "role",
        "message_field_content": "content",
        **base_dataset_extra,
    }
    config["datasets"] = [dataset_entry]
    config["output_dir"] = output_dir
    config["sequence_len"] = max_seq_length
    config["micro_batch_size"] = batch_size
    config["gradient_accumulation_steps"] = grad_accum
    config["num_epochs"] = epochs
    config["learning_rate"] = lr

    if wandb_project:
        config["wandb_project"] = wandb_project

    return config


def main():
    parser = argparse.ArgumentParser()
    # Training Args
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--output-dir", default="out/qwen3-8b")

    # Data Args (We just need the path to the prepared JSONL)
    parser.add_argument("--data-file", required=True, help="Path to traces.jsonl")

    # Hyperparams
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument(
        "--base-config", default="train/configs/axolotl_qwen3_8b_persistent.yaml"
    )
    parser.add_argument("--wandb-project", default=None, help="Wandb project name")

    args = parser.parse_args()

    # 1. Generate Config
    print(f"Generating Axolotl config for data: {args.data_file}")
    config = generate_axolotl_config(
        base_config=Path(args.base_config),
        data_path=Path(args.data_file).resolve(),
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        model=args.model,
        wandb_project=args.wandb_project,
    )

    config_path = Path(args.output_dir) / "axolotl_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config saved to: {config_path}")

    # 2. Save manifest
    manifest = build_manifest(args, config, config_path)
    manifest_path = Path(args.output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Manifest saved to: {manifest_path}")

    # 3. Run Axolotl
    print("Starting Training...")

    # Disable telemetry to avoid bugs
    os.environ["AXOLOTL_DO_NOT_TRACK"] = "1"

    cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", str(config_path)]

    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env={**os.environ})


if __name__ == "__main__":
    main()
