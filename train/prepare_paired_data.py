#!/usr/bin/env python3
"""
Prepares PAIRED training data from two sets of .trace.json files (Persistent & Stateless).
Pipeline: Set Intersection -> Dual Validation (AND logic) -> Load -> Truncate -> Save to dual outputs.
Ensures 100% structural parity with the unpaired prepare_data.py script for downstream compatibility.
"""

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from dataloader import (
    load_codeact_messages,
    load_trace_messages,
    truncate_messages,
)
from tqdm import tqdm
from trace_validator import TraceValidator
from transformers import AutoTokenizer  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="Prepare paired datasets to eliminate confounds."
    )

    # 1. Dual Input & Dual Output Arguments
    parser.add_argument(
        "--trace-root-persistent",
        required=True,
        help="Path to persistent traces zip extraction",
    )
    parser.add_argument(
        "--trace-root-stateless",
        required=True,
        help="Path to stateless traces zip extraction",
    )
    parser.add_argument(
        "--output-dir-persistent",
        required=True,
        help="Output dir for paired persistent data",
    )
    parser.add_argument(
        "--output-dir-stateless",
        required=True,
        help="Output dir for paired stateless data",
    )

    # Shared Arguments
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument(
        "--trace-format", default="codeact", choices=["react", "codeact"]
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip all validation"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.5, help="Min score (0.0-1.0)."
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # 2. Setup Dual Output Directories
    out_p = Path(args.output_dir_persistent)
    out_s = Path(args.output_dir_stateless)

    for d in [out_p, out_s]:
        d.mkdir(parents=True, exist_ok=True)
        (d / "txt").mkdir(parents=True, exist_ok=True)

    # 3. Initialize Components
    validator = None
    if not args.no_validate:
        print(f"Validation Enabled: min_score >= {args.min_score}")
        validator = TraceValidator(min_score=args.min_score)

    print(f"Loading Tokenizer: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 4. File Discovery & Set Intersection
    print("Scanning traces and building Hash Maps...")
    dict_p = {f.name: f for f in Path(args.trace_root_persistent).rglob("*.trace.json")}
    dict_s = {f.name: f for f in Path(args.trace_root_stateless).rglob("*.trace.json")}

    # Get strict intersection of filenames
    common_filenames = sorted(list(set(dict_p.keys()) & set(dict_s.keys())))
    random.shuffle(common_filenames)

    print(f"Found {len(dict_p)} persistent traces and {len(dict_s)} stateless traces.")
    print(f"Intersection resulted in {len(common_filenames)} PAIRED problem instances.")

    loader = (
        load_codeact_messages if args.trace_format == "codeact" else load_trace_messages
    )

    valid_count = 0
    skipped_count = 0
    skipped_reasons = defaultdict(int)

    # Dictionaries to track token usage for each environment separately
    token_counts_map_p = {}
    token_counts_map_s = {}

    TOKEN_OVERHEAD_PER_MESSAGE = 4
    TRUNCATION_SAFETY_MARGIN = 100

    def get_token_count(msgs):
        return sum(
            len(tokenizer.encode(m["content"], add_special_tokens=False))
            + TOKEN_OVERHEAD_PER_MESSAGE
            for m in msgs
        )

    def write_preview_file(output_dir, filename, messages, token_count):
        """Generates a human-readable preview of a trace."""
        preview_lines = [
            f"=== SOURCE: {filename} ===",
            f"=== TOKENS: {token_count} ===",
        ]
        for m in messages:
            preview_lines.append(f"[{m['role'].upper()}]:\n{m['content']}")
        preview_lines.append("-" * 40)
        with open(output_dir / "txt" / f"{filename}.txt", "w") as f_txt:
            f_txt.write("\n\n".join(preview_lines))

    print(f"Processing {len(common_filenames)} paired traces...")

    # 5. Dual Processing Loop
    with (
        open(out_p / "traces.jsonl", "w") as f_jsonl_p,
        open(out_s / "traces.jsonl", "w") as f_jsonl_s,
    ):
        for filename in tqdm(common_filenames):
            tf_p = dict_p[filename]
            tf_s = dict_s[filename]

            # STEP A: Strict Dual Validation (AND Logic)
            if validator:
                is_valid_p, reason_p = validator.validate(tf_p)
                is_valid_s, reason_s = validator.validate(tf_s)

                # If either fails, discard the pair to prevent confounds
                if not (is_valid_p and is_valid_s):
                    skipped_count += 1
                    if not is_valid_p and not is_valid_s:
                        skipped_reasons["failed_both_environments"] += 1
                    elif not is_valid_p:
                        skipped_reasons[f"failed_persistent_only_{reason_p}"] += 1
                    else:
                        skipped_reasons[f"failed_stateless_only_{reason_s}"] += 1
                    continue

            # STEP B: Load Text
            msgs_p = loader(tf_p)
            msgs_s = loader(tf_s)

            if not msgs_p or not msgs_s:
                skipped_count += 1
                skipped_reasons["load_failed_empty"] += 1
                continue

            # STEP C: Truncate (Tokenization)
            safe_limit = args.max_seq_length - TRUNCATION_SAFETY_MARGIN
            proc_p, fail_p = truncate_messages(msgs_p, safe_limit, tokenizer)
            proc_s, fail_s = truncate_messages(msgs_s, safe_limit, tokenizer)

            if fail_p or fail_s or proc_p is None or proc_s is None:
                skipped_count += 1
                skipped_reasons["truncation_error_on_pair"] += 1
                continue

            # STEP D: Final Structural Validity Checks
            if (
                len(proc_p) < 2
                or len(proc_s) < 2
                or proc_p[-1]["role"] != "assistant"
                or proc_s[-1]["role"] != "assistant"
            ):
                skipped_count += 1
                skipped_reasons["structural_check_failed"] += 1
                continue

            # STEP E: Save Paired Data
            f_jsonl_p.write(json.dumps({"messages": proc_p}) + "\n")
            f_jsonl_s.write(json.dumps({"messages": proc_s}) + "\n")
            valid_count += 1

            # STEP F: Token Counting
            t_count_p = get_token_count(proc_p)
            t_count_s = get_token_count(proc_s)
            token_counts_map_p[filename] = t_count_p
            token_counts_map_s[filename] = t_count_s

            # STEP G: Generate preview TXT files for both environments
            # Persistent Preview
            write_preview_file(out_p, filename, proc_p, t_count_p)
            write_preview_file(out_s, filename, proc_s, t_count_s)

            if args.max_samples > 0 and valid_count >= args.max_samples:
                break

    # 6. Generate Master Manifests
    # Extract EXACT validator configuration to preserve experiment state
    validator_config = {"enabled": False}
    if validator:
        validator_config = {
            "min_score": validator.min_score,
            "require_finish": validator.require_finish,
            "max_loop_similarity": validator.max_loop_similarity,
            "loop_window": validator.loop_window,
            "max_error_ratio": validator.max_error_ratio,
        }

    # Helper function to compute token statistics for a given token map
    def get_token_stats(token_map):
        vals = list(token_map.values())
        return {
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
            "mean": round(statistics.mean(vals), 2) if vals else 0,
            "total": sum(vals),
        }

    def build_and_save_manifest(output_dir, token_counts_map, base_metrics):
        """Builds and saves the manifest and tokens.json file for a dataset."""
        manifest = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "arguments": vars(args),
            "validator_config": validator_config,
            "metrics": base_metrics,
            "token_stats": get_token_stats(token_counts_map),
            "paths": {
                "jsonl": str((output_dir / "traces.jsonl").absolute()),
                "tokens": str((output_dir / "tokens.json").absolute()),
                "txt_dir": str((output_dir / "txt").absolute()),
            },
        }
        with open(output_dir / "manifest.json", "w") as f_man:
            json.dump(manifest, f_man, indent=2)
        with open(output_dir / "tokens.json", "w") as f_tok:
            json.dump(token_counts_map, f_tok, indent=2)

    # Base metrics shared by both manifests
    # Maintaining EXACT schema parity with the unpaired script for downstream parsers
    base_metrics = {
        "total_files_found": len(common_filenames),  # Semantically: total pairs found
        "processed": len(common_filenames),  # Semantically: total pairs processed
        "valid": valid_count,  # Semantically: valid pairs
        "skipped": skipped_count,  # Semantically: skipped pairs
        "yield_rate": round((valid_count / len(common_filenames) * 100), 2)
        if common_filenames
        else 0,
        "skipped_reasons": dict(skipped_reasons),
    }

    # Build and save manifests for Persistent and Stateless datasets
    build_and_save_manifest(out_p, token_counts_map_p, base_metrics)
    build_and_save_manifest(out_s, token_counts_map_s, base_metrics)

    print(f"\nDone. PAIRED Valid: {valid_count}, Skipped: {skipped_count}")
    print("Manifests and Token distributions saved to both output directories.")


if __name__ == "__main__":
    main()
