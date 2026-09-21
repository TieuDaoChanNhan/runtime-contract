#!/usr/bin/env python3
"""
Prepares training data from .trace.json files.
Pipeline: Validation -> Load -> Truncate -> Save.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument(
        "--trace-format", default="codeact", choices=["react", "codeact"]
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Validation Args
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip all validation"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.5, help="Min score (0.0-1.0)."
    )

    args = parser.parse_args()

    # Setup
    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    txt_dir = out_dir / "txt"
    jsonl_path = out_dir / "traces.jsonl"
    tokens_path = out_dir / "tokens.json"
    manifest_path = out_dir / "manifest.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Components
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

    # 2. File Discovery
    print(f"Scanning traces in {args.trace_root}...")
    trace_files = sorted(list(Path(args.trace_root).rglob("*.trace.json")))
    total_found = len(trace_files)

    random.shuffle(trace_files)

    loader = (
        load_codeact_messages if args.trace_format == "codeact" else load_trace_messages
    )

    valid_count = 0
    skipped_count = 0
    skipped_reasons = defaultdict(int)
    token_counts_map = {}
    assistant_token_counts_map = {}

    def get_token_count(msgs):
        return sum(
            len(tokenizer.encode(m["content"], add_special_tokens=False)) + 4
            for m in msgs
        )

    def get_assistant_token_count(msgs):
        return sum(
            len(tokenizer.encode(m["content"], add_special_tokens=False)) + 4
            for m in msgs
            if m["role"] == "assistant"
        )

    print(f"Processing {len(trace_files)} traces...")

    with open(jsonl_path, "w") as f_jsonl:
        for tf in tqdm(trace_files):
            # STEP 1: Fast Validation (Outcome & Structure)
            if validator:
                is_valid, reason = validator.validate(tf)
                if not is_valid:
                    skipped_count += 1
                    skipped_reasons[reason] += 1
                    continue

            # STEP 2: Load Text
            messages = loader(tf)
            if not messages:
                skipped_count += 1
                skipped_reasons["load_failed_empty"] += 1
                continue

            # STEP 3: Truncate (Tokenization)
            safe_limit = args.max_seq_length - 100

            # --- FIX START ---
            # Unpack the tuple (processed_msgs, failure_reason)
            processed, failure_reason = truncate_messages(
                messages, safe_limit, tokenizer
            )

            if failure_reason:
                skipped_count += 1
                skipped_reasons[failure_reason] += 1
                continue

            if not processed:
                skipped_count += 1
                skipped_reasons["truncation_unknown"] += 1
                continue
            # --- FIX END ---

            # STEP 4: Final Validity Checks
            if len(processed) < 2:
                skipped_count += 1
                skipped_reasons["trace_too_short"] += 1
                continue

            # Note: truncate_messages now guarantees assistant ending or returns error,
            # but a double check is safe.
            if processed[-1]["role"] != "assistant":
                skipped_count += 1
                skipped_reasons["ends_with_user"] += 1
                continue

            # Stats & Save
            t_count = get_token_count(processed)
            a_count = get_assistant_token_count(processed)
            token_counts_map[tf.name] = t_count
            assistant_token_counts_map[tf.name] = a_count

            f_jsonl.write(json.dumps({"messages": processed}) + "\n")
            valid_count += 1

            # Readable Preview
            # processed is now strictly List[dict], so iteration works correctly
            preview_lines = [f"=== SOURCE: {tf.name} ===", f"=== TOKENS: {t_count} ==="]
            for m in processed:
                role = m["role"].upper()
                content = m["content"]
                preview_lines.append(f"[{role}]:\n{content}")
            preview_lines.append("-" * 40)

            with open(txt_dir / (tf.name + ".txt"), "w") as f_txt:
                f_txt.write("\n\n".join(preview_lines))

            if args.max_samples > 0 and valid_count >= args.max_samples:
                break

    # 3. Generate Manifest

    # Extract ACTUAL validator configuration
    validator_config = {}
    if validator:
        validator_config = {
            "min_score": validator.min_score,
            "require_finish": validator.require_finish,
            "max_loop_similarity": validator.max_loop_similarity,
            "loop_window": validator.loop_window,
            "max_error_ratio": validator.max_error_ratio,
        }
    else:
        validator_config = {"enabled": False}

    token_values = list(token_counts_map.values())
    assistant_values = list(assistant_token_counts_map.values())
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "validator_config": validator_config,
        "metrics": {
            "total_files_found": total_found,
            "processed": len(trace_files),
            "valid": valid_count,
            "skipped": skipped_count,
            "yield_rate": round((valid_count / len(trace_files) * 100), 2)
            if trace_files
            else 0,
            "skipped_reasons": dict(skipped_reasons),
        },
        "token_stats": {
            "min": min(token_values) if token_values else 0,
            "max": max(token_values) if token_values else 0,
            "mean": round(statistics.mean(token_values), 2) if token_values else 0,
            "total": sum(token_values),
        },
        "assistant_token_stats": {
            "min": min(assistant_values) if assistant_values else 0,
            "max": max(assistant_values) if assistant_values else 0,
            "mean": round(statistics.mean(assistant_values), 2)
            if assistant_values
            else 0,
            "total": sum(assistant_values),
            "pct_of_total": round(sum(assistant_values) / sum(token_values) * 100, 1)
            if token_values
            else 0,
        },
        "paths": {
            "jsonl": str(jsonl_path.absolute()),
            "tokens": str(tokens_path.absolute()),
            "txt_dir": str(txt_dir.absolute()),
        },
    }

    with open(manifest_path, "w") as f_man:
        json.dump(manifest, f_man, indent=2)
    with open(tokens_path, "w") as f_tok:
        json.dump(token_counts_map, f_tok, indent=2)

    print(f"\nDone. Valid: {valid_count}, Skipped: {skipped_count}")
    if skipped_reasons:
        print("Skipped Reasons Summary:")
        print(json.dumps(dict(skipped_reasons), indent=2))
    if token_values:
        print(
            f"\nToken stats (all roles):      total={sum(token_values):,}  mean={statistics.mean(token_values):,.0f}"
        )
        print(
            f"Token stats (assistant-only):  total={sum(assistant_values):,}  mean={statistics.mean(assistant_values):,.0f}  ({manifest['assistant_token_stats']['pct_of_total']}% of total)"
        )
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
