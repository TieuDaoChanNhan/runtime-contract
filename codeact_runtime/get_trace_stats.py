import argparse
import json
import os
import re
import statistics
from glob import glob


def get_trace_files(directory):
    return glob(os.path.join(directory, "*.trace.json"))


def count_tool_functions(code_snippet):
    if not code_snippet:
        return 0
    return len(
        re.findall(r"\b(inspect|take_item|list_items|finish)\s*\(", code_snippet)
    )


def parse_pair(trace_filepath):
    result_filepath = trace_filepath.replace(".trace.json", ".json")

    # 1. Initialize Metrics
    metrics = {
        "steps": 0,
        "tool_calls": 0,
        "total_tokens": 0,
        "is_solved": None,
        "score": None,
        "capacity_utilization": None,
        "inspected_count": None,
    }

    # 2. Load Trace Data (Execution Metrics)
    try:
        with open(trace_filepath, "r", encoding="utf-8") as f:
            trace_data = json.load(f)

            # Steps
            summary = trace_data.get("summary", {})
            metrics["steps"] = summary.get("num_steps", 0)
            if metrics["steps"] == 0:
                metrics["steps"] = sum(
                    1
                    for e in trace_data.get("events", [])
                    if e.get("type") == "StepEvent"
                )

            # Tokens
            if "token_usage" in summary:
                metrics["total_tokens"] = summary["token_usage"].get("total_tokens", 0)

            # Tool Calls
            for event in trace_data.get("events", []):
                if event.get("type") == "StepEvent":
                    metrics["tool_calls"] += count_tool_functions(
                        event.get("data", {}).get("code")
                    )
    except Exception as e:
        print(e)
        return None

    # 3. Load Result Data (Quality & Diagnosis Metrics)
    if os.path.exists(result_filepath):
        try:
            with open(result_filepath, "r", encoding="utf-8") as f:
                res_data = json.load(f)
                if "result" in res_data:
                    res = res_data["result"]
                    metrics["is_solved"] = 1 if res.get("is_solved") is True else 0
                    metrics["score"] = res.get("score", 0.0)

                    # --- NEW METRICS ---
                    inner_metrics = res.get("metrics", {})
                    weight = inner_metrics.get("weight_used", 0)
                    capacity = inner_metrics.get("capacity", 0)
                    metrics["inspected_count"] = inner_metrics.get("inspected_count", 0)

                    if capacity > 0:
                        metrics["capacity_utilization"] = (weight / capacity) * 100
                    else:
                        metrics["capacity_utilization"] = 0
        except Exception as e:
            print(e)
            pass

    return metrics


def process_directory(directory):
    files = get_trace_files(directory)
    agg = {
        k: []
        for k in [
            "steps",
            "tool_calls",
            "total_tokens",
            "is_solved",
            "score",
            "capacity_utilization",
            "inspected_count",
        ]
    }

    print(f"Processing {len(files)} traces in {directory}...")
    for f in files:
        m = parse_pair(f)
        if m:
            for k in agg:
                if m[k] is not None:
                    agg[k].append(m[k])

    results = {}
    for k, v in agg.items():
        if not v:
            results[k] = 0
        elif k == "is_solved":
            results[k] = (sum(v) / len(v)) * 100
        else:
            results[k] = statistics.mean(v)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir1")
    parser.add_argument("dir2")
    args = parser.parse_args()

    r1 = process_directory(args.dir1)
    r2 = process_directory(args.dir2)

    name1, name2 = args.dir1, args.dir2
    print(f"\n{'Metric':<35} | {name1:<20} | {name2:<20}")
    print("-" * 80)
    print(
        f"{'Teacher Success Rate (%)':<35} | {r1['is_solved']:<20.1f} | {r2['is_solved']:<20.1f}"
    )
    print(
        f"{'Avg Teacher Optimality (0-1)':<35} | {r1['score']:<20.4f} | {r2['score']:<20.4f}"
    )
    print(
        f"{'Avg Capacity Utilization (%)':<35} | {r1['capacity_utilization']:<20.1f} | {r2['capacity_utilization']:<20.1f}"
    )
    print(
        f"{'Avg Items Inspected':<35} | {r1['inspected_count']:<20.1f} | {r2['inspected_count']:<20.1f}"
    )
    print("-" * 80)
    print(f"{'Avg Steps / Episode':<35} | {r1['steps']:<20.2f} | {r2['steps']:<20.2f}")
    print(
        f"{'Avg Tool Calls / Episode':<35} | {r1['tool_calls']:<20.2f} | {r2['tool_calls']:<20.2f}"
    )
    print(
        f"{'Avg Total Tokens / Episode':<35} | {r1['total_tokens']:<20.0f} | {r2['total_tokens']:<20.0f}"
    )
