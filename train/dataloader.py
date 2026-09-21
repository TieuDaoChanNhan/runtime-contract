"""Trace dataset loader and truncation logic."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple


class TraceFormat(str, Enum):
    REACT = "react"
    CODEACT = "codeact"


def load_trace_messages(trace_path: Path) -> Optional[List[dict]]:
    """Extracts ReAct-style messages from a trace file."""
    try:
        data = json.loads(trace_path.read_text())
    except Exception:
        return None

    events = data.get("events", [])
    if not isinstance(events, list):
        return None

    sys_prompts = []
    task = None
    step_events = []

    for e in events:
        if e["type"] == "SystemPromptEvent":
            p = e["data"].get("prompts")
            if isinstance(p, list):
                sys_prompts.extend([s for s in p if isinstance(s, str)])
            elif isinstance(p, str):
                sys_prompts.append(p)
        elif e["type"] == "StartEvent":
            if not task:
                task = e["data"].get("task")
        elif e["type"] == "StepEvent":
            step_events.append(e["data"])

    if not task or not step_events:
        return None

    messages = []
    if sys_prompts:
        messages.append({"role": "system", "content": "\n\n".join(sys_prompts)})
    messages.append({"role": "user", "content": task})

    for i, step in enumerate(step_events):
        asst_text = step.get("assistant_text", "").strip()
        if not asst_text:
            continue
        messages.append({"role": "assistant", "content": asst_text})

        # Skip observation for the very last step
        if i < len(step_events) - 1:
            res = step.get("interpreter_result", {})
            tools = res.get("results") or res.get("tool_calls") or []

            obs_parts = []
            if tools:
                for t in tools:
                    t_name = t.get("tool", "unknown")
                    t_out = t.get("output", "")
                    obs_parts.append(f"Observation: [{t_name}] {t_out}")

            if obs_parts:
                messages.append({"role": "user", "content": "\n".join(obs_parts)})

    return messages if len(messages) >= 3 else None


def load_codeact_messages(trace_path: Path) -> Optional[List[dict]]:
    """Extracts CodeAct-style messages matching the inference runtime."""
    try:
        data = json.loads(trace_path.read_text())
    except Exception:
        return None

    events = data.get("events", [])
    if not isinstance(events, list):
        return None

    sys_prompts = []
    task = None
    step_events = []

    for e in events:
        if e["type"] == "SystemPromptEvent":
            p = e["data"].get("prompts")
            if isinstance(p, list):
                sys_prompts.extend([s for s in p if isinstance(s, str)])
            elif isinstance(p, str):
                sys_prompts.append(p)
        elif e["type"] == "StartEvent":
            if not task:
                task = e["data"].get("task")
        elif e["type"] == "StepEvent":
            step_events.append(e["data"])

    if not task or not step_events:
        return None

    messages = []
    if sys_prompts:
        messages.append({"role": "system", "content": "\n\n".join(sys_prompts)})
    messages.append({"role": "user", "content": task})

    for i, step in enumerate(step_events):
        asst_text = step.get("assistant_text", "").strip()
        if not asst_text:
            continue
        messages.append({"role": "assistant", "content": asst_text})

        # Skip observation for the very last step
        if i < len(step_events) - 1:
            res = step.get("interpreter_result", {}) or {}
            out = res.get("output")
            err = res.get("error")

            content_dict = {}
            if out is not None:
                content_dict["output"] = out
            if err:
                content_dict["error"] = err

            if content_dict:
                messages.append({"role": "user", "content": json.dumps(content_dict)})

    return messages if len(messages) >= 3 else None


def truncate_messages(
    messages: List[dict], max_tokens: int, tokenizer: Any
) -> Tuple[Optional[List[dict]], Optional[str]]:
    """
    SKELETON TRUNCATION STRATEGY

    Returns:
        (processed_messages, None) on success
        (None, reason_string) on failure
    """
    if not messages:
        return None, "empty_message_list"

    def get_cost(text):
        return len(tokenizer.encode(text, add_special_tokens=False)) + 4

    def count_total(msgs):
        return sum(get_cost(m["content"]) for m in msgs)

    # --- Case 0: Fits in context ---
    if count_total(messages) <= max_tokens:
        # Sanity Check: Must end with Assistant
        if messages[-1]["role"] == "assistant":
            return messages, None

        # If ends with User, try dropping the last message (observation)
        # to expose the Assistant's last action as the label.
        if len(messages) > 1 and messages[-2]["role"] == "assistant":
            return messages[:-1], None

        return None, "fits_but_ends_with_user_unfixable"

    # --- Case 1: Truncation Required ---

    # 1. Identify Sections
    head_indices = [0]
    if len(messages) > 1 and messages[1]["role"] == "user":
        head_indices = [0, 1]

    # Tail: Keep last 2 turns. Ensure we don't overlap with Head.
    # Note: We enforce ending with Assistant here.
    tail_indices = []
    if messages[-1]["role"] == "assistant":
        tail_count = 2
        tail_start = max(head_indices[-1] + 1, len(messages) - tail_count)
        tail_indices = list(range(tail_start, len(messages)))
    elif len(messages) > 1 and messages[-2]["role"] == "assistant":
        # Ends with user, so we grab the assistant before it
        tail_indices = [len(messages) - 2]
    else:
        return None, "trace_ends_with_user_unfixable"

    middle_indices = [
        i
        for i in range(len(messages))
        if i not in head_indices and i not in tail_indices
    ]

    # 2. Check Feasibility
    head_msgs = [messages[i] for i in head_indices]
    tail_msgs = [messages[i] for i in tail_indices]

    min_tokens = count_total(head_msgs) + count_total(tail_msgs)
    if min_tokens > max_tokens:
        return None, "trunc_head_tail_exceeds_limit"

    remaining_budget = max_tokens - min_tokens

    # 3. Fill Middle (Backwards)
    middle_to_keep = []

    for i in reversed(middle_indices):
        msg = messages[i]
        cost_full = get_cost(msg["content"])

        if cost_full <= remaining_budget:
            middle_to_keep.append(msg)
            remaining_budget -= cost_full
        else:
            # Masking logic for User messages
            if msg["role"] == "user":
                masked_content = json.dumps(
                    {"output": "[... Output Omitted for Brevity ...]"}
                )
                cost_mask = get_cost(masked_content)

                if cost_mask <= remaining_budget:
                    new_msg = msg.copy()
                    new_msg["content"] = masked_content
                    middle_to_keep.append(new_msg)
                    remaining_budget -= cost_mask
                else:
                    break
            else:
                break

    middle_to_keep.reverse()
    result = head_msgs + middle_to_keep + tail_msgs

    if not result:
        return None, "trunc_result_empty"

    if result[-1]["role"] != "assistant":
        return None, "trunc_result_ends_with_user"

    return result, None
