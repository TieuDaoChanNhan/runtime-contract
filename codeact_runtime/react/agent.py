import json
import logging
import re
from dataclasses import dataclass

from codeact_runtime.codeact.agent import FinishTool
from codeact_runtime.codeact.events import (
    ErrorEvent,
    EventBus,
    FinishEvent,
    StartEvent,
    StepEvent,
    SystemPromptEvent,
)
from codeact_runtime.codeact.tool import Tool
from codeact_runtime.llm import LlmProxy, TokenUsage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a simple ReAct-style autonomous agent.

Follow this format strictly:

Thought: <brief reasoning>
Action: <tool name>
Action Input: <JSON object of arguments>

Rules:
- Take exactly one action per turn.
- Only use the tools listed below.
- Action Input must be valid JSON.

Termination:
- When you are completely done, call:
  Action: finish
  Action Input: {}
"""


# Added: memory instructions (only enabled when memory_mode=True)
MEMORY_PROMPT_APPENDIX = """
Optional persistent memory (enabled in this run):

On TOOL TURNS ONLY, you MAY include an additional line after Action Input:

Memory: <JSON object>

This JSON object will be applied as a patch to persistent memory AFTER tool execution:
- Keys map to JSON-serializable values.
- Use null to delete a key.
- Keep memory concise (store only what you will need later).
- Memory MUST be a JSON object (not a list / string).

Format (memory is optional):
Thought: ...
Action: ...
Action Input: {...}
Memory: {...}

Do NOT include Memory when calling the finish tool.
"""


BATCH_ACTIONS_PROMPT_APPENDIX = """
Multiple tool calls per turn are allowed (enabled in this run).

You may output up to {max_actions_per_turn} Action blocks in a single turn, like:

Thought: ...
Action: <tool name>
Action Input: <JSON object>
Action: <tool name>
Action Input: <JSON object>
Memory: <JSON object>   # optional, applied after all actions

Rules:
- Provide between 1 and {max_actions_per_turn} Action blocks.
- Each Action must be followed by exactly one Action Input JSON object.
- You will receive ONE combined Observation for the entire batch on the next turn.
- If you call finish, it MUST be the LAST Action in the message.
"""


@dataclass(frozen=True)
class ToolCallSpec:
    tool: str
    tool_input: dict[str, object]


@dataclass(frozen=True)
class ParsedReAct:
    actions: list[ToolCallSpec]
    memory_update: dict[str, object] | None


def _parse_react_response(text: str) -> ParsedReAct:
    # --- Memory (optional) ---
    memory_update: dict[str, object] | None = None
    mem_match = re.search(r"^Memory:\s*", text, re.MULTILINE)
    mem_start = mem_match.start() if mem_match else len(text)

    if mem_match:
        mem_raw = text[mem_match.end() :].strip()
        if mem_raw:
            try:
                parsed_mem = json.loads(mem_raw)
                if isinstance(parsed_mem, dict):
                    memory_update = parsed_mem
            except json.JSONDecodeError:
                memory_update = None  # ignore invalid memory blob

    # --- Actions (required) ---
    region = text[:mem_start]
    lines = region.splitlines()

    actions: list[ToolCallSpec] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Action:"):
            tool_name = line[len("Action:") :].strip()
            i += 1

            if i >= len(lines) or not lines[i].lstrip().startswith("Action Input:"):
                # malformed; skip this action
                continue

            # capture JSON blob that starts after "Action Input:"
            first = lines[i]
            json_part = first.split("Action Input:", 1)[1].lstrip()
            i += 1

            # continue until next Action: or Memory: or end
            while i < len(lines):
                peek = lines[i].lstrip()
                if peek.startswith("Action:") or peek.startswith("Memory:"):
                    break
                json_part += "\n" + lines[i]
                i += 1

            json_part = json_part.strip()
            try:
                tool_input = json.loads(json_part) if json_part else {}
            except json.JSONDecodeError:
                continue

            # your prompt says JSON object => dict
            if not isinstance(tool_input, dict):
                continue

            actions.append(ToolCallSpec(tool=tool_name, tool_input=tool_input))
            continue

        i += 1

    return ParsedReAct(actions=actions, memory_update=memory_update)


class SimpleReAct:
    def __init__(
        self,
        llm_proxy: LlmProxy,
        max_num_turns: int,
        tools: list[Tool],
        bus: EventBus,
        *,
        memory_mode: bool = True,
        memory_max_chars: int = 4000,
        history_keep_last: int = 8,
        compact_history_with_memory: bool = False,
        max_tool_calls: int | None = None,
    ):
        """
        memory_mode:
          - False: identical behavior to your original agent (full message_history grows).
          - True: enables an explicit persistent JSON memory that the model can update via `Memory: {...}`.
                  Optionally compacts message history to make memory actually matter.

        memory_max_chars:
          - Hard cap on serialized memory JSON length (approx budget control).
            Oldest keys are evicted deterministically if exceeded.

        history_keep_last:
          - When compact_history_with_memory=True, keep only the last N non-system messages
            (in addition to system/tool/memory messages and the initial user task).
            This prevents “full transcript” from being the real memory.

        compact_history_with_memory:
          - If True, truncate history when memory_mode is enabled.
        """
        self.llm_proxy = llm_proxy
        self.max_num_turns = max_num_turns
        self.bus = bus

        self.memory_mode = memory_mode
        self.memory_max_chars = memory_max_chars
        self.history_keep_last = history_keep_last
        self.compact_history_with_memory = compact_history_with_memory

        self.finish_tool = FinishTool()
        self.tools = list(tools) + [self.finish_tool]
        self.tool_map = {tool.name: tool for tool in self.tools}

        self.memory: dict[str, object] = {}
        self._memory_message: dict[str, str] | None = None

        system_prompt = SYSTEM_PROMPT + (
            MEMORY_PROMPT_APPENDIX if self.memory_mode else ""
        )

        self.tool_prompt = "Available tools:\n" + "\n".join(
            [tool.full_python_doc() for tool in self.tools]
        )

        self.max_tool_calls = max_tool_calls
        self.multi_action_mode = max_tool_calls is not None and max_tool_calls > 0

        if self.multi_action_mode:
            system_prompt = system_prompt.replace(
                "- Take exactly one action per turn.",
                "- Take one or more actions per turn (see below).",
            )
            system_prompt += BATCH_ACTIONS_PROMPT_APPENDIX.format(
                max_actions_per_turn=self.max_tool_calls
            )

        self.message_history = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": self.tool_prompt},
        ]

        if self.memory_mode:
            # Keep a stable dict object so we can mutate its "content" in-place.
            self._memory_message = {
                "role": "system",
                "content": self._render_memory_system_message(),
            }
            self.message_history.append(self._memory_message)

        logger.info(
            "Starting SimpleReAct agent with %s tools (memory_mode=%s)",
            len(self.tools),
            self.memory_mode,
        )

    def _render_memory_system_message(self) -> str:
        # Compact JSON to reduce token footprint; stable key order for reproducibility.
        mem_json = json.dumps(
            self.memory,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            "Persistent memory (JSON). Treat this as your state across turns.\n"
            f"{mem_json}"
        )

    def _apply_memory_update(self, update: dict[str, object]) -> None:
        """
        Apply patch semantics:
          - value is None => delete key
          - else => set/update key (move-to-end for deterministic eviction behavior)
        """
        for k, v in update.items():
            if v is None:
                self.memory.pop(k, None)
            else:
                # Move-to-end semantics to make eviction deterministic and “recently used” keys survive longer.
                if k in self.memory:
                    self.memory.pop(k)
                self.memory[k] = v

        # Enforce memory budget (deterministic eviction of oldest keys).
        self._enforce_memory_budget()

        # Refresh the system memory message content.
        if self._memory_message is not None:
            self._memory_message["content"] = self._render_memory_system_message()

    def _enforce_memory_budget(self) -> None:
        if self.memory_max_chars <= 0:
            # treat as "no memory"
            self.memory.clear()
            return

        def _serialized_len() -> int:
            return len(
                json.dumps(
                    self.memory,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )

        # Evict oldest keys until within budget.
        while self.memory and _serialized_len() > self.memory_max_chars:
            oldest_key = next(iter(self.memory))
            self.memory.pop(oldest_key, None)

    def _maybe_compact_history(self) -> None:
        """
        In memory mode, optionally keep history short so that:
          - long-term state is the explicit memory dict, not the ever-growing transcript
        """
        if not (self.memory_mode and self.compact_history_with_memory):
            return

        # Indices of all system messages (system prompt, tool prompt, memory prompt)
        system_idxs = [
            i for i, m in enumerate(self.message_history) if m.get("role") == "system"
        ]

        # Keep the initial user task message (first user message after system messages)
        first_user_idx = next(
            (i for i, m in enumerate(self.message_history) if m.get("role") == "user"),
            None,
        )

        # Keep last N non-system messages for recency
        non_system_idxs = [
            i for i, m in enumerate(self.message_history) if m.get("role") != "system"
        ]
        tail_idxs = (
            non_system_idxs[-self.history_keep_last :]
            if self.history_keep_last > 0
            else []
        )

        keep = set(system_idxs)
        if first_user_idx is not None:
            keep.add(first_user_idx)
        keep.update(tail_idxs)

        self.message_history = [
            m for i, m in enumerate(self.message_history) if i in keep
        ]

    async def run(self, prompt: str):
        def _append_error_observation(msg: str) -> None:
            # Keep the same "Observation: <json>" shape as normal tool results.
            observation = json.dumps(
                [{"tool": "__error__", "input": {}, "output": msg}],
                ensure_ascii=False,
            )
            self.message_history.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )

        total_usage = TokenUsage(0, 0, 0)
        try:
            # Reset per-episode state
            self.finish_tool.is_finished = False
            # Reset memory per run (typical for benchmarking episodes).
            if self.memory_mode:
                self.memory.clear()
                if self._memory_message is not None:
                    self._memory_message["content"] = (
                        self._render_memory_system_message()
                    )

            await self.bus.emit(
                SystemPromptEvent(
                    prompts=[
                        message["content"]
                        for message in self.message_history
                        if message.get("role") == "system"
                    ]
                )
            )
            await self.bus.emit(StartEvent(task=prompt))

            self.message_history.append({"content": prompt, "role": "user"})
            self._maybe_compact_history()

            for turn_idx in range(self.max_num_turns):
                completion = await self.llm_proxy.complete(
                    messages=list(self.message_history)
                )
                if completion.finish_reason != "stop":
                    raise RuntimeError(
                        f"Unexpected finish reason: {completion.finish_reason}"
                    )

                response_msg = completion.text
                completion_token_usage = completion.token_usage
                total_usage = total_usage + completion_token_usage

                self.message_history.append(
                    {"role": "assistant", "content": response_msg}
                )

                parsed = _parse_react_response(response_msg or "")

                if not parsed.actions:
                    _append_error_observation(
                        "Format error: missing Action block(s). Use Thought/Action/Action Input."
                    )
                    await self.bus.emit(
                        StepEvent(
                            turn=turn_idx,
                            assistant_text=response_msg,
                            code=None,
                            interpreter_result={"error": "Missing Action(s)."},
                            token_usage=completion_token_usage,
                        )
                    )
                    self._maybe_compact_history()
                    continue

                if not self.multi_action_mode and len(parsed.actions) > 1:
                    _append_error_observation(
                        "Format error: You provided multiple actions, but only one action per turn is allowed."
                    )
                    await self.bus.emit(
                        StepEvent(
                            turn=turn_idx,
                            assistant_text=response_msg,
                            code=None,
                            interpreter_result={
                                "error": "Multiple actions not allowed."
                            },
                            token_usage=completion_token_usage,
                        )
                    )
                    self._maybe_compact_history()
                    continue

                # 5) Enforce multi-action count if enabled (use Observation, not a bare string)
                if self.multi_action_mode and self.max_tool_calls is not None:
                    if len(parsed.actions) > self.max_tool_calls:
                        _append_error_observation(
                            f"Too many actions in one turn ({len(parsed.actions)} > {self.max_tool_calls}). "
                            f"Submit at most {self.max_tool_calls} actions."
                        )
                        await self.bus.emit(
                            StepEvent(
                                turn=turn_idx,
                                assistant_text=response_msg,
                                code=None,
                                interpreter_result={
                                    "error": "Too many actions in one turn."
                                },
                                token_usage=completion_token_usage,
                            )
                        )
                        self._maybe_compact_history()
                        continue

                # 6) Enforce finish placement / multiplicity
                finish_positions = [
                    i for i, c in enumerate(parsed.actions) if c.tool == "finish"
                ]
                if finish_positions:
                    if len(finish_positions) > 1:
                        _append_error_observation(
                            "Format error: finish() must be called at most once."
                        )
                        await self.bus.emit(
                            StepEvent(
                                turn=turn_idx,
                                assistant_text=response_msg,
                                code=None,
                                interpreter_result={
                                    "error": "Multiple finish actions."
                                },
                                token_usage=completion_token_usage,
                            )
                        )
                        self._maybe_compact_history()
                        continue

                    if finish_positions[0] != len(parsed.actions) - 1:
                        _append_error_observation(
                            "Format error: if you call finish(), it must be the LAST action in the message."
                        )
                        await self.bus.emit(
                            StepEvent(
                                turn=turn_idx,
                                assistant_text=response_msg,
                                code=None,
                                interpreter_result={"error": "finish not last."},
                                token_usage=completion_token_usage,
                            )
                        )
                        self._maybe_compact_history()
                        continue

                tool_results = []
                for call in parsed.actions:
                    tool = self.tool_map.get(call.tool)

                    if tool is None:
                        tool_output = f"Error: Unknown tool requested: {call.tool}"
                    else:
                        try:
                            # Execute the tool and capture the result
                            tool_output = await tool(**call.tool_input)
                        except Exception as e:
                            # Capture the crash and turn it into an observation
                            tool_output = f"ToolRuntimeException: {str(e)}"

                    tool_results.append(
                        {
                            "tool": call.tool,
                            "input": call.tool_input,
                            "output": tool_output,
                        }
                    )

                # Now apply memory patch (after tools)
                if self.memory_mode and parsed.memory_update:
                    self._apply_memory_update(parsed.memory_update)

                observation = json.dumps(tool_results, ensure_ascii=False)
                self.message_history.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )

                await self.bus.emit(
                    StepEvent(
                        turn=turn_idx,
                        assistant_text=response_msg,
                        code=None,
                        interpreter_result={
                            "tool_calls": tool_results,
                            **(
                                {"memory": dict(self.memory)}
                                if self.memory_mode
                                else {}
                            ),
                        },
                        token_usage=completion_token_usage,
                    )
                )
                self._maybe_compact_history()

                if self.finish_tool.is_finished:
                    await self.bus.emit(
                        FinishEvent(reason="finish_tool", token_usage=total_usage)
                    )
                    return

            await self.bus.emit(
                FinishEvent(reason="max_turns", token_usage=total_usage)
            )
        except Exception as exc:
            await self.bus.emit(ErrorEvent(error=str(exc)))
            await self.bus.emit(FinishEvent(reason="error", token_usage=total_usage))
            raise exc
