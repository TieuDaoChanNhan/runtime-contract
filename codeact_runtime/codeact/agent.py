import json
import logging
import re
from datetime import datetime, timezone

from codeact_runtime.codeact.events import (
    ErrorEvent,
    EventBus,
    FinishEvent,
    ModelCallEvent,
    ModelResponseEvent,
    StartEvent,
    StepEvent,
    SystemPromptEvent,
    UserObservationEvent,
)
from codeact_runtime.codeact.interpreter import AsyncInterpreter
from codeact_runtime.codeact.tool import Tool
from codeact_runtime.llm import LlmProxy, TokenUsage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a CodeAct-style autonomous agent.

You solve tasks by alternating between:
1. Natural-language reasoning (plain text), and
2. Executable simple Python code blob (inside fenced code blocks).

Each step (output) can include at most 1 (one) code block.

Be concise in your reasoning and code.

When you are finished solving the task, ensure that you output a Python code block which calls the `finish` tool. 
Call the `finish` tool ONLY after completely solving the task, NOT on every turn.

Execution rules:
- Python code blocks are executed sequentially.
- Only expressions that are printed or explicitly returned are visible to you.
- Variable assignments alone do NOT produce observable output.
- Do not use variable names that conflict with tool names.

Output discipline:
- If a value will be needed for later reasoning or decisions, you MUST print it
  (e.g., via `print(...)`) or make it the final expression in the code block.
- Do not rely on implicit interpreter state visibility.

Tool usage:
- All tool calls must occur inside Python code blocks.
- Do not fabricate tool outputs; rely only on observed execution results.

Error handling:
- If execution fails or a needed value is missing, explain why and rerun with
  corrected code.

Completion:
- When the task is complete, provide a final plain-text answer.
- Do not emit further code after completion.

Output Structure:
You must strictly follow this format for every single turn:

1. Reflect upon the previous observation, explain your reasoning, what you want to verify, or what the next step is.
2. A single executable Python block.

You prioritize observability and correctness over brevity.
"""

FEW_SHOT_EXAMPLES = """--- EXAMPLE: PERSISTENT STATE ---
Task: Store items and sum values.

Turn 1
Assistant:
1. I will initialize the global list `items` and print its length and contents.
```python
items = [10, 20]

def foo(items):
    return items + items

print(len(items))
```
User: {"observation": {"success": true, "result": null, "output": "2\n", "error": null}, "runtime_state": {"runtime": "persistent", "active_globals": ["items", "foo"], "last_step_globals": ["items", "foo"]}}

Turn 2
Assistant:

I can see `items` and `foo` in active globals, so I will reuse them directly.

```python
items = foo(items)
total = sum(items)
print(f"Total: {total}")

finish()
```
User: {"observation": {"success": true, "result": null, "output": "Total: 60\n", "error": null}, "runtime_state": {"runtime": "persistent", "active_globals": ["items", "foo", "total"], "last_step_globals": ["items", "foo", "total"]}} --- EXAMPLE END --- 

--- EXAMPLE: ITERATIVE EXPLORATION ---
Task: Inspect 10 items from a list, 2 at a time.

Turn 1
Assistant:
1. I will define a helper function `batch_inspect` to handle the logic. Then I will run the first batch.
```python
import json

# Setup data
items = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
inspected = []

def batch_inspect():
    # Calculate indices based on current state
    start = len(inspected)
    # Stop if done
    if start >= len(items):
        print("Done!")
        return
        
    # Process next 2
    batch = items[start : start + 2]
    for x in batch:
        # Mock inspection logic
        inspected.append(x)
        print(f"Inspected {x}")
    
    print(f"Progress: {len(inspected)}/{len(items)}")

# Run first time
batch_inspect()
```
User: {"observation": {"success": true, "result": null, "output": "Inspected 100\nInspected 200\nProgress: 2/10\n", "error": null}, "runtime_state": {"runtime": "persistent", "active_globals": ["items", "inspected", "batch_inspect"], "last_step_globals": ["items", "inspected", "batch_inspect"]}}

Turn 2
Assistant:

I can see `batch_inspect` in active globals, so I will call it to process the next items.
```python
batch_inspect()
```
User: {"observation": {"success": true, "result": null, "output": "Inspected 300\nInspected 400\nProgress: 4/10\n", "error": null}, "runtime_state": {"runtime": "persistent", "active_globals": ["items", "inspected", "batch_inspect"], "last_step_globals": ["items", "inspected", "batch_inspect"]}} ---


--- EXAMPLE: BATCH PROCESSING ---
Task: Process a list of numbers 3 at a time, keeping only even numbers, and report the running sum.

Turn 1
Assistant:
1. I will define a "fat" helper function `run_step()` that handles the fetching, filtering, and reporting all in one go.
```python
# Setup
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
processed_count = 0
valid_evens = []

def run_step():
    global processed_count
    
    # 1. FETCH / ACT
    batch = numbers[processed_count : processed_count + 3]
    if not batch:
        print("Done.")
        return

    # 2. UPDATE STATE
    for n in batch:
        print(f"Saw: {n}")
        if n % 2 == 0:
            valid_evens.append(n)
    
    processed_count += len(batch)
    
    # 3. ANALYZE & REPORT (Inside the function!)
    current_sum = sum(valid_evens)
    print(f"--- Report ---")
    print(f"Progress: {processed_count}/{len(numbers)}")
    print(f"Valid Evens: {valid_evens}")
    print(f"Running Sum: {current_sum}")

# Execute first step
run_step()
```
User: {"observation": {"success": true, "result": null, "output": "Saw: 1\nSaw: 2\nSaw: 3\n--- Report ---\nProgress: 3/9\nValid Evens: [2]\nRunning Sum: 2\n", "error": null}, "runtime_state": {"runtime": "persistent", "active_globals": ["numbers", "processed_count", "run_step", "valid_evens"], "last_step_globals": ["numbers", "processed_count", "run_step", "valid_evens"]}} --- EXAMPLE END ---


"""

STATELESS_FEWSHOT_EXAMPLES = """--- EXAMPLE: RESET STATE ---
Task: Store items and sum values.

Turn 1
Assistant:
1. I will initialize the list `items` and print its contents so I can retrieve them next turn.
```python
items = [10, 20]

def foo(items):
    return items + items

# State must be printed to survive the reset
print(f"STATE: items={items}")
print(len(items))
```
User: {"observation": {"success": true, "result": null, "output": "STATE: items=[10, 20]\n2\n", "error": null}, "runtime_state": {"runtime": "reset", "active_globals": [], "last_step_globals": ["items", "foo"]}}

Turn 2
Assistant:

The environment has reset and `active_globals` is empty. I will read the previous observation, redefine `items` and `foo`, then compute the total.

```python
# Re-initializing based on previous observation
items = [10, 20]

def foo(items):
    return items + items

items = foo(items)
total = sum(items)
print(f"Total: {total}")

finish()
```
User: {"observation": {"success": true, "result": null, "output": "Total: 60\n", "error": null}, "runtime_state": {"runtime": "reset", "active_globals": [], "last_step_globals": ["items", "foo", "total"]}} --- EXAMPLE END ---

--- EXAMPLE: ITERATIVE EXPLORATION ---
Task: Inspect 10 items from a list, 2 at a time.

Turn 1
Assistant:

I will define the items and process the first batch. Because state resets, I must explicitly print my progress and the inspected list.

```python

# Setup data
items = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
inspected = []

# Process first 2
batch = items[0:2]
for x in batch:
    # Mock inspection logic
    inspected.append(x)
    print(f"Inspected {x}")
    
print(f"Progress: {len(inspected)}/{len(items)}")
# Print explicit state for the next turn
print(f"STATE: inspected={inspected}")
```
User: {"observation": {"success": true, "result": null, "output": "Inspected 100\nInspected 200\nProgress: 2/10\nSTATE: inspected=[100, 200]\n", "error": null}, "runtime_state": {"runtime": "reset", "active_globals": [], "last_step_globals": ["batch", "inspected", "items", "x"]}}

Turn 2
Assistant:

Variables were reset and `active_globals` is empty. From the previous observation, I see inspected=[100, 200]. I will redefine data, resume at index 2, and process the next batch.

```python
import json

items = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# Restoring state from previous output
inspected = [100, 200] 

start = len(inspected)
batch = items[start : start + 2]
for x in batch:
    inspected.append(x)
    print(f"Inspected {x}")

print(f"Progress: {len(inspected)}/{len(items)}")
print(f"STATE: inspected={inspected}")
```
User: {"observation": {"success": true, "result": null, "output": "Inspected 300\nInspected 400\nProgress: 4/10\nSTATE: inspected=[100, 200, 300, 400]\n", "error": null}, "runtime_state": {"runtime": "reset", "active_globals": [], "last_step_globals": ["batch", "inspected", "items", "start", "x"]}} --- EXAMPLE END ---

--- EXAMPLE: BATCH PROCESSING ---
Task: Process a list of numbers 3 at a time, keeping only even numbers, and report the running sum.

Turn 1
Assistant:

I will process the first batch of 3 numbers. I must print my processed_count and valid_evens so I can reconstruct them later.

```python
# Setup
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
processed_count = 0
valid_evens = []

# 1. FETCH / ACT
batch = numbers[processed_count : processed_count + 3]

# 2. UPDATE STATE
for n in batch:
    print(f"Saw: {n}")
    if n % 2 == 0:
        valid_evens.append(n)

processed_count += len(batch)
current_sum = sum(valid_evens)

# 3. ANALYZE & REPORT
print(f"--- Report ---")
print(f"Progress: {processed_count}/{len(numbers)}")
print(f"Running Sum: {current_sum}")
# Explicit state handoff
print(f"STATE: processed_count={processed_count}, valid_evens={valid_evens}")
```
User: {"observation": {"success": true, "result": null, "output": "Saw: 1\nSaw: 2\nSaw: 3\n--- Report ---\nProgress: 3/9\nRunning Sum: 2\nSTATE: processed_count=3, valid_evens=[2]\n", "error": null}, "runtime_state": {"runtime": "reset", "active_globals": [], "last_step_globals": ["batch", "n", "numbers", "processed_count", "valid_evens"]}} --- EXAMPLE END ---
"""


class FinishTool(Tool):
    name: str = "finish"
    doc: str = "Call when the task execution is finished"
    arg_doc: dict[str, str] = {}

    is_finished: bool = False

    def __init__(self):
        super().__init__()

    async def run(self) -> None:
        self.is_finished = True


def extract_code_blocks(text: str) -> list[str]:
    """
    Extract all fenced code blocks from text.

    Returns a list of code strings without the backticks or language tags.
    """
    pattern = re.compile(
        r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```",
        re.DOTALL,
    )
    return [block.strip() for block in pattern.findall(text)]


class CodeAct:
    def __init__(
        self,
        llm_proxy: LlmProxy,
        max_num_turns: int,
        tools: list[Tool],
        bus: EventBus,
        persistent_state: bool = True,
        max_tool_calls: int | None = None,
        state_carryover_on_cap: bool = False,
        announce_carryover: bool = False,
        llm_seed: int | None = None,
    ):
        self.llm_seed = llm_seed
        self.llm_proxy = llm_proxy
        self.max_num_turns = max_num_turns
        self.finish_tool = FinishTool()
        self.persistent_state = persistent_state
        # Cap-boundary carryover changes the RUNTIME only. The prompt, the few-shot
        # examples, the cap error and the runtime_state header below stay exactly those of
        # the reset condition, so the policy is never told that a truncated turn's
        # workspace survived -- the only thing that differs is whether it did.
        self.state_carryover_on_cap = state_carryover_on_cap
        # ...unless the carryover is ANNOUNCED, the one variant that does change what the
        # policy reads: the banner then reports the bindings that really are still live
        # when it writes its next block, exactly as the persistent runtime's banner does.
        # It separates the informational half of the recovery gate from the behavioural one.
        self.announce_carryover = announce_carryover

        self.tools = tools + [self.finish_tool]

        self.tool_prompt = "Available tools:\n" + "\n".join(
            [x.full_python_doc() for x in self.tools]
        )
        if max_tool_calls is not None:
            self.tool_prompt += f"\nSTRICT LIMIT: Do not exceed {max_tool_calls} tool calls in one turn. Use loops carefully and batch your work across multiple turns if needed.\n"

        if persistent_state:
            state_rule = (
                "1. Globals persist eternally. Once you define `x = 1`, it is available forever. NEVER re-import libraries.\n"
                "2. NEVER paste code from previous steps."
            )
        else:
            state_rule = "1. Runtime state resets every turn. Python variables DO NOT persist. You must redefine variables and re-import libraries every step."

        # 2. STRATEGY: Shared instructions for BOTH regimes (The "Pep Talk")
        # This ensures the baseline isn't artificially handicapped by poor prompting.
        shared_strategy = """CRITICAL RULES FOR THIS SESSION:
    - Assume the environment is valid.
    - Output discipline: Print values you need to see.

    PLANNING & EFFICIENCY:
    - Write reusable helper functions and then use them.
    - Plan your logic steps ahead of time.
    - Code efficiency is very important.
        """

        # Combine them
        runtime_note = f"""
Runtime Execution Mode: {"PERSISTENT" if persistent_state else "RESET"}

{state_rule}
{shared_strategy}
"""

        self.message_history = [
            {"content": SYSTEM_PROMPT, "role": "system"},
            {"role": "system", "content": runtime_note},
        ]

        # Use few-shot examples appropriate for the mode
        # (Or share them if you want to test zero-shot adaptation)
        if persistent_state:
            self.message_history.append(
                {"role": "system", "content": FEW_SHOT_EXAMPLES}
            )
        else:
            # Inject the matched Reset examples
            self.message_history.append(
                {"role": "system", "content": STATELESS_FEWSHOT_EXAMPLES}
            )

        self.message_history.append({"content": self.tool_prompt, "role": "system"})

        self.interpreter = AsyncInterpreter(
            persistent_state=persistent_state,
            max_tool_calls=max_tool_calls,
            carryover_on_cap=state_carryover_on_cap,
        )
        for tool in self.tools:
            self.interpreter.register_tool(tool.name, tool)
        self.bus = bus

        logger.info(f"Starting agent with {len(self.tools)} tools")

    async def run(self, prompt: str):
        total_usage = TokenUsage(0, 0, 0)
        interpreter_globals = {}
        try:
            logger.info("Starting task run")

            await self.bus.emit(StartEvent(task=prompt))
            await self.bus.emit(
                SystemPromptEvent(
                    prompts=[
                        message["content"]
                        for message in self.message_history
                        if message.get("role") == "system"
                    ]
                )
            )

            last_step_globals = []

            # --- TURN 0 INJECTION ---
            # Bundle the Task Prompt with the initial Turn 0 state
            initial_state_header = {
                "runtime": "persistent" if self.persistent_state else "reset",
                "active_globals": [],
                "last_step_globals": [],
            }
            initial_user_msg = json.dumps(
                {"task": prompt, "runtime_state": initial_state_header}
            )
            self.message_history.append({"role": "user", "content": initial_user_msg})

            for turn_idx in range(self.max_num_turns):
                if self.finish_tool.is_finished:
                    logger.info("Task execution finished")
                    await self.bus.emit(
                        FinishEvent(reason="finish_tool", token_usage=total_usage)
                    )
                    return

                call_started_at = datetime.now(timezone.utc)
                try:
                    # Pass the actual self.message_history! No temporary copies needed.
                    # seed is passed only when one is configured, so a proxy written
                    # against the older signature keeps working for every unseeded run
                    # (the loop swallows call errors, so a TypeError here would silently
                    # burn the whole episode rather than fail loudly)
                    if self.llm_seed is None:
                        completion = await self.llm_proxy.complete(
                            messages=self.message_history
                        )
                    else:
                        completion = await self.llm_proxy.complete(
                            messages=self.message_history, seed=self.llm_seed
                        )
                except Exception as llm_error:
                    error_msg = f"LLM Call Failed: {str(llm_error)}"
                    logger.warning(error_msg)
                    await self.bus.emit(ErrorEvent(error=error_msg))
                    self.message_history.append(
                        {"role": "user", "content": json.dumps({"error": error_msg})}
                    )
                    continue

                if completion.finish_reason != "stop":
                    error_msg = f"LLM ended unexpectedly (reason: {completion.finish_reason}). Please continue."
                    await self.bus.emit(ErrorEvent(error=error_msg))
                    self.message_history.append(
                        {"role": "user", "content": json.dumps({"error": error_msg})}
                    )
                    continue

                response_msg = completion.text
                completion_token_usage = completion.token_usage
                total_usage = total_usage + completion_token_usage

                await self.bus.emit(
                    ModelCallEvent(
                        timestamp=call_started_at.isoformat(),
                        prompt_tokens=completion_token_usage.prompt_tokens,
                    )
                )
                await self.bus.emit(
                    ModelResponseEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        completion_tokens=completion_token_usage.completion_tokens,
                        total_tokens=completion_token_usage.total_tokens,
                    )
                )

                code_blocks = extract_code_blocks(response_msg)
                code = None
                interpreter_result = None

                if code_blocks:
                    # 1. Isolate the first block & Reasoning
                    code = code_blocks[0]
                    reasoning = response_msg.split("```")[0].strip()
                    clean_content = f"{reasoning}\n\n```python\n{code}\n```"

                    self.message_history.append(
                        {"role": "assistant", "content": clean_content}
                    )

                    # 2. Execute Code
                    interpreter_result = await self.interpreter.run_code(code)

                    if not interpreter_result["success"]:
                        await self.bus.emit(
                            ErrorEvent(error=interpreter_result.get("error") or "")
                        )

                    # 3. Handle Globals State Tracking
                    interpreter_globals = interpreter_result.get("globals", {})
                    step_globals = sorted(interpreter_globals.keys())

                    if self.persistent_state:
                        active_globals = step_globals
                        last_step_globals = step_globals
                    else:
                        announced = (
                            self.announce_carryover
                            and self.interpreter.has_checkpoint
                        )
                        active_globals = step_globals if announced else []
                        last_step_globals = step_globals

                    state_header = {
                        "runtime": "persistent" if self.persistent_state else "reset",
                        "active_globals": active_globals,
                        "last_step_globals": last_step_globals,
                    }

                    interpreter_result_copy = interpreter_result.copy()
                    interpreter_result_copy.pop("globals", {})

                    combined = {
                        "observation": interpreter_result_copy,
                        "runtime_state": state_header,
                    }
                    if len(code_blocks) > 1:
                        combined["system_note"] = (
                            "Multiple blocks detected. Only the first was executed."
                        )

                    self.message_history.append(
                        {"role": "user", "content": json.dumps(combined, default=str)}
                    )
                    await self.bus.emit(
                        UserObservationEvent(
                            turn=turn_idx, content=json.dumps(combined, default=str)
                        )
                    )
                else:
                    error_msg = "Error: No code block found. Please provide an executable block."

                    interpreter_result_copy = {
                        "success": False,
                        "result": None,
                        "output": "",
                        "error": error_msg,
                    }

                    # Preserve a consistent runtime_state payload
                    # no code block ran this turn, so a pending checkpoint was not
                    # consumed: under announced carryover it is still live next turn
                    announced = (
                        self.announce_carryover and self.interpreter.has_checkpoint
                    )
                    state_header = {
                        "runtime": "persistent" if self.persistent_state else "reset",
                        "active_globals": last_step_globals
                        if (self.persistent_state or announced)
                        else [],
                        "last_step_globals": last_step_globals,
                    }

                    combined = {
                        "observation": interpreter_result_copy,
                        "runtime_state": state_header,
                        "system_note": "No code block found; model must retry with a single fenced python block.",
                    }

                    self.message_history.append(
                        {"role": "assistant", "content": response_msg}
                    )
                    self.message_history.append(
                        {"role": "user", "content": json.dumps(combined)}
                    )

                    await self.bus.emit(ErrorEvent(error=error_msg))
                    await self.bus.emit(
                        UserObservationEvent(
                            turn=turn_idx, content=json.dumps(combined)
                        )
                    )

                await self.bus.emit(
                    StepEvent(
                        turn=turn_idx,
                        assistant_text=response_msg,
                        code=code,
                        interpreter_result=interpreter_result_copy,
                        token_usage=completion_token_usage,
                    )
                )

            await self.bus.emit(
                FinishEvent(reason="max_turns", token_usage=total_usage)
            )

        except Exception as e:
            await self.bus.emit(ErrorEvent(error=str(e)))
            await self.bus.emit(FinishEvent(reason="error", token_usage=total_usage))
            raise e
