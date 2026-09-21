import asyncio
import builtins
import concurrent.futures
import io
import json
import sys
import traceback
from typing import Any, Awaitable, Callable, Dict, Optional


class ToolCallLimitError(RuntimeError):
    def __init__(self, message: str, partial_output: str = ""):
        super().__init__(message)
        self.partial_output = partial_output


class AsyncInterpreter:
    def __init__(
        self,
        *,
        persistent_state: bool = False,
        max_tool_calls: Optional[int] = None,
        carryover_on_cap: bool = False,
    ):
        self.persistent_state = persistent_state
        # Cap-boundary carryover ("interruption checkpointing"): a RESET runtime that keeps
        # the globals of a turn the tool-call cap terminated, and only that turn's, for
        # exactly the next turn. Every other turn boundary still wipes the workspace. It
        # isolates the state destroyed at the cap-forced boundary from everything else that
        # differs between the persistent and reset runtimes.
        if carryover_on_cap and persistent_state:
            raise ValueError(
                "carryover_on_cap applies to a reset runtime; a persistent one already "
                "keeps state across every boundary."
            )
        self.carryover_on_cap = carryover_on_cap
        self._carry_env: Optional[Dict[str, Any]] = None
        # observability for tests/harnesses; never shown to the policy
        self.carried_in_last_run = False
        self.cap_hit_last_run = False
        self.stdout = io.StringIO()
        if max_tool_calls is not None and max_tool_calls < 1:
            raise ValueError("max_tool_calls must be a positive integer or None.")
        self._max_tool_calls = max_tool_calls
        self._tool_call_count = 0
        self._tool_call_lock = asyncio.Lock()

        # This executor runs the user's synchronous code
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # We hold a reference to the main event loop to schedule tool calls
        self._loop = asyncio.get_event_loop()

        # The environment (globals) for the user code
        self._base_env: Dict[str, Any] = {"__builtins__": builtins}
        self._base_env["json"] = json
        self._env: Optional[Dict[str, Any]] = None

        # Registry of original async tools
        self._async_tools: Dict[str, Callable[..., Awaitable[Any]]] = {}

        # Cache the set of builtin names for quick lookup (protection logic)
        self._builtin_names = set(dir(builtins))

        if self.persistent_state:
            self._ensure_persistent_env()

    def _ensure_persistent_env(self) -> None:
        if self._env is None:
            self._env = dict(self._base_env)

    def register_tool(self, name: str, tool: Callable[..., Awaitable[Any]]) -> None:
        """
        Registers an async tool. It injects a SYNCHRONOUS wrapper into the user's env.
        """
        self._async_tools[name] = tool

        # The bridge: Sync wrapper running in user thread -> Async tool in main loop
        def sync_tool_wrapper(*args, **kwargs):
            # Check limit BEFORE scheduling the tool
            if self._max_tool_calls is not None:
                # This returns a Future. .result() raises ToolCallLimitError if limit hit.
                asyncio.run_coroutine_threadsafe(
                    self._check_and_increment_tool_call(), self._loop
                ).result()

            # 1. Schedule the async tool to run on the main event loop
            future = asyncio.run_coroutine_threadsafe(tool(*args, **kwargs), self._loop)  # type: ignore
            # 2. Block this thread until the result is ready
            try:
                return future.result()
            except Exception as e:
                # Re-raise exceptions from the tool in the user's thread
                raise e

        self._base_env[name] = sync_tool_wrapper
        if self.persistent_state and self._env is not None:
            self._env[name] = sync_tool_wrapper

    @property
    def has_checkpoint(self) -> bool:
        """Whether a truncated turn's workspace is waiting to be handed to the next block.

        A caller that ANNOUNCES the carryover reads this rather than the last run's cap
        flag, because a turn that produced no code at all consumes no checkpoint: the
        state is still there when the policy finally emits one."""
        return self._carry_env is not None

    def _restore_env(self, env: Dict[str, Any]) -> None:
        """Re-bind the tools and un-shadow the builtins in an env that outlives its turn.

        Applies to any surviving workspace: the persistent runtime's env and, under
        cap-boundary carryover, the checkpointed one."""
        for tool_name in self._async_tools.keys():
            if tool_name in self._base_env:
                env[tool_name] = self._base_env[tool_name]

        keys_to_remove = []
        for k in env.keys():
            if k in self._base_env:
                continue
            if k in self._builtin_names:
                keys_to_remove.append(k)

        for k in keys_to_remove:
            del env[k]

    def _get_persisted_globals(self, active_env: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a clean view of user-defined variables from the given environment."""
        reserved = set(self._base_env.keys()) | {"__builtins__"}
        import types

        clean = {}
        for k, v in active_env.items():  # <--- Iterate over the passed-in env
            if k in reserved or k.startswith("__"):
                continue
            if isinstance(v, (types.ModuleType)):
                continue
            clean[k] = repr(v)
        return clean

    def _execute_in_thread(self, code: str, env: Dict[str, Any]) -> Any:
        """
        This function runs inside the ThreadPoolExecutor.
        It is fully synchronous.
        """
        # 1. Create a local buffer for this specific thread/execution
        out_capture = io.StringIO()

        # 2. Define a shadow 'print' function that writes to our local buffer
        #    instead of the global sys.stdout (which is shared by all threads).
        def safe_print(*args, sep=" ", end="\n", file=None, flush=False):
            # If the user tries to print to stdout (default), hijack it to our buffer.
            if file is None or file == sys.stdout:
                file = out_capture

            try:
                # Call the real built-in print, but force the file argument
                builtins.print(*args, sep=sep, end=end, file=file, flush=flush)
            except Exception:
                # If writing to the buffer fails for some reason, ignore it
                # to prevent crashing the agent logic.
                pass

        # 3. Inject the safe print into the environment
        env["print"] = safe_print

        try:
            # 4. Execute the code.
            # We NO LONGER use contextlib.redirect_stdout here.
            exec(code, env)

            return {
                "output": out_capture.getvalue(),
                "success": True,
                "error": None,
            }

        except ToolCallLimitError as e:
            # Catch the limit error, capture partial output, and return failure.
            # cap_hit marks the termination itself, so a caller (cap-boundary carryover)
            # never has to re-detect it by matching the error string.
            return {
                "output": out_capture.getvalue(),
                "success": False,
                "error": str(e),
                "cap_hit": True,
            }

        except Exception as e:
            # --- ERROR CLEANING LOGIC ---
            # We want to show the user where the error happened in THEIR code (<string>),
            # filtering out the internal frames of the interpreter infrastructure.
            tb_list = traceback.extract_tb(e.__traceback__)
            user_frames = [frame for frame in tb_list if frame.filename == "<string>"]

            if user_frames:
                clean_trace = "Traceback (most recent call last):\n" + "".join(
                    traceback.format_list(user_frames)
                )
            else:
                clean_trace = ""

            error_details = "".join(traceback.format_exception_only(type(e), e))
            final_error_msg = f"{clean_trace}{error_details}".strip()

            return {
                "output": out_capture.getvalue(),
                "success": False,
                "error": final_error_msg,
            }

    async def _check_and_increment_tool_call(self) -> None:
        async with self._tool_call_lock:
            if self._max_tool_calls is None:
                return
            if self._tool_call_count >= self._max_tool_calls:
                raise ToolCallLimitError(
                    f"Tool call limit exceeded: allowed {self._max_tool_calls} per run."
                )
            self._tool_call_count += 1

    async def run_code(self, code: str) -> Dict[str, Any]:
        """
        Runs the user code in a separate thread.
        """
        carried_in = False
        if self.persistent_state:
            env = self._env
            if env is None:
                env = self._env = dict(self._base_env)
        elif self._carry_env is not None:
            # a checkpoint survives exactly ONE turn: consume it here, so an ordinary
            # (non-truncated) turn is followed by an ordinary reset
            env, self._carry_env, carried_in = self._carry_env, None, True
        else:
            env = dict(self._base_env)
        self.carried_in_last_run = carried_in
        self.cap_hit_last_run = False

        async with self._tool_call_lock:
            self._tool_call_count = 0

        # Per-turn hook: let tools reset any per-turn state (e.g. a per-turn
        # probe cap that forces work to be spread across turns). One run_code
        # call == one agent turn, so this fires exactly once per turn.
        for tool in self._async_tools.values():
            hook = getattr(tool, "on_turn_start", None)
            if hook is not None:
                hook()

        result_payload = None

        try:
            # Run blocking execution. The thread catches ToolCallLimitError and returns dict.
            result_payload = await self._loop.run_in_executor(  # type: ignore
                self._executor, self._execute_in_thread, code, env
            )
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "output": "",
                "error": str(e),
                "globals": self._get_persisted_globals(env),
            }
        finally:
            # SELF-HEALING: Restore tools and builtins
            if self.persistent_state and self._env is not None:
                self._restore_env(self._env)

        assert result_payload is not None
        self.cap_hit_last_run = bool(result_payload.get("cap_hit"))
        if self.carryover_on_cap and self.cap_hit_last_run:
            # the cap truncated this block: keep its workspace for the next turn only.
            # Consecutive truncations chain, because the next turn checkpoints in turn.
            self._restore_env(env)
            self._carry_env = env

        return {
            "success": result_payload["success"],
            "result": None,
            "output": result_payload["output"],
            "error": result_payload["error"],
            "globals": self._get_persisted_globals(env),
        }
