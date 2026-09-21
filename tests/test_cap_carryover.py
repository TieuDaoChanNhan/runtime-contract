"""Cap-boundary carryover: the reset runtime that keeps ONLY a truncated turn's workspace.

The condition exists to isolate one suspected mediator of the mismatch collapse -- the
partially built workspace that the cap-forced turn boundary wipes -- so the runtime is the
only thing it may change. The prompt, the few-shot examples, the cap error and the
`runtime_state` header the policy sees stay exactly those of the reset condition; these
tests pin both halves of that: the state does survive, and the policy is not told so.
"""

import json

import pytest

from codeact_runtime.codeact.agent import CodeAct
from codeact_runtime.codeact.events import EventBus
from codeact_runtime.codeact.interpreter import AsyncInterpreter
from codeact_runtime.codeact.tool import Tool
from codeact_runtime.llm import LLMResult, TokenUsage

CAP = 2


def _interp(**kwargs) -> AsyncInterpreter:
    interp = AsyncInterpreter(persistent_state=False, max_tool_calls=CAP, **kwargs)

    async def ping(value: int = 0) -> int:
        return value

    interp.register_tool("ping", ping)
    return interp


# a block whose loop is cut short by the cap, after CAP calls have landed in `acquired`
TRUNCATED = "acquired = []\nfor i in range(5):\n    acquired.append(ping(i))\n"
EXTEND = "for i in range(5):\n    acquired.append(ping(i))\n"


@pytest.mark.asyncio
async def test_truncated_turn_survives_into_exactly_one_following_turn():
    interp = _interp(carryover_on_cap=True)

    first = await interp.run_code(TRUNCATED)
    assert first["success"] is False
    assert "Tool call limit exceeded" in first["error"]

    # the next turn continues from the truncated turn's workspace, tools included
    second = await interp.run_code("print(len(acquired)); print(ping(9))")
    assert second["success"] is True, second["error"]
    assert second["output"].splitlines() == [str(CAP), "9"]

    # ...and that is the ONLY turn it survives into: this one ended normally
    third = await interp.run_code("print(acquired)")
    assert third["success"] is False
    assert "NameError" in third["error"]


@pytest.mark.asyncio
async def test_consecutive_truncations_chain():
    interp = _interp(carryover_on_cap=True)

    await interp.run_code(TRUNCATED)
    second = await interp.run_code(EXTEND)
    assert "Tool call limit exceeded" in second["error"]

    third = await interp.run_code("print(len(acquired))")
    assert third["success"] is True, third["error"]
    assert third["output"].strip() == str(2 * CAP)


@pytest.mark.asyncio
async def test_ordinary_turns_still_reset():
    interp = _interp(carryover_on_cap=True)

    await interp.run_code("held = 1\nprint(ping(0))")
    after = await interp.run_code("print(held)")
    assert after["success"] is False
    assert "NameError" in after["error"]


@pytest.mark.asyncio
async def test_plain_reset_runtime_wipes_the_truncated_turn():
    interp = _interp()  # carryover off: the cell the intervention is measured against

    first = await interp.run_code(TRUNCATED)
    assert "Tool call limit exceeded" in first["error"]

    second = await interp.run_code("print(len(acquired))")
    assert second["success"] is False
    assert "NameError" in second["error"]


@pytest.mark.asyncio
async def test_carried_workspace_is_repaired_like_a_persistent_one():
    """A name shadowing a builtin must not ride the checkpoint into the next turn."""
    interp = _interp(carryover_on_cap=True)

    await interp.run_code("len = 5\n" + TRUNCATED)
    second = await interp.run_code("print(len(acquired))")
    assert second["success"] is True, second["error"]
    assert second["output"].strip() == str(CAP)
    assert "len" not in second["globals"]


def test_carryover_is_rejected_on_a_persistent_runtime():
    with pytest.raises(ValueError):
        AsyncInterpreter(persistent_state=True, max_tool_calls=CAP, carryover_on_cap=True)


# --------------------------------------------------------------------------------------
# agent level: the observations the policy sees must be the reset condition's
# --------------------------------------------------------------------------------------


class _Ping(Tool):
    name: str = "ping"
    doc: str = "Return the integer passed in."
    arg_doc: dict[str, str] = {"value": "the integer to return"}

    async def run(self, value: int) -> int:
        return value


class _ScriptedProxy:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.seeds: list[int | None] = []

    async def complete(
        self, *, messages, temperature=None, max_tokens=None, seed=None, extra=None
    ) -> LLMResult:
        self.seeds.append(seed)
        text = self._replies.pop(0) if self._replies else "```python\nfinish()\n```"
        return LLMResult(
            text=text,
            finish_reason="stop",
            cache_hit=False,
            token_usage=TokenUsage(0, 0, 0),
        )


REPLIES = [
    "start\n```python\n" + TRUNCATED + "```",
    "continue\n```python\nprint(len(acquired))\n```",
    "done\n```python\nfinish()\n```",
]


async def _run_agent(*, carryover: bool, announce: bool = False, seed: int | None = None):
    proxy = _ScriptedProxy(REPLIES)
    agent = CodeAct(
        llm_proxy=proxy,
        max_num_turns=4,
        tools=[_Ping()],
        bus=EventBus(listeners=[]),
        persistent_state=False,
        max_tool_calls=CAP,
        state_carryover_on_cap=carryover,
        announce_carryover=announce,
        llm_seed=seed,
    )
    await agent.run("count the pings")
    observations = [
        json.loads(m["content"])
        for m in agent.message_history
        if m["role"] == "user" and "observation" in m["content"]
    ]
    return proxy, observations


@pytest.mark.asyncio
async def test_agent_tells_the_policy_nothing_the_reset_cell_would_not():
    _, reset_obs = await _run_agent(carryover=False)
    _, carry_obs = await _run_agent(carryover=True)

    assert len(reset_obs) == len(carry_obs) == 3
    for reset_turn, carry_turn in zip(reset_obs, carry_obs):
        # the two cells differ in what the runtime KEEPS, never in what it announces:
        # same reset banner, same empty active set. (`last_step_globals` reports what the
        # executed turn actually defined, so it does move -- the state really is there.)
        assert carry_turn["runtime_state"]["runtime"] == "reset"
        assert carry_turn["runtime_state"]["active_globals"] == []
        assert (
            carry_turn["runtime_state"]["runtime"]
            == reset_turn["runtime_state"]["runtime"]
        )
        assert (
            carry_turn["runtime_state"]["active_globals"]
            == reset_turn["runtime_state"]["active_globals"]
        )

    # both cells are truncated identically on turn 1
    for obs in (reset_obs[0], carry_obs[0]):
        assert "Tool call limit exceeded" in obs["observation"]["error"]

    # and only the carryover cell can continue from what the truncation left behind
    assert "NameError" in reset_obs[1]["observation"]["error"]
    assert carry_obs[1]["observation"]["error"] is None
    assert carry_obs[1]["observation"]["output"].strip() == str(CAP)


@pytest.mark.asyncio
async def test_decoding_seed_is_passed_through_every_call():
    proxy, _ = await _run_agent(carryover=True, seed=4242)
    assert proxy.seeds and set(proxy.seeds) == {4242}


@pytest.mark.asyncio
async def test_announced_carryover_reports_the_bindings_that_really_survive():
    """The announced variant is the ONE that changes what the policy reads."""
    _, silent = await _run_agent(carryover=True)
    _, announced = await _run_agent(carryover=True, announce=True)

    # turn 0 was truncated, so its bindings are live when the next block is written --
    # every one of them, the loop variable the cap interrupted included
    assert announced[0]["runtime_state"]["active_globals"] == ["acquired", "i"]
    assert silent[0]["runtime_state"]["active_globals"] == []
    # the runtime is still the reset one; only the active set is now truthful
    assert announced[0]["runtime_state"]["runtime"] == "reset"

    # turn 1 ended normally and consumed the checkpoint: nothing survives it
    assert announced[1]["runtime_state"]["active_globals"] == []

    # and announcing changes nothing about what the runtime keeps
    assert announced[1]["observation"]["output"].strip() == str(CAP)
    assert silent[1]["observation"]["output"].strip() == str(CAP)


def test_announcing_without_carrying_is_rejected():
    from codeact_runtime.benchmark.config import AgentConfig

    def build(**overrides):
        # validated from a mapping: pydantic fills the defaults that a direct call would
        # have to spell out
        return AgentConfig.model_validate(
            {
                "name": "x",
                "llm": {"model": "openai/persistent"},
                "max_tool_calls": CAP,
                "persistent_state": False,
                **overrides,
            }
        )

    with pytest.raises(ValueError):
        build(announce_carryover=True)
    # ...and the pair is accepted
    build(announce_carryover=True, state_carryover_on_cap=True)


@pytest.mark.asyncio
async def test_an_unseeded_run_never_hands_a_proxy_the_seed_keyword():
    """Some older callers predate the seed parameter. The agent loop
    catches call errors, so passing an unsupported keyword would not fail loudly -- it
    would burn every turn and report an empty episode."""

    class OldStyleProxy:
        """Deliberately predates `seed`, so it does NOT satisfy the LlmProxy protocol."""

        def __init__(self, replies: list[str]):
            self._inner = _ScriptedProxy(replies)
            self.seeds: list[int | None] = []

        async def complete(self, *, messages, temperature=None, max_tokens=None, extra=None):
            self.seeds.append(None)
            return await self._inner.complete(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            )

    proxy = OldStyleProxy(REPLIES)
    agent = CodeAct(
        llm_proxy=proxy,  # type: ignore[arg-type]  # legacy signature is the point
        max_num_turns=4,
        tools=[_Ping()],
        bus=EventBus(listeners=[]),
        persistent_state=False,
        max_tool_calls=CAP,
        state_carryover_on_cap=True,
    )
    await agent.run("count the pings")
    executed = [
        m for m in agent.message_history if m["role"] == "user" and "observation" in m["content"]
    ]
    assert len(executed) == 3
    assert proxy.seeds == [None, None, None]

