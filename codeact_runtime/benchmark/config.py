from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from codeact_runtime.config import (
    LLMConfig,
)
from codeact_runtime.families.knapsack import KnapsackEnv
from codeact_runtime.families.navigation import CEGISNavEnv
from codeact_runtime.families.rule_diagnosis import RuleDiagnosisEnv

FAMILY_ENVS = {
    "knapsack": KnapsackEnv,
    "navigation": CEGISNavEnv,
    "rule_diagnosis": RuleDiagnosisEnv,
}


class AgentConfig(BaseModel):
    name: str
    agent_type: Literal["codeact", "react"] = "codeact"
    llm: LLMConfig
    max_turns: int = Field(12, ge=1)
    timeout_s: float = Field(300.0, ge=1.0)
    persistent_state: bool = True
    max_tool_calls: int | None = None
    # Third runtime condition (cap-boundary carryover): reset every turn EXCEPT after one
    # the tool-call cap truncated, whose globals survive into the next turn only. Requires
    # persistent_state: false -- a persistent runtime keeps state at every boundary anyway.
    state_carryover_on_cap: bool = False
    # Announced variant: the runtime_state banner reports the bindings a carried checkpoint
    # leaves live. The only variant that changes what the policy READS.
    announce_carryover: bool = False

    @model_validator(mode="after")
    def validate_carryover(self) -> "AgentConfig":
        if self.announce_carryover and not self.state_carryover_on_cap:
            raise ValueError(
                "announce_carryover requires state_carryover_on_cap: there would be "
                "nothing surviving to announce."
            )
        if self.state_carryover_on_cap:
            if self.persistent_state:
                raise ValueError(
                    "state_carryover_on_cap requires persistent_state: false."
                )
            if self.max_tool_calls is None:
                raise ValueError(
                    "state_carryover_on_cap needs a per-turn cap: with max_tool_calls "
                    "unset no turn is ever truncated, so the condition is a no-op."
                )
            if self.agent_type != "codeact":
                raise ValueError("state_carryover_on_cap is implemented for codeact.")
        return self


class BenchmarkConfig(BaseModel):
    output_dir: Path
    run_name: str = "codeact"
    seed_start: int = 0
    max_concurrent_runs: int = Field(4, ge=1)
    cache_db_path: Path
    agents: list[AgentConfig]
    task_families: list[str] | None = None

    @field_validator("task_families")
    @classmethod
    def validate_task_families(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("task_families must contain at least one family.")
        invalid = sorted(set(value) - set(FAMILY_ENVS.keys()))
        if invalid:
            raise ValueError(
                "task_families must be one of: "
                + ", ".join(sorted(FAMILY_ENVS.keys()))
                + f". Invalid: {', '.join(invalid)}"
            )
        return value
