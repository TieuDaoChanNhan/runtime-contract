import hashlib
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generic, List, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from codeact_runtime.codeact.tool import Tool


def opaque_id(rng: random.Random, *, prefix: str = "id", n_hex: int = 12) -> str:
    """Deterministically generate an opaque-ish ID.

    We deliberately avoid sequential integers to better match the "opaque handle"
    motivation in the design draft.
    """
    # Use random bits and hash to make IDs look less structured while remaining deterministic.
    bits = rng.getrandbits(128)
    h = hashlib.sha256(str(bits).encode("utf-8")).hexdigest()[:n_hex]
    return f"{prefix}_{h}"


# Generic Type Variables for the variable parts of the task
T_Public = TypeVar("T_Public")
T_Private = TypeVar("T_Private")
T_Ref = TypeVar("T_Ref")


class TaskData(BaseModel, Generic[T_Public, T_Private, T_Ref]):
    """
    Generic container for any task.
    Pydantic handles the parsing of nested generics automatically.
    """

    family: str
    seed: int
    difficulty: Dict[str, Any]

    public: T_Public
    private: T_Private
    reference: T_Ref

    schema_version: str = "codeact_runtime.task.v1"
    task_id: str | None = None
    run_name: str | None = None
    generated_at: datetime | None = None  # TODO: refactor to avoid Nones everywhere

    # Natural Language wrapper (e.g. pre-generated prompts)
    nl: Any | None = None  # currently unused


@dataclass
class TaskResult:
    """Standardized output for any task execution."""

    is_solved: bool  # Binary success/failure
    score: float  # Continuous score (0.0 to 1.0)
    metrics: dict[str, Any]  # Domain-specific details (e.g. "gap", "probes")


@runtime_checkable
class TaskEnvironment(Protocol):
    """
    Interface for any environment that codeact_runtime agents can interact with.
    """

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> "TaskEnvironment":
        """Factory method to initialize the environment from a task file/dict."""
        ...

    def get_goal_prompt(self) -> str:
        """Returns the initial prompt/goal description to be sent to the Agent."""
        ...

    def get_tools(self) -> list[Tool]:
        """Returns the list of instantiated Tool objects bound to this environment."""
        ...

    def evaluate(self) -> TaskResult:
        """
        Calculate final metrics, comparing agent performance against
        ground truth or optimal solutions.
        """
        ...


@dataclass(frozen=True)
class ToolSpec:
    name: str
    signature: str
    description: str
    args_schema: Dict[str, Any]
    returns_schema: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature,
            "description": self.description,
            "args_schema": self.args_schema,
            "returns_schema": self.returns_schema,
        }


def tool_contract_dict(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
    return [t.as_dict() for t in tools]
