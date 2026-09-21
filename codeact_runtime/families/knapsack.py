import asyncio
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from codeact_runtime.codeact.tool import Tool

from ..config import KnapsackConfig
from .base import TaskData, TaskResult, opaque_id


class KnapsackPublic(BaseModel):
    item_ids: List[str]
    capacity: int
    allowed_classes: List[str]
    inspect_budget: int


class KnapsackItem(BaseModel):
    weight: int
    value: int
    # Alias handles the JSON key "class" mapping to Python attr "cls"
    cls: str = Field(alias="class")


class KnapsackPrivate(BaseModel):
    # Maps item_id -> payload
    items: Dict[str, KnapsackItem]


class KnapsackReference(BaseModel):
    optimal_item_ids: List[str]
    optimal_value: int
    optimal_weight: int


class KnapsackTaskData(TaskData[KnapsackPublic, KnapsackPrivate, KnapsackReference]):
    pass


@dataclass
class KnapsackEnv:
    capacity: int
    allowed_classes: set[str]
    inspect_budget: int
    items: dict[str, KnapsackItem]
    reference: KnapsackReference

    taken: list[str] = field(default_factory=list)
    total_weight: int = 0
    total_value: int = 0
    inspected_count: int = 0
    _inspect_cache: dict[str, str] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @classmethod
    def from_task(cls, task_dict: dict[str, Any]) -> "KnapsackEnv":
        task = KnapsackTaskData.model_validate(task_dict)
        public = task.public

        items: dict[str, KnapsackItem] = task.private.items

        return cls(
            capacity=public.capacity,
            allowed_classes=set(public.allowed_classes),
            inspect_budget=public.inspect_budget,
            items=items,
            reference=task.reference,
        )

    def get_goal_prompt(self) -> str:
        return """
Goal
- Knapsack: select a subset of items to maximize total value, subject to a hard capacity constraint.

What you are given:
- capacity C (maximum total weight): {capacity}
- inspect_budget B — a strict limit on how many DISTINCT items you may inspect: {inspect_budget}
- list of classes that items may belong to: {item_classes}
- The list of candidate items ID's is accessible via the list_items() tool.

Rules
- Do not assume any item properties without inspecting.
- Never take an item unless you have inspected it.
- Never exceed capacity C. Maintain an explicit running total of current_weight in a variable and update it immediately after each take.
- You may stop early if you believe further inspections are not worth the budget.
- Because inspection is budgeted, you should spend inspections to discover high-value allowed-class items. Prefer exploring across the list (e.g., inspect every k-th ID or a small random sample) rather than only the first IDs.
- Finish inspecting items before you start taking.
- Do not call the `finish` tool before taking items.

Strategy & Failure Conditions:
1. Aggressive Exploration: You have a massive budget ({inspect_budget}). You MUST use it. Inspecting only 10-20 items is a FAILURE.
   - Goal: Inspect at least 75% of the items (or until budget runs low) to find the absolute best candidates.
2. Maximize Capacity: Your goal is not just to "take valid items," but to **fill the knapsack**.
   - If you finish with significant empty capacity (e.g., >20% empty) while you still have budget to find more items, you have FAILED the task.

""".format(
            capacity=self.capacity,
            item_classes=set([x.cls for x in self.items.values()]),
            inspect_budget=self.inspect_budget,
        )

    def get_tools(self) -> list[Tool]:
        return [InspectTool(self), TakeTool(self), ListItemsTool(self)]

    def evaluate(self) -> TaskResult:
        # 1. Validation: Did we exceed capacity?
        if self.total_weight > self.capacity:
            return TaskResult(
                is_solved=False,
                score=0.0,
                metrics={
                    "error": "Capacity exceeded",
                    "weight": self.total_weight,
                    "limit": self.capacity,
                },
            )

        # 2. Calculation
        opt_val = self.reference.optimal_value
        agent_val = self.total_value
        gap = max(0, opt_val - agent_val)

        # Avoid division by zero
        score = (
            (agent_val / opt_val) if opt_val > 0 else (1.0 if agent_val == 0 else 0.0)
        )

        # 3. Solved condition: strictly equal to optimal (or better, theoretically impossible)
        is_solved = gap == 0

        return TaskResult(
            is_solved=is_solved,
            score=score,
            metrics={
                "agent_value": agent_val,
                "optimal_value": opt_val,
                "optimality_gap": gap,
                "weight_used": self.total_weight,
                "capacity": self.capacity,
                "items_taken_count": len(self.taken),
                "inspected_count": self.inspected_count,
            },
        )

    async def inspect_item_json(self, item_id: str) -> str:
        async with self._lock:
            if item_id not in self.items:
                raise ValueError(f"Unknown item_id: {item_id}")

            if item_id in self._inspect_cache:
                return self._inspect_cache[item_id]

            if self.inspected_count >= self.inspect_budget:
                raise RuntimeError(
                    f"Inspect budget exceeded (budget={self.inspect_budget}, used={self.inspected_count})"
                )

            it = self.items[item_id]
            payload = {
                "weight": it.weight,
                "value": it.value,
                "class": it.cls,
            }
            out = json.dumps(payload, separators=(",", ":"), sort_keys=True)

            self._inspect_cache[item_id] = out
            self.inspected_count += 1
            return out

    async def take_item(self, item_id: str) -> None:
        async with self._lock:
            if item_id not in self.items:
                raise ValueError(f"Unknown item_id: {item_id}")
            if item_id in self.taken:
                raise ValueError(f"Item already taken: {item_id}")

            it = self.items[item_id]

            if self.allowed_classes and it.cls not in self.allowed_classes:
                raise PermissionError(f"Item class '{it.cls}' not allowed.")

            if self.total_weight + it.weight > self.capacity:
                raise RuntimeError(
                    f"Capacity exceeded: current={self.total_weight}, item={it.weight}, capacity={self.capacity}"
                )

            self.taken.append(item_id)
            self.total_weight += it.weight
            self.total_value += it.value


class InspectTool(Tool):
    name: str = "inspect"
    doc: str = """Return the item's properties as a JSON-encoded string with exactly these keys:
  {"weight": <int>, "value": <int>, "class": <str>}

Example:
  inspect("item_0263322e3f1c") -> '{"class":"A","value":13,"weight":12}'

You MUST parse the returned string using `json.loads(...)` (do not use regex).
Example:
  import json
  d = json.loads(inspect(item_id))
  weight = d["weight"]; value = d["value"]; cls = d["class"]
"""
    arg_doc: dict[str, str] = {"item_id": "the opaque item id to inspect"}

    def __init__(self, env: KnapsackEnv):
        self._env = env
        super().__init__()

    async def run(self, item_id: str) -> str:
        return await self._env.inspect_item_json(item_id)


class TakeTool(Tool):
    name: str = "take_item"
    doc: str = (
        "Select an item for your knapsack. The environment records your selection."
    )
    arg_doc: dict[str, str] = {"item_id": "the item id to take"}

    def __init__(self, env: KnapsackEnv):
        self._env = env
        super().__init__()

    async def run(self, item_id: str) -> None:
        await self._env.take_item(item_id)


class ListItemsTool(Tool):
    name: str = "list_items"
    doc: str = """Returns a list of all item IDs available for inspection.

    Example:
    list_items() -> '["A", "B", "C"]'
        
    Example:
    import json
    d = json.loads(list_items())"""
    arg_doc: dict = {}

    def __init__(self, env: KnapsackEnv):
        self._env = env
        super().__init__()

    async def run(self) -> str:
        # Return the keys as a list formatted as a JSON string
        return json.dumps(list(self._env.items.keys()))


def _solve_01_knapsack(
    items: List[Tuple[str, int, int]], capacity: int
) -> Tuple[int, List[str]]:
    """0/1 knapsack DP.

    Args:
        items: list of (id, weight, value)
        capacity: max weight

    Returns:
        (best_value, chosen_ids)
    """
    n = len(items)
    # dp[i][w] = best value using first i items with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    keep = [[False] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        _id, wt, val = items[i - 1]
        for w in range(capacity + 1):
            best = dp[i - 1][w]
            take_val = -1
            if wt <= w:
                take_val = dp[i - 1][w - wt] + val
            if take_val > best:
                dp[i][w] = take_val
                keep[i][w] = True
            else:
                dp[i][w] = best

    # Reconstruct
    w = capacity
    chosen: List[str] = []
    for i in range(n, 0, -1):
        if keep[i][w]:
            _id, wt, _ = items[i - 1]
            chosen.append(_id)
            w -= wt
    chosen.reverse()
    return dp[n][capacity], chosen


# Assuming other imports (KnapsackConfig, KnapsackTaskData, etc.) are present


def sample_knapsack_instance(seed: int, cfg: KnapsackConfig) -> KnapsackTaskData:
    """Generate a single knapsack instance that requires combinatorial logic.

    Features:
    - Feasibility-Based Budgeting: Ensures statistical solvability.
    - Dominance Check: Rejects 'Golden Ticket' items (no single item > 40% of value).
    - Minimum Complexity: Requires optimal set size >= 3.
    """

    for attempt in range(200):
        rng = random.Random(seed + attempt * 1_000_003)

        # 1. Setup Basic Parameters
        n_items = rng.randint(cfg.n_items_range[0], cfg.n_items_range[1])
        classes = list(cfg.classes)
        allowed = random.sample(classes, k=cfg.num_allowed_classes)

        prob_valid = cfg.num_allowed_classes / len(classes)
        avg_weight = statistics.mean(cfg.weight_range)

        # 2. Generate Items
        items: Dict[str, Dict[str, Any]] = {}
        item_ids: List[str] = []

        for i in range(n_items):
            iid = opaque_id(rng, prefix="item")
            item_ids.append(iid)

            # Force at least one allowed item at index 0
            if i == 0:
                cls = rng.choice(allowed)
            else:
                cls = rng.choice(classes)

            wt = rng.randint(cfg.weight_range[0], cfg.weight_range[1])
            val = rng.randint(cfg.value_range[0], cfg.value_range[1])
            items[iid] = {"weight": wt, "value": val, "class": cls}

        allowed_items = [
            (iid, d["weight"], d["value"])
            for iid, d in items.items()
            if d["class"] in allowed
        ]

        # Need enough raw material to form a complex solution
        if len(allowed_items) < 5:
            continue

        # 3. Set Capacity (Tighter constraints for complexity)
        total_allowed_w = sum(w for _, w, _ in allowed_items)
        ratio = rng.uniform(cfg.capacity_ratio_range[0], cfg.capacity_ratio_range[1])
        capacity = max(1, int(round(total_allowed_w * ratio)))

        min_w = min(w for _, w, _ in allowed_items)
        if capacity < min_w:
            capacity = min_w

        # 4. Solve Optimal
        best_value, chosen = _solve_01_knapsack(allowed_items, capacity)

        # --- NEW CHECKS START ---

        # A. Minimum Complexity: Solution must rely on multiple items
        if len(chosen) < 3:
            continue

        # B. Dominance Check: No single item should be the "hero"
        # We want a team effort, not a superstar + support crew.
        chosen_vals = [items[i]["value"] for i in chosen]
        max_single_val = max(chosen_vals)

        # If one item provides > 40% of the total score, it's too dominant.
        if (max_single_val / best_value) > 0.40:
            continue

        # --- NEW CHECKS END ---

        chosen_w = sum(items[i]["weight"] for i in chosen)

        # 5. Calculate Budget (Feasibility Logic)
        estimated_items_to_fill = capacity / avg_weight
        target_valid_finds = estimated_items_to_fill * 1.5
        stat_budget = target_valid_finds / max(prob_valid, 0.01)

        # B. Ground Truth Floor (The "Edge" Case)
        # We must allow the agent to inspect at least enough items to hold the optimal set.
        # We add a small 10% buffer so they don't have to be psychic.
        deterministic_floor = len(chosen) * 1.1

        # Take the safer of the two
        raw_budget = max(stat_budget, deterministic_floor)

        # C. Final Clamping
        inspect_budget = int(math.ceil(raw_budget))
        inspect_budget = max(5, min(inspect_budget, n_items))

        return KnapsackTaskData(
            family="knapsack",
            seed=seed,
            difficulty={
                "n_items": n_items,
                "optimal_set_size": len(chosen),
                "capacity": capacity,
                "p_valid": round(prob_valid, 2),
                "budget_coverage": round(inspect_budget / n_items, 2),
                "max_item_dominance": round(max_single_val / best_value, 2),
            },
            public=KnapsackPublic(
                item_ids=item_ids,
                capacity=capacity,
                allowed_classes=allowed,
                inspect_budget=inspect_budget,
            ),
            private=KnapsackPrivate(
                items={k: KnapsackItem.model_validate(v) for k, v in items.items()}
            ),
            reference=KnapsackReference(
                optimal_item_ids=chosen,
                optimal_value=best_value,
                optimal_weight=chosen_w,
            ),
        )

    raise RuntimeError(
        f"Failed to sample a balanced knapsack instance (seed={seed}). "
        "Try loosening the dominance check or increasing N."
    )
