# type: ignore

"""CEGIS-style Navigation family with pydantic TaskData integration.

Hidden mechanics:
- Locked directed edges requiring keys.
- Keys discovered only on arrival (or via probe).
- Trap nodes that teleport the agent and invalidate plans.

Execution semantics:
- neighbors(node_id)  FREE   — reveals full outgoing edge list for any node.
- probe(node_id)      BUDGETED — reveals hidden properties (traps, keys, edge locks)
                                 for any node without physically moving there.
- move(dst)           COSTLY  — physical movement; consumes step budget.
- try_path(path)      FREE    — pure path validator; no state change.

The agent can build a complete graph topology cheaply, but must spend its
probe budget strategically to discover which nodes on candidate paths are
traps or key-bearing before committing to physical movement.
"""

import asyncio
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from codeact_runtime.codeact.tool import Tool
from codeact_runtime.config import NavigationConfig
from codeact_runtime.families.base import TaskData, TaskResult


@dataclass(frozen=True)
class CEGISNavInstance:
    seed: int
    n: int
    start: int
    goal: int
    adj: Dict[int, List[int]]
    locked_edges: Dict[Tuple[int, int], str]
    key_nodes: Dict[int, str]
    traps: Dict[int, int]
    probe_budget: int
    # Hop radius for the `key_nearby` hint. Default 1 preserves legacy
    # behavior for tasks generated before this field was added.
    key_hint_radius: int = 1
    # Nodes per neighbors() call; 50 is the batched default every existing task uses.
    neighbors_batch_max: int = 50


class NavigationPublic(BaseModel):
    start: int
    goal: int
    n: int
    max_steps: int
    probe_budget: int
    notes: Optional[str] = None
    key_hint_radius: int = 1
    # How many nodes one neighbors() call may query — the RECONSTRUCTION BANDWIDTH of the
    # free topology channel. Default 50 preserves the batched interface every existing
    # task was generated and evaluated under; setting it to 1 makes rebuilding an n-node
    # map cost n calls instead of ceil(n/50), which is the intervention of Sec. 5.
    neighbors_batch_max: int = 50


class NavigationPrivate(BaseModel):
    adjacency: Dict[int, List[int]]
    locked_edges: List[Dict[str, Any]]
    key_nodes: List[Dict[str, Any]]
    traps: List[Dict[str, Any]]


class NavigationReference(BaseModel):
    shortest_path_len_with_key: int
    one_solution_path_with_key: List[int]


class NavigationTaskData(
    TaskData[NavigationPublic, NavigationPrivate, NavigationReference]
):
    pass


def _bfs_path(
    adj: Dict[int, List[int]],
    src: int,
    dst: int,
    forbidden: Optional[Set[int]] = None,
) -> Optional[List[int]]:
    """Shortest path on directed graph; can exclude forbidden nodes."""
    blocked = forbidden or set()
    if src == dst:
        return [src]
    if src in blocked or dst in blocked:
        return None

    q: deque[int] = deque([src])
    parent: Dict[int, Optional[int]] = {src: None}

    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v in parent:
                continue
            if v in blocked:
                continue
            parent[v] = u
            if v == dst:
                path = [dst]
                cur = dst
                while parent[cur] is not None:
                    cur = parent[cur]  # type: ignore[index]
                    path.append(cur)
                path.reverse()
                return path
            q.append(v)
    return None


def _reachable_nodes(adj: Dict[int, List[int]], src: int) -> Set[int]:
    seen: Set[int] = {src}
    q: deque[int] = deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def _sorted_adj(adj_sets: Dict[int, Set[int]]) -> Dict[int, List[int]]:
    return {u: sorted(vs) for u, vs in adj_sets.items()}


def _permute_instance_labels(
    n: int,
    start: int,
    goal: int,
    adj: Dict[int, List[int]],
    locked_edges: Dict[Tuple[int, int], str],
    key_nodes: Dict[int, str],
    traps: Dict[int, int],
    rng: random.Random,
) -> Tuple[
    int,
    int,
    Dict[int, List[int]],
    Dict[Tuple[int, int], str],
    Dict[int, str],
    Dict[int, int],
]:
    """Apply a deterministic random relabeling of node IDs to reduce structural leakage."""
    labels = list(range(n))
    rng.shuffle(labels)
    perm = {old: labels[old] for old in range(n)}

    adj_p: Dict[int, List[int]] = {
        perm[u]: sorted(perm[v] for v in vs) for u, vs in adj.items()
    }
    locked_p = {(perm[u], perm[v]): lid for (u, v), lid in locked_edges.items()}
    keys_p = {perm[node]: lid for node, lid in key_nodes.items()}
    traps_p = {perm[node]: perm[dst] for node, dst in traps.items()}
    return perm[start], perm[goal], adj_p, locked_p, keys_p, traps_p


def _lock_aware_reachable(inst: CEGISNavInstance, held_keys: Set[str]) -> bool:
    q: deque[int] = deque([inst.start])
    seen: Set[int] = {inst.start}
    while q:
        u = q.popleft()
        if u == inst.goal:
            return True
        for v in inst.adj.get(u, []):
            lock_id = inst.locked_edges.get((u, v))
            if lock_id is not None and lock_id not in held_keys:
                continue
            if v not in seen:
                seen.add(v)
                q.append(v)
    return False


def _audit_instance(inst: CEGISNavInstance) -> None:
    gate_edges = [edge for edge, lid in inst.locked_edges.items() if lid == "L0"]
    if len(gate_edges) != 1:
        raise RuntimeError("Expected exactly one L0 locked gate edge")

    gate_u, gate_v = gate_edges[0]
    if gate_v != inst.goal:
        raise RuntimeError("L0 gate must enter goal")

    l0_nodes = [node for node, lid in inst.key_nodes.items() if lid == "L0"]
    if not l0_nodes:
        raise RuntimeError("L0 key node missing")
    key_node = l0_nodes[0]

    if _bfs_path(inst.adj, inst.start, key_node) is None:
        raise RuntimeError("L0 key must be reachable from start")
    if _bfs_path(inst.adj, inst.start, gate_u) is None:
        raise RuntimeError("Gate must be reachable from start")

    if _lock_aware_reachable(inst, held_keys=set()):
        raise RuntimeError("Goal must be unreachable without L0")
    if not _lock_aware_reachable(inst, held_keys={"L0"}):
        raise RuntimeError("Goal must be reachable with L0")

    forbidden = set(inst.traps.keys()) | {inst.goal}
    if gate_u in forbidden:
        raise RuntimeError("Gate node cannot be a trap")
    if key_node in forbidden:
        raise RuntimeError("L0 key node cannot be a trap")

    if _bfs_path(inst.adj, inst.start, key_node, forbidden=forbidden) is None:
        raise RuntimeError("Key must be reachable without stepping on traps")
    if _bfs_path(inst.adj, key_node, gate_u, forbidden=forbidden) is None:
        raise RuntimeError("Gate must be reachable from key without stepping on traps")


def _check_state_pressure(
    inst: CEGISNavInstance,
    probe_budget: int,
    *,
    reject_singleton_l0_hint: bool,
) -> Optional[str]:
    """Structural acceptance gate beyond `_audit_instance`.

    Returns None when the instance forces the agent to track durable state
    (decoy ambiguity in the `key_nearby` hint, plus a tight enough probe
    budget that probing every candidate is infeasible). Otherwise returns a
    short reason string used by the sampling loop to retry. When the flag is
    False this is a no-op so legacy tiers keep their behavior.
    """
    if not reject_singleton_l0_hint:
        return None

    real_keys = {n for n, k in inst.key_nodes.items() if k.startswith("L")}
    decoys = {n for n, k in inst.key_nodes.items() if k.startswith("D")}
    all_keys = real_keys | decoys

    # `key_nearby` semantics — must mirror `_key_within_hops(..., max_hops=R)`
    # where R is the configured hint radius stored on the instance.
    radius = max(1, inst.key_hint_radius)
    hint_set: Set[int] = set()
    decoys_reachable: Dict[int, Set[int]] = {}  # node -> reachable key targets within R
    real_reachable: Dict[int, Set[int]] = {}
    for u in inst.adj:
        if _key_within_hops(inst.adj, u, all_keys, max_hops=radius):
            hint_set.add(u)
            # Track which keys are reachable for the decoy_only / l0_only audits.
            ds = {k for k in decoys if _key_within_hops(inst.adj, u, {k}, max_hops=radius)}
            ls = {k for k in real_keys if _key_within_hops(inst.adj, u, {k}, max_hops=radius)}
            decoys_reachable[u] = ds
            real_reachable[u] = ls
    if not hint_set:
        return "empty_hint_set"

    decoy_only = [u for u in hint_set if decoys_reachable[u] and not real_reachable[u]]
    if not decoy_only:
        return "no_decoy_only_hint_node"

    l0_only = [u for u in hint_set if real_reachable[u] and not decoys_reachable[u]]
    if l0_only and len(l0_only) == len(hint_set):
        return "all_hint_nodes_localize_l0"

    # Probing every key_nearby node and every interior reference-path node would
    # solve the task without strategic allocation. Require the budget to be
    # strictly smaller so the agent has to *choose* what to probe.
    ref_path = _build_reference_path(inst) or []
    interior = set(ref_path[1:-1])
    full_audit = len(hint_set | interior)
    if probe_budget >= full_audit:
        return "budget_not_tight"

    return None


def generate_instance(
    seed: int,
    n_min: int = 35,
    n_max: int = 65,
    extra_edges_factor: float = 3.0,
    n_locks: int = 1,
    trap_prob: float = 0.3,
    probe_budget: Optional[int] = None,
    decoy_count_range: Tuple[int, int] = (1, 3),
    trap_count_range: Optional[Tuple[int, int]] = None,
    key_hint_radius: int = 1,
) -> CEGISNavInstance:
    if n_min < 3 or n_max < n_min:
        raise ValueError("Require 3 <= n_min <= n_max")
    if extra_edges_factor < 0:
        raise ValueError("extra_edges_factor must be >= 0")
    if n_locks < 1:
        raise ValueError("n_locks must be >= 1")
    if not (0.0 <= trap_prob <= 1.0):
        raise ValueError("trap_prob must be in [0, 1]")
    if decoy_count_range[0] < 1 or decoy_count_range[1] < decoy_count_range[0]:
        raise ValueError("decoy_count_range must be (lo>=1, hi>=lo)")
    if trap_count_range is not None and (
        trap_count_range[0] < 0 or trap_count_range[1] < trap_count_range[0]
    ):
        raise ValueError("trap_count_range must be (lo>=0, hi>=lo)")
    if key_hint_radius < 1 or key_hint_radius > 3:
        raise ValueError("key_hint_radius must be in [1, 3]")

    rng = random.Random(seed)

    n = rng.randint(n_min, n_max)
    start = 0
    goal = n - 1

    adj_sets: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        adj_sets[i].add((i + 1) % n)

    for i in range(n):
        if rng.random() < 0.18:
            adj_sets[i].add((i - 1) % n)

    extra = int(extra_edges_factor * n)
    for _ in range(extra):
        u = rng.randrange(n)
        v = rng.randrange(n - 1)
        if v >= u:
            v += 1
        adj_sets[u].add(v)

    for u in range(n):
        adj_sets[u].discard(goal)

    adj_sets[n - 2].add(0)

    gate = rng.randrange(1, n - 1)
    adj_sets[gate].add(goal)
    locked_edges: Dict[Tuple[int, int], str] = {(gate, goal): "L0"}

    lock_ids = [f"L{i}" for i in range(max(1, n_locks))]
    for lock_id in lock_ids[1:]:
        for _ in range(20):
            u = rng.randrange(n)
            if not adj_sets[u]:
                continue
            v = rng.choice(sorted(adj_sets[u]))
            if (u, v) == (gate, goal) or v == goal:
                continue
            locked_edges[(u, v)] = lock_id
            break

    adj = _sorted_adj(adj_sets)
    reachable = sorted(_reachable_nodes(adj, start))

    # 1. Get the natural direct path from start to gate
    key_candidates = [x for x in reachable if x not in {start, goal, gate}]

    if not key_candidates:
        raise RuntimeError("No reachable key candidate")

    key_nodes: Dict[int, str] = {rng.choice(key_candidates): "L0"}
    key_node = next(iter(key_nodes.keys()))

    used_nodes = set(key_nodes)
    for lock_id in lock_ids[1:]:
        extras = [x for x in reachable if x not in used_nodes and x != goal]
        if not extras:
            break
        node = rng.choice(extras)
        key_nodes[node] = lock_id
        used_nodes.add(node)

    decoy_count = rng.randint(decoy_count_range[0], decoy_count_range[1])
    decoy_candidates = [
        x for x in reachable if x not in used_nodes and x not in {start, goal, gate}
    ]
    rng.shuffle(decoy_candidates)

    # Bias one decoy into L0's `key_nearby` footprint so probing a `key_nearby`
    # hit is no longer a guaranteed L0 locator. l0_hint_set is the set of nodes
    # whose 1-hop adjacency contains the L0 key node — same definition as
    # `_key_within_hops(..., max_hops=1)`. A decoy "near L0's hint footprint"
    # is one that some hint node also sees in its 1-hop, i.e. there is a node
    # H with both L0 and the decoy in adj[H]. When such a placement exists,
    # use it for the first decoy; remaining decoys go anywhere.
    l0_hint_set = {u for u in adj if key_node in adj.get(u, [])}
    near_l0_pool = [
        x for x in decoy_candidates
        if any(x in adj.get(h, []) for h in l0_hint_set)
    ]
    placement: List[int] = []
    if near_l0_pool and decoy_count > 0:
        first = near_l0_pool[0]
        placement.append(first)
        for x in decoy_candidates:
            if x == first:
                continue
            if len(placement) >= decoy_count:
                break
            placement.append(x)
    else:
        placement = decoy_candidates[:decoy_count]
    for i, node in enumerate(placement):
        key_nodes[node] = f"D{i}"
        used_nodes.add(node)

    traps: Dict[int, int] = {}
    forbidden_critical = {start, goal, gate, *key_nodes.keys()}
    path_to_gate = _bfs_path(adj, start, gate) or []
    candidates = [x for x in path_to_gate[1:] if x not in forbidden_critical]
    rng.shuffle(candidates)

    if trap_count_range is not None:
        # Multi-trap mode: place up to k traps that don't block the reference
        # path. Each trap must individually preserve trap-free reachability of
        # the key and the gate. Best-effort: if the candidate pool is exhausted
        # before lo traps are placed, return what we got (the sampling loop
        # will surface this via state_pressure_warning when applicable).
        target_k = rng.randint(trap_count_range[0], trap_count_range[1])
        for trap_node in candidates:
            if len(traps) >= target_k:
                break
            forbidden = set(traps.keys()) | {trap_node, goal}
            if _bfs_path(adj, start, key_node, forbidden=forbidden) is None:
                continue
            if _bfs_path(adj, key_node, gate, forbidden=forbidden) is None:
                continue
            traps[trap_node] = start
    else:
        # Legacy probabilistic single-trap path. Preserved so easy/hard tiers
        # configured without `trap_count_range` behave exactly as before.
        if rng.random() < trap_prob:
            for trap_node in candidates[:10]:
                forbidden = {trap_node, goal}
                if _bfs_path(adj, start, key_node, forbidden=forbidden) is None:
                    continue
                if _bfs_path(adj, key_node, gate, forbidden=forbidden) is None:
                    continue
                traps[trap_node] = start
                break

    start_p, goal_p, adj_p, locked_p, key_nodes_p, traps_p = _permute_instance_labels(
        n=n,
        start=start,
        goal=goal,
        adj=adj,
        locked_edges=locked_edges,
        key_nodes=key_nodes,
        traps=traps,
        rng=rng,
    )

    # Probe budget: deferred; caller injects after computing effective_horizon.
    # If not provided, fall back to a reasonable default.
    resolved_probe_budget = probe_budget if probe_budget is not None else max(5, n // 6)

    inst = CEGISNavInstance(
        seed=seed,
        n=n,
        start=start_p,
        goal=goal_p,
        adj=adj_p,
        locked_edges=locked_p,
        key_nodes=key_nodes_p,
        traps=traps_p,
        probe_budget=resolved_probe_budget,
        key_hint_radius=key_hint_radius,
    )
    _audit_instance(inst)
    return inst


def public_spec(instance: CEGISNavInstance) -> Dict[str, Any]:
    return {
        "seed": instance.seed,
        "n": instance.n,
        "start": instance.start,
        "goal": instance.goal,
        "probe_budget": instance.probe_budget,
    }


def private_spec(instance: CEGISNavInstance) -> Dict[str, Any]:
    return {
        **public_spec(instance),
        "adjacency": {u: list(vs) for u, vs in sorted(instance.adj.items())},
        "locked_edges": [
            {"from": u, "to": v, "lock_id": lock_id}
            for (u, v), lock_id in sorted(instance.locked_edges.items())
        ],
        "key_nodes": [
            {"node": node, "lock_id": lock_id}
            for node, lock_id in sorted(instance.key_nodes.items())
        ],
        "traps": [
            {"node": node, "teleport_to": dst}
            for node, dst in sorted(instance.traps.items())
        ],
    }


def render_task_text(pub: Dict[str, Any]) -> str:
    return (
        "Navigate from start to goal in a directed graph. Outgoing neighbors are freely "
        "queryable for any node, but locks, keys, and traps are hidden. Use the budgeted "
        "probe() tool to inspect node properties before committing to physical movement. "
        f"Start={pub['start']}, Goal={pub['goal']}, Nodes={pub['n']}, "
        f"Probe budget={pub['probe_budget']}."
    )


@dataclass
class CEGISNavEnv:
    instance: CEGISNavInstance
    max_steps: int

    current: int = field(init=False)
    keys: Set[str] = field(default_factory=set)
    collected_key_nodes: Set[int] = field(default_factory=set)
    probed_nodes: Set[int] = field(default_factory=set)
    probes_used: int = 0
    last_event: Optional[Dict[str, Any]] = None
    steps_taken: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self.current = self.instance.start

    @classmethod
    def from_task(cls, task_dict: dict[str, Any]) -> "CEGISNavEnv":
        task = NavigationTaskData.model_validate(task_dict)
        instance = CEGISNavInstance(
            seed=task.seed,
            n=task.public.n,
            start=task.public.start,
            goal=task.public.goal,
            adj={int(k): list(v) for k, v in task.private.adjacency.items()},
            locked_edges={
                (int(row["from"]), int(row["to"])): str(row["lock_id"])
                for row in task.private.locked_edges
            },
            key_nodes={
                int(row["node"]): str(row["lock_id"]) for row in task.private.key_nodes
            },
            traps={
                int(row["node"]): int(row["teleport_to"]) for row in task.private.traps
            },
            probe_budget=task.public.probe_budget,
            key_hint_radius=task.public.key_hint_radius,
            neighbors_batch_max=task.public.neighbors_batch_max,
        )
        return cls(instance=instance, max_steps=task.public.max_steps)

    def get_goal_prompt(self) -> str:
        return """
Goal
- Reach the goal node in a directed graph containing hidden locks, keys, and traps.

What you are given:
- Start node: {start}
- Goal node: {goal}
- Total nodes in graph: {n}
- max_steps: strict limit on physical moves: {max_steps}
- probe_budget: number of probe() calls available: {probe_budget}
- Available tools: neighbors(nodes), probe(node_id), move(dst), try_path(path), status(), at_goal()

Tools
- neighbors(nodes)    — FREE, no budget cost. Returns outgoing edges for {batch_phrase}.
                        {batch_advice}
- probe(node_id)      — BUDGETED (costs 1 probe). Reveals hidden properties of a node:
                          * whether it is a trap and where it teleports to
                          * whether it holds a key and which key
                          * which outgoing edges from that node are locked and which lock they require
                        Probe nodes on your candidate paths before physically moving through them.
                        You have {probe_budget} probes total — use them strategically.
- move(dst)           — COSTLY (costs 1 step). Physical movement. Consumes your step budget.
- try_path(path)      — FREE. Validates a path against known state (keys currently held, known edges).
                        Does NOT reveal unknown locks or traps on un-probed nodes.
- status()            — Returns current node, keys held, probes used, steps taken.
- at_goal()           — Returns True if you have reached the goal.

Strategy
1. Build the full graph: {batch_strategy} to get all outgoing edges. This is free.
   neighbors() returns a dict keyed by node ID: {{node_id: {{"neighbors": [...], "key_nearby": bool}}}}
   key_nearby=True means a key node (real or decoy) is reachable within {key_hint_radius} hop(s) from this node.
   Example: result = neighbors(nodes={batch_example}); adj[nid] = result[nid]["neighbors"]
2. Find a candidate path to goal using BFS on the adjacency you just built.
3. BEFORE probing the path, probe all key_nearby=True nodes first.
   At hint radius {key_hint_radius}, key_nearby is True for nodes whose {key_hint_radius}-hop neighborhood contains a key.
   Probe each directly — if has_key=True, record the node and its key_id.
   This ensures you know the key location even if path-probing consumes the rest of your budget.
   At larger hint radius the key_nearby set is bigger; prioritize nodes nearest to your candidate path.
4. Probe nodes on the BFS path to discover traps and locked edges (with remaining probe budget).
   - Check probe result field 'locked_outgoing' (NOT 'locked_edges') for locked edges FROM that node.
   - If locked_outgoing contains an entry, note its lock_id (e.g. "L0") and the 'to' node.
   - If is_trap is True, replan around it. Stop probing once budget is nearly exhausted.
5. Identify which key you need: the lock's lock_id (e.g. "L0"). Keys starting with "D" are decoys.
   You already probed the key_nearby nodes in step 3 — check those results for the matching key.
   If the key_id was not found in key_nearby nodes, probe their outgoing neighbors with any budget left.
6. Execute: move to the key node (collecting the key), then move to the gate, then to the goal.
   If you haven't probed the full path, use try_path() to check for known obstacles before moving.
7. If probes run out before you have a fully safe plan, attempt the best available path rather
   than stalling — a risky move is better than taking no action.

Rules
- Never call move() to blindly explore. Every step counts.
- Always check the result dict returned by move() — do not assume success.
- try_path() will not warn you about locks or traps on nodes you have not probed.
- Never wait for user input or ask for permission to act — make the best decision you can and execute it.
""".strip().format(
            start=self.instance.start,
            goal=self.instance.goal,
            n=self.instance.n,
            max_steps=self.max_steps,
            probe_budget=self.instance.probe_budget,
            key_hint_radius=self.instance.key_hint_radius,
            **self._batch_prompt_fragments(),
        )

    def _batch_prompt_fragments(self) -> Dict[str, str]:
        """Prompt wording for the configured reconstruction bandwidth. The batched text is
        the wording every existing evaluation used; the unbatched text must not merely drop
        the batch advice but state the one-node-per-call rule, or the agent spends its first
        turns discovering the limit through errors."""
        limit = self.instance.neighbors_batch_max
        if limit == 1:
            return {
                "batch_phrase": "ONE node per call (pass a single-element list)",
                "batch_advice": (
                    "Each node costs a separate call, so the map is expensive to rebuild:\n"
                    "                        query only the nodes you actually need."
                ),
                "batch_strategy": "call neighbors() once per node",
                "batch_example": "[0]",
            }
        return {
            "batch_phrase": f"a list of nodes (up to {limit} at once)",
            "batch_advice": (
                "Call this in batches to efficiently build a complete graph map "
                "before doing anything else."
            ),
            "batch_strategy": "call neighbors() in batches",
            "batch_example": "[0,1,2]",
        }

    def get_tools(self) -> List[Tool]:
        return [
            NeighborsTool(self),
            ProbeTool(self),
            MoveTool(self),
            TryPathTool(self),
            StatusTool(self),
            AtGoalTool(self),
        ]

    def reset(self) -> None:
        self.current = self.instance.start
        self.keys.clear()
        self.collected_key_nodes.clear()
        self.probed_nodes.clear()
        self.probes_used = 0
        self.last_event = None
        self.steps_taken = 0

    def evaluate(self) -> TaskResult:
        solved = self.at_goal()
        if solved:
            score = 1.0
        else:
            inst = self.instance
            traps = set(inst.traps.keys())
            real_key_node = next(
                (n for n, k in inst.key_nodes.items() if k.startswith("L")), None
            )
            has_real_key = any(k.startswith("L") for k in self.keys)

            if has_real_key or real_key_node is None:
                # Phase 2: measure progress from key_node toward goal.
                d_now = _nav_bfs_dist(
                    inst.adj, inst.locked_edges, self.keys, traps,
                    self.current, inst.goal,
                )
                ref_src = real_key_node if real_key_node is not None else self.current
                d_ref = _nav_bfs_dist(
                    inst.adj, inst.locked_edges, self.keys, traps,
                    ref_src, inst.goal,
                )
                phase2 = (
                    max(0.0, 1.0 - d_now / d_ref)
                    if 0 < d_ref < float("inf") else 0.0
                )
                score = 0.5 + 0.5 * phase2
            else:
                # Phase 1: measure progress from start toward key.
                d_now = _nav_bfs_dist(
                    inst.adj, inst.locked_edges, self.keys, traps,
                    self.current, real_key_node,
                )
                d_ref = _nav_bfs_dist(
                    inst.adj, inst.locked_edges, self.keys, traps,
                    inst.start, real_key_node,
                )
                phase1 = (
                    max(0.0, 1.0 - d_now / d_ref)
                    if 0 < d_ref < float("inf") else 0.0
                )
                score = 0.5 * phase1

        return TaskResult(
            is_solved=solved,
            score=score,
            metrics={
                "current": self.current,
                "goal": self.instance.goal,
                "keys": sorted(self.keys),
                "steps_taken": self.steps_taken,
                "max_steps": self.max_steps,
                "probes_used": self.probes_used,
                "probe_budget": self.instance.probe_budget,
            },
        )

    def _remaining_key_nodes(self) -> Dict[int, str]:
        return {
            node: key_id
            for node, key_id in self.instance.key_nodes.items()
            if node not in self.collected_key_nodes
        }

    def neighbors(self, nodes: Optional[List[int]] = None) -> Dict[int, Dict[str, Any]]:
        targets = [self.current] if nodes is None else nodes
        limit = self.instance.neighbors_batch_max
        if len(targets) > limit:
            raise ValueError(
                f"neighbors() supports at most {limit} "
                f"{'node' if limit == 1 else 'nodes'} per call"
            )

        remaining_keys = self._remaining_key_nodes()
        result = {}
        for t in targets:
            nbrs = list(self.instance.adj.get(t, []))
            result[t] = {
                "neighbors": nbrs,
                "hints": {
                    nbr: {
                        "key_nearby": _key_within_hops(
                            self.instance.adj, nbr, remaining_keys,
                            max_hops=self.instance.key_hint_radius,
                        )
                    }
                    for nbr in nbrs
                },
            }
        return result
    
    def probe(self, node_id: int) -> dict[str, Any]:
        """
        Reveal hidden properties of node_id without physically moving there.
        Costs 1 unit of probe_budget. Repeated probes of the same node still
        cost budget (cache externally if needed).

        Returns:
          - node: the queried node id
          - is_trap: bool
          - teleport_to: int | None  (destination if trap)
          - has_key: bool
          - key_id: str | None
          - locked_outgoing: list[{"to": int, "lock_id": str}]
              locked edges whose SOURCE is node_id
          - probes_remaining: int
          - ok: False with reason "budget_exceeded" if budget is gone
        """
        if self.probes_used >= self.instance.probe_budget:
            return {
                "ok": False,
                "reason": "budget_exceeded",
                "probes_used": self.probes_used,
                "probe_budget": self.instance.probe_budget,
            }

        if node_id < 0 or node_id >= self.instance.n:
            return {
                "ok": False,
                "reason": "invalid_node",
                "node": node_id,
                "probes_remaining": self.instance.probe_budget - self.probes_used,
            }

        self.probes_used += 1
        self.probed_nodes.add(node_id)

        is_trap = node_id in self.instance.traps
        teleport_to = self.instance.traps.get(node_id)
        remaining_keys = self._remaining_key_nodes()
        has_key = node_id in remaining_keys
        key_id = remaining_keys.get(node_id)

        locked_outgoing = [
            {"to": v, "lock_id": lid}
            for (u, v), lid in self.instance.locked_edges.items()
            if u == node_id
        ]

        locked_outgoing.sort(key=lambda r: (r["to"], r["lock_id"]))

        return {
            "ok": True,
            "node": node_id,
            "is_trap": is_trap,
            "teleport_to": teleport_to,
            "has_key": has_key,
            "key_id": key_id,
            "locked_outgoing": locked_outgoing,
            "probes_remaining": self.instance.probe_budget - self.probes_used,
        }

    def status(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "keys": sorted(self.keys),
            "last_event": self.last_event,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "probes_used": self.probes_used,
            "probe_budget": self.instance.probe_budget,
            "probes_remaining": self.instance.probe_budget - self.probes_used,
        }

    def at_goal(self) -> bool:
        return self.current == self.instance.goal

    def move(self, dst: int) -> dict[str, bool | str | int]:
        if self.steps_taken >= self.max_steps:
            self.last_event = {
                "ok": False,
                "reason": "max_steps_exceeded",
                "steps_taken": self.steps_taken,
                "max_steps": self.max_steps,
            }
            return self.last_event

        u = self.current
        if dst not in self.instance.adj.get(u, []):
            self.steps_taken += 1
            self.last_event = {"ok": False, "reason": "no_edge", "from": u, "to": dst}
            return self.last_event

        lock_id = self.instance.locked_edges.get((u, dst))
        if lock_id is not None and lock_id not in self.keys:
            self.steps_taken += 1
            self.last_event = {
                "ok": False,
                "reason": "locked",
                "lock_id": lock_id,
                "from": u,
                "to": dst,
            }
            return self.last_event

        self.steps_taken += 1
        self.current = dst

        if dst in self.instance.traps:
            tele = self.instance.traps[dst]
            self.current = tele
            self.last_event = {
                "ok": False,
                "reason": "trap",
                "trap_node": dst,
                "teleport_to": tele,
            }
            return self.last_event

        if dst in self.instance.key_nodes and dst not in self.collected_key_nodes:
            key_id = self.instance.key_nodes[dst]
            self.keys.add(key_id)
            self.collected_key_nodes.add(dst)
            self.last_event = {
                "ok": True,
                "moved_to": dst,
                "event": "found_key",
                "key": key_id,
                "node": dst,
            }
            return self.last_event

        self.last_event = {"ok": True, "moved_to": dst}
        return self.last_event

    def try_path(self, path: List[int]) -> dict[str, bool | str | int | None]:
        """
        Validate a path against currently known state.
        NOTE: try_path only reflects locks and traps on nodes that have been
        probed. Un-probed nodes are treated as safe/unlocked by this checker.
        Probe first if you want reliable predictions.
        """
        if not path or path[0] != self.current:
            return {
                "ok": False,
                "reason": "bad_path_start",
                "expected_start": self.current,
                "given_start": path[0] if path else None,
            }

        held = set(self.keys)
        remaining_keys = self._remaining_key_nodes()
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            if v not in self.instance.adj.get(u, []):
                return {"ok": False, "step": i, "reason": "no_edge", "from": u, "to": v}

            # Only surface lock/trap info for probed nodes.
            if u in self.probed_nodes:
                lock_id = self.instance.locked_edges.get((u, v))
                if lock_id is not None and lock_id not in held:
                    return {
                        "ok": False,
                        "step": i,
                        "reason": "locked",
                        "lock_id": lock_id,
                        "from": u,
                        "to": v,
                    }

            if v in self.probed_nodes and v in self.instance.traps:
                return {
                    "ok": False,
                    "step": i,
                    "reason": "trap",
                    "trap_node": v,
                    "teleport_to": self.instance.traps[v],
                }

            if v in self.probed_nodes:
                key_id = remaining_keys.get(v)
                if key_id is not None:
                    held.add(key_id)

        return {
            "ok": True,
            "final": path[-1],
            "would_reach_goal": path[-1] == self.instance.goal,
        }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class NeighborsTool(Tool):
    name: str = "neighbors"
    # `doc` is overridden per-instance in __init__ to reflect the configured
    # key_hint_radius on the bound env. The class-level default assumes the
    # legacy 1-hop hint; instances on tiers with radius > 1 update it below.
    doc: str = (
        "Return outgoing neighbors for one or more nodes. FREE — no budget cost. "
        "Pass a list of node IDs to inspect them remotely without moving (max 50 per call). "
        "Omit nodes parameter to query the current node.\n\n"
        "Returns a dict keyed by node ID. Each value has:\n"
        "  - 'neighbors': list of outgoing neighbor node IDs\n"
        "  - 'key_nearby': bool — True if a key node is reachable within 1 hop of this node\n\n"
        "Example usage:\n"
        "  result = neighbors(nodes=[5, 12, 7])\n"
        "  for node_id, data in result.items():\n"
        "      print(node_id, data['neighbors'], data['key_nearby'])\n\n"
        "key_nearby=True means a key node (real or decoy) is reachable within the configured radius — "
        "probe those neighbors to find it. Does NOT reveal which key is the real goal key."
    )

    arg_doc = {"nodes": "A list of node IDs to query for neighbors."}

    def __init__(self, env: CEGISNavEnv):
        self._env = env
        super().__init__()
        radius = self._env.instance.key_hint_radius
        hop_phrase = (
            "within 1 hop" if radius == 1
            else f"within {radius} hops (i.e., reachable through up to {radius} outgoing edges)"
        )
        # The batch limit is part of the tool contract, so it belongs in the signature the
        # agent reads -- not only in the error it gets after exceeding it.
        limit = self._env.instance.neighbors_batch_max
        if limit == 1:
            batch_phrase = (
                "Pass a single-element list to inspect ONE node remotely without moving; "
                "this tool accepts only one node per call. "
            )
            example_nodes = "[5]"
        else:
            batch_phrase = (
                "Pass a list of node IDs to inspect them remotely without moving "
                f"(max {limit} per call). "
            )
            example_nodes = "[5, 12, 7]"
        self.doc = (
            "Return outgoing neighbors for one or more nodes. FREE — no budget cost. "
            + batch_phrase
            + "Omit nodes parameter to query the current node.\n\n"
            "Returns a dict keyed by node ID. Each value has:\n"
            "  - 'neighbors': list of outgoing neighbor node IDs\n"
            f"  - 'key_nearby': bool — True if a key node is reachable {hop_phrase} from this node\n\n"
            "Example usage:\n"
            f"  result = neighbors(nodes={example_nodes})\n"
            "  for node_id, data in result.items():\n"
            "      print(node_id, data['neighbors'], data['key_nearby'])\n\n"
            f"key_nearby=True means a key node (real or decoy) is reachable {hop_phrase} — "
            "probe those neighbors to find it. Does NOT reveal which key is the real goal key."
        )

    async def run(
        self, nodes: list[int] | None = None
    ) -> dict:
        async with self._env._lock:
            raw = self._env.neighbors(nodes)
            remaining_keys = self._env._remaining_key_nodes()
            return {
                node_id: {
                    "neighbors": data["neighbors"],
                    "key_nearby": _key_within_hops(
                        self._env.instance.adj, node_id, remaining_keys,
                        max_hops=self._env.instance.key_hint_radius,
                    ),
                }
                for node_id, data in raw.items()
            }


class ProbeTool(Tool):
    name: str = "probe"
    doc: str = (
        "Reveal hidden properties of a node without physically moving there. "
        "Costs 1 probe from your probe_budget.\n\n"
        "Returns a dict with these fields:\n"
        "  - 'is_trap' (bool): True if stepping on this node triggers a teleport\n"
        "  - 'has_key' (bool): True if this node holds a key\n"
        "  - 'key_id' (str|None): The key's ID (e.g. 'L0', 'D0') if has_key is True\n"
        "  - 'locked_outgoing' (list): Locked edges FROM this node, each as {'to': int, 'lock_id': str}\n"
        "  - 'probes_remaining' (int): Budget left after this call\n\n"
        "IMPORTANT: The field is 'locked_outgoing', NOT 'locked_edges'. Check locked_outgoing to "
        "discover which outgoing edges require a key."
    )
    arg_doc: dict[str, str] = {"node_id": "The node ID to probe."}

    def __init__(self, env: CEGISNavEnv):
        self._env = env
        super().__init__()

    async def run(self, node_id: int) -> dict[str, Any]:
        async with self._env._lock:
            return self._env.probe(node_id)


class MoveTool(Tool):
    name: str = "move"
    doc: str = (
        "Attempt a stateful move from current node to dst.\n"
        "Returns a dict with keys:\n"
        "- 'ok' (bool): True if move succeeded, False otherwise.\n"
        "- 'moved_to' (int): The node you arrived at (if successful).\n"
        "- 'reason' (str): Explanation if the move failed (e.g., 'locked', 'no_edge', 'trap')."
    )
    arg_doc: dict[str, str] = {"dst": "destination node id"}

    def __init__(self, env: CEGISNavEnv):
        self._env = env
        super().__init__()

    async def run(self, dst: int) -> dict[str, bool | str | int]:
        async with self._env._lock:
            return self._env.move(dst)


class TryPathTool(Tool):
    name: str = "try_path"
    doc: str = (
        "Validate a path against currently known state (probed nodes only).\n"
        "IMPORTANT: Locks and traps on un-probed nodes are NOT detected by this tool.\n"
        "Probe candidate path nodes first for reliable validation.\n"
        "Returns a dict with keys:\n"
        "- 'ok' (bool): True if path appears traversable given known state.\n"
        "- 'reason' (str): Why it failed (if ok is False).\n"
        "- 'final' (int): The last node in the path."
    )
    arg_doc: dict[str, str] = {"path": "path list where path[0] equals current node"}

    def __init__(self, env: CEGISNavEnv):
        self._env = env
        super().__init__()

    async def run(self, path: list[int]) -> dict[str, bool | str | int | None]:
        async with self._env._lock:
            return self._env.try_path(path)


class StatusTool(Tool):
    name: str = "status"
    doc: str = (
        "Return current node, keys held, probe usage, and step usage.\n"
        "Returns a dict with keys:\n"
        "- 'current' (int): The node you are currently on.\n"
        "- 'keys' (list[str]): Keys you are holding.\n"
        "- 'steps_taken' (int): Physical moves used so far.\n"
        "- 'max_steps' (int): Maximum allowed physical moves.\n"
        "- 'probes_used' (int): Probes consumed so far.\n"
        "- 'probe_budget' (int): Total probe budget.\n"
        "- 'probes_remaining' (int): Probes still available."
    )
    arg_doc: dict[str, str] = {}

    def __init__(self, env: CEGISNavEnv):
        self._env = env
        super().__init__()

    async def run(self) -> dict[str, Any]:
        async with self._env._lock:
            return self._env.status()


class AtGoalTool(Tool):
    name: str = "at_goal"
    doc: str = "Return whether current node equals goal."
    arg_doc: dict[str, str] = {}

    def __init__(self, env: CEGISNavEnv):
        self._env = env
        super().__init__()

    async def run(self) -> bool:
        async with self._env._lock:
            return self._env.at_goal()


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _nav_bfs_dist(
    adj: Dict[int, List[int]],
    locked_edges: Dict[Tuple[int, int], str],
    held_keys: set,
    traps: set,
    src: int,
    dst: int,
) -> float:
    """BFS shortest distance src→dst: lock-aware and trap-avoiding.

    Locked edges whose lock_id is not in held_keys are impassable.
    Trap nodes are forbidden as intermediate steps (but src and dst are allowed).
    Returns float('inf') if dst is unreachable.
    """
    if src == dst:
        return 0
    forbidden = traps - {src, dst}
    from collections import deque
    q: deque = deque([(src, 0)])
    visited = {src}
    while q:
        u, d = q.popleft()
        for v in adj.get(u, []):
            if v in visited or v in forbidden:
                continue
            lid = locked_edges.get((u, v))
            if lid is not None and lid not in held_keys:
                continue
            if v == dst:
                return d + 1
            visited.add(v)
            q.append((v, d + 1))
    return float("inf")


# ---------------------------------------------------------------------------
# Instance generation helpers
# ---------------------------------------------------------------------------


def generate_instances(
    count: int, seed0: int = 0, **kwargs: Any
) -> List[CEGISNavInstance]:
    return [generate_instance(seed0 + i, **kwargs) for i in range(count)]


def _build_reference_path(inst: CEGISNavInstance) -> Optional[list[int]]:
    gate_u = next(
        u for (u, v), lid in inst.locked_edges.items() if lid == "L0" and v == inst.goal
    )
    key_node = next(node for node, lid in inst.key_nodes.items() if lid == "L0")

    forbidden = set(inst.traps.keys()) | {inst.goal}
    to_key = _bfs_path(inst.adj, inst.start, key_node, forbidden=forbidden)
    to_gate = _bfs_path(inst.adj, key_node, gate_u, forbidden=forbidden)
    if to_key is None or to_gate is None:
        return None
    return to_key + to_gate[1:] + [inst.goal]

def _key_within_hops(
    adj: Dict[int, List[int]],
    start: int,
    key_nodes: Dict[int, str],
    max_hops: int = 1,
) -> bool:
    """BFS check: is any key node reachable from start within max_hops steps?"""
    if start in key_nodes:
        return True
    visited = {start}
    frontier = [start]
    for _ in range(max_hops):
        next_frontier = []
        for u in frontier:
            for v in adj.get(u, []):
                if v in key_nodes:
                    return True
                if v not in visited:
                    visited.add(v)
                    next_frontier.append(v)
        frontier = next_frontier
    return False


def _compute_probe_budget(
    effective_horizon: int,
    n: int,
    n_traps: int,
    cap_fraction: float = 0.3,
) -> int:
    # Uncapped baseline: probing roughly the optimal start→L0→gate corridor.
    # 1.5x gives budget for the nominal path plus key-search probes.
    # At 1.0x the budget was fully consumed by path probing, leaving nothing
    # to probe key candidates after discovering a locked edge.
    base = int(math.ceil(1.5 * max(1, effective_horizon)))
    buffer = 2 + n_traps  # 1 spare per trap + 2 general slack
    budget = base + buffer

    # Cap at `cap_fraction * n` of nodes — prevents exhaustive graph-wide probing.
    # Tighter values (e.g. 0.25) force strategic allocation on medium tier.
    hard_cap = int(math.ceil(cap_fraction * n))
    budget = min(budget, hard_cap)

    return max(5, budget)


def sample_navigation_instance(seed: int, cfg: NavigationConfig) -> NavigationTaskData:
    """Family factory entrypoint returning pydantic TaskData."""
    rng = random.Random(seed)
    target_horizon = rng.randint(cfg.horizon_range[0], cfg.horizon_range[1])

    n_min = max(35, cfg.horizon_range[0] + cfg.extra_nodes_range[0] + 20)
    n_max = max(n_min, cfg.horizon_range[1] + cfg.extra_nodes_range[1] + 30)
    extra_factor = max(0.5, cfg.extra_edge_factor_range[1] * 1.2)

    # Three nested fallbacks tracked in order of preference:
    #   best_full     — passes horizon + structural state-pressure gate
    #   best_horizon  — passes horizon only (used when state-pressure never satisfied)
    #   best_any      — closest by horizon-gap (used when nothing lands in horizon band)
    best_full: Optional[Tuple[CEGISNavInstance, List[int], int, int]] = None
    best_horizon: Optional[Tuple[CEGISNavInstance, List[int], int, int]] = None
    best_any: Optional[Tuple[CEGISNavInstance, List[int], int, int, int]] = None
    last_pressure_reason: Optional[str] = None

    for attempt in range(500):
        candidate_seed = seed + attempt * 1_000_003
        inst = generate_instance(
            seed=candidate_seed,
            n_min=n_min,
            n_max=n_max,
            extra_edges_factor=extra_factor,
            n_locks=1,
            trap_prob=cfg.trap_prob,
            decoy_count_range=cfg.decoy_count_range,
            trap_count_range=cfg.trap_count_range,
            key_hint_radius=cfg.key_hint_radius,
            # probe_budget injected below after path is known
        )
        path = _build_reference_path(inst)
        if path is None:
            continue

        effective_horizon = max(0, len(path) - 1)
        gap = abs(effective_horizon - target_horizon)

        candidate_budget = _compute_probe_budget(
            effective_horizon=effective_horizon,
            n=inst.n,
            n_traps=len(inst.traps),
            cap_fraction=cfg.probe_cap_fraction,
        )
        candidate_budget = max(5, round(candidate_budget * cfg.probe_budget_multiplier))

        if best_any is None or gap < best_any[4]:
            best_any = (inst, path, candidate_budget, candidate_seed, gap)

        in_horizon = cfg.horizon_range[0] <= effective_horizon <= cfg.horizon_range[1]
        if not in_horizon:
            continue

        if best_horizon is None:
            best_horizon = (inst, path, candidate_budget, candidate_seed)

        reason = _check_state_pressure(
            inst,
            candidate_budget,
            reject_singleton_l0_hint=cfg.reject_singleton_l0_hint,
        )
        # If a trap count range was requested but graceful fallback under-placed,
        # treat that as a state-pressure miss so the loop tries another seed.
        if reason is None and cfg.trap_count_range is not None:
            t_lo, _ = cfg.trap_count_range
            if len(inst.traps) < t_lo:
                reason = "trap_shortfall"
        if reason is None:
            best_full = (inst, path, candidate_budget, candidate_seed)
            break
        last_pressure_reason = reason

    state_pressure_warning: Optional[str] = None
    if best_full is not None:
        inst, path, probe_budget, best_instance_seed = best_full
    elif best_horizon is not None:
        inst, path, probe_budget, best_instance_seed = best_horizon
        state_pressure_warning = last_pressure_reason or "state_pressure_unmet"
    elif best_any is not None:
        inst, path, probe_budget, best_instance_seed, _ = best_any
        state_pressure_warning = "horizon_unmet"
    else:
        raise RuntimeError(
            "Failed to construct navigation instance with valid reference path"
        )

    effective_horizon = max(0, len(path) - 1)
    # Rebuild instance with the chosen probe_budget (dataclass is frozen).
    inst = CEGISNavInstance(
        seed=inst.seed,
        n=inst.n,
        start=inst.start,
        goal=inst.goal,
        adj=inst.adj,
        locked_edges=inst.locked_edges,
        key_nodes=inst.key_nodes,
        traps=inst.traps,
        probe_budget=probe_budget,
        key_hint_radius=cfg.key_hint_radius,
    )

    max_steps = int(math.ceil(cfg.max_steps_multiplier * max(1, effective_horizon)))

    priv = private_spec(inst)
    difficulty: Dict[str, Any] = {
        "target_horizon": target_horizon,
        "effective_horizon": effective_horizon,
        "instance_seed": best_instance_seed,
        "n_nodes": inst.n,
        "n_edges": sum(len(v) for v in inst.adj.values()),
        "n_locks": len(inst.locked_edges),
        "n_keys": len(inst.key_nodes),
        "n_traps": len(inst.traps),
        "probe_budget": probe_budget,
    }
    if state_pressure_warning is not None:
        difficulty["state_pressure_warning"] = state_pressure_warning
    return NavigationTaskData(
        family="navigation",
        seed=seed,
        difficulty=difficulty,
        public=NavigationPublic(
            start=inst.start,
            goal=inst.goal,
            n=inst.n,
            max_steps=max_steps,
            probe_budget=probe_budget,
            key_hint_radius=cfg.key_hint_radius,
            neighbors_batch_max=cfg.neighbors_batch_max,
            notes=(
                "Graph topology is freely queryable via neighbors(nodes)"
                + (
                    " in batches. "
                    if cfg.neighbors_batch_max > 1
                    else ", one node per call. "
                )
                + "Hidden node properties (traps, keys, locks) require probe()."
            ),
        ),
        private=NavigationPrivate(
            adjacency=priv["adjacency"],
            locked_edges=priv["locked_edges"],
            key_nodes=priv["key_nodes"],
            traps=priv["traps"],
        ),
        reference=NavigationReference(
            shortest_path_len_with_key=effective_horizon,
            one_solution_path_with_key=path,
        ),
    )
