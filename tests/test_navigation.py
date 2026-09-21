# type: ignore

import math
import unittest

from codeact_runtime.config import NavigationConfig
from codeact_runtime.families.navigation import (
    CEGISNavEnv,
    _bfs_path,
    _build_reference_path,
    _check_state_pressure,
    _compute_probe_budget,
    generate_instance,
    generate_instances,
    private_spec,
    public_spec,
    render_task_text,
    sample_navigation_instance,
)


class NavigationTests(unittest.TestCase):
    def test_determinism(self) -> None:
        a = generate_instance(seed=42)
        b = generate_instance(seed=42)
        self.assertEqual(private_spec(a), private_spec(b))

    def test_bulk_generation_count(self) -> None:
        tasks = generate_instances(2000, seed0=10)
        self.assertEqual(2000, len(tasks))

    def test_node_ids_are_permuted_not_backbone_order(self) -> None:
        # Stronger regression: numeric successor edges should be far from ubiquitous.
        ratios = []
        for seed in range(80):
            inst = generate_instance(seed=seed, trap_prob=0.0)
            count = sum(1 for u in range(inst.n) if ((u + 1) % inst.n) in inst.adj[u])
            ratios.append(count / inst.n)
        # If IDs leaked backbone ordering, this would be ~1.0.
        self.assertLess(sum(ratios) / len(ratios), 0.35)

    def test_trap_bug_regression_instances_remain_solvable(self) -> None:
        # Regression for trap-on-gate/key and trap-cutoff unsatisfiable cases.
        for seed in range(300):
            inst = generate_instance(seed=seed, trap_prob=0.3)
            gate_u = next(
                u
                for (u, v), lid in inst.locked_edges.items()
                if lid == "L0" and v == inst.goal
            )
            key_node = next(node for node, lid in inst.key_nodes.items() if lid == "L0")
            self.assertNotIn(gate_u, inst.traps)
            self.assertNotIn(key_node, inst.traps)

    def test_prekey_reachability_forbids_goal_and_traps(self) -> None:
        for seed in range(300):
            inst = generate_instance(seed=seed, trap_prob=0.3)
            gate_u = next(
                u
                for (u, v), lid in inst.locked_edges.items()
                if lid == "L0" and v == inst.goal
            )
            key_node = next(node for node, lid in inst.key_nodes.items() if lid == "L0")
            forbidden = set(inst.traps.keys()) | {inst.goal}
            self.assertIsNotNone(
                _bfs_path(inst.adj, inst.start, key_node, forbidden=forbidden)
            )
            self.assertIsNotNone(
                _bfs_path(inst.adj, key_node, gate_u, forbidden=forbidden)
            )

    def test_key_never_placed_on_start(self) -> None:
        for seed in range(300):
            inst = generate_instance(seed=seed)
            self.assertNotIn(inst.start, inst.key_nodes)

    def test_goal_unreachable_without_key_but_reachable_with_key(self) -> None:
        inst = generate_instance(seed=123, trap_prob=0.0, n_locks=1)
        env = CEGISNavEnv(inst, max_steps=10_000)

        (gate_u, goal), lock_id = next(
            ((u, v), lid)
            for (u, v), lid in inst.locked_edges.items()
            if lid == "L0" and v == inst.goal
        )
        self.assertEqual(inst.goal, goal)

        from collections import deque

        def safe_path(src: int, dst: int):
            q = deque([src])
            parent = {src: None}
            while q:
                u = q.popleft()
                if u == dst:
                    out = [u]
                    while parent[out[-1]] is not None:
                        out.append(parent[out[-1]])
                    out.reverse()
                    return out
                for v in inst.adj[u]:
                    if v in parent:
                        continue
                    if v in inst.traps and v != dst:
                        continue
                    if v == inst.goal and v != dst:
                        continue
                    parent[v] = u
                    q.append(v)
            return None

        to_gate = safe_path(inst.start, gate_u)
        self.assertIsNotNone(to_gate)
        for step in to_gate[1:]:
            step_result = env.move(step)
            self.assertTrue(
                step_result["ok"], msg=f"failed while approaching gate: {step_result}"
            )

        self.assertEqual(gate_u, env.status()["current"])
        blocked = env.move(inst.goal)
        self.assertFalse(blocked["ok"])
        self.assertEqual("locked", blocked["reason"])
        self.assertEqual(lock_id, blocked["lock_id"])

        key_node = next(node for node, lid in inst.key_nodes.items() if lid == "L0")
        to_key = safe_path(env.current, key_node)
        if to_key is None:
            env.reset()
            to_key = safe_path(env.current, key_node)
        self.assertIsNotNone(to_key)
        for step in to_key[1:]:
            step_result = env.move(step)
            self.assertTrue(
                step_result["ok"], msg=f"failed while approaching key: {step_result}"
            )

        self.assertIn("L0", env.status()["keys"])

        to_gate_again = safe_path(env.current, gate_u)
        self.assertIsNotNone(to_gate_again)
        for step in to_gate_again[1:]:
            step_result = env.move(step)
            self.assertTrue(
                step_result["ok"], msg=f"failed while returning to gate: {step_result}"
            )

        final = env.move(inst.goal)
        self.assertTrue(final["ok"])
        self.assertEqual(inst.goal, env.status()["current"])

    def test_try_path_returns_first_failure_and_is_pure(self) -> None:
        inst = generate_instance(seed=7, trap_prob=1.0)
        env = CEGISNavEnv(inst, max_steps=10_000)

        state_before = env.status().copy()
        path = [env.current]
        if env.neighbors():
            first_node = list(env.neighbors().keys())[0]
            path.append(env.neighbors()[first_node]["neighbors"][0])

        _ = env.try_path(path)
        state_after = env.status().copy()
        self.assertEqual(state_before, state_after)

        bad = env.try_path([env.current, inst.n + 1000])
        self.assertFalse(bad["ok"])
        self.assertEqual("no_edge", bad["reason"])
        self.assertEqual(0, bad["step"])

    def test_try_path_bad_path_start(self) -> None:
        inst = generate_instance(seed=9)
        env = CEGISNavEnv(inst, max_steps=10_000)
        out = env.try_path([inst.start + 1])
        self.assertFalse(out["ok"])
        self.assertEqual("bad_path_start", out["reason"])

    def test_move_no_edge_failure(self) -> None:
        inst = generate_instance(seed=10)
        env = CEGISNavEnv(inst, max_steps=10_000)
        impossible = next(
            v for v in range(inst.n) if v not in set(inst.adj[env.current])
        )
        out = env.move(impossible)
        self.assertFalse(out["ok"])
        self.assertEqual("no_edge", out["reason"])

    def test_trap_behavior_when_present(self) -> None:
        inst = generate_instance(seed=0, trap_prob=1.0)
        if not inst.traps:
            for s in range(1, 200):
                inst = generate_instance(seed=s, trap_prob=1.0)
                if inst.traps:
                    break
        self.assertTrue(inst.traps)

        trap_node, tele = next(iter(inst.traps.items()))
        env = CEGISNavEnv(inst, max_steps=10_000)

        from collections import deque

        q = deque([inst.start])
        parent = {inst.start: None}
        while q and trap_node not in parent:
            u = q.popleft()
            for v in inst.adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)

        self.assertIn(trap_node, parent)
        path = [trap_node]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        path.reverse()

        for step in path[1:-1]:
            env.move(step)

        ev = env.move(trap_node)
        self.assertFalse(ev["ok"])
        self.assertEqual("trap", ev["reason"])
        self.assertEqual(tele, env.status()["current"])

    def test_step_limit_enforced(self) -> None:
        inst = generate_instance(seed=8, trap_prob=0.0)
        env = CEGISNavEnv(inst, max_steps=1)
        n0 = env.neighbors()
        self.assertTrue(n0)
        first_node = list(n0.keys())[0]
        dst = n0[first_node]["neighbors"][0]
        _ = env.move(dst)  # uses the 1 allowed step → ok: True
        out = env.move(dst)  # over budget → ok: False
        self.assertFalse(out["ok"])
        self.assertEqual("max_steps_exceeded", out["reason"])

    def test_public_private_render_and_wrapper(self) -> None:
        inst = generate_instance(seed=11)
        pub = public_spec(inst)
        priv = private_spec(inst)
        txt = render_task_text(pub)

        self.assertIn("start", pub)
        self.assertIn("goal", pub)
        self.assertIn("locked_edges", priv)
        self.assertIn("Navigate", txt)

        cfg = NavigationConfig(num_tasks=1)
        wrapped = sample_navigation_instance(11, cfg=cfg)
        self.assertEqual("navigation", wrapped.family)
        self.assertEqual(
            wrapped.public.start, wrapped.reference.one_solution_path_with_key[0]
        )
        self.assertGreaterEqual(
            wrapped.public.max_steps, wrapped.reference.shortest_path_len_with_key
        )
        self.assertIn("instance_seed", wrapped.difficulty)

    def test_wrapper_effective_horizon_hard_like_config_not_collapsed(self) -> None:
        cfg = NavigationConfig(
            num_tasks=1,
            horizon_range=(25, 40),
            extra_nodes_range=(20, 35),
            extra_edge_factor_range=(0.35, 0.6),
            max_steps_multiplier=3.4,
        )
        in_range = 0
        for seed in range(80):
            task = sample_navigation_instance(seed, cfg)
            eff = task.reference.shortest_path_len_with_key
            if cfg.horizon_range[0] <= eff <= cfg.horizon_range[1]:
                in_range += 1
        self.assertGreaterEqual(in_range, 35)

    def test_wrapper_effective_horizon_tracks_config_range(self) -> None:
        cfg = NavigationConfig(
            num_tasks=1,
            horizon_range=(12, 14),
            extra_nodes_range=(0, 3),
            extra_edge_factor_range=(0.05, 0.15),
            max_steps_multiplier=3.0,
        )
        in_range = 0
        for seed in range(80):
            task = sample_navigation_instance(seed, cfg)
            eff = task.reference.shortest_path_len_with_key
            if cfg.horizon_range[0] <= eff <= cfg.horizon_range[1]:
                in_range += 1
        self.assertGreaterEqual(in_range, 60)

    def test_reference_path_wrapper_valid_across_many_seeds(self) -> None:
        cfg = NavigationConfig(num_tasks=1)
        for seed in range(200):
            task = sample_navigation_instance(seed, cfg)
            env = CEGISNavEnv.from_task(task.model_dump(mode="json"))
            out = env.try_path(task.reference.one_solution_path_with_key)
            self.assertTrue(out["ok"], msg=f"seed={seed}, out={out}")

    def test_reference_path_is_trap_free_and_executable(self) -> None:
        cfg = NavigationConfig(num_tasks=1)
        task = sample_navigation_instance(77, cfg)
        env = CEGISNavEnv.from_task(task.model_dump(mode="json"))

        out = env.try_path(task.reference.one_solution_path_with_key)
        self.assertTrue(out["ok"], msg=f"reference path should be executable: {out}")

    def test_probe_budget_is_capped(self):
        # Huge horizon-driven budget, small n cap should apply
        b = _compute_probe_budget(effective_horizon=60, n=100, n_traps=0)
        self.assertLessEqual(b, int(math.ceil(0.30 * 100)))

    def test_probe_budget_not_capped_when_under_limit(self):
        b = _compute_probe_budget(effective_horizon=10, n=200, n_traps=0)
        # base=ceil(1.5*10)=15, +2 buffer=17, cap=60 — should not clamp
        self.assertEqual(b, int(math.ceil(1.5 * 10)) + 2)

    def test_instance_contains_decoy_keys(self) -> None:
        for seed in range(50):
            inst = generate_instance(seed=seed, trap_prob=0.0)
            decoy_ids = [key_id for key_id in inst.key_nodes.values() if key_id.startswith("D")]
            self.assertGreaterEqual(len(decoy_ids), 1)


def _easy_nav_cfg() -> NavigationConfig:
    return NavigationConfig(
        num_tasks=1,
        horizon_range=(5, 10),
        extra_nodes_range=(4, 14),
        extra_edge_factor_range=(0.55, 0.85),
        max_steps_multiplier=1.25,
        trap_prob=0.45,
        probe_budget_multiplier=0.85,
        decoy_count_range=(2, 3),
        reject_singleton_l0_hint=True,
    )


def _hint_set(inst) -> set[int]:
    """Set of nodes where `key_nearby` is True under inst.key_hint_radius.

    Mirrors the runtime `_key_within_hops(...)` semantics used to populate
    the `key_nearby` hint in NeighborsTool — and the same semantics used by
    `_check_state_pressure` for its acceptance audit.
    """
    from codeact_runtime.families.navigation import _key_within_hops
    keys = set(inst.key_nodes.keys())
    radius = max(1, getattr(inst, "key_hint_radius", 1))
    return {u for u in inst.adj if _key_within_hops(inst.adj, u, keys, max_hops=radius)}


class StatePressureAcceptanceTests(unittest.TestCase):
    """Structural acceptance gate for the easy navigation tier."""

    def test_decoy_near_l0_hint_when_required(self) -> None:
        cfg = _easy_nav_cfg()
        ok = 0
        for seed in range(200):
            task = sample_navigation_instance(seed, cfg)
            if "state_pressure_warning" not in task.difficulty:
                ok += 1
        self.assertGreaterEqual(
            ok, 190,
            msg=f"only {ok}/200 seeds satisfy structural acceptance",
        )

    def test_no_singleton_l0_hint_set(self) -> None:
        cfg = _easy_nav_cfg()
        for seed in range(200):
            task = sample_navigation_instance(seed, cfg)
            if "state_pressure_warning" in task.difficulty:
                continue  # fallback path; not required to satisfy structural invariants
            env = CEGISNavEnv.from_task(task.model_dump(mode="json"))
            inst = env.instance
            real_keys = {n for n, k in inst.key_nodes.items() if k.startswith("L")}
            decoys = {n for n, k in inst.key_nodes.items() if k.startswith("D")}
            hint = _hint_set(inst)
            if not hint:
                continue
            decoy_only = [
                u for u in hint
                if any(v in decoys for v in inst.adj.get(u, []))
                and not any(v in real_keys for v in inst.adj.get(u, []))
            ]
            self.assertTrue(
                decoy_only,
                msg=f"seed={seed}: no decoy-only hint node — key_nearby uniquely localizes L0",
            )

    def test_probe_budget_below_full_audit(self) -> None:
        cfg = _easy_nav_cfg()
        ok = 0
        for seed in range(200):
            task = sample_navigation_instance(seed, cfg)
            if "state_pressure_warning" in task.difficulty:
                continue
            env = CEGISNavEnv.from_task(task.model_dump(mode="json"))
            inst = env.instance
            ref = _build_reference_path(inst) or []
            full_audit = len(_hint_set(inst) | set(ref[1:-1]))
            if task.public.probe_budget < full_audit:
                ok += 1
        self.assertGreaterEqual(
            ok, 190,
            msg=f"only {ok}/200 seeds have probe_budget < full audit",
        )

    def test_check_state_pressure_passes_for_accepted(self) -> None:
        """Round-trip: instances without warning must satisfy the gate directly."""
        cfg = _easy_nav_cfg()
        for seed in range(50):
            task = sample_navigation_instance(seed, cfg)
            if "state_pressure_warning" in task.difficulty:
                continue
            env = CEGISNavEnv.from_task(task.model_dump(mode="json"))
            reason = _check_state_pressure(
                env.instance,
                task.public.probe_budget,
                reject_singleton_l0_hint=True,
            )
            self.assertIsNone(reason, msg=f"seed={seed}: {reason}")


def _medium_nav_cfg() -> NavigationConfig:
    """Mirrors medium.json navigation block after the medium-tier tuning."""
    return NavigationConfig(
        num_tasks=1,
        horizon_range=(10, 25),
        extra_nodes_range=(8, 18),
        extra_edge_factor_range=(0.18, 0.35),
        max_steps_multiplier=1.3,
        trap_count_range=(1, 2),
        probe_cap_fraction=0.25,
        decoy_count_range=(3, 5),
        key_hint_radius=2,
        reject_singleton_l0_hint=True,
    )


class MediumNavigationAcceptanceTests(unittest.TestCase):
    """Multi-trap + hint-radius + probe-cap acceptance for the medium navigation tier."""

    def test_medium_multi_trap(self) -> None:
        """At least ≥190/200 tasks have 1–2 traps (rare graceful fallback ok)."""
        cfg = _medium_nav_cfg()
        ok = 0
        for seed in range(200):
            task = sample_navigation_instance(seed, cfg)
            n_traps = task.difficulty["n_traps"]
            if 1 <= n_traps <= 2:
                ok += 1
        self.assertGreaterEqual(
            ok, 190,
            msg=f"only {ok}/200 seeds have n_traps in [1, 2]",
        )

    def test_medium_hint_radius_2(self) -> None:
        """Hint set at radius=2 covers a region, not a singleton."""
        cfg = _medium_nav_cfg()
        for seed in range(200):
            task = sample_navigation_instance(seed, cfg)
            self.assertEqual(task.public.key_hint_radius, 2)
            env = CEGISNavEnv.from_task(task.model_dump(mode="json"))
            self.assertEqual(env.instance.key_hint_radius, 2)
            # With 3-5 decoys + L0, hint_set should be at least 3 nodes wide.
            hs = _hint_set(env.instance)
            self.assertGreaterEqual(
                len(hs), 3,
                msg=f"seed={seed}: hint_set has only {len(hs)} nodes",
            )

    def test_medium_probe_cap_fraction(self) -> None:
        """Probe budget never exceeds ceil(0.25 * n_nodes) — the configured cap."""
        cfg = _medium_nav_cfg()
        import math as _math
        for seed in range(200):
            task = sample_navigation_instance(seed, cfg)
            n = task.difficulty["n_nodes"]
            cap = _math.ceil(0.25 * n)
            # `probe_budget_multiplier` defaults to 1.0 here, so the capped
            # output of `_compute_probe_budget` is the upper bound (modulo the
            # max(5, ...) floor on tiny graphs).
            self.assertLessEqual(
                task.public.probe_budget, max(5, cap),
                msg=f"seed={seed}: probe_budget {task.public.probe_budget} > cap {cap} for n={n}",
            )


class NeighborsBandwidthTests(unittest.TestCase):
    """`neighbors_batch_max` is the reconstruction-bandwidth knob behind the Sec. 5
    navigation intervention: it decides whether rebuilding an n-node map costs
    ceil(n/50) calls or n of them, against the same per-turn tool-call cap."""

    def _task(self, batch_max: int | None) -> dict:
        task = sample_navigation_instance(7, NavigationConfig(num_tasks=1)).model_dump(
            mode="json"
        )
        if batch_max is not None:
            task["public"]["neighbors_batch_max"] = batch_max
        return task

    def test_default_is_the_batched_interface_every_existing_task_uses(self) -> None:
        # a task JSON written before the field existed must behave exactly as before
        legacy = self._task(None)
        legacy["public"].pop("neighbors_batch_max", None)
        env = CEGISNavEnv.from_task(legacy)
        self.assertEqual(50, env.instance.neighbors_batch_max)
        self.assertEqual(50, len(env.neighbors(list(range(50)))))
        with self.assertRaises(ValueError):
            env.neighbors(list(range(51)))

    def test_unbatched_env_accepts_one_node_and_rejects_two(self) -> None:
        env = CEGISNavEnv.from_task(self._task(1))
        self.assertEqual([0], list(env.neighbors([0])))
        self.assertEqual([env.current], list(env.neighbors()))  # implicit current node
        with self.assertRaises(ValueError) as caught:
            env.neighbors([0, 1])
        self.assertIn("at most 1 node per call", str(caught.exception))

    def test_the_agent_is_told_the_limit_up_front(self) -> None:
        """Not only enforced: stated in both surfaces the agent reads, so the cost is
        planned for rather than discovered through errors."""
        from codeact_runtime.families.navigation import NeighborsTool

        unbatched = CEGISNavEnv.from_task(self._task(1))
        prompt, doc = unbatched.get_goal_prompt(), NeighborsTool(unbatched).doc
        self.assertIn("ONE node per call", prompt)
        self.assertIn("once per node", prompt)
        self.assertNotIn("up to 50", prompt)
        self.assertIn("only one node per call", doc)

        batched = CEGISNavEnv.from_task(self._task(50))
        self.assertIn("up to 50 at once", batched.get_goal_prompt())
        self.assertIn("in batches", batched.get_goal_prompt())
        self.assertIn("max 50 per call", NeighborsTool(batched).doc)

    def test_only_the_interface_changes_not_the_instance(self) -> None:
        """The intervention must hold the latent graph, budgets and scoring fixed."""
        batched, unbatched = (
            CEGISNavEnv.from_task(self._task(b)).instance for b in (50, 1)
        )
        for field in ("seed", "n", "start", "goal", "probe_budget", "key_hint_radius"):
            self.assertEqual(getattr(batched, field), getattr(unbatched, field), field)
        self.assertEqual(batched.adj, unbatched.adj)
        self.assertEqual(batched.locked_edges, unbatched.locked_edges)
        self.assertEqual(batched.key_nodes, unbatched.key_nodes)
        self.assertEqual(batched.traps, unbatched.traps)

    def test_generator_carries_the_configured_bandwidth(self) -> None:
        task = sample_navigation_instance(
            7, NavigationConfig(num_tasks=1, neighbors_batch_max=1)
        )
        self.assertEqual(1, task.public.neighbors_batch_max)
        self.assertIn("one node per call", task.public.notes or "")


if __name__ == "__main__":
    unittest.main()
