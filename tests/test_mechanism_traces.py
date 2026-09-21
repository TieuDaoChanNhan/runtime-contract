"""Unit tests for the mechanism analyses (scripts/mechanism_traces.py).

The replay harness itself is exercised end-to-end by running the script over the recorded
cap-sweep traces (it self-checks against the published knapsack anchors). What is worth
pinning down in a unit test is the logic that turns an executed-call log into the numbers:
the novel/replay classifier and the intermediate-hypothesis scorer.
"""

import unittest
from typing import Any

from codeact_runtime.config import RuleDiagnosisConfig
from codeact_runtime.families.rule_diagnosis import (
    RuleDiagnosisEnv,
    sample_rule_diagnosis_instance,
)
from scripts.mechanism_traces import (  # type: ignore[reportMissingImports]
    _recorded_witnesses,
    _score_hypothesis,
    classify,
    resume_stats,
)


def _call(tool: str, *args: Any, out: Any = None, **kwargs: Any) -> dict:
    """One entry of the replay harness's executed-call log."""
    return {
        "turn": 0,
        "tool": tool,
        "args": args,
        "kwargs": kwargs,
        "ok": True,
        "out": out,
    }


class TestClassify(unittest.TestCase):
    def test_knapsack_repeat_inspect_is_replay(self):
        calls = [
            _call("list_items"),  # first listing acquires state
            _call("inspect", "a"),
            _call("inspect", "b"),
            _call("inspect", "a"),  # repeat -> replay
            _call("list_items"),  # re-listing -> replay
            _call("take_item", "a"),  # progress
        ]
        self.assertEqual(
            classify("knapsack", calls),
            ["novel", "novel", "novel", "replay", "replay", "novel"],
        )

    def test_navigation_batched_neighbors_counts_fresh_nodes(self):
        calls = [
            _call("neighbors", nodes=[1, 2, 3]),
            _call("neighbors", nodes=[2, 3]),  # all already mapped -> replay
            _call("neighbors", nodes=[3, 4]),  # one new node -> novel
            _call("probe", node_id=7, out={"ok": True, "node": 7}),
            _call(
                "probe", node_id=7, out={"ok": True, "node": 7}
            ),  # re-probe -> replay
            _call("move", dst=2, out={"ok": True, "moved_to": 2}),
            _call("move", dst=2, out={"ok": True, "moved_to": 2}),  # revisit -> replay
            _call(
                "move", dst=9, out={"ok": False, "reason": "no_edge"}
            ),  # failed -> other
            _call("status"),
        ]
        self.assertEqual(
            classify("navigation", calls),
            [
                "novel",
                "replay",
                "novel",
                "novel",
                "replay",
                "novel",
                "replay",
                "other",
                "other",
            ],
        )
        self.assertEqual(calls[0]["fresh_nodes"], 3)
        self.assertEqual(calls[2]["fresh_nodes"], 1)

    def test_calls_the_environment_refused_are_not_progress(self):
        """A tool that raised (unknown id, budget spent, overweight take) acquired
        nothing, in any family, so it must not count as novel or as replay."""
        failed = {**_call("inspect", "a"), "ok": False, "out": None}
        calls = [
            _call("inspect", "a"),
            failed,  # would previously have counted as a repeat of "a"
            {**_call("take_item", "b"), "ok": False, "out": None},
            _call("take_item", "c"),
        ]
        self.assertEqual(
            classify("knapsack", calls), ["novel", "other", "other", "novel"]
        )

    def test_rule_rejected_schema_is_not_a_hypothesis_check(self):
        calls = [
            _call("check", hypothesis={"family": "bogus"}, out={"status": "error"}),
            _call(
                "check",
                hypothesis={"m": 8, "family": "affine_mod", "a": 1, "b": 0},
                out={"status": "fail", "x": 3, "y_pred": 1},
            ),
        ]
        self.assertEqual(classify("rule_diagnosis", calls), ["other", "novel"])

    def test_navigation_repeat_trap_is_replay(self):
        """Re-entering a trap the episode already discovered acquires nothing new."""
        trap = {"ok": False, "reason": "trap", "teleport_to": 2}
        calls = [
            {**_call("move", dst=7, out=trap), "context": {"current": 0}},
            {**_call("move", dst=7, out=trap), "context": {"current": 2}},
        ]
        self.assertEqual(classify("navigation", calls), ["novel", "replay"])

    def test_navigation_rejected_probe_acquires_nothing(self):
        """A probe the environment refuses (budget gone, bad node) reveals nothing, so it
        must not count as novel progress -- these are frequent once the budget runs out."""
        calls = [
            _call("probe", node_id=3, out={"ok": True, "node": 3}),
            _call("probe", node_id=3, out={"ok": True, "node": 3}),  # charged again
            _call("probe", node_id=8, out={"ok": False, "reason": "budget_exceeded"}),
            _call("probe", node_id=9, out={"ok": False, "reason": "invalid_node"}),
        ]
        self.assertEqual(
            classify("navigation", calls), ["novel", "replay", "other", "other"]
        )

    def test_navigation_move_back_to_the_occupied_node_is_replay(self):
        """The agent occupies its source node before moving, so returning there is not
        novel movement -- including on the episode's first move, out of the start node."""
        calls = [
            {
                **_call("move", dst=5, out={"ok": True, "moved_to": 5}),
                "context": {"current": 0},
            },
            {
                **_call("move", dst=0, out={"ok": True, "moved_to": 0}),
                "context": {"current": 5},
            },
            {
                **_call(
                    "move", dst=7, out={"ok": False, "reason": "trap", "teleport_to": 2}
                ),
                "context": {"current": 0},
            },
            {
                **_call("move", dst=2, out={"ok": True, "moved_to": 2}),
                "context": {"current": 2},
            },
        ]
        # 5 is new; 0 is the start, already occupied; stepping into an unseen trap DID
        # acquire state (it revealed the trap and moved the agent to 2), so it is novel,
        # and moving to 2 afterwards is a return rather than new ground.
        self.assertEqual(
            classify("navigation", calls), ["novel", "replay", "novel", "replay"]
        )

    def test_navigation_parameterless_neighbors_uses_the_current_node(self):
        """neighbors() with no argument queries the agent's current node, which the
        recorder captures as context -- otherwise the call looks like it mapped nothing."""
        calls = [
            {**_call("neighbors"), "context": {"current": 4}},
            {**_call("neighbors"), "context": {"current": 4}},  # same node -> replay
            _call("neighbors", nodes=[4, 5]),  # 5 is new -> novel
        ]
        self.assertEqual(classify("navigation", calls), ["novel", "replay", "novel"])
        self.assertEqual(calls[0]["nodes"], 1)
        self.assertEqual(calls[2]["fresh_nodes"], 1)

    def test_rule_repeat_probe_and_repeat_hypothesis_are_replay(self):
        hyp = {"m": 8, "family": "affine_mod", "a": 3, "b": 1}
        calls = [
            _call("test_input", 5),
            _call("test_input", 5),  # uncached repeat -> replay
            _call("test_input", 6),
            _call("check", hypothesis=hyp),
            _call(
                "check", hypothesis=dict(hyp)
            ),  # same hypothesis re-checked -> replay
        ]
        self.assertEqual(
            classify("rule_diagnosis", calls),
            ["novel", "replay", "novel", "novel", "replay"],
        )


def _turn(k=1, caphit=False, code="", surviving=()):
    return {
        "k": k,
        "executed": True,
        "caphit": caphit,
        "rec_caphit": caphit,
        "code": code,
        "surviving": list(surviving),
        "err": "",
        "rec_err": "",
    }


class TestResumeStats(unittest.TestCase):
    """Gate 2: after the cap truncates a block, does the next turn continue or start over?"""

    def _episode(self, calls, turns):
        return {"calls": calls, "turns": turns}

    def test_continuing_agent_neither_re_enumerates_nor_replays(self):
        calls = [
            {**_call("inspect", "a"), "turn": 0},
            {
                **_call("inspect", "b"),
                "turn": 1,
            },  # first call of the post-cap turn is new
        ]
        turns = [
            _turn(caphit=True, surviving=["inspected"]),
            _turn(code="inspected['b']"),
        ]
        (row,) = resume_stats("knapsack", [self._episode(calls, turns)])
        self.assertTrue(row["resume"])
        self.assertFalse(row["re_enumerates"])
        self.assertEqual(0, row["prefix_replay"])
        self.assertTrue(row["refs_surviving"])  # named a binding that survived

    def test_restarting_agent_re_derives_its_index_first(self):
        calls = [
            {**_call("list_items"), "turn": 0},
            {**_call("inspect", "a"), "turn": 0},
            {**_call("list_items"), "turn": 1},  # re-derives the catalogue
            {**_call("inspect", "a"), "turn": 1},  # then re-acquires what it held
            {**_call("inspect", "b"), "turn": 1},  # only now something new
        ]
        turns = [_turn(caphit=True), _turn()]
        (row,) = resume_stats("knapsack", [self._episode(calls, turns)])
        self.assertFalse(row["resume"])
        self.assertTrue(row["re_enumerates"])
        self.assertEqual(2, row["prefix_replay"])

    def test_a_derived_arm_uses_its_parent_index_call(self):
        """navigation_batch2 must count neighbors() as navigation's index call."""
        calls = [
            {**_call("neighbors", nodes=[1, 2]), "turn": 0},
            {**_call("neighbors", nodes=[1, 2]), "turn": 1},
            {**_call("neighbors", nodes=[3]), "turn": 1},
        ]
        turns = [_turn(caphit=True), _turn()]
        (row,) = resume_stats("navigation_batch2", [self._episode(calls, turns)])
        self.assertTrue(row["re_enumerates"])

    def test_a_cap_hit_on_the_final_turn_yields_no_transition(self):
        calls = [{**_call("inspect", "a"), "turn": 0}]
        self.assertEqual(
            [], resume_stats("knapsack", [self._episode(calls, [_turn(caphit=True)])])
        )


class TestRecordedWitnesses(unittest.TestCase):
    """check() draws its witness at random, so the repair statistics must use the witness
    the ORIGINAL episode saw, recovered from what that turn printed."""

    def test_recovers_witness_and_its_prediction_in_execution_order(self):
        """x and y_pred travel together: y_pred is the hypothesis's prediction AT x, so
        restoring one without the other would hand back a result no episode ever saw."""
        steps = [
            ("code", "", "{'status': 'fail', 'x': 493, 'y_pred': 2}\n"),
            ("code", "", "no check output here"),
            (
                "code",
                "",
                "{'status': 'fail', 'x': 12, 'y_pred': 0}\n{\"status\": \"fail\", \"x\": 700, \"y_pred\": 3}",
            ),
        ]
        self.assertEqual(
            _recorded_witnesses(steps),
            {
                0: [{"x": 493, "y_pred": 2}],
                1: [],
                2: [{"x": 12, "y_pred": 0}, {"x": 700, "y_pred": 3}],
            },
        )

    def test_passing_checks_contribute_no_witness(self):
        self.assertEqual(
            _recorded_witnesses([("code", "", "{'status': 'pass'}")]), {0: []}
        )

    def test_a_witness_printed_without_its_prediction_is_not_recovered(self):
        """Half a result is not a result -- such a call stays marked as replay-drawn."""
        self.assertEqual(
            _recorded_witnesses([("code", "", "{'status': 'fail', 'x': 493}")]), {0: []}
        )


class TestScoreHypothesis(unittest.TestCase):
    def _env(self) -> RuleDiagnosisEnv:
        cfg: dict[str, Any] = dict(
            probe_budget_range=(18, 20),
            mod_m_choices=[7, 11, 13],
            domain_range=(0, 999),
        )
        task = sample_rule_diagnosis_instance(42, RuleDiagnosisConfig(**cfg))
        return RuleDiagnosisEnv.from_task(task.model_dump())

    def test_true_rule_scores_one_and_does_not_mutate_env(self):
        env = self._env()
        truth = {"m": env.m, "family": env.family, **env.params}
        scored = _score_hypothesis(env, truth)
        assert scored is not None
        self.assertEqual(scored["functional"], 1.0)
        self.assertEqual(scored["boundary_f1"], 1.0)
        self.assertEqual(scored["composite"], 1.0)
        # scoring an intermediate hypothesis must not end or otherwise touch the episode
        self.assertFalse(env.done)
        self.assertEqual(env.probes_used, 0)

    def test_wrong_rule_scores_below_truth(self):
        env = self._env()
        wrong = _score_hypothesis(
            env, {"m": env.m, "family": "affine_mod", "a": 1, "b": 0}
        )
        assert wrong is not None
        self.assertLess(wrong["composite"], 1.0)

    def test_invalid_schema_returns_none(self):
        env = self._env()
        self.assertIsNone(_score_hypothesis(env, {"family": "not_a_family"}))


if __name__ == "__main__":
    unittest.main()
