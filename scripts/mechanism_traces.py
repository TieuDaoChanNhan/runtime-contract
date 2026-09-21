#!/usr/bin/env python3
"""Mechanism analyses measured on the ALREADY-RECORDED traces (no GPU, no new runs).

Replaces the
nominal cross-family replay-demand proxy R with quantities measured in the unit the
harness actually enforces -- tool calls inside one action block -- and on a scale that is
comparable across the three task families.

Everything here comes from a family-generic *trace-conditioned replay*: each episode's
recorded CodeAct code blocks are re-executed against the real environment (no LLM) under
the same persistent/stateless interpreter semantics and the same per-turn tool-call cap,
with every registered tool wrapped in a recorder. That gives the true executed call
sequence per turn, which static parsing of the code cannot (calls sit inside loops that
break on tool feedback). It generalizes scripts/replay_analysis.py, which wrapped only
knapsack's inspect().

Sections
  0  replay fidelity: turns where the replay's error state diverges from the recording
  1  cap exposure at the SLACK cap: K_t per action block, E_25=Pr(K_t>25), O_25=E[(K_t-25)+]
  2  replay vs novel progress at both caps (common table across families)
  3  navigation stage decomposition (topology / key / gate / goal / traps / duplicate probes)
  3b navigation counterfactual: cap exposure if neighbors() were unbatched (design check)
  4  rule-diagnosis hypothesis-repair trajectory: H = E[dscore | counterexample returned]
  5  operational currency: P->S over P->P ratios (tokens, calls, turns, duplicates, errors)
  6  first-block intended width: the announced cap vs the enforced cap
  7  resume vs restart after a cap hit: the differential-resumability gate

Run: uv run python scripts/mechanism_traces.py [--families knapsack,navigation,rule_diagnosis]
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter
from typing import Any

from codeact_runtime.benchmark.config import FAMILY_ENVS
from codeact_runtime.codeact.agent import FinishTool
from codeact_runtime.codeact.interpreter import AsyncInterpreter, ToolCallLimitError
from codeact_runtime.families.rule_diagnosis import (
    _compute_structural_metrics,
    _parse_hypothesis_input,
)

CAP_BIND, CAP_SLACK = 25, 80
# A recorded block that loops without tool calls cannot be interrupted: it runs in the
# interpreter's worker thread, which asyncio.wait_for cannot cancel and which would then
# keep a core busy and block interpreter exit. We therefore treat it as fatal rather than
# pretending to skip it -- see _abort_on_hung_block. No block in the released cap-sweep
# corpus hits this; every one of them ran to completion in its original episode too.
BLOCK_TIMEOUT_S = 120.0

# arm -> task instances and the run directory of one evaluated cell. An "arm" is usually
# just a family, but navigation has two: the batched interface every earlier evaluation
# used, and the unbatched intervention arm of Sec. 5. Both run the SAME environment code
# and the same 16 graphs, so `env` names the family whose env class and trace filenames
# apply, while the key names the arm.
FAMILIES = {
    "knapsack": {
        "tasks": "experiments/cap_sweep/knapsack/task_defs/tasks/knapsack",
        "runs": "experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap{cap}",
    },
    "navigation": {
        "tasks": "experiments/cap_sweep/navigation/task_defs/tasks/navigation",
        "runs": "experiments/cap_sweep/navigation/qwen3_8b/main/nav_{cell}_cap{cap}",
    },
    "navigation_batch2": {
        "tasks": "experiments/cap_sweep/navigation/task_defs_batch2/tasks/navigation",
        "runs": "experiments/cap_sweep/navigation/qwen3_8b/batch2/navb2_{cell}_cap{cap}",
        "env": "navigation",
    },
    "rule_diagnosis": {
        "tasks": "experiments/cap_sweep/rule_diagnosis/task_defs/tasks/rule_diagnosis",
        "runs": "experiments/cap_sweep/rule_diagnosis/qwen3_8b/main/rule_{cell}_cap{cap}",
    },
}
SHORT = {
    "knapsack": "knap",
    "navigation": "nav",
    "navigation_batch2": "navb2",
    "rule_diagnosis": "rule",
}


def _env_family(arm: str) -> str:
    """The family whose env class, trace filenames and task ids this arm uses."""
    return str(FAMILIES[arm].get("env", arm))


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _traces(family, cell, cap):
    base = FAMILIES[family]["runs"].format(cell=cell, cap=cap)
    fam = _env_family(family)
    return sorted(glob.glob(f"{base}/**/{fam}-{fam}-*.trace.json", recursive=True))


def _task_for(family, trace_path):
    """Pair a trace with its task JSON by index, verifying a public field appears in the
    goal text so a mis-paired tasks_root fails loudly instead of scoring nonsense."""
    idx = int(os.path.basename(trace_path).split("-")[-1].split(".")[0])
    tj = f"{FAMILIES[family]['tasks']}/{_env_family(family)}-{idx:010d}.json"
    if not os.path.exists(tj):
        return None
    task = _load(tj)
    goal = str((_load(trace_path).get("summary") or {}).get("task") or "")
    pub = task.get("public", {})
    # dispatch on the ENV family: a derived arm (e.g. navigation_batch2) shares navigation's
    # goal text, so keying this on the arm name would check it against the wrong family
    family = _env_family(family)
    if family == "knapsack":
        m = re.search(r"total weight\):\s*(\d+)", goal)
        ok = m is not None and int(m.group(1)) == pub["capacity"]
    elif family == "navigation":
        ok = (
            f"Goal node: {pub['goal']}" in goal
            and f"Total nodes in graph: {pub['n']}" in goal
        )
    else:
        ok = (
            f"Modulus m: {pub['m']}" in goal
            and f"Probe budget: {pub['probe_budget']}" in goal
        )
    if not ok:
        raise SystemExit(f"task/trace mismatch: {trace_path} vs {tj}")
    return task


def _steps(trace_path):
    """(code, recorded_error, recorded_stdout) per StepEvent, in turn order."""
    out = []
    for e in _load(trace_path).get("events", []):
        if e.get("type") != "StepEvent":
            continue
        d = e.get("data") or {}
        res = d.get("interpreter_result") or {}
        out.append(
            (
                d.get("code") or "",
                str(res.get("error") or ""),
                str(res.get("output") or ""),
            )
        )
    return out


def _observed_globals(trace_path):
    """turn index -> the interpreter globals the harness reported as still ALIVE after that
    turn. Under a persistent runtime this is what carries into the next turn; under a reset
    it is empty by construction. Read from the observation the model itself saw, so it
    reflects the episode rather than our replay."""
    out = {}
    for e in _load(trace_path).get("events", []):
        if e.get("type") != "UserObservationEvent":
            continue
        d = e.get("data") or {}
        try:
            state = json.loads(d.get("content") or "{}").get("runtime_state") or {}
        except (json.JSONDecodeError, TypeError):
            continue
        out[d.get("turn")] = list(state.get("active_globals") or [])
    return out


# The witness a failing check() returns is drawn at random inside the first wrong run
# (rule_diagnosis.py::check), so a replay cannot reproduce the original draw from a seed.
# Where the episode printed the check result we recover the RECORDED witness and force the
# replay to return it, which keeps both the branching and the repair statistics faithful;
# calls whose witness cannot be recovered are marked and excluded from those statistics.
_WITNESS_RE = re.compile(
    r"['\"]status['\"]\s*:\s*['\"]fail['\"][^}]*?['\"]x['\"]\s*:\s*(\d+)"
    r"[^}]*?['\"]y_pred['\"]\s*:\s*(-?\d+)"
)


def _recorded_witnesses(steps):
    """turn index -> the failing-check results that turn printed, in execution order.

    Both fields are recovered: substituting the recorded x while keeping the replay's
    y_pred would hand the episode a result it never saw (y_pred is the hypothesis's
    prediction AT x, so the pair must travel together)."""
    return {
        i: [{"x": int(x), "y_pred": int(y)} for x, y in _WITNESS_RE.findall(out)]
        for i, (_, _, out) in enumerate(steps)
    }


def _arg_of(args, kwargs, name, idx=0):
    if name in kwargs:
        return kwargs[name]
    return args[idx] if len(args) > idx else None


def _hypothesis_key(env, raw):
    """Canonical identity of a rule hypothesis: the env's normalized (family, params,
    exceptions). Falls back to the raw spelling when the schema is invalid, since the
    environment rejects those without telling the agent anything either way."""
    try:
        return json.dumps(
            env._normalize_hypothesis(_parse_hypothesis_input(raw)),
            sort_keys=True,
            default=str,
        )
    except Exception:
        return repr(raw)


class _CountingInterpreter(AsyncInterpreter):
    """Counts calls the per-turn cap REJECTED, at the point of rejection.

    Inferring cap hits from the block's top-level error misses every turn whose recorded
    code wrapped its tool calls in try/except: the ToolCallLimitError is swallowed inside
    the program, so the turn looks complete while its demand was actually truncated."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rejected = 0

    async def _check_and_increment_tool_call(self) -> None:
        try:
            await super()._check_and_increment_tool_call()
        except ToolCallLimitError:
            self.rejected += 1
            raise


# ---------------------------------------------------------------- replay harness
def _abort_on_hung_block(trace_path, turn_index):
    """Stop the whole analysis when a recorded block exceeds BLOCK_TIMEOUT_S.

    The block is executing in a ThreadPoolExecutor worker that neither wait_for nor
    shutdown(wait=False) can terminate, so continuing would report numbers from a
    half-replayed corpus while a runaway thread burns a core and blocks exit. Exit
    immediately and loudly instead; os._exit skips the executor's atexit join, which
    would otherwise hang on that same thread."""
    sys.stderr.write(
        f"\nFATAL: replay of {trace_path} turn {turn_index} exceeded "
        f"{BLOCK_TIMEOUT_S:.0f}s and cannot be interrupted (the block is running in an\n"
        "interpreter worker thread). Results would be incomplete, so the analysis is\n"
        "aborting rather than reporting partial numbers.\n"
    )
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(3)


class _Recorder:
    """Wraps a Tool so every executed call is logged. The interpreter charges the
    per-turn cap BEFORE invoking the tool, so a call rejected by the cap never reaches
    here -- exactly the semantics we want for K_t (executed calls per action block).

    `context` (optional) captures env state that the call's arguments do not carry, e.g.
    the node navigation's parameterless neighbors() implicitly queries.
    `witness_queue` (optional) replaces a failing check()'s randomly drawn witness with
    the one the original episode saw.
    `fingerprint` (optional) records a canonical identity for the call's arguments, so
    two spellings of the same hypothesis are recognised as the same query."""

    def __init__(
        self, tool, sink, turn, context=None, witness_queue=None, fingerprint=None
    ):
        self._tool, self._sink, self._turn = tool, sink, turn
        self._context, self._witness_queue = context, witness_queue
        self._fingerprint = fingerprint
        self.name = tool.name

    def __getattr__(self, item):  # forwards on_turn_start for tools that define it
        return getattr(self._tool, item)

    def _apply_recorded_witness(self, rec, out):
        if self._witness_queue is None or not isinstance(out, dict):
            return out
        if out.get("status") != "fail" or "x" not in out:
            return out
        queue = self._witness_queue.get(rec["turn"]) or []
        if not queue:
            rec["witness_source"] = "replayed"
            return out
        out = dict(
            out, **queue.pop(0)
        )  # both x and its y_pred, as the episode saw them
        rec["witness_source"] = "recorded"
        return out

    async def __call__(self, *args, **kwargs):
        rec = {"turn": self._turn[0], "tool": self.name, "args": args, "kwargs": kwargs}
        if self._context is not None:
            rec["context"] = self._context()
        if self._fingerprint is not None:
            rec["hyp_key"] = self._fingerprint(args, kwargs)
        self._sink.append(rec)
        try:
            out = await self._tool(*args, **kwargs)
        except Exception as exc:
            rec["ok"], rec["out"], rec["err"] = False, None, repr(exc)
            raise
        rec["ok"], rec["out"] = True, self._apply_recorded_witness(rec, out)
        return rec["out"]


async def first_block_intent(family, trace_path, persistent):
    """Calls the episode's FIRST action block would have made under no cap at all.

    The cap treatment moves two coupled things: the announced STRICT LIMIT line in the
    prompt, and the enforced runtime limit. On the first block those can be separated,
    because nothing the cap did can have reached the model yet -- it has seen only the
    announced number. Executing that block with the cap removed therefore recovers the
    plan the announced number induced, and comparing the c=25-prompt plan with the
    c=80-prompt plan says whether the integer alone changes what the policy attempts.

    Only the first block qualifies: from the second turn on, the trajectory has already
    absorbed cap-produced observations, so an uncapped replay would be counterfactual."""
    task = _task_for(family, trace_path)
    if task is None:
        return None
    random.seed(0)
    env = FAMILY_ENVS[_env_family(family)].from_task(task)
    calls, turn = [], [0]
    steps = _steps(trace_path)
    if not steps or not steps[0][0]:
        return None
    interp = _CountingInterpreter(persistent_state=persistent, max_tool_calls=None)
    for tool in list(env.get_tools()) + [FinishTool()]:
        interp.register_tool(tool.name, _Recorder(tool, calls, turn))
    try:
        await asyncio.wait_for(interp.run_code(steps[0][0]), BLOCK_TIMEOUT_S)
    except asyncio.TimeoutError:
        _abort_on_hung_block(trace_path, 0)
    finally:
        interp._executor.shutdown(wait=False)
    return len(calls)


async def replay(family, trace_path, persistent, cap, carryover=False):
    """Re-execute one episode's recorded blocks against the real env. Returns the env
    (final state), the executed-call log, per-turn K_t, and a divergence count."""
    task = _task_for(family, trace_path)
    if task is None:
        return None
    random.seed(0)  # only reached for checks whose recorded witness is unrecoverable
    env = FAMILY_ENVS[_env_family(family)].from_task(task)
    calls, turn = [], [0]
    steps = _steps(trace_path)
    env_fam = _env_family(family)
    witnesses = _recorded_witnesses(steps) if env_fam == "rule_diagnosis" else None
    # `carryover` replays the cap-boundary-carryover cell: a turn the cap truncated hands
    # its workspace to the next turn, so the recorded blocks must see the same runtime the
    # episode was produced under or their errors will not match the recording.
    interp = _CountingInterpreter(
        persistent_state=persistent, max_tool_calls=cap, carryover_on_cap=carryover
    )
    for tool in list(env.get_tools()) + [FinishTool()]:
        # Navigation calls whose meaning depends on env state rather than arguments:
        # neighbors() defaults to the current node, and a move's source is the node the
        # agent already occupies. Capture the current node before each such call.
        context = (
            (lambda: {"current": env.current})
            if (env_fam == "navigation" and tool.name in ("neighbors", "move"))
            else None
        )
        # Two spellings of one hypothesis (intervals in another order, ignored extra
        # fields) are the same oracle query, and the env's own normalizer is what decides
        # that -- so identity is taken from its output, not from the raw argument.
        fingerprint = (
            (lambda a, kw, env=env: _hypothesis_key(env, _arg_of(a, kw, "hypothesis")))
            if (env_fam == "rule_diagnosis" and tool.name in ("check", "submit"))
            else None
        )
        interp.register_tool(
            tool.name,
            _Recorder(
                tool,
                calls,
                turn,
                context=context,
                witness_queue=witnesses if tool.name == "check" else None,
                fingerprint=fingerprint,
            ),
        )

    surviving = _observed_globals(trace_path)
    per_turn, diverged, nocode = [], 0, 0
    try:
        for i, (code, rec_err, _stdout) in enumerate(steps):
            turn[0], before = i, len(calls)
            rejected_before = interp.rejected
            err = ""
            if code:
                try:
                    res = await asyncio.wait_for(interp.run_code(code), BLOCK_TIMEOUT_S)
                    err = str(res.get("error") or "")
                except asyncio.TimeoutError:
                    _abort_on_hung_block(trace_path, i)
            else:
                # The model emitted no fenced block that turn; the harness answered with a
                # retry prompt, so there is no execution to compare against.
                nocode += 1
            if code and bool(err) != bool(rec_err):
                diverged += 1
            per_turn.append(
                {
                    "k": len(calls) - before,
                    "executed": bool(code),
                    "code": code,
                    # names still bound after this turn, as reported to the model
                    "surviving": surviving.get(i, []),
                    # counted at the rejection itself, so a block that catches the
                    # exception cannot hide its truncation
                    "caphit": interp.rejected > rejected_before,
                    "rec_caphit": "Tool call limit exceeded" in rec_err,
                    "err": err,
                    "rec_err": rec_err,
                }
            )
    finally:
        interp._executor.shutdown(wait=False)
    # An episode is faithful unless a failing check() had to fall back on a randomly drawn
    # witness: from that point on the recorded code may branch, or carry state, on a value
    # the original episode never saw, so EVERY later call is counterfactual -- not just the
    # repair transition. Such episodes are dropped from all replay-derived statistics.
    unfaithful = next(
        (c["turn"] for c in calls if c.get("witness_source") == "replayed"), None
    )
    return {
        "trace": trace_path,
        "task": task,
        "env": env,
        "calls": calls,
        "turns": per_turn,
        "diverged": diverged,
        "nocode": nocode,
        "faithful": unfaithful is None,
        "unfaithful_from_turn": unfaithful,
    }


async def replay_cell(family, cell, cap):
    persistent = cell[1] == "P"
    # any cell whose name carries "ckpt" ran the cap-boundary-carryover runtime, announced
    # or not -- the announcement changes the banner, not the semantics the replay reproduces
    # (the cap-boundary-carryover condition; see the paper's replication appendix)
    carryover = "ckpt" in cell
    out = []
    for t in _traces(family, cell, cap):
        ep = await replay(family, t, persistent, cap, carryover)
        if ep:
            out.append(ep)
    return out


def _task_ids(family):
    """Every task instance the family defines, as zero-padded index strings.

    Cross-checked against the generator manifest's num_tasks so that a partial extraction
    which removed task JSONs (and their traces) cannot redefine 'complete' downwards."""
    root = FAMILIES[family]["tasks"]
    ids = {
        os.path.basename(f).split("-")[-1].split(".")[0]
        for f in glob.glob(f"{root}/{_env_family(family)}-*.json")
    }
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(root)), "_cfg.json")
    if os.path.exists(cfg_path):
        declared = ((_load(cfg_path).get(_env_family(family)) or {}) or {}).get(
            "num_tasks"
        )
        # the generator emits exactly ids 0..num_tasks-1 (generator.py), so compare the
        # SET: a missing instance alongside a stale higher-numbered one would otherwise
        # keep the count right and silently swap a task out of the cohort
        expected = {f"{i:010d}" for i in range(declared or 0)}
        if declared and ids != expected:
            raise SystemExit(
                f"{family}: {cfg_path} declares num_tasks={declared} but {root} holds a "
                f"different set -- missing {sorted(expected - ids)[:5]}, "
                f"unexpected {sorted(ids - expected)[:5]}."
            )
    return ids


def _episode_ids(episodes):
    return {os.path.basename(e["trace"]).split("-")[-1].split(".")[0] for e in episodes}


def _require_complete(data, families):
    """Refuse to report means over a truncated corpus.

    Each cell is checked against the family's FULL task set, not just against its sibling
    cells: an archive missing the same instances everywhere would keep the cells mutually
    consistent while quietly shrinking every mean."""
    problems = []
    for fam in families:
        expected = _task_ids(fam)
        if not expected:
            problems.append(f"{fam}: no task instances under {FAMILIES[fam]['tasks']}")
            continue
        for cell in ("PP", "PS"):
            for cap in (CAP_BIND, CAP_SLACK):
                eps = data.get((fam, cell, cap)) or []
                where = FAMILIES[fam]["runs"].format(cell=cell, cap=cap)
                if not eps:
                    problems.append(f"{fam} {cell}@{cap}: no episodes under {where}")
                    continue
                missing = expected - _episode_ids(eps)
                if missing:
                    problems.append(
                        f"{fam} {cell}@{cap}: {len(eps)}/{len(expected)} tasks replayed, "
                        f"missing {sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}"
                    )
                # a trace without its scored result would silently become a NaN score in
                # sections 2, 3 and 5 rather than an error
                unscored = [
                    os.path.basename(e["trace"])
                    for e in eps
                    if math.isnan(_episode_score(e))
                ]
                if unscored:
                    problems.append(
                        f"{fam} {cell}@{cap}: {len(unscored)} episodes have no scored "
                        f"result JSON, e.g. {sorted(unscored)[:3]}"
                    )
    if problems:
        raise SystemExit(
            "incomplete replay corpus -- refusing to report partial means:\n  "
            + "\n  ".join(problems)
            + "\nUnpack the eval-data mirror into experiments/cap_sweep/ (see the Released "
            "artifacts section of the README)."
        )


def _paired_faithful(data, families):
    """Restrict every cell to the tasks that replayed faithfully in ALL of a family's
    cells, so the paired design survives the exclusion: dropping a task from one cell
    only would leave sections 1-5 comparing means over different cohorts."""
    out = dict(data)
    for fam in families:
        cells = [k for k in data if k[0] == fam]
        if not cells:
            continue
        keep = set.intersection(
            *(
                {_episode_ids([e]).pop() for e in data[k] if e["faithful"]}
                for k in cells
            )
        )
        for k in cells:
            out[k] = [e for e in data[k] if _episode_ids([e]).pop() in keep]
    return out


# ------------------------------------------------------- call classification
def _arg(rec, name, idx=0):
    if name in rec["kwargs"]:
        return rec["kwargs"][name]
    return rec["args"][idx] if len(rec["args"]) > idx else None


def _nav_nodes(rec):
    """Nodes a neighbors() call actually queried. Called with no argument it queries the
    agent's current node (navigation.py::neighbors), which the recorder captured."""
    nodes = _arg(rec, "nodes")
    if nodes is None:
        current = (rec.get("context") or {}).get("current")
        return [] if current is None else [current]
    return nodes if isinstance(nodes, list) else [nodes]


def classify(family, calls):
    """Tag each executed call novel / replay / other, per the review's definitions.

    A call the environment REFUSED acquires nothing, so it is 'other' whichever family it
    belongs to: a tool that raised (inspect of an unknown id, inspect/probe budget spent,
    an overweight take_item, an out-of-domain test_input), a navigation probe answered
    with ok=false, or a check()/submit() rejected for an invalid schema. Section 3 counts
    navigation's refused probes separately, as wasted calls.

      knapsack   replay = repeated inspect(id) or repeated list_items();
                 novel  = first inspection of an item, first listing, or a take_item the
                          environment accepted.
      navigation replay = neighbors() whose nodes were all mapped before, a repeat
                          probe(node) that still charged budget, or a move back onto an
                          already-occupied node (the source node counts as occupied);
                 novel  = neighbors() mapping >=1 new node, a first successful probe, a
                          move onto a node not yet occupied.
      rule       replay = repeat test_input(x) or a check()/submit() of a hypothesis
                          already checked; novel = new probe, new hypothesis check.
    Free bookkeeping calls (status, at_goal, try_path, finish) are 'other'."""
    # tools belong to the ENV family, so a derived arm classifies like its parent
    family = _env_family(family) if family in FAMILIES else family
    seen, kinds, acquired = set(), [], set()
    for rec in calls:
        t, kind, touched = rec["tool"], "other", []
        if rec.get("ok") is False:
            # the tool raised: no state was acquired, and nothing was re-acquired either
            kinds.append(kind)
            rec["kind"] = kind
            continue
        if family == "knapsack":
            if t == "inspect":
                key = ("i", _arg(rec, "item_id"))
                touched = [key]
                kind = "replay" if key in seen else "novel"
                seen.add(key)
            elif t == "list_items":
                touched = [("l",)]
                kind = "replay" if ("l",) in seen else "novel"
                seen.add(("l",))
            elif t == "take_item":
                touched = [("take", _arg(rec, "item_id"))]
                kind = "novel"
        elif family == "navigation":
            if t == "neighbors":
                nodes = _nav_nodes(rec)
                touched = [("n", n) for n in nodes]
                fresh = [n for n in nodes if ("n", n) not in seen]
                kind = "novel" if fresh else "replay"
                seen.update(touched)
                rec["fresh_nodes"], rec["nodes"] = len(fresh), len(nodes)
            elif t == "probe":
                out = rec.get("out") if isinstance(rec.get("out"), dict) else {}
                if not out.get("ok"):
                    kind = "other"  # budget exhausted or invalid node: reveals nothing
                else:
                    key = ("p", _arg(rec, "node_id"))
                    touched = [key]
                    kind = "replay" if key in seen else "novel"
                    seen.add(key)
            elif t == "move":
                out = rec.get("out") if isinstance(rec.get("out"), dict) else {}
                # the agent already occupies the source node, so returning to it is not
                # novel movement even on the episode's very first move
                src = (rec.get("context") or {}).get("current")
                if src is not None:
                    seen.add(("m", src))
                dst = _arg(rec, "dst")
                touched = [("m", dst)]
                if out.get("ok"):
                    kind = "replay" if ("m", dst) in seen else "novel"
                    seen.add(("m", dst))
                elif out.get("reason") == "trap":
                    # not a refused move: the agent really stepped onto dst, learned it is
                    # a trap and where it teleports, and now occupies the far node. First
                    # time through, that is acquisition; a repeat is replay.
                    fresh = ("m", dst) not in seen or (
                        "m",
                        out.get("teleport_to"),
                    ) not in seen
                    kind = "novel" if fresh else "replay"
                    seen.update((("m", dst), ("m", out.get("teleport_to"))))
                else:
                    kind = "other"
        elif family == "rule_diagnosis":
            if t == "test_input":
                key = ("t", _arg(rec, "x"))
                touched = [key]
                kind = "replay" if key in seen else "novel"
                seen.add(key)
            elif t in ("check", "submit"):
                out = rec.get("out") if isinstance(rec.get("out"), dict) else {}
                if out.get("status") == "error":
                    kind = "other"  # schema rejected: the oracle told the agent nothing
                else:
                    # the recorder's canonical key, so re-checking one hypothesis spelled
                    # differently counts as the repeat it is
                    key = ("h", rec.get("hyp_key") or repr(_arg(rec, "hypothesis")))
                    touched = [key]
                    kind = "replay" if key in seen else "novel"
                    seen.add(key)
        kinds.append(kind)
        rec["kind"] = kind
        # the entities this call touched, in the family's own acquisition unit -- a
        # neighbors() call covering k nodes touches k of them, not one
        rec["entities"] = touched
        rec["new_entities"] = [k for k in touched if k not in acquired]
        acquired.update(touched)
    return kinds


# ------------------------------------------------------------------ reporting
# Every printed table is also accumulated here so --json can hand the numbers to
# scripts/build_paper_numbers.py without replaying the traces a second time.
RESULTS: dict[str, dict] = {}


def _emit(section, key, payload):
    RESULTS.setdefault(section, {})[key] = payload
    return payload


def _json_safe(value):
    """Undefined metrics are NaN in the tables; JSON has no NaN, so they become null."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(x, n=2):
    return (
        "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"
    )


def _episode_tokens(ep):
    summary = _load(ep["trace"]).get("summary") or {}
    return (summary.get("token_usage") or {}).get("total_tokens") or 0


def _episode_score(ep):
    """Outcome score as the benchmark recorded it (the replay never re-scores)."""
    result_path = ep["trace"].replace(".trace.json", ".json")
    if not os.path.exists(result_path):
        return float("nan")
    score = (_load(result_path).get("result") or {}).get("score")
    return float("nan") if score is None else float(score)


def section0(data):
    print(
        "=== 0. replay fidelity (recorded vs replayed error state, per executed turn) ==="
    )
    print(
        f"  {'cell':16} {'episodes':>8} {'turns':>6} {'no code':>8} {'diverged':>9} "
        f"{'dropped':>8}"
    )
    for (fam, cell, cap), eps in sorted(data.items()):
        turns = sum(len(e["turns"]) for e in eps)
        row = _emit(
            "fidelity",
            f"{SHORT[fam]}_{cell}_{cap}",
            {
                "episodes": len(eps),
                "turns": turns,
                "nocode": sum(e["nocode"] for e in eps),
                "diverged": sum(e["diverged"] for e in eps),
                "dropped_unfaithful": sum(1 for e in eps if not e["faithful"]),
            },
        )
        print(
            f"  {SHORT[fam] + ' ' + cell + '@' + str(cap):16} {row['episodes']:>8} {row['turns']:>6} "
            f"{row['nocode']:>8} {row['diverged']:>9} {row['dropped_unfaithful']:>8}"
        )
    print(
        "  divergence = the replayed block errored where the recording did not, or vice versa;\n"
        "  expected only where a block's control flow depends on a non-deterministic reply (rule's\n"
        "  check() draws its witness at random). 'no code' turns emitted no fenced block, so there\n"
        "  is nothing to execute or compare. 'dropped' = episodes where a failing check() fell back\n"
        "  on a random witness, making every later call counterfactual; they are excluded from every\n"
        "  table below, and this one reports them.\n"
    )


def _nominal_Rc(family):
    """The paper's a-priori replay-demand proxy R/c, for contrast with measured exposure:
    knapsack R = inspect_budget, rule R = probe_budget, navigation R = ceil(n/50)."""
    vals = []
    for f in sorted(glob.glob(f"{FAMILIES[family]['tasks']}/*.json")):
        pub = _load(f).get("public", {})
        if family == "knapsack":
            vals.append(pub["inspect_budget"])
        elif family == "rule_diagnosis":
            vals.append(pub["probe_budget"])
        else:
            # navigation: the map rebuilds in ceil(n / batch) calls, so the intervention
            # arm's nominal demand is n itself
            vals.append(math.ceil(pub["n"] / pub.get("neighbors_batch_max", 50)))
    return (statistics.mean(vals) / CAP_BIND) if vals else float("nan")


def section1(data):
    print(f"=== 1. cap exposure measured on the SLACK-cap (c={CAP_SLACK}) runs ===")
    print(
        f"  Would a c={CAP_BIND} cap have split the same action? K_t = executed tool calls per block."
    )
    print(
        f"  {'cell':16} {'turns':>6} {'mean K':>7} {'median':>7} {'max':>5} "
        f"{'E_25':>6} {'E_25/ep':>8} {'O_25 >=':>8} {'share >=':>9} {'censored':>9}"
    )
    for (fam, cell, cap), eps in sorted(data.items()):
        if cap != CAP_SLACK:
            continue
        # K_t is defined per ACTION BLOCK, so a turn that emitted no code is not a
        # zero-call action -- it is not an action at all, and counting it would dilute
        # every exposure statistic below
        turns = [t for e in eps for t in e["turns"] if t["executed"]]
        ks = [t["k"] for t in turns]
        if not ks:
            continue
        # A turn that hit the c=80 limit is right-censored: its executed k stops at the
        # cap, so O_25 and the deferred share computed from it are LOWER BOUNDS. E_25 is
        # unaffected -- a censored turn already exceeds 25 by construction.
        censored = sum(1 for t in turns if t["caphit"] or t["rec_caphit"])
        over = [max(0, k - CAP_BIND) for k in ks]
        # pooled over blocks (above) weights long episodes more; the inferential unit
        # everywhere else in the paper is the task, so also report the episode-averaged
        # exposure -- mean over episodes of that episode's share of wide blocks
        per_ep = [
            sum(1 for t in e["turns"] if t["executed"] and t["k"] > CAP_BIND)
            / max(sum(1 for t in e["turns"] if t["executed"]), 1)
            for e in eps
        ]
        row = _emit(
            "cap_exposure",
            f"{SHORT[fam]}_{cell}_{cap}",
            {
                "turns": len(ks),
                "mean_K": statistics.mean(ks),
                "median_K": statistics.median(ks),
                "max_K": max(ks),
                "E25": sum(1 for k in ks if k > CAP_BIND) / len(ks),
                "E25_per_episode": statistics.mean(per_ep) if per_ep else float("nan"),
                "O25_lower": statistics.mean(over),
                "deferred_call_share_lower": sum(over) / max(sum(ks), 1),
                "censored_turns": censored,
                "censored_frac": censored / len(ks),
                "Rc_nominal": _nominal_Rc(fam),
            },
        )
        print(
            f"  {SHORT[fam] + ' ' + cell + '@' + str(cap):16} {row['turns']:>6} {row['mean_K']:>7.1f} "
            f"{row['median_K']:>7.1f} {row['max_K']:>5} {row['E25']:>6.2f} "
            f"{row['E25_per_episode']:>8.2f} {row['O25_lower']:>8.1f} "
            f"{row['deferred_call_share_lower']:>9.2f} "
            f"{row['censored_turns']:>4} ({row['censored_frac'] * 100:>2.0f}%)"
        )
    print(
        "  E_25 = Pr(K_t > 25); O_25 = E[(K_t-25)+]; share = fraction of all executed calls that a\n"
        "  25-call cap would have deferred out of their turn. Turns that hit the c=80 limit are\n"
        "  right-censored (true demand >= 80), so O_25 and the share are lower bounds; E_25 is exact\n"
        "  because a censored turn exceeds 25 either way. R/c is the paper's a-priori proxy, shown for\n"
        "  contrast: it is NOT what the agents' realized per-turn demand looks like.\n"
    )


def section2(data):
    print(
        "=== 2. replay vs novel progress, both caps (common table across families) ==="
    )
    print(
        f"  {'cell':16} {'calls/ep':>8} {'replay':>7} {'novel/turn':>10} {'dry turns':>9} "
        f"{'resume gap':>10} {'caphit%':>8} {'(rec)':>6} {'turns':>6} {'tokens/ep':>10} {'score':>6}"
    )
    for cap_group in (CAP_BIND, CAP_SLACK):
        _section2_block(data, cap_group)
    print(
        "  replay = repeated-call fraction of classified calls; dry turns = turns acquiring nothing\n"
        "  novel; resume gap = turns from a cap-hit turn until novel progress resumes, with '+Nc' the\n"
        "  cap hits after which progress never resumed (right-censored, excluded from the mean rather\n"
        "  than scored as the remaining episode length); (rec) is the cap-hit rate in the ORIGINAL\n"
        "  recording, a cross-check on the replay.\n"
    )


def _section2_block(data, cap_group):
    for (fam, cell, cap), eps in sorted(data.items()):
        if cap != cap_group:
            continue
        rows = []
        for e in eps:
            classify(fam, e["calls"])
            per_turn_novel = Counter()
            for rec in e["calls"]:
                if rec["kind"] == "novel":
                    per_turn_novel[rec["turn"]] += 1
            n_turns = len(e["turns"]) or 1
            nrep = sum(1 for c in e["calls"] if c["kind"] == "replay")
            nnov = sum(1 for c in e["calls"] if c["kind"] == "novel")
            dry = [i for i in range(n_turns) if per_turn_novel[i] == 0]
            # Turns until novel progress resumes after a turn that hit the cap. A cap hit
            # with no later novel call is RIGHT-CENSORED -- the episode ended before
            # progress was observed to resume -- so it is counted, not folded into the
            # mean as if the remaining turns were the true gap.
            gaps, censored_gaps = [], 0
            for i, t in enumerate(e["turns"]):
                if not t["caphit"]:
                    continue
                nxt = next(
                    (j for j in range(i + 1, n_turns) if per_turn_novel[j] > 0), None
                )
                if nxt is None:
                    censored_gaps += 1
                else:
                    gaps.append(nxt - i)
            rows.append(
                {
                    "calls": len(e["calls"]),
                    "replay": nrep / max(nrep + nnov, 1),
                    "novel_per_turn": nnov / n_turns,
                    "dry": len(dry) / n_turns,
                    "gap": statistics.mean(gaps) if gaps else None,
                    "gap_censored": censored_gaps,
                    "caphit": sum(1 for t in e["turns"] if t["caphit"]) / n_turns,
                    "caphit_rec": sum(1 for t in e["turns"] if t["rec_caphit"])
                    / n_turns,
                    "turns": n_turns,
                    "tokens": _episode_tokens(e),
                    "score": _episode_score(e),
                }
            )
        if not rows:
            continue

        def m(k, rows=rows):
            """Mean over episodes; 'gap' is undefined for a cell that never hits the cap."""
            vals = [r[k] for r in rows if r[k] is not None]
            return statistics.mean(vals) if vals else float("nan")

        row = _emit(
            "replay_novel",
            f"{SHORT[fam]}_{cell}_{cap}",
            {
                **{
                    k: m(k)
                    for k in (
                        "calls",
                        "replay",
                        "novel_per_turn",
                        "dry",
                        "gap",
                        "caphit",
                        "caphit_rec",
                        "turns",
                        "tokens",
                        "score",
                    )
                },
                "gap_censored": sum(r["gap_censored"] for r in rows),
            },
        )
        print(
            f"  {SHORT[fam] + ' ' + cell + '@' + str(cap):16} {row['calls']:>8.1f} {row['replay']:>7.2f} "
            f"{row['novel_per_turn']:>10.2f} {row['dry']:>9.2f} {_fmt(row['gap'], 1):>7}"
            f"{'+' + str(row['gap_censored']) + 'c':>5} "
            f"{row['caphit'] * 100:>7.0f}% {row['caphit_rec'] * 100:>5.0f}% {row['turns']:>6.1f} "
            f"{row['tokens'] / 1000:>9.0f}k {row['score']:>6.2f}"
        )


def _nav_required_key(task):
    """The lock guarding the gate edge, and the node holding its key."""
    locked = task["private"].get("locked_edges") or []
    if not locked:
        return None, None, None
    row = locked[0]
    lock_id = str(row["lock_id"])
    key_node = next(
        (
            int(k["node"])
            for k in (task["private"].get("key_nodes") or [])
            if str(k["lock_id"]) == lock_id
        ),
        None,
    )
    return lock_id, key_node, (int(row["from"]), int(row["to"]))


def section3(data):
    if not any(_env_family(fam) == "navigation" for fam, _, _ in data):
        return
    print(
        "=== 3. navigation stage decomposition (does the scalar score hide a stage effect?) ==="
    )
    print(
        f"  {'cell':14} {'map%':>6} {'nodes/call':>10} {'keyfound':>9} {'keyheld':>8} {'gate':>6} "
        f"{'goal':>6} {'traps':>6} {'dup pr.':>8} {'spent pr.':>10} {'steps/max':>10} {'score':>6}"
    )
    for (fam, cell, cap), eps in sorted(data.items()):
        if _env_family(fam) != "navigation":
            continue
        rows = []
        for e in eps:
            env, task = e["env"], e["task"]
            lock_id, key_node, gate = _nav_required_key(task)
            classify(fam, e["calls"])
            mapped, probed, visited = set(), set(), set()
            traps = dup = spent = 0
            nodes_per_call = []
            for rec in e["calls"]:
                if rec["tool"] == "neighbors":
                    # a rejected batch (e.g. more than the environment's 50-node limit)
                    # returns no adjacency at all, so it maps nothing and is not a batch
                    if rec.get("ok") is False:
                        continue
                    nodes = _nav_nodes(rec)
                    nodes_per_call.append(len(nodes))
                    mapped.update(nodes)
                elif rec["tool"] == "probe":
                    n = _arg(rec, "node_id")
                    out = rec.get("out") if isinstance(rec.get("out"), dict) else {}
                    if not out.get("ok"):
                        spent += 1  # budget gone or bad node: a pure wasted call
                    else:
                        dup += n in probed
                        probed.add(n)
                elif rec["tool"] == "move" and isinstance(rec.get("out"), dict):
                    traps += rec["out"].get("reason") == "trap"
                    if rec["out"].get("ok"):
                        visited.add(_arg(rec, "dst"))
            rows.append(
                {
                    "score": _episode_score(e),
                    "map": len(mapped) / task["public"]["n"],
                    "nodes_per_call": statistics.mean(nodes_per_call)
                    if nodes_per_call
                    else float("nan"),
                    "keyfound": key_node in probed,
                    "keyheld": lock_id in env.keys if lock_id else False,
                    "gate": gate is not None and gate[0] in visited,
                    "goal": env.at_goal(),
                    "traps": traps,
                    "dup": dup,
                    "spent": spent,
                    "steps": env.steps_taken / max(env.max_steps, 1),
                }
            )
        if not rows:
            continue

        def m(k, rows=rows):
            vals = [
                float(r[k])
                for r in rows
                if r[k] is not None and not math.isnan(float(r[k]))
            ]
            return statistics.mean(vals) if vals else float("nan")

        row = _emit(
            "nav_stages",
            f"{SHORT[fam]}_{cell}_{cap}",
            {
                k: m(k)
                for k in (
                    "map",
                    "nodes_per_call",
                    "keyfound",
                    "keyheld",
                    "gate",
                    "goal",
                    "traps",
                    "dup",
                    "spent",
                    "steps",
                    "score",
                )
            },
        )
        print(
            f"  {SHORT[fam] + ' ' + cell + '@' + str(cap):14} {row['map'] * 100:>5.0f}% "
            f"{row['nodes_per_call']:>10.1f} "
            f"{row['keyfound']:>9.2f} {row['keyheld']:>8.2f} {row['gate']:>6.2f} {row['goal']:>6.2f} "
            f"{row['traps']:>6.2f} {row['dup']:>8.2f} {row['spent']:>10.2f} {row['steps']:>10.2f} "
            f"{row['score']:>6.2f}"
        )
    print(
        "  map% = share of nodes whose adjacency the episode pulled; nodes/call = realized neighbors()\n"
        "  batch size (the bandwidth the unbatched intervention removes); keyfound = probed the node\n"
        "  holding the gate key; keyheld = ended holding it; gate = stood on the gate's source node;\n"
        "  dup pr. = re-probes of an already-probed node; spent pr. = probes after the budget ran out.\n"
    )


def section3b(data):
    """Manipulation check for the proposed unbatched-navigation intervention: holding the
    recorded behaviour fixed, recompute K_t as if neighbors() returned ONE node per call.
    This is a design check, not an outcome -- an agent facing that interface would adapt --
    but it says whether the interface change can move the cap exposure at all."""
    if not any(fam == "navigation" for fam, _, _ in data):
        return
    print(
        f"=== 3b. navigation counterfactual on the SLACK-cap (c={CAP_SLACK}) runs: "
        "what if neighbors() were unbatched? ==="
    )
    print(
        f"  {'cell':10} {'turns':>6} {'mean K':>7} {'mean K1':>8} {'E_25':>6} {'E_25 (1/call)':>14} "
        f"{'max K1':>7} {'censored':>9}"
    )
    for (fam, cell, cap), eps in sorted(data.items()):
        # only the slack-cap runs: at c=25 the recorded K_t is truncated at the cap, so
        # both the observed and the counterfactual exposure would be degenerate.
        if fam != "navigation" or cap != CAP_SLACK:
            continue
        ks, k1s, censored = [], [], 0
        for e in eps:
            calls_per_turn, nodes_per_turn = Counter(), Counter()
            for rec in e["calls"]:
                # a rejected batch (over the 50-node limit) revealed no topology, so an
                # unbatched interface would not have replaced it with one call per node
                if rec["tool"] == "neighbors" and rec.get("ok") is not False:
                    calls_per_turn[rec["turn"]] += 1
                    nodes_per_turn[rec["turn"]] += len(_nav_nodes(rec))
            for i, turn in enumerate(e["turns"]):
                if not turn["executed"]:
                    continue  # no action block to expand
                if turn["caphit"] or turn["rec_caphit"]:
                    # only the executed prefix was recorded, so the expansion would be a
                    # lower bound on a lower bound -- leave it out and count it instead
                    censored += 1
                    continue
                ks.append(turn["k"])
                k1s.append(turn["k"] - calls_per_turn[i] + nodes_per_turn[i])
        if not ks:
            continue
        row = _emit(
            "nav_unbatched_counterfactual",
            f"{cell}_{cap}",
            {
                "turns": len(ks),
                "mean_K": statistics.mean(ks),
                "mean_K_unbatched": statistics.mean(k1s),
                "E25": sum(1 for k in ks if k > CAP_BIND) / len(ks),
                "E25_unbatched": sum(1 for k in k1s if k > CAP_BIND) / len(k1s),
                "max_K_unbatched": max(k1s),
                "censored_turns": censored,
            },
        )
        print(
            f"  {cell + '@' + str(cap):10} {row['turns']:>6} {row['mean_K']:>7.1f} "
            f"{row['mean_K_unbatched']:>8.1f} {row['E25']:>6.2f} {row['E25_unbatched']:>14.2f} "
            f"{row['max_K_unbatched']:>7} {row['censored_turns']:>9}"
        )
    print(
        "  K1 = the same turn's calls with each neighbors(nodes) expanded to one call per node.\n"
        "  Turns that hit the c=80 limit are excluded and counted instead: only their executed\n"
        "  prefix was recorded, so expanding it would bound an already-truncated turn. Where max K1\n"
        "  exceeds 80 the unbatched interface would bind at the SLACK cap too, which would attenuate\n"
        "  the intervention's cap contrast rather than sharpen it.\n"
    )


def _score_hypothesis(env, raw):
    """Score an intermediate hypothesis exactly as submit() would, without mutating env."""
    try:
        fam, params, exc = env._normalize_hypothesis(_parse_hypothesis_input(raw))
    except Exception:
        return None
    total = env.x_max - env.x_min + 1
    matched = sum(
        1
        for x in range(env.x_min, env.x_max + 1)
        if env._f(x) == (exc[x] if x in exc else env._base_output(x, fam, params))
    )
    functional = matched / total if total else 0.0
    true_iv = (
        env.params.get("intervals", []) if env.family == "stepwise_composition" else []
    )
    sub_iv = params.get("intervals", []) if fam == "stepwise_composition" else []
    bf1, facc = _compute_structural_metrics(true_iv, sub_iv, env.m)
    penalty = 0.05 * max(0, len(sub_iv) - len(true_iv))
    return {
        "functional": functional,
        "boundary_f1": bf1,
        "family_acc": facc,
        "penalty": penalty,
        "composite": max(
            0.0, min(1.0, 0.60 * functional + 0.25 * bf1 + 0.15 * facc - penalty)
        ),
    }


def section4(data):
    if not any(_env_family(fam) == "rule_diagnosis" for fam, _, _ in data):
        return
    print(
        "=== 4. rule diagnosis: hypothesis-repair trajectory (is check() actually self-healing?) ==="
    )
    print(
        f"  {'cell':14} {'checks/ep':>9} {'witn.pairs':>10} {'H comp':>7} {'H func':>7} {'H bndF1':>8} "
        f"{'fixed':>6} {'n:ok/drop':>9} {'regress':>8} {'new ti':>7} {'rep ti':>7}"
    )
    for (fam, cell, cap), eps in sorted(data.items()):
        if _env_family(fam) != "rule_diagnosis":
            continue
        rows = []
        for e in eps:
            env = FAMILY_ENVS[fam].from_task(e["task"])  # fresh, unmutated scorer
            classify(fam, e["calls"])
            hyps = []  # (index in calls, scores, witness x or None, witness is recorded?)
            for i, rec in enumerate(e["calls"]):
                if rec["tool"] not in ("check", "submit"):
                    continue
                sc = _score_hypothesis(env, _arg(rec, "hypothesis"))
                out = rec.get("out") if isinstance(rec.get("out"), dict) else {}
                failed = out.get("status") == "fail"
                # submit()'s witness is deterministic (first mismatch); check()'s is random,
                # so it only counts as recorded when recovered from the episode's stdout.
                faithful = (
                    rec["tool"] == "submit" or rec.get("witness_source") == "recorded"
                )
                hyps.append(
                    (i, sc, out.get("x") if failed else None, failed and faithful)
                )
            deltas, fixed, regressed, dropped = [], [], [], 0
            for (i, sc, wx, faithful), (j, sc2, _, _) in zip(hyps, hyps[1:]):
                if sc is None or sc2 is None or wx is None:
                    continue
                if not faithful:
                    # The episode saw a witness this replay could not recover, and the
                    # next hypothesis may have been COMPUTED from it -- so the replayed
                    # h_{j+1} need not be the recorded one. Such a transition is dropped
                    # from every witness-conditioned statistic, not just from 'fixed'.
                    dropped += 1
                    continue
                deltas.append((sc, sc2))
                regressed.append(sc2["functional"] < sc["functional"])
                # did the next hypothesis fix the returned counterexample point?
                nfam, nparams, nexc = env._normalize_hypothesis(
                    _parse_hypothesis_input(_arg(e["calls"][j], "hypothesis"))
                )
                pred = nexc[wx] if wx in nexc else env._base_output(wx, nfam, nparams)
                fixed.append(pred == env._f(wx))
            rows.append(
                {
                    "checks": sum(1 for c in e["calls"] if c["tool"] == "check"),
                    "witness": len(deltas),
                    "dropped": dropped,
                    "H": [b["composite"] - a["composite"] for a, b in deltas],
                    "Hf": [b["functional"] - a["functional"] for a, b in deltas],
                    "Hb": [b["boundary_f1"] - a["boundary_f1"] for a, b in deltas],
                    "fixed": fixed,
                    "regress": regressed,
                    "new_ti": sum(
                        1
                        for c in e["calls"]
                        if c["tool"] == "test_input" and c["kind"] == "novel"
                    ),
                    # only calls the classifier actually labelled replay: a test_input the
                    # environment refused is neither novel nor a repeat
                    "rep_ti": sum(
                        1
                        for c in e["calls"]
                        if c["tool"] == "test_input" and c["kind"] == "replay"
                    ),
                }
            )
        if not rows:
            continue

        def flat(k, rows=rows):
            vals = [float(v) for r in rows for v in r[k]]
            return statistics.mean(vals) if vals else float("nan")

        def m(k, rows=rows):
            return statistics.mean(r[k] for r in rows)

        row = _emit(
            "rule_repair",
            f"{cell}_{cap}",
            {
                "checks": m("checks"),
                "witness_pairs": m("witness"),
                "H_composite": flat("H"),
                "H_functional": flat("Hf"),
                "H_boundary_f1": flat("Hb"),
                "fixed_rate": flat("fixed"),
                "scored_n": sum(len(r["fixed"]) for r in rows),
                "dropped_n": sum(r["dropped"] for r in rows),
                "regress_rate": flat("regress"),
                "new_test_input": m("new_ti"),
                "repeat_test_input": m("rep_ti"),
            },
        )
        print(
            f"  {cell + '@' + str(cap):14} {row['checks']:>9.1f} {row['witness_pairs']:>10.1f} "
            f"{row['H_composite']:>+7.3f} {row['H_functional']:>+7.3f} {row['H_boundary_f1']:>+8.3f} "
            f"{row['fixed_rate']:>6.2f} {row['scored_n']:>4}/{row['dropped_n']:<4} "
            f"{row['regress_rate']:>8.2f} {row['new_test_input']:>7.1f} "
            f"{row['repeat_test_input']:>7.1f}"
        )
    print(
        "  H = E[score(h_{j+1}) - score(h_j) | a counterexample was returned]; 'fixed' = the next\n"
        "  hypothesis predicts f(x) correctly at that witness. A transition counts only when the\n"
        "  episode's own witness was recoverable from its stdout and the replay was forced to return\n"
        "  it: where it was not, the next hypothesis may have been computed from a witness this replay\n"
        "  never saw, so the transition is dropped from ALL of these statistics. n = scored/dropped.\n"
    )


def section5(data):
    print("=== 5. operational currency: P->S relative to P->P, per family and cap ===")
    print(
        f"  {'family@cap':14} {'tokens':>7} {'calls':>7} {'turns':>7} {'dupfrac PS/PP':>14} "
        f"{'NameErr PS':>11} {'quality gap':>12}"
    )
    for fam in FAMILIES:
        for cap in (CAP_BIND, CAP_SLACK):
            cells = {}
            for cl in ("PP", "PS"):
                eps = data.get((fam, cl, cap))
                if not eps:
                    continue
                agg = []
                for e in eps:
                    classify(fam, e["calls"])
                    classified = [
                        c for c in e["calls"] if c["kind"] in ("novel", "replay")
                    ]
                    agg.append(
                        {
                            "tokens": _episode_tokens(e),
                            "calls": len(e["calls"]),
                            "turns": len(e["turns"]) or 1,
                            "dup": sum(1 for c in classified if c["kind"] == "replay")
                            / max(len(classified), 1),
                            "nameerr": sum(
                                1 for t in e["turns"] if "NameError" in t["rec_err"]
                            ),
                            "score": _episode_score(e),
                        }
                    )
                cells[cl] = agg
            if len(cells) != 2:
                continue

            def mean(cl, k, cells=cells):
                return statistics.mean(r[k] for r in cells[cl])

            def ratio(k, mean=mean):
                base = mean("PP", k)
                return mean("PS", k) / base if base else float("nan")

            row = _emit(
                "op_currency",
                f"{SHORT[fam]}_{cap}",
                {
                    "token_ratio": ratio("tokens"),
                    "call_ratio": ratio("calls"),
                    "turn_ratio": ratio("turns"),
                    "dup_PS": mean("PS", "dup"),
                    "dup_PP": mean("PP", "dup"),
                    "nameerr_PS": mean("PS", "nameerr"),
                    "quality_gap": mean("PP", "score") - mean("PS", "score"),
                },
            )
            print(
                f"  {SHORT[fam] + '@' + str(cap):14} {row['token_ratio']:>6.1f}x {row['call_ratio']:>6.1f}x "
                f"{row['turn_ratio']:>6.1f}x {row['dup_PS']:>6.2f}/{row['dup_PP']:<7.2f} "
                f"{row['nameerr_PS']:>11.2f} {row['quality_gap']:>+12.3f}"
            )
    print(
        "  A ratio >> 1 with a ~0 quality gap is the mismatch being paid in the EFFICIENCY currency.\n"
    )


async def section6(families):
    """Announced cap vs enforced cap, on the one block where they can be told apart."""
    print(
        "=== 6. first-block intended width: does the ANNOUNCED cap change the plan? ==="
    )
    print(
        f"  {'arm/cell':16} {'n':>3} {'K_0 @25-prompt':>15} {'K_0 @80-prompt':>15} "
        f"{'paired diff':>12} {'95% CI':>18}"
    )
    for fam in families:
        for cell in ("PS", "PP"):
            persistent = cell[1] == "P"
            widths = {}
            for cap in (CAP_BIND, CAP_SLACK):
                for trace in _traces(fam, cell, cap):
                    idx = os.path.basename(trace).split("-")[-1].split(".")[0]
                    k = await first_block_intent(fam, trace, persistent)
                    if k is not None:
                        widths.setdefault(idx, {})[cap] = k
            paired = [
                (v[CAP_BIND], v[CAP_SLACK]) for v in widths.values() if len(v) == 2
            ]
            if not paired:
                continue
            diffs = [b - s for b, s in paired]
            random.seed(0)
            n = len(diffs)
            boots = sorted(
                sum(diffs[random.randrange(n)] for _ in range(n)) / n
                for _ in range(5000)
            )
            lo, hi = boots[125], boots[4874]
            row = _emit(
                "first_block_intent",
                f"{SHORT[fam]}_{cell}",
                {
                    "n": n,
                    "mean_K0_bind_prompt": statistics.mean(b for b, _ in paired),
                    "mean_K0_slack_prompt": statistics.mean(s for _, s in paired),
                    "mean_diff": statistics.mean(diffs),
                    "ci_lo": lo,
                    "ci_hi": hi,
                },
            )
            print(
                f"  {SHORT[fam] + ' ' + cell:16} {row['n']:>3} "
                f"{row['mean_K0_bind_prompt']:>15.1f} {row['mean_K0_slack_prompt']:>15.1f} "
                f"{row['mean_diff']:>+12.1f} {'[' + f'{lo:+.1f},{hi:+.1f}' + ']':>18}"
            )
    print(
        "  K_0 = calls the episode's FIRST block makes when replayed with the cap REMOVED, so it\n"
        "  is the plan the announced number induced, before any cap-produced observation could\n"
        "  reach the model. A difference near 0 means the two prompts induce the same opening\n"
        "  plan, and what separates the cells afterwards is enforcement, not the integer.\n"
    )


_IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")


def _is_index_rederivation(arm, rec):
    """Does this call re-derive the policy's index over the task, rather than ask the one
    local question a continuing policy needs anyway?

    Knapsack is unambiguous: `list_items` enumerates the catalogue and has no local
    variant, so re-issuing it after an interruption IS starting over.

    Navigation is not. `neighbors` doubles as the mapping sweep and as the lookup an agent
    standing on a node makes before it moves -- called with no argument it queries exactly
    the current node (navigation.py). Counting the latter as a re-derivation would label a
    policy that continues normally (query where I am, then move somewhere new) as
    restarting, so a call that asks about nothing beyond the node the agent occupies does
    not count. What remains is re-querying adjacency elsewhere: the sweep."""
    fam = _env_family(arm)
    if fam == "knapsack":
        return rec["tool"] == "list_items"
    if fam == "navigation":
        if rec["tool"] != "neighbors":
            return False
        current = (rec.get("context") or {}).get("current")
        here = {current} if current is not None else set()
        return bool(set(_nav_nodes(rec)) - here)
    return False


def _instance_of(ep, pos):
    """The task instance an episode ran, used to cluster the bootstrap below.

    Falls back on the episode's position when there is no trace path, so an episode built
    by hand (a unit test) still gets a cluster of its own."""
    trace = ep.get("trace")
    return os.path.basename(trace).split("-")[-1].split(".")[0] if trace else str(pos)


def resume_stats(family, episodes):
    """After the cap truncates a block, does the next turn CONTINUE or START OVER?

    Exposure says whether the cap intervenes; it does not say whether the interruption is
    survivable. That depends on what the policy does next, and only a policy that resumes
    can benefit from state a persistent runtime preserved. So for every turn that hit the
    cap we look at the next executed turn and ask what its first classified call is:

      resume     its first classified call acquires something NEW -- the turn extends the
                 working set instead of rebuilding it
      restart    it re-acquires already-held state first, and `prefix_replay` counts how
                 many such calls precede the first novel one

    `refs_surviving` is meaningful only under a persistent runtime, where names do survive:
    it asks whether the next block even mentions one of them. It is an identifier scan, so
    it shows the block NAMED a surviving binding, not that it used the value -- the
    behavioural resume rate above needs no such parsing."""
    rows = []
    for pos, ep in enumerate(episodes):
        classify(family, ep["calls"])
        # the instance this episode ran, so the differential below can resample TASKS
        # rather than cap-hit events -- events within one episode are not independent
        inst = _instance_of(ep, pos)
        by_turn, recs_by_turn = {}, {}
        for rec in ep["calls"]:
            if rec["kind"] in ("novel", "replay"):
                by_turn.setdefault(rec["turn"], []).append(rec["kind"])
                recs_by_turn.setdefault(rec["turn"], []).append(rec)
        turns = ep["turns"]
        for i, turn in enumerate(turns):
            if not turn["caphit"]:
                continue
            nxt = next(
                (j for j in range(i + 1, len(turns)) if turns[j]["executed"]), None
            )
            if nxt is None:
                continue  # the episode ended on the truncated turn: nothing to resume into
            kinds = by_turn.get(nxt, [])
            prefix = next((x for x, k in enumerate(kinds) if k == "novel"), len(kinds))
            surviving = set(turns[i]["surviving"])
            pre_novel = recs_by_turn.get(nxt, [])[:prefix]
            # Entities, not calls: one neighbors() call can acquire many nodes while a
            # probe acquires one, so counting calls would compare unlike units and would
            # let replayed moves inflate a "topology rebuilt" figure. `held` is what the
            # episode had acquired before this turn; `refetched` is how much of that the
            # turn re-acquires before touching anything new.
            held_keys = {
                k
                for j in range(nxt)
                for r in recs_by_turn.get(j, [])
                for k in r["new_entities"]
            }
            refetched = {k for r in pre_novel for k in r["entities"]} & held_keys
            row = {
                "instance": inst,
                "resume": bool(kinds) and kinds[0] == "novel",
                "prefix_replay": prefix,
                "held": len(held_keys),
                "refetched": len(refetched),
                "refetch_frac": (
                    len(refetched) / len(held_keys) if held_keys else float("nan")
                ),
                "re_enumerates": any(
                    _is_index_rederivation(family, r) for r in pre_novel
                ),
                "acted": bool(kinds),
            }
            if surviving:
                row["refs_surviving"] = bool(
                    surviving & set(_IDENT_RE.findall(turns[nxt]["code"]))
                )
            rows.append(row)
    return rows


def terminal_caphits(episodes):
    """Cap hits the episode never got to answer: no executed turn follows them.

    `resume_stats` cannot score these -- there is no next turn to classify -- so every
    quantity built on it, `R_c` included, is conditional on a post-cap TRANSITION and not
    merely on a cap hit. The two differ, and they can differ by different amounts in the
    two cells (a mismatched episode that exhausts its turn budget is likely to end ON a
    truncated turn), so we count them and report them rather than let the conditioning
    hide behind a column labelled `cap hits`."""
    n = 0
    for ep in episodes:
        turns = ep["turns"]
        for i, turn in enumerate(turns):
            if turn["caphit"] and not any(t["executed"] for t in turns[i + 1 :]):
                n += 1
    return n


def _by_instance(rows):
    out = {}
    for r in rows:
        out.setdefault(r["instance"], []).append(r)
    return out


def _rmean(rows, key):
    vals = [
        float(r[key])
        for r in rows
        if key in r and not (isinstance(r[key], float) and math.isnan(r[key]))
    ]
    return statistics.mean(vals) if vals else float("nan")


def _resume_contrast(ps, pp):
    """PS-minus-PP for the restarting measures, PP-minus-PS for the progress one, so a
    positive number always means 'the matched runtime continues better'."""
    return {
        "Rc": _rmean(ps, "re_enumerates") - _rmean(pp, "re_enumerates"),
        "dresume": _rmean(pp, "resume") - _rmean(ps, "resume"),
        "dprefix": _rmean(ps, "prefix_replay") - _rmean(pp, "prefix_replay"),
    }


def resume_delta(
    rows_ps, rows_pp, cohort=None, resamples=5000
) -> dict[str, Any] | None:
    """Gate 2 as one number: how much MORE the mismatched cell restarts after a truncation.

    Gate 2 is a claim about a DIFFERENCE between two runtimes, so the matched cell's
    restart rate cannot carry it on its own: at b=2 both cells restart, and only the
    contrast says that neither is advantaged. Hence

        R_c = Pr(re-derives the index | P->S, post-cap transition)
              - Pr(same | P->P, post-cap transition),

    conditional on the TRANSITION and not merely on the cap hit, because a cap hit the
    episode never answers cannot be scored (see `terminal_caphits`), with `dresume` the
    same contrast on whether that turn acquires anything novel and `dprefix` on how many
    already-held calls precede its first novel one.

    The interval resamples TASK INSTANCES, not transitions: one episode contributes many
    and they share whatever that episode's policy decided, so treating them as
    independent would understate it. `cohort` is the set of instances that RAN, and it is
    what gets resampled -- an instance that produced no transition in either cell is still
    a draw that contributes nothing, and dropping it would condition the interval on
    having produced one. It matters most exactly where the estimate is weakest: at
    navigation, 16 instances ran and 9 produced a transition. Resamples in which either
    cell draws nothing leave the contrast undefined and are counted, not silently
    dropped."""
    ps_by, pp_by = _by_instance(rows_ps), _by_instance(rows_pp)
    insts = sorted(set(cohort) if cohort is not None else (set(ps_by) | set(pp_by)))
    if not insts or not rows_ps or not rows_pp:
        return None
    point = _resume_contrast(rows_ps, rows_pp)
    random.seed(0)
    n = len(insts)
    boots = {k: [] for k in point}
    undefined = 0
    for _ in range(resamples):
        draw = [insts[random.randrange(n)] for _ in range(n)]
        ps = [r for i in draw for r in ps_by.get(i, ())]
        pp = [r for i in draw for r in pp_by.get(i, ())]
        if not ps or not pp:
            undefined += 1
            continue
        for k, v in _resume_contrast(ps, pp).items():
            if v == v:
                boots[k].append(v)
    out: dict[str, Any] = {
        "n_ps": len(rows_ps),
        "n_pp": len(rows_pp),
        # instances RESAMPLED (everything that ran) vs instances that produced a scorable
        # transition -- when these differ, the second is a post-treatment subset and only
        # the first is a valid cluster universe
        "n_instances": n,
        "n_instances_with_events": len(set(ps_by) | set(pp_by)),
        "undefined_resamples": undefined,
    }
    for k, v in point.items():
        b = sorted(boots[k])
        out[k] = v
        # same percentile convention as every other interval here: for 5000 defined
        # resamples this is exactly boots[125] and boots[4874]
        out[f"{k}_ci"] = (
            [b[int(0.025 * len(b))], b[int(0.975 * len(b)) - 1]]
            if len(b) >= 100
            else None
        )
    return out


def section7(data):
    """Gate 2 of the mechanism: differential resumability."""
    print(
        "=== 7. after the cap truncates a block, does the next turn resume or restart? ==="
    )
    print(
        f"  {'cell':16} {'cap hits':>9} {'re-derives':>11} {'prefix calls':>13} "
        f"{'re-fetched':>11} {'/held':>7} {'frac':>6} {'resume':>7} {'refs surviving':>15}"
    )
    for (fam, cell, cap), eps in sorted(data.items()):
        rows = resume_stats(fam, eps)
        if not rows:
            continue

        def mean(key, rows=rows):
            vals = [
                float(r[key])
                for r in rows
                if key in r and not (isinstance(r[key], float) and math.isnan(r[key]))
            ]
            return statistics.mean(vals) if vals else float("nan")

        refs = [r["refs_surviving"] for r in rows if "refs_surviving" in r]
        row = _emit(
            "resume",
            f"{SHORT[fam]}_{cell}_{cap}",
            {
                "n": len(rows),
                "resume_rate": mean("resume"),
                "re_enumerates": mean("re_enumerates"),
                "prefix_replay": mean("prefix_replay"),
                "refetch_frac": mean("refetch_frac"),
                "held": mean("held"),
                "refetched": mean("refetched"),
                "refs_surviving": statistics.mean(map(float, refs)) if refs else None,
                "refs_n": len(refs),
            },
        )
        refs_col = (
            f"{_fmt(row['refs_surviving'], 2)} (n={row['refs_n']})"
            if refs
            else "n/a (reset)"
        )
        print(
            f"  {SHORT[fam] + ' ' + cell + '@' + str(cap):16} {row['n']:>9} "
            f"{row['re_enumerates']:>11.2f} {row['prefix_replay']:>13.1f} "
            f"{row['refetched']:>11.1f} {row['held']:>7.1f} "
            f"{_fmt(row['refetch_frac'], 2):>6} {row['resume_rate']:>7.2f} {refs_col:>15}"
        )
    print(
        "  re-derives = the pre-novel prefix re-derives the policy's INDEX over the task (knapsack\n"
        "  list_items; navigation a neighbors() sweep of nodes it is not standing on, since the\n"
        "  one-node lookup before a move is what continuing looks like) -- the signature of starting\n"
        "  over rather than continuing;\n"
        "  resume = the next turn's first classified call acquires something new (strict, and sensitive\n"
        "  to one cheap re-issued call at the top of a turn);\n"
        "  prefix calls = calls issued before that first novel one; re-fetched and held count ENTITIES\n"
        "  in each family's own acquisition unit (items, nodes, probe points) -- one neighbors() call\n"
        "  covers many nodes -- so frac is the share of already-acquired state the turn re-acquires\n"
        "  before touching anything new. refs surviving applies only where a persistent runtime left\n"
        "  bindings alive, and is an identifier scan (named, not necessarily used).\n"
    )

    # The differential the two-gate claim actually rests on. Printed after the levels
    # because it is a function of them: a high MATCHED restart rate is not by itself a
    # failed gate 2 (nav b=2), and a low one is not by itself a passed gate 2 -- only the
    # contrast separates mismatch-specific failure from generic cap failure.
    print(
        "  --- gate 2 as a contrast (P->S minus P->P, positive = matched continues better)"
    )
    print(
        f"  {'family':16} {'R_c (re-derives)':>26} {'d resume':>22} {'d prefix calls':>24}"
    )
    for fam in sorted({k[0] for k in data}):
        ps = data.get((fam, "PS", CAP_BIND))
        pp = data.get((fam, "PP", CAP_BIND))
        if not ps or not pp:
            continue
        rows_ps, rows_pp = resume_stats(fam, ps), resume_stats(fam, pp)
        # the cluster universe is every instance that RAN, including those that produced
        # no scorable transition: resampling only the productive ones would condition the
        # interval on having produced one
        cohort = {_instance_of(e, i) for i, e in enumerate(ps)} | {
            _instance_of(e, i) for i, e in enumerate(pp)
        }
        term = {
            "terminal_ps": terminal_caphits(ps),
            "terminal_pp": terminal_caphits(pp),
        }
        if not rows_ps or not rows_pp:
            # no scorable transition in one of the cells: gate 2 is not measurable, not zero
            _emit(
                "resume_delta",
                f"{SHORT[fam]}_{CAP_BIND}",
                {
                    "n_ps": len(rows_ps),
                    "n_pp": len(rows_pp),
                    "measurable": False,
                    **term,
                },
            )
            print(
                f"  {SHORT[fam]:16} {'-- (transitions ' + str(len(rows_ps)) + '/' + str(len(rows_pp)) + ')':>26}"
            )
            continue
        d = resume_delta(rows_ps, rows_pp, cohort=cohort)
        assert d is not None  # both cells have rows, so the contrast is defined
        row = _emit(
            "resume_delta",
            f"{SHORT[fam]}_{CAP_BIND}",
            {**d, **term, "measurable": True},
        )

        def est(key, row=row):
            ci = row.get(f"{key}_ci")
            span = f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else "[--]"
            return f"{row[key]:+.2f} {span}"

        print(
            f"  {SHORT[fam]:16} {est('Rc'):>26} {est('dresume'):>22} "
            f"{est('dprefix'):>24}   n={row['n_ps']}/{row['n_pp']} transitions "
            f"(+{row['terminal_ps']}/{row['terminal_pp']} unanswered), "
            f"{row['n_instances']} instances resampled, "
            f"{row['n_instances_with_events']} with events"
        )
    print(
        "  R_c = Pr(re-derives the index | P->S, post-cap transition) - Pr(same | P->P). It is\n"
        "  conditional on the TRANSITION, not on the cap hit: 'unanswered' counts cap hits the\n"
        "  episode never got to answer (no executed turn follows), which cannot be scored.\n"
        "  Intervals resample TASK INSTANCES -- every instance that ran, not only those that\n"
        "  produced a transition -- since one episode contributes many correlated transitions.\n"
    )


# ------------------------------------------------------------------ self-check
def selfcheck(data, requested):
    """The knapsack PS@25 replay must reproduce the published anchors before any new
    number here is trusted (main.tex: 681 executed inspects, 25.7 distinct items)."""
    eps = data.get(("knapsack", "PS", CAP_BIND))
    if not eps:
        if "knapsack" in requested:  # _require_complete should already have caught this
            raise SystemExit("self-check impossible: knapsack PS@25 has no episodes")
        print(
            "=== self-check: not run (knapsack was not among the requested families)\n"
        )
        return
    ins = statistics.mean(
        sum(1 for c in e["calls"] if c["tool"] == "inspect") for e in eps
    )
    uniq = statistics.mean(
        len({_arg(c, "item_id") for c in e["calls"] if c["tool"] == "inspect"})
        for e in eps
    )
    ok = abs(ins - 681) <= 35 and abs(uniq - 25.7) <= 2.0
    _emit(
        "selfcheck",
        "knap_PS_25",
        {"inspects_per_ep": ins, "distinct_per_ep": uniq, "ok": ok},
    )
    print("=== self-check against published knapsack replay anchors ===")
    print(
        f"  executed inspects/ep {ins:.1f} (paper 681)   distinct items/ep {uniq:.1f} (paper 25.7)"
        f"   -> {'OK' if ok else 'MISMATCH'}\n"
    )
    if not ok:
        raise SystemExit(
            "self-check failed: replay does not reproduce the published anchors"
        )


async def run_all(families=None, json_path=None, verbose=True):
    """Replay every cell of the requested families and print all six tables.

    Importable so scripts/analyze_paper.py can reproduce these numbers alongside the
    paper's other estimands (the whole sweep replays in well under a minute)."""
    requested = families or list(FAMILIES)
    unknown = [f for f in requested if f not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}; known: {list(FAMILIES)}")
    # a second call in the same process must not inherit the first call's tables
    RESULTS.clear()
    data = {}
    for fam in requested:
        for cap in (CAP_BIND, CAP_SLACK):
            for cl in ("PP", "PS"):
                eps = await replay_cell(fam, cl, cap)
                if eps:
                    data[(fam, cl, cap)] = eps
                    if verbose:
                        print(
                            f"  [replayed] {fam} {cl}@{cap}: {len(eps)} episodes",
                            flush=True,
                        )
    print()
    _require_complete(data, requested)
    selfcheck(data, requested)
    section0(data)  # fidelity reports on every episode, faithful or not
    # every other table runs on the tasks that replayed faithfully in ALL cells of their
    # family, keeping the four cells on one cohort
    faithful = _paired_faithful(data, requested)
    section1(faithful)
    section2(faithful)
    section3(faithful)
    section3b(faithful)
    section4(faithful)
    section5(faithful)
    section7(faithful)
    await section6([f for f in requested if any(k[0] == f for k in data)])
    if json_path:
        with open(json_path, "w") as fh:
            # allow_nan=False + explicit nulls: an undefined metric (e.g. the resume gap
            # in a cell that never hits the cap) must not become the non-standard NaN
            # token that strict JSON parsers reject.
            json.dump(
                _json_safe(RESULTS), fh, indent=1, sort_keys=True, allow_nan=False
            )
        print(f"  [wrote] {json_path}")
    return RESULTS


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument(
        "--json", default=None, help="also write every table to this JSON path"
    )
    args = ap.parse_args()
    asyncio.run(run_all(args.families.split(","), args.json))


if __name__ == "__main__":
    _main()
