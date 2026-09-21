#!/usr/bin/env python3
"""Paper re-analyses. Reproduces every
number in the paper from experiments/cap_sweep/ (no GPU, no new runs).

Estimands (paired over the common task ids of the four cells):
  G_P(c) = PP-PS                          persistent model's runtime gap
  D(c)   = (PP-PS)-(SP-SS)                train x runtime interaction at cap c
  T      = D(c_bind)-D(c_slack)           train x runtime x cap moderation (confirmatory)
  A_P    = G_P(bind)-G_P(slack)           budget-specific amplification (cross-family, sec 5)
Cells XY = X-trained adapter on Y runtime, X,Y in {P,S}.

Sections: A1 (interaction T), A2 (operational metrics / confound rebuttal), A4 (model-
selection flip), A5 (dense log(c) trend), A7/A8 (mechanism triangulation + per-instance R),
A11 (measured cross-family mechanism tables: cap exposure, replay vs novel progress,
navigation stages, rule hypothesis repair, operational currency).
"""

import asyncio
import glob
import json
import os
import re
import statistics
import random
import math
from collections import Counter

random.seed(0)

CAP_BIND, CAP_SLACK, MAXT, KFAM = 25, 80, 40, "knapsack"
SEEDS = ["0", "777", "1337"]


# ---------- loading ----------
def _scores(path, fam, metric="score"):
    out = {}
    for f in glob.glob(f"{path}/**/{fam}-{fam}-*.json", recursive=True):
        if f.endswith(".trace.json"):
            continue
        r = json.load(open(f)).get("result", {}) or {}
        v = (
            r.get("score")
            if metric == "score"
            else (r.get("metrics") or {}).get(metric)
        )
        if v is not None:
            out[os.path.basename(f)] = float(v)
    return out


def knap_dir(seed, cl, cap):
    return (
        f"experiments/cap_sweep/knapsack/qwen3_8b/main/{cl}_cap{cap}"
        if seed == "0"
        else f"experiments/cap_sweep/knapsack/qwen3_8b/seeds/s{seed}_{cl}_cap{cap}"
    )


def knap_cells(seed, cap):
    c = {cl: _scores(knap_dir(seed, cl, cap), KFAM) for cl in ("PP", "PS", "SP", "SS")}
    if not all(c.values()):
        return None
    common = sorted(set.intersection(*[set(v) for v in c.values()]))
    return {cl: [c[cl][k] for k in common] for cl in c}, common


def _bootci(vals, nb=5000):
    n = len(vals)
    b = sorted(sum(vals[random.randrange(n)] for _ in range(n)) / n for _ in range(nb))
    return b[int(0.025 * nb)], b[int(0.975 * nb)]


# ---------- A1: per-seed G_P/D/T + hierarchical bootstrap ----------
def a1():
    rows = {}
    for s in SEEDS:
        b, sl = knap_cells(s, CAP_BIND), knap_cells(s, CAP_SLACK)
        if not b or not sl:
            continue
        (cb, _), (cs, _) = b, sl

        def m(d, cl):
            return statistics.mean(d[cl])

        Db = (m(cb, "PP") - m(cb, "PS")) - (m(cb, "SP") - m(cb, "SS"))
        Ds = (m(cs, "PP") - m(cs, "PS")) - (m(cs, "SP") - m(cs, "SS"))
        rows[s] = dict(
            GPb=m(cb, "PP") - m(cb, "PS"),
            GPs=m(cs, "PP") - m(cs, "PS"),
            Db=Db,
            Ds=Ds,
            T=Db - Ds,
            cb=cb,
            cs=cs,
        )
    print("=== A1: per-seed G_P, D, T (knapsack) ===")
    print(
        f"  {'seed':>5} {'GP_bind':>8} {'GP_slk':>7} {'A_P':>7} {'D_bind':>7} {'D_slk':>7} {'T':>7}"
    )
    for s, r in rows.items():
        print(
            f"  {s:>5} {r['GPb']:>8.3f} {r['GPs']:>7.3f} {r['GPb'] - r['GPs']:>7.3f} "
            f"{r['Db']:>7.3f} {r['Ds']:>7.3f} {r['T']:>7.3f}"
        )
    # hierarchical bootstrap of pooled T
    seeds = list(rows)
    Ts = []
    for _ in range(5000):
        bs = [seeds[random.randrange(len(seeds))] for _ in seeds]
        st = []
        for s in bs:
            cb, cs = rows[s]["cb"], rows[s]["cs"]
            nb, ns = len(cb["PP"]), len(cs["PP"])
            ib = [random.randrange(nb) for _ in range(nb)]
            iss = [random.randrange(ns) for _ in range(ns)]

            def mb(cl, cb=cb, ib=ib, nb=nb):
                return sum(cb[cl][i] for i in ib) / nb

            def ms(cl, cs=cs, iss=iss, ns=ns):
                return sum(cs[cl][i] for i in iss) / ns

            st.append(
                ((mb("PP") - mb("PS")) - (mb("SP") - mb("SS")))
                - ((ms("PP") - ms("PS")) - (ms("SP") - ms("SS")))
            )
        Ts.append(statistics.mean(st))
    Ts.sort()
    print(
        f"  D(cap{CAP_BIND}) 3-seed mean {statistics.mean([r['Db'] for r in rows.values()]):.3f} "
        f"(sd {statistics.pstdev([r['Db'] for r in rows.values()]):.3f}); "
        f"D(cap{CAP_SLACK}) mean {statistics.mean([r['Ds'] for r in rows.values()]):.3f}"
    )
    print(
        f"  pooled T = {statistics.mean(Ts):+.3f}  hierarchical-bootstrap 95% CI "
        f"[{Ts[125]:+.3f}, {Ts[4874]:+.3f}]  excludes 0: {Ts[125] > 0}\n"
    )


# ---------- A2: operational metrics (confound rebuttal), knapsack seed0 ----------
def a2():
    def status(cell, cap):
        st = {}
        for f in glob.glob(
            f"experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap{cap}/**/{KFAM}-{KFAM}-*.json",
            recursive=True,
        ):
            if f.endswith(".trace.json"):
                continue
            d = json.load(open(f))
            r = d.get("result", {}) or {}
            st[os.path.basename(f)] = dict(
                status=d.get("status"),
                insp=(r.get("metrics") or {}).get("inspected_count"),
                score=r.get("score"),
            )
        return st

    def trmet(cell, cap):
        rows = []
        for t in glob.glob(
            f"experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap{cap}/**/{KFAM}-{KFAM}-*.trace.json",
            recursive=True,
        ):
            tr = json.load(open(t))
            s = tr.get("summary", {}) or {}
            steps = [e for e in tr.get("events", []) if e.get("type") == "StepEvent"]
            n = s.get("num_steps") or len(steps)

            def err(e):
                return str(
                    (e.get("data", {}).get("interpreter_result") or {}).get("error")
                    or ""
                )

            caphit = sum(1 for e in steps if "Tool call limit exceeded" in err(e))
            nameerr = sum(1 for e in steps if "NameError" in err(e))
            rows.append(
                dict(
                    turns=n,
                    caphit=caphit / max(n, 1),
                    nameerr=nameerr,
                    tokens=(s.get("token_usage") or {}).get("total_tokens"),
                    finish=s.get("finish_reason"),
                    base=os.path.basename(t).replace(".trace.json", ".json"),
                )
            )
        return rows

    print(
        "=== A2: operational metrics (knapsack seed0) -> confound rebuttal (Table 1) ==="
    )
    print(
        f"  max total calls = {MAXT}xc (nonbinding; task needs ~50). realized-call proxy = inspected_count."
    )
    print(
        f"  {'cell@cap':10} {'turns':>6} {'caphit%':>7} {'NameErr':>7} {'tokens':>8} {'insp':>5} {'fin/turn/to':>12} {'score':>6}"
    )
    for cap in (CAP_BIND, CAP_SLACK):
        for cell in ("PP", "PS", "SP", "SS"):
            rs, st = trmet(cell, cap), status(cell, cap)
            if not rs:
                continue

            def g(k, rs=rs):
                return statistics.mean(r[k] for r in rs if r[k] is not None)

            insp = statistics.mean(
                st[r["base"]]["insp"]
                for r in rs
                if st.get(r["base"], {}).get("insp") is not None
            )
            sc = statistics.mean(st[r["base"]]["score"] for r in rs if r["base"] in st)
            term = Counter(
                "timeout"
                if st.get(r["base"], {}).get("status") == "timeout"
                else (r["finish"] or "none")
                for r in rs
            )
            print(
                f"  {cell}@{cap:<7} {g('turns'):>6.1f} {g('caphit') * 100:>6.0f}% {g('nameerr'):>7.2f} "
                f"{g('tokens'):>8.0f} {insp:>5.1f} {term.get('finish_tool', 0):>3}/{term.get('max_turns', 0):>2}/{term.get('timeout', 0):<2} {sc:>6.2f}"
            )
    print()


# ---------- A4: model-selection flip ----------
def a4():
    print("=== A4: model-selection flip (stateless runtime; PS vs SS) ===")
    for cap in (CAP_BIND, CAP_SLACK):
        d = []
        for s in SEEDS:
            c = knap_cells(s, cap)
            if c:
                d += [ps - ss for ps, ss in zip(c[0]["PS"], c[0]["SS"])]
        mu = statistics.mean(d)
        print(
            f"  cap{cap}: mean(PS-SS)={mu:+.3f} -> better under stateless deploy: "
            f"{'persistent-trained' if mu > 0 else 'stateless-trained'} (n={len(d)})"
        )
    print()


# ---------- A5: dense log(c) trend ----------
def a5():
    print("=== A5: dense P->S (mismatched) and P->P (matched) dose-response vs log(cap) ===")
    caps = (10, 20, 25, 30, 40, 50, 80, 100)
    ps = {c: _scores(f"experiments/cap_sweep/knapsack/qwen3_8b/dense/persistent_cap{c}", KFAM) for c in caps}
    pp = {c: _scores(f"experiments/cap_sweep/knapsack/qwen3_8b/dense/matched_cap{c}", KFAM) for c in caps}
    common = sorted(set.intersection(*[set(ps[c]) for c in caps],
                                     *[set(pp[c]) for c in caps]))
    if not common:
        print("  (dense per-task results absent -- unpack the eval-data mirror into "
              "experiments/cap_sweep/, see README)\n")
        return
    xs = [math.log(c) for c in caps]
    mx = sum(xs) / len(xs)

    def fe_slope(ids, d):  # task-fixed-effects slope of Q on log(cap) (matches build_dense)
        num = den = 0.0
        for t in ids:
            yi = [d[c][t] for c in caps]
            yb = sum(yi) / len(yi)
            for x, y in zip(xs, yi):
                num += (x - mx) * (y - yb)
                den += (x - mx) ** 2
        return num / den

    def diff(ids):  # interaction = paired slope contrast beta_PS - beta_PP
        return fe_slope(ids, ps) - fe_slope(ids, pp)

    NB = 5000
    random.seed(0)
    dboots = sorted(diff([common[random.randrange(len(common))] for _ in common]) for _ in range(NB))
    print("  PS(cap):", {c: round(statistics.mean([ps[c][t] for t in common]), 3) for c in caps})
    print("  PP(cap):", {c: round(statistics.mean([pp[c][t] for t in common]), 3) for c in caps})
    print(
        f"  slope PS={fe_slope(common, ps):+.3f}  PP={fe_slope(common, pp):+.3f}  "
        f"contrast beta_PS-beta_PP={diff(common):+.3f} "
        f"[95% CI {dboots[int(0.025 * NB)]:+.3f}, {dboots[int(0.975 * NB)]:+.3f}] (n={len(common)})\n"
    )


# ---------- A7/A8: mechanism triangulation ----------
def a7a8():
    def cells(base, prefix, fam, metric):
        return {
            (cl, cap): _scores(f"{base}/{prefix}{cl}_cap{cap}", fam, metric)
            for cl in ("PP", "PS")
            for cap in (CAP_BIND, CAP_SLACK)
        }

    def AP(c):
        common = sorted(
            set(c[("PP", 25)])
            & set(c[("PS", 25)])
            & set(c[("PP", 80)])
            & set(c[("PS", 80)])
        )
        d = [
            (c[("PP", 25)][k] - c[("PS", 25)][k])
            - (c[("PP", 80)][k] - c[("PS", 80)][k])
            for k in common
        ]
        gpb = statistics.mean(c[("PP", 25)][k] - c[("PS", 25)][k] for k in common)
        gps = statistics.mean(c[("PP", 80)][k] - c[("PS", 80)][k] for k in common)
        lo, hi = _bootci(d)
        return statistics.mean(d), lo, hi, gpb, gps

    def Rmean(td, fam, field, tr):
        vals = [
            tr(json.load(open(f)).get("public", {})[field])
            for f in glob.glob(f"experiments/cap_sweep/{td}/tasks/{fam}/*.json")
            if field in (json.load(open(f)).get("public", {}) or {})
        ]
        return statistics.mean(vals) if vals else float("nan")

    specs = [
        (
            "knapsack",
            "knapsack/qwen3_8b/main",
            "",
            "knapsack",
            "score",
            "knapsack/task_defs",
            "inspect_budget",
            lambda x: x,
            "cached,1/call / none",
        ),
        (
            "rule(score)",
            "rule_diagnosis/qwen3_8b/main",
            "rule_",
            "rule_diagnosis",
            "score",
            "rule_diagnosis/task_defs",
            "probe_budget",
            lambda x: x,
            "budgeted,uncached / free check()",
        ),
        (
            "rule(bndF1)",
            "rule_diagnosis/qwen3_8b/main",
            "rule_",
            "rule_diagnosis",
            "boundary_f1",
            "rule_diagnosis/task_defs",
            "probe_budget",
            lambda x: x,
            "budgeted,uncached / free check()",
        ),
        (
            "navigation",
            "navigation/qwen3_8b/main",
            "nav_",
            "navigation",
            "score",
            "navigation/task_defs",
            "n",
            lambda x: math.ceil(x / 50),
            "free+batched / n/a",
        ),
    ]
    print(
        "=== A7/A8: mechanism triangulation (A_P, per-instance R, channel) — Table 2 ==="
    )
    print(
        f"  {'family(metric)':14} {'GPbind':>7} {'GPslk':>6} {'A_P':>7} {'A_P 95%CI':>16} {'R':>5} {'R/c':>5}  channel / self-heal"
    )
    for name, base, pfx, fam, metric, td, field, tr, chan in specs:
        c = cells(f"experiments/cap_sweep/{base}", pfx, fam, metric)
        ap, lo, hi, gpb, gps = AP(c)
        R = Rmean(td, fam, field, tr)
        print(
            f"  {name:14} {gpb:>7.3f} {gps:>6.3f} {ap:>+7.3f} [{lo:>+5.2f},{hi:>+5.2f}] {R:>5.1f} {R / CAP_BIND:>5.2f}  {chan}"
        )
    print(
        "  a-priori R/c orders nav<<knap~rule; empirical A_P: only knapsack CI excludes 0 (rule underpowered, nav null).\n"
    )


# ---------- A6: dense per-cap bootstrap CIs (appendix dense table) ----------
def a6_dense():
    print("=== A6: dense P->S per-cap n, mean, bootstrap 95% CI (App. dense) ===")
    random.seed(0)  # deterministic CIs independent of call order
    for cap in (10, 20, 25, 30, 40, 50, 80, 100):
        v = list(
            _scores(f"experiments/cap_sweep/knapsack/qwen3_8b/dense/persistent_cap{cap}", KFAM).values()
        )
        if v:
            lo, hi = _bootci(v)
            print(
                f"  cap{cap:>3}: n={len(v):>2} mean={statistics.mean(v):.3f} CI[{lo:.3f},{hi:.3f}]"
            )
    print()


# ---------- A9: per-seed cell means + D + model-selection M(c), dM ----------
def a9_seedcells():
    print("=== A9: per-seed cell means, D, M(c)=PS-SS, dM=M(slack)-M(bind) (App. stats) ===")
    random.seed(0)  # deterministic CIs independent of call order
    msel = {}
    for seed in SEEDS:
        for cap in (CAP_BIND, CAP_SLACK):
            c = knap_cells(seed, cap)
            if not c:
                print(f"  seed {seed:>4} cap{cap}: MISSING")
                continue
            cells, common = c
            m = {cl: statistics.mean(cells[cl]) for cl in cells}
            dstat = (m["PP"] - m["PS"]) - (m["SP"] - m["SS"])
            msel[(seed, cap)] = [ps - ss for ps, ss in zip(cells["PS"], cells["SS"])]
            print(
                f"  seed {seed:>4} cap{cap}: n={len(common):>2} PP={m['PP']:.3f} PS={m['PS']:.3f} "
                f"SP={m['SP']:.3f} SS={m['SS']:.3f} | D={dstat:+.3f}"
            )
    for seed in SEEDS:
        if (seed, CAP_BIND) in msel and (seed, CAP_SLACK) in msel:
            mb = statistics.mean(msel[(seed, CAP_BIND)])
            ms = statistics.mean(msel[(seed, CAP_SLACK)])
            print(f"  seed {seed:>4}: M(bind)={mb:+.3f} M(slack)={ms:+.3f} dM={ms - mb:+.3f}")
    for cap, tag in ((CAP_BIND, "bind"), (CAP_SLACK, "slack")):
        if ("0", cap) in msel:
            lo, hi = _bootci(msel[("0", cap)])
            print(f"  seed0 M({tag}) paired-boot 95% CI [{lo:+.3f},{hi:+.3f}]")
    print()


# ---------- A10: progress-vs-turn curve (appendix T_max) ----------
def a10_progress():
    print("=== A10: cumulative inspect/list_items by turn, PS vs PP @cap_bind (App. T_max) ===")

    def curve(cell):
        ins, lst, finals = {}, {}, []
        pat = f"experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap{CAP_BIND}/**/{KFAM}-{KFAM}-*.trace.json"
        for t in glob.glob(pat, recursive=True):
            with open(t) as fh:
                tr = json.load(fh)
            steps = [e for e in tr.get("events", []) if e.get("type") == "StepEvent"]
            ci = cl = 0
            for ti, e in enumerate(steps):
                code = (e.get("data", {}) or {}).get("code") or ""
                ci += len(re.findall(r"\binspect\s*\(", code))
                cl += len(re.findall(r"\blist_items\s*\(", code))
                ins.setdefault(ti, []).append(ci)
                lst.setdefault(ti, []).append(cl)
            rf = os.path.join(
                os.path.dirname(t),
                os.path.basename(t).replace(".trace.json", ".json"),
            )
            if os.path.exists(rf):
                with open(rf) as fh:
                    mm = (json.load(fh).get("result", {}) or {}).get("metrics") or {}
                if mm.get("inspected_count") is not None:
                    finals.append(mm["inspected_count"])
        return ins, lst, finals

    for cell in ("PS", "PP"):
        ins, lst, finals = curve(cell)
        fm = statistics.mean(finals) if finals else float("nan")
        print(f"  {cell}@{CAP_BIND} (distinct inspected_count mean={fm:.1f}):")
        for ti in (0, 5, 10, 20, 30, 39):
            if ti in ins:
                print(
                    f"    t{ti:>2}: ins={statistics.mean(ins[ti]):>6.1f}  lst={statistics.mean(lst[ti]):>5.1f}"
                )
    print()


# ---------- A12: navigation reconstruction-bandwidth intervention ----------
def a12_bandwidth():
    """Sec. 5 Prediction 1, tested WITHIN navigation: does making erased state expensive to
    rebuild make the per-turn cap bite?

    The two arms are the same 16 graphs, the same adapter, the same budgets and scoring;
    only `neighbors_batch_max` differs (50 vs 2, preselected by scripts/nav_batch_sweep.py),
    so rebuilding an n-node map costs ceil(n/50) calls in one arm and ceil(n/2) in the other. Every quantity below is paired on the
    task id, and the bootstrap resamples tasks -- each draw carries a task's eight scores
    (2 arms x 2 cells x 2 caps) together.

        A_P(arm) = [Q_PP(25) - Q_PS(25)] - [Q_PP(80) - Q_PS(80)]
        D_batch  = A_P(reduced) - A_P(batched)     > 0 supports the capacity condition
    """
    arms = {
        "batched": "experiments/cap_sweep/navigation/qwen3_8b/main/nav_{cell}_cap{cap}",
        "reduced": "experiments/cap_sweep/navigation/qwen3_8b/batch2/navb2_{cell}_cap{cap}",
    }
    cells = {}
    for arm, tpl in arms.items():
        for cl in ("PP", "PS"):
            for cap in (CAP_BIND, CAP_SLACK):
                cells[(arm, cl, cap)] = _scores(
                    tpl.format(cell=cl, cap=cap), "navigation"
                )
    missing = [k for k, v in cells.items() if not v]
    print("=== A12: navigation reconstruction-bandwidth intervention (D_batch) ===")
    if missing:
        print(f"  cells absent: {missing}\n  (run the navb2_* shards first)\n")
        return
    common = sorted(set.intersection(*[set(v) for v in cells.values()]))

    def gap(arm, cap, t):  # G_P = matched minus mismatched, on one task
        return cells[(arm, "PP", cap)][t] - cells[(arm, "PS", cap)][t]

    def amp(arm, t):  # binding-cap amplification of the runtime gap
        return gap(arm, CAP_BIND, t) - gap(arm, CAP_SLACK, t)

    per_task = [amp("reduced", t) - amp("batched", t) for t in common]
    random.seed(0)
    lo, hi = _bootci(per_task)
    print(f"  {'arm':10} {'Q_PP(25)':>9} {'Q_PS(25)':>9} {'Q_PP(80)':>9} {'Q_PS(80)':>9} "
          f"{'G_P(25)':>8} {'G_P(80)':>8} {'A_P':>7}")
    for arm in arms:
        m = {
            (cl, cap): statistics.mean(cells[(arm, cl, cap)][t] for t in common)
            for cl in ("PP", "PS")
            for cap in (CAP_BIND, CAP_SLACK)
        }
        gb, gs = m[("PP", CAP_BIND)] - m[("PS", CAP_BIND)], m[("PP", CAP_SLACK)] - m[("PS", CAP_SLACK)]
        print(
            f"  {arm:10} {m[('PP', CAP_BIND)]:>9.3f} {m[('PS', CAP_BIND)]:>9.3f} "
            f"{m[('PP', CAP_SLACK)]:>9.3f} {m[('PS', CAP_SLACK)]:>9.3f} "
            f"{gb:>8.3f} {gs:>8.3f} {gb - gs:>+7.3f}"
        )
    d = statistics.mean(per_task)
    print(
        f"  D_batch = A_P(reduced-bandwidth) - A_P(batched) = {d:+.3f} "
        f"[95% CI {lo:+.3f}, {hi:+.3f}] (n={len(common)} paired tasks, "
        f"{'excludes' if lo > 0 or hi < 0 else 'includes'} 0)"
    )
    print(
        "  Positive => within one task, cutting reconstruction bandwidth makes the binding cap\n"
        "  amplify the runtime mismatch. Read with the manipulation check in A11 section 1:\n"
        "  the intervention is only meaningful if the reduced arm's realized K_t crosses 25.\n"
    )


# ---------- A11: measured cross-family mechanism tables ----------
def a11_mechanism():
    """Cap exposure, replay-vs-novel progress, navigation stages, rule hypothesis repair,
    and operational currency -- all measured by replaying the recorded code blocks through
    the real environments (scripts/mechanism_traces.py). These replace the nominal R/c
    proxy as the cross-family evidence in Sec. 5; the replay self-checks against the
    published knapsack anchors before printing anything."""
    try:
        # `python -m scripts.analyze_paper`, or imported by build_paper_numbers.py
        from scripts.mechanism_traces import run_all  # type: ignore[reportMissingImports]
    except ModuleNotFoundError:
        # the documented `python scripts/analyze_paper.py`: sys.path[0] is scripts/
        from mechanism_traces import run_all  # type: ignore[reportMissingImports]

    print(
        "=== A11: measured mechanism tables "
        "(cap exposure, replay/novel, stages, repair) ==="
    )
    asyncio.run(run_all())


if __name__ == "__main__":
    a1()
    a2()
    a4()
    a5()
    a7a8()
    a6_dense()
    a9_seedcells()
    a10_progress()
    a11_mechanism()
    a12_bandwidth()
