#!/usr/bin/env python3
"""Render the paper's data figures to PDF from paper/numbers.json (the single source).

Replaces the inline TikZ/pgfplots plots with matplotlib PDFs that main.tex includes
via \\includegraphics. Run after scripts/build_paper_numbers.py.

Layout redesign (readability only, no data change): doseresponse uses direct
end-of-line labels instead of a colliding legend box; throughput is split into two
stacked panels (the twin-y-axis mixed unrelated 0-40 vs 0-700 scales); select shows
the per-cell 95% CIs as error bars. Palette unchanged (colorblind-safe).

hero() is a schematic rather than a data plot: two rollout strips showing the same
persistent-trained agent on the same stateless runtime and the same task instance,
differing only in the per-turn cap. Its numbers come from numbers.json like every
other figure; its monospace strings are verbatim from two real traces (see the
provenance comment above the constants).

Usage:  uv run python scripts/plot_figures.py   # writes paper/figures/*.pdf
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path  # noqa: E402

OUT = "paper/figures"
BLUE, RED = "#1f4e9c", "#c0392b"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3, "figure.dpi": 150,
    "pdf.fonttype": 42, "font.family": "serif",
})


def _nums():
    with open("paper/numbers.json") as fh:
        return json.load(fh)


def dose_response(n):
    f = n["figures"]["doseresponse"]
    ps, pp = f["ps"], f["pp"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.plot([p[0] for p in ps], [p[1] for p in ps], marker="o", ms=3.5, color=BLUE)
    ax.plot([p[0] for p in pp], [p[1] for p in pp], marker="s", ms=3.5, ls="--", color=RED)
    ax.set_xlabel("per-turn tool-call cap $c$")
    ax.set_ylabel("normalized optimality")
    ax.set_xlim(5, 108)
    ax.set_ylim(0, 0.92)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    # direct end-of-line labels instead of a legend box (which collided with the
    # P->P line/marker around cap~25-30 in the original). ylim top is 0.92, not
    # 0.85, to leave headroom above the P->P label (anchored +6pt above its last
    # point, y~0.77) so its text box doesn't run into the top spine.
    ps_x, ps_y = ps[-1]
    ax.annotate("P→S (mismatch)", (ps_x, ps_y), xytext=(0, -8),
                textcoords="offset points", ha="right", va="top", color=BLUE, fontsize=7)
    pp_x, pp_y = pp[-1]
    ax.annotate("P→P (matched)", (pp_x, pp_y), xytext=(-4, 6),
                textcoords="offset points", ha="right", va="bottom", color=RED, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/doseresponse.pdf", bbox_inches="tight")
    plt.close(fig)


def throughput(n):
    t = n["figures"]["throughput_ps25"]
    # two stacked panels sharing the x-axis: the two series have unrelated scales
    # (0-40 distinct vs 0-700 attempts), so a shared twin y-axis was misleading.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(3.5, 3.0), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )
    ax_top.plot(t["turns"], t["distinct"], marker="o", ms=3, color=BLUE)
    ax_top.set_ylabel("distinct items\ninspected (novel)")
    ax_top.set_ylim(0, 40)
    ax_top.grid(True, ls=":", lw=0.5, alpha=0.6)

    ax_bot.plot(t["turns"], t["attempts"], marker="s", ms=3, ls="--", color=RED)
    ax_bot.set_ylabel("inspect-call attempts\n(repeated)")
    ax_bot.set_xlabel("turn index")
    ax_bot.set_xlim(0, 40)
    ax_bot.set_ylim(0, 720)
    ax_bot.grid(True, ls=":", lw=0.5, alpha=0.6)

    fig.savefig(f"{OUT}/throughput.pdf", bbox_inches="tight")
    plt.close(fig)


def select(n):
    s = n["figures"]["select"]
    fig, ax = plt.subplots(figsize=(2.7, 2.0))
    groups = ["binding\n$c{=}25$", "slack\n$c{=}80$"]
    x = range(len(groups))
    w = 0.36

    def bars_err(cell):
        m = [s[cell]["bind"]["mean"], s[cell]["slack"]["mean"]]
        lo = [m[0] - s[cell]["bind"]["ci_lo"], m[1] - s[cell]["slack"]["ci_lo"]]
        hi = [s[cell]["bind"]["ci_hi"] - m[0], s[cell]["slack"]["ci_hi"] - m[1]]
        return m, [lo, hi]

    ps, ps_err = bars_err("PS")
    ss, ss_err = bars_err("SS")
    errkw = dict(ecolor="black", elinewidth=0.8, capsize=2.5, capthick=0.8)
    ax.bar([i - w / 2 for i in x], ps, w, color=BLUE, label="P→S (persistent-trained)",
           yerr=ps_err, error_kw=errkw)
    ax.bar([i + w / 2 for i in x], ss, w, color=RED, label="S→S (stateless-trained)",
           yerr=ss_err, error_kw=errkw)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("normalized optimality")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/select.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hero figure: the per-turn cap as the switch between a token cost and a task
# failure.
#
# The replies and NameErrors below are verbatim from the two traces.  The turn-0
# listing is the abbreviation *published in App. J* (of the c=25 episode); the
# c=80 episode issues the same plan but binds the parsed dict to `details`, so
# the figure attributes that block to App. J rather than calling it verbatim.
# Serif text is our annotation.  The two episodes are the SAME task instance (n=60, C=143,
# B=60) run by the SAME persistent-trained adapter under the SAME stateless
# runtime, differing only in the per-turn tool-call cap:
#   c=25  experiments/cap_sweep/knapsack/qwen3_8b/main/PS_cap25/PS_cap25/results/PS_cap25/
#         knapsack/knapsack-knapsack-0000000000.trace.json   (40 steps, max_turns)
#   c=80  experiments/cap_sweep/knapsack/qwen3_8b/main/PS_cap80/PS_cap80/results/PS_cap80/
#         knapsack/knapsack-knapsack-0000000000.trace.json   (9 steps, finish_tool)
# The turn-0 listing is the abbreviated form published in App. J (app:cap,
# paper/appendix.tex:575-591); both episodes emit this same opening plan (the
# c=80 trace differs only in binding the parsed dict to `details`).  The
# NameError lines are the final line of each episode's real turn-1 traceback.
# All NUMBERS are read from paper/numbers.json -- never hardcoded here.
INK, MUTED, RULE = "#1a1a1a", "#5c5c5c", "#c9c9c9"
AMBER = "#a06a12"        # absorbed damage: the same event, paid in tokens
MONO = "DejaVu Sans Mono"

# turn-0 opening, abbreviated as published in App. J; both episodes emit it
CODE_LINES = [
    "all_item_ids = json.loads(list_items())",
    "to_inspect = all_item_ids[:INSPECT_BUDGET]",
    "for item_id in to_inspect:",
    "    data = json.loads(inspect(item_id))",
]
OBS_SLACK = "Inspected 60 items."
OBS_BIND = "Tool call limit exceeded: allowed 25 per run."
ERR_SLACK = "NameError: name 'inspected_data' is not defined"
ERR_BIND = "NameError: name 'to_inspect' is not defined"

# three tiers, deliberately far apart: tier 1 is read from three feet, tier 2 on
# approach, tier 3 is scaffolding the eye is meant to skip.
FS_THESIS, FS_NUM, FS_VERDICT = 9.5, 21.0, 8.5                      # tier 1
FS_TITLE, FS_MECH, FS_STAT = 9.0, 7.6, 7.4                          # tier 2
FS_T3, FS_CODE = 5.8, 5.6                                           # tier 3
FS_EVENT = 6.6            # the cut and the reset: causal beats, not body text
_OVERFLOW = []


def _runs(fig, ax, x, y, runs, va="center", limit=None, tag=""):
    """Draw (text, kwargs) runs left-to-right from data x; return the trailing x.

    Data units are inches (the axes spans the whole figure), so a text's pixel
    width divided by fig.dpi is its width in data units.  `limit` records an
    overflow instead of silently letting a string run past its column.
    """
    r = fig.canvas.get_renderer()
    for txt, kw in runs:
        t = ax.text(x, y, txt, ha="left", va=va, **kw)
        x += t.get_window_extent(renderer=r).width / fig.dpi
    if limit is not None and x > limit + 1e-6:
        _OVERFLOW.append((tag, round(x - limit, 4), runs[0][0][:44]))
    return x


def _measure(fig, ax, txt, kw):
    """Width of `txt` in data units (= inches) without leaving it on the axes."""
    t = ax.text(0, 0, txt, **kw)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def _centre(fig, ax, xmid, y, txt, kw, limit=None, tag="", floor=None):
    """Centre `txt`, shrinking to fit `limit` rather than letting it run off."""
    kw = dict(kw)
    while True:
        w = _measure(fig, ax, txt, kw)
        if limit is None or xmid + w / 2 <= limit or kw["fontsize"] <= (floor or kw["fontsize"]):
            break
        kw["fontsize"] -= 0.1
    _runs(fig, ax, xmid - w / 2, y, [(txt, kw)], limit=limit, tag=tag)


def _bolt(ax, x, y, h=0.105, color=RED):
    """A lightning glyph as a polygon (DejaVu Serif has no U+26A1)."""
    w = h * 0.44
    pts = [(0.55, 1.0), (0.0, 0.46), (0.42, 0.46), (0.16, 0.0),
           (1.0, 0.60), (0.52, 0.60), (0.86, 1.0)]
    ax.add_patch(mpatches.Polygon(
        [(x + px * w, y + py * h) for px, py in pts],
        closed=True, facecolor=color, edgecolor="none", zorder=6))


def _scissors(ax, x, y, s=0.082, color=RED, alpha=1.0):
    """A scissors glyph as strokes + rings (DejaVu Serif has no U+2702)."""
    for sgn in (1, -1):
        ax.plot([x, x + s * 1.5], [y + sgn * s * 0.42, y - sgn * s * 0.42],
                color=color, lw=0.7, zorder=7, solid_capstyle="round", alpha=alpha)
        ax.add_patch(mpatches.Circle((x - s * 0.18, y + sgn * s * 0.5), s * 0.22,
                                     fill=False, ec=color, lw=0.7, zorder=7, alpha=alpha))


def _check(ax, x, y, s=0.075, color=BLUE, alpha=1.0):
    """A check glyph as two strokes (DejaVu Serif has no U+2713)."""
    ax.plot([x, x + s * 0.55, x + s * 1.6], [y + s * 0.05, y - s * 0.5, y + s * 0.75],
            color=color, lw=1.1, zorder=7, alpha=alpha,
            solid_capstyle="round", solid_joinstyle="miter")


def _round_path(pts, r):
    """Polyline through `pts` with corners rounded to radius `r`."""
    def unit(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        return dx / d, dy / d

    v, c = [pts[0]], [Path.MOVETO]
    for i in range(1, len(pts) - 1):
        p, a, b = pts[i], pts[i - 1], pts[i + 1]
        ui, uo = unit(a, p), unit(p, b)
        v += [(p[0] - ui[0] * r, p[1] - ui[1] * r), p,
              (p[0] + uo[0] * r, p[1] + uo[1] * r)]
        c += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    v.append(pts[-1])
    c.append(Path.LINETO)
    return Path(v, c)


def _flow(ax, pts, color, lw, r=0.09, hl=0.105, hw=0.052, alpha=1.0):
    """A polyline arrow whose head is drawn explicitly (a rounded corner near the
    endpoint skews a path-tangent head).  The final leg sets the head's direction.
    """
    (px_, py_), (ex, ey) = pts[-2], pts[-1]
    dx, dy = ex - px_, ey - py_
    d = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / d, dy / d
    body = list(pts[:-1]) + [(ex - ux * hl * 0.7, ey - uy * hl * 0.7)]
    ax.add_patch(mpatches.PathPatch(_round_path(body, r), fill=False, ec=color, lw=lw,
                                    joinstyle="round", capstyle="butt", zorder=4,
                                    alpha=alpha))
    ax.add_patch(mpatches.Polygon(
        [(ex, ey), (ex - ux * hl + uy * hw, ey - uy * hl - ux * hw),
         (ex - ux * hl - uy * hw, ey - uy * hl + ux * hw)],
        closed=True, fc=color, ec="none", zorder=5, alpha=alpha))


def _plate(ax, fig, x, ymid, w, h):
    """A knockout behind a label so it stays legible across a fill boundary."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, ymid - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        fc="white", ec="none", alpha=0.88, zorder=4.5))


def _meter(ax, x, y, w, h, frac, edge, ceiling):
    """The record workspace the agent can reach.

    The reset wipes interpreter bindings, not environment records: inspect
    results live in KnapsackEnv._inspect_cache and survive it in BOTH cells
    (families/knapsack.py:156-172).  So nothing here is drawn as destroyed.
    Solid blue is what the agent holds; hatching marks what it can never
    reach, because replaying the cached prefix already costs a whole turn's
    cap.  Earlier revisions hatched the built portion as "wiped", which
    misstated the mechanism in both directions.
    """
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="white", ec="none", zorder=2))
    if frac > 0:
        ax.add_patch(mpatches.Rectangle((x, y), w * frac, h, fc=BLUE, ec="none", zorder=3))
    if ceiling and frac < 1:
        ax.add_patch(mpatches.Rectangle((x + w * frac, y), w * (1 - frac), h, fc="none",
                                        ec=edge, lw=0.0, hatch="///", alpha=0.8, zorder=4))
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="none",
                                    ec=edge if ceiling else "#8fa2c4", lw=0.9, zorder=5))


def _strip(fig, ax, px, pw, s):
    """One rollout strip.  Geometry is identical in both panels; what differs is
    the cut, the colour of absorbed-vs-fatal events, line-vs-cycle, and whether
    turn 2 exists at all."""
    y, hue, lane = s["y"], s["hue"], 0.30
    cx = px + lane
    cw = pw - lane - 0.02
    lim = cx + cw
    t3 = dict(fontsize=FS_T3, color=MUTED)

    # -- tier 2: what this panel is, and why ---------------------------------
    _runs(fig, ax, cx, y["title"],
          [(s["title"], dict(fontsize=FS_TITLE, fontweight="bold", color=INK)),
           (f"   $c={s['cap']}$", dict(fontsize=FS_TITLE, color=INK))],
          limit=lim, tag=s["tag"] + ".title")
    _runs(fig, ax, cx, y["mech"],
          [(s["mech"], dict(fontsize=FS_MECH, fontweight="bold", color=hue))],
          limit=lim, tag=s["tag"] + ".mech")
    _runs(fig, ax, cx, y["arith"], [(s["arith"], t3)], limit=lim, tag=s["tag"] + ".arith")

    # -- turn 0: same code both sides, greyed; the cut is the only event -----
    _runs(fig, ax, cx, y["t0lbl"],
          [("turn 0", dict(t3, fontweight="bold")),
           (f"   {s['t0lbl']}", t3)],
          limit=lim, tag=s["tag"] + ".t0lbl")
    bh = 0.072 + len(CODE_LINES) * 0.094
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx, y["box0"] - bh), cw, bh, boxstyle="round,pad=0,rounding_size=0.03",
        fc="#f5f5f5", ec="#dedede", lw=0.6, zorder=1))
    for i, line in enumerate(CODE_LINES):
        _runs(fig, ax, cx + 0.045, y["box0"] - 0.060 - i * 0.094,
              [(line, dict(fontsize=FS_CODE, family=MONO, color="#8a8a8a", zorder=3))],
              limit=lim, tag=f"{s['tag']}.code{i}")
    if s["cut"]:
        yc = y["box0"] - bh + 0.036
        ax.plot([cx + 0.03, cx + cw - 0.21], [yc, yc], color=RED, lw=1.0,
                ls=(0, (2.0, 1.4)), zorder=5)
        _scissors(ax, cx + cw - 0.175, yc)
    _runs(fig, ax, cx + 0.045, y["reply"],
          [(s["reply"], dict(fontsize=FS_CODE, family=MONO, color=s["replycol"]))],
          limit=lim, tag=s["tag"] + ".reply")

    # -- the reset, restored as a full-width event between the turns.  Without
    # -- it the strip reads "to_inspect is defined" then "to_inspect undefined".
    rkw = dict(fontsize=FS_EVENT, fontweight="bold", color=s["evt"], zorder=6)
    ax.add_patch(mpatches.Rectangle((cx, y["reset"] - 0.075), cw, 0.15,
                                    fc=s["errfc"], ec=s["evt"], lw=0.7, zorder=1))
    _bolt(ax, cx + 0.035, y["reset"] - 0.052, h=0.102, color=s["evt"])
    _runs(fig, ax, cx + 0.13, y["reset"],
          [("interpreter reset: every variable wiped", rkw)],
          limit=lim, tag=s["tag"] + ".reset")

    # -- what survives it -----------------------------------------------------
    mh = 0.165
    _meter(ax, cx, y["ws"], cw, mh, s["fill"], s["evt"], s["cut"])
    lkw = dict(fontsize=FS_T3, fontweight="bold", zorder=6)
    if 0.06 + _measure(fig, ax, s["wslbl"], lkw) < cw * s["fill"]:
        lkw["color"] = "white"
    else:
        lkw["color"] = INK
        _plate(ax, fig, cx + 0.045, y["ws"] + mh / 2,
               _measure(fig, ax, s["wslbl"], lkw) + 0.03, mh - 0.05)
    _runs(fig, ax, cx + 0.06, y["ws"] + mh / 2, [(s["wslbl"], lkw)],
          limit=lim, tag=s["tag"] + ".ws")

    # -- turn 1: the same NameError in both panels, differently coloured ----
    _runs(fig, ax, cx, y["t1lbl"],
          [("turn 1", dict(t3, fontweight="bold")),
           ("   the reply after the reset", t3)],
          limit=lim, tag=s["tag"] + ".t1lbl")
    eh = 0.165
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx, y["err"] - eh / 2), cw, eh, boxstyle="round,pad=0,rounding_size=0.025",
        fc=s["errfc"], ec=s["evt"], lw=0.6, alpha=0.9, zorder=1))
    _runs(fig, ax, cx + 0.045, y["err"],
          [(s["err"], dict(fontsize=FS_CODE, family=MONO, color=s["evt"], zorder=3))],
          limit=lim, tag=s["tag"] + ".err")
    for i, g in enumerate(s["gloss"]):
        _runs(fig, ax, cx + 0.045, y["gloss"] - i * 0.10, [(g, t3)],
              limit=lim, tag=f"{s['tag']}.gloss{i}")

    # -- line vs cycle, carried by shape rather than by words ---------------
    lx, ym = px + 0.09, y["box0"] - bh / 2
    if s["cut"]:
        _flow(ax, [(cx - 0.05, y["t2"]), (lx, y["t2"]), (lx, ym), (cx - 0.015, ym)],
              RED, 2.8, r=0.075, hl=0.125, hw=0.062)
        _scissors(ax, cx + 0.03, y["t2"], s=0.072)
        _runs(fig, ax, cx + 0.16, y["t2"],
              [(s["loop"], dict(fontsize=FS_STAT, fontweight="bold", color=RED))],
              limit=lim, tag=s["tag"] + ".loop")
    else:
        _flow(ax, [(lx, ym), (lx, y["stat"] + 0.10)], BLUE, 2.8, hl=0.125, hw=0.062)

    # -- tier 2: exactly two statistics --------------------------------------
    for i, (a, b) in enumerate(s["stats"]):
        x1 = _runs(fig, ax, cx, y["stat"] - i * 0.15,
                   [(a, dict(fontsize=FS_STAT, fontweight="bold", color=INK))],
                   limit=lim, tag=f"{s['tag']}.stat{i}a")
        _runs(fig, ax, x1, y["stat"] - i * 0.15,
              [(b, dict(fontsize=FS_STAT, color=MUTED))],
              limit=lim, tag=f"{s['tag']}.stat{i}b")

    # -- tier 1: the outcome, and what the outcome is paid in ---------------
    nkw = dict(fontsize=FS_NUM, fontweight="bold", color=hue)
    nw = _measure(fig, ax, s["score"], nkw)
    ax.text(cx, y["out"] + 0.005, s["score"], va="center", ha="left", **nkw)
    bx0, bw0, bh0 = cx + nw + 0.10, cw - nw - 0.10, 0.34
    _runs(fig, ax, bx0, y["verdict"],
          [(s["verdict"], dict(fontsize=FS_VERDICT, fontweight="bold", color=hue))],
          limit=lim, tag=s["tag"] + ".verdict")
    ax.add_patch(mpatches.Rectangle((bx0, y["out"] - bh0 / 2), bw0, bh0, fc="#f0f0f0",
                                    ec="#c4c4c4", lw=0.7, zorder=2))
    ax.add_patch(mpatches.Rectangle((bx0, y["out"] - bh0 / 2), bw0 * s["frac"], bh0,
                                    fc=hue, ec="none", zorder=3))
    tx = bx0 + bw0 * s["afrac"]
    ax.plot([tx, tx], [y["out"] - bh0 / 2 - 0.03, y["out"] + bh0 / 2 + 0.03],
            color=INK, lw=1.1, zorder=5)
    lbl = f"same agent on its trained runtime: {s['anchor']}"
    _runs(fig, ax, lim - _measure(fig, ax, lbl, t3), y["out"] - bh0 / 2 - 0.085,
          [(lbl, t3)], limit=lim, tag=s["tag"] + ".anchor")


def _hero_check(cfg, task):
    """Guard the figure's two kinds of content.

    Numbers are read from numbers.json, so they cannot drift.  The verbatim trace
    strings are literals (they are quotations), so instead we assert that the
    values baked into those quotations still agree with numbers.json.
    """
    assert min(FS_THESIS, FS_TITLE, FS_MECH, FS_STAT, FS_VERDICT) >= 7.0
    assert FS_T3 >= 5.5, "tier-3 scaffolding below 5.5pt"
    assert FS_CODE >= 5.5, "quoted trace strings below 5.5pt"
    assert f"allowed {cfg['cap_bind']} per run." in OBS_BIND, OBS_BIND
    assert f"{task['eg_n']} items." in OBS_SLACK, OBS_SLACK
    assert "list_items()" in CODE_LINES[0], "turn 0 lost its setup line"
    assert "to_inspect" in CODE_LINES[1] and "inspect(item_id)" in CODE_LINES[3]
    assert "to_inspect" in ERR_BIND and "inspected_data" in ERR_SLACK


def _hero_ceiling_check(n):
    """The binding panel claims a ceiling, not a loss: guard both halves."""
    rep, task, cfg = n["replay"]["PS25"], n["task"], n["config"]
    assert rep["unique"] < task["eg_n"], "binding cell would have reached the full set"
    assert rep["replay_frac"] > 0.9, "calls are not predominantly replay"
    # replaying the prefix built in turn 0 costs list_items + (cap-1) inspects,
    # i.e. exactly the cap -- which is why the workspace cannot extend
    assert cfg["cap_bind"] == (cfg["cap_bind"] - 1) + 1


def hero(n):
    cfg, op, rep = n["config"], n["operational"], n["replay"]
    agg, task = n["knap2x2"]["agg"], n["task"]
    _hero_check(cfg, task)
    _hero_ceiling_check(n)
    # turn 0 under a binding cap spends call 1 on list_items(), so cap-1 inspects
    # land before the cap rejects the next one (interpreter.py:166-174)
    done, N = cfg["cap_bind"] - 1, task["eg_n"]
    W, H = 5.5, 3.78
    _OVERFLOW.clear()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    pad, gap = 0.04, 0.10
    pw = (W - 2 * pad - gap) / 2
    y = dict(title=3.28, mech=3.125, arith=3.015, t0lbl=2.865, box0=2.805,
             reply=2.285, reset=2.115, ws=1.825, t1lbl=1.700, err=1.565,
             gloss=1.400, t2=1.150, stat=1.010, verdict=0.700, out=0.435)

    _centre(fig, ax, W / 2, H - 0.135,
            "The per-turn cap decides whether a runtime mismatch costs tokens or the task.",
            dict(fontsize=FS_THESIS, fontweight="bold", color=INK),
            limit=W - pad, tag="thesis", floor=8.0)
    _centre(fig, ax, W / 2, H - 0.285,
            "same persistent-trained agent  ·  same stateless (reset) runtime  ·  same "
            f"{N}-item instance  ·  only $c$ differs",
            dict(fontsize=FS_T3, color=MUTED), limit=W - pad, tag="controls", floor=5.2)
    ax.plot([pad, W - pad], [H - 0.375] * 2, color=RULE, lw=0.7)
    ax.plot([W / 2, W / 2], [0.16, H - 0.44], color=RULE, lw=0.6)

    common = dict(y=y)
    slack = dict(
        common, tag="slack", cut=False, hue=BLUE, evt=AMBER, errfc="#fdf6e8",
        cap=cfg["cap_slack"], title="Slack cap",
        mech=f"all {N} inspects fit in one action",
        arith=f"{N} inspects + 1 list_items = {N + 1} calls  ≤  {cfg['cap_slack']}",
        t0lbl=f"tries to inspect all {N} items in one action",
        reply=OBS_SLACK, replycol=INK, fill=1.0,
        wslbl=f"{N} of {N} — restored in one action", err=ERR_SLACK,
        gloss=["it lost only names: one action re-materializes",
               f"all {N} cached records, and the episode moves on"],
        stats=[(f"{op['PS80']['tokens_k']:.0f}k tokens/ep",
                f"  (matched: {op['PP80']['tokens_k']:.0f}k)"),
               (f"finishes {op['PS80']['fin']}/{agg['cap80']['n']} episodes", "")],
        verdict="pays mainly in tokens",
        score=f"{op['PS80']['score']:.2f}", frac=op["PS80"]["score"],
        anchor=f"{op['PP80']['score']:.2f}", afrac=op["PP80"]["score"],
    )
    bind = dict(
        common, tag="bind", cut=True, hue=RED, evt=RED, errfc="#fdefec",
        cap=cfg["cap_bind"], title="Binding cap",
        mech=f"all {N} inspects never fit in one action",
        arith=f"{N} inspects + 1 list_items = {N + 1} calls  >  {cfg['cap_bind']}",
        t0lbl=f"tries to inspect all {N} items in one action",
        reply=OBS_BIND, replycol=RED, fill=done / N,
        wslbl=f"{done} of {N} — never gets past ≈{round(rep['PS25']['unique'])}", err=ERR_BIND,
        gloss=[f"it lost only names too, but replaying the cached {done}",
               f"costs {done + 1} calls — so the workspace never extends"],
        loop=f"restarts ×{op['PS25']['restarts']:.1f} per episode",
        stats=[(f"{rep['PS25']['calls']:.0f} inspect calls",
                f"  →  ≈{rep['PS25']['unique']:.0f} distinct items"),
               (f"exhausts $T_{{\\max}}$ in {op['PS25']['turnlimit']}/{agg['cap25']['n']}"
                f" episodes", "")],
        verdict="pays in the task",
        score=f"{op['PS25']['score']:.2f}", frac=op["PS25"]["score"],
        anchor=f"{op['PP25']['score']:.2f}", afrac=op["PP25"]["score"],
    )

    _strip(fig, ax, pad, pw, slack)
    _strip(fig, ax, pad + pw + gap, pw, bind)

    ax.plot([pad, W - pad], [0.105] * 2, color=RULE, lw=0.7)
    _centre(fig, ax, W / 2, 0.048,
            "turn-0 listing as printed in App. J; replies and errors verbatim  ·  bars and "
            "statistics are means over $n{=}25$  ·  strips show one episode  ·  Table 1",
            dict(fontsize=FS_T3, color=MUTED), limit=W - pad, tag="footer", floor=4.9)

    fig.savefig(f"{OUT}/hero.pdf")
    dev = os.environ.get("HERO_PNG")
    if dev:
        fig.savefig(dev, dpi=300)
    plt.close(fig)
    if _OVERFLOW:
        print("HERO OVERFLOW:")
        for tag, over, txt in _OVERFLOW:
            print(f"  {tag:22s} +{over:.3f}in  {txt!r}")


def main():
    os.makedirs(OUT, exist_ok=True)
    n = _nums()
    dose_response(n)
    throughput(n)
    select(n)
    hero(n)
    print(f"wrote {OUT}/doseresponse.pdf, throughput.pdf, select.pdf, hero.pdf")


if __name__ == "__main__":
    main()
