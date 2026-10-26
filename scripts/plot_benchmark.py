#!/usr/bin/env python3
"""
Plot mean hit probability against mean time to fire for every benchmarked
policy, with the HP-vs-time Pareto front (paper figure).

Reads the JSON written by compare_traditional_vs_rl.py (pooled summary,
best rule per family and the Pareto front) and writes PDF and PNG.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent

INK = "#0b0b0b"
INK_MUTED = "#52514e"
# Validated categorical slots 1-3 (all-pairs safe for a scatter) for the three
# rule families shown by default; the covariance-gated family (hidden by
# default) is drawn in neutral ink. Each family also has its own marker so
# identity survives greyscale print and CVD.
FAMILIES = (
    ("Threshold_", "Fixed threshold", "#2a78d6", "o"),
    ("Decaying_", "Decaying threshold", "#eb6834", "^"),
    ("Plateau_", "Plateau rule", "#1baf7a", "s"),
    ("CovGated_", "Covariance-gated threshold", INK_MUTED, "v"),
)
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def policy_family(name):
    """Index into FAMILIES for a rule-based policy, or None for agents/oracle."""
    for i, (prefix, *_rest) in enumerate(FAMILIES):
        if name.startswith(prefix):
            return i
    return None


def display_name(name):
    """Readable label for a policy name."""
    if name.startswith("Threshold_"):
        return f"Fixed {name.split('_')[1]}"
    if name.startswith("Decaying_"):
        _, start, end = name.split("_")
        return f"Decaying {start}→{end}"
    if name.startswith("Plateau_"):
        _, min_hp, window, eps = name.split("_")
        return f"Plateau ≥{min_hp}, k={window[1:]}"
    if name.startswith("CovGated_"):
        _, p_thr, tau = name.split("_")
        return f"Cov-gated {p_thr}, τ={tau}"
    if name == "Oracle":
        return "Hindsight oracle"
    return name.replace("_", " ")


def plot_pareto(pooled, front, out_paths, labelled=(), dqn_name="DQN", oracle_name="Oracle",
                include_gated=False):
    """Scatter of pooled mean HP vs mean time to fire.

    pooled: list of dicts with policy, mean_hp, mean_fire_time_s, firing_rate_pct.
    front: policy names on the Pareto front, ordered by time.
    labelled: policy names to label directly (DQN and oracle are always labelled).
    Policies that time out in some episodes are drawn hollow (not front candidates).
    The covariance-gated family coincides with the fixed thresholds in this
    benchmark and is hidden unless include_gated=True; the front is still
    computed over all policies.
    Returns the matplotlib Figure.
    """
    by_name = {p["policy"]: p for p in pooled if p.get("mean_fire_time_s") is not None}
    families = [f for f in FAMILIES if include_gated or f[0] != "CovGated_"]
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    hidden = {f[0] for f in FAMILIES} - {f[0] for f in families}

    def visible(name):
        return name in by_name and not any(name.startswith(h) for h in hidden)

    # Draw the front through visible policies only, so every step has a marker.
    front_pts = [(by_name[n]["mean_fire_time_s"], by_name[n]["mean_hp"]) for n in front if visible(n)]
    if front_pts:
        xs, ys = zip(*front_pts)
        ax.plot(xs, ys, drawstyle="steps-post", color=INK_MUTED, lw=1.0, ls="--", zorder=1)

    for i, (prefix, _label, color, marker) in enumerate(FAMILIES):
        if all(f[0] != prefix for f in families):
            continue
        pts = [p for p in by_name.values() if policy_family(p["policy"]) == i]
        for full in (True, False):
            sel = [p for p in pts if (p["firing_rate_pct"] >= 100.0) == full]
            if not sel:
                continue
            ax.scatter([p["mean_fire_time_s"] for p in sel], [p["mean_hp"] for p in sel], s=22,
                       marker=marker, facecolors=color if full else "none", edgecolors=color,
                       linewidths=1.0 if not full else 0.6, zorder=3)

    others = [p for p in by_name.values()
              if policy_family(p["policy"]) is None and p["policy"] not in (dqn_name, oracle_name)]
    if others:
        ax.scatter([p["mean_fire_time_s"] for p in others], [p["mean_hp"] for p in others], s=14,
                   marker="x", color=INK_MUTED, linewidths=0.8, zorder=3)

    for name, marker, size in ((dqn_name, "*", 90), (oracle_name, "D", 28)):
        if name in by_name:
            p = by_name[name]
            ax.scatter([p["mean_fire_time_s"]], [p["mean_hp"]], s=size, marker=marker, color=INK,
                       edgecolors=SURFACE, linewidths=0.8, zorder=4)

    for name in dict.fromkeys([dqn_name, oracle_name, *labelled]):
        if not visible(name):
            continue
        p = by_name[name]
        # DQN above-left, oracle left, rules below-right: keeps labels off the front line.
        arrow = None
        text = display_name(name)
        if name in (dqn_name, oracle_name):
            offset, ha = ((-4, 6), "right") if name == dqn_name else ((-6, 0), "right")
        elif name.startswith("Decaying_"):
            # Placed in the empty area below-left; a leader line names the exact point.
            offset, ha = ((-30, -45), "right")
            text = text.replace(" ", "\n", 1)
            arrow = {"arrowstyle": "-", "color": INK_MUTED, "lw": 0.6, "shrinkB": 3}
        else:
            offset, ha = ((4, -10), "left")
        ax.annotate(text, (p["mean_fire_time_s"], p["mean_hp"]), xytext=offset,
                    textcoords="offset points", fontsize=7, color=INK, ha=ha, va="center",
                    arrowprops=arrow)

    ax.set_xlabel("Mean time to fire (s)", color=INK)
    ax.set_ylabel("Mean hit probability\n(timeouts scored as 0)", color=INK)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)

    handles = [Line2D([], [], ls="", marker=m, color=c, markersize=5, label=l) for _p, l, c, m in families]
    handles += [
        Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor=INK_MUTED, markersize=5,
               label="Rule with timeouts"),
        Line2D([], [], ls="", marker="*", color=INK, markersize=8, label="DQN"),
        Line2D([], [], ls="", marker="D", color=INK, markersize=4, label="Hindsight oracle"),
        Line2D([], [], color=INK_MUTED, ls="--", lw=1.0, label="Pareto front"),
    ]
    if others:
        handles.insert(-1, Line2D([], [], ls="", marker="x", color=INK_MUTED, markersize=5, label="Other agents"))
    fig.legend(handles=handles, fontsize=6, frameon=False, loc="lower center", ncol=3,
               handletextpad=0.3, columnspacing=1.0, labelspacing=0.4)

    fig.tight_layout(rect=(0, 0.11, 1, 1))
    for path in out_paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, facecolor=SURFACE)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_json", type=Path,
                        default=ROOT / "results" / "benchmark_traditional_vs_rl.json")
    parser.add_argument("--output_stem", type=Path,
                        default=ROOT / "results" / "benchmark_hp_vs_time")
    args = parser.parse_args(argv)

    data = json.loads(args.results_json.read_text(encoding="utf-8"))
    best = [b["policy"] for b in data["best_fixed"].values()]
    paths = [args.output_stem.with_suffix(".pdf"), args.output_stem.with_suffix(".png")]
    plot_pareto(data["pooled"], data["pareto_front_hp_vs_time"], paths, labelled=best)
    for p in paths:
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
