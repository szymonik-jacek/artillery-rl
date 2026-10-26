#!/usr/bin/env python3
"""
Figure: hit probability against mean firing time, test set and random set 2.

All 82 rules (grey, marker by family; the four tuned rules outlined and
labelled), the four agents (mean over training runs, large; single runs,
small) and the hindsight oracle. HP is unconditional (no firing scores 0);
the firing time is the mean over the episodes with a shot. Rules that never
fire have no firing time and are not drawn.

Writes docs/paper2_rl_methodology/figures/hp_vs_time.pdf (and .png).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_numbers import AGENTS, ORACLE, RAW, TUNED_RULES  # noqa: E402

from rl_training.utils.benchmark_metrics import pooled_summary  # noqa: E402

OUT_DIR = ROOT / "docs/paper2_rl_methodology/figures"
PANELS = (("Test set", ROOT / "results/stage4e_eval/evaluation_100" / RAW),
          ("Random set 2", ROOT / "results/stage4e_eval/heldout2" / RAW))

# Validated categorical pair (dataviz reference palette, light): colour = configuration,
# fill = agent within the configuration; every agent is also labelled directly.
AGENT_STYLE = {
    "Base-0": {"color": "#2a78d6", "filled": False},
    "Base": {"color": "#2a78d6", "filled": True},
    "Ext-steps": {"color": "#eb6834", "filled": False},
    "Ext-sec": {"color": "#eb6834", "filled": True},
}
RULE_COLOR, RULE_EDGE = "#b4b2ab", "#52514e"
FAMILIES = {"Threshold_": ("o", "Fixed"), "Decaying_": ("s", "Decaying"),
            "Plateau_": ("^", "Plateau"), "CovGated_": ("D", "Cov.-gated")}
TEXT, MUTED = "#0b0b0b", "#52514e"


def summaries(path):
    pooled = pooled_summary(pd.read_csv(path).to_dict("records")).set_index("policy")
    return pooled[["mean_hp", "mean_fire_time_s"]]


def draw(ax, title, s):
    tuned = set(TUNED_RULES.values())
    for prefix, (marker, _) in FAMILIES.items():
        rules = s[s.index.str.startswith(prefix)].dropna()
        ax.scatter(rules.mean_fire_time_s, rules.mean_hp, s=16, marker=marker, color=RULE_COLOR,
                   edgecolors="none", zorder=2)
        for name, row in rules[rules.index.isin(tuned)].iterrows():
            ax.scatter(row.mean_fire_time_s, row.mean_hp, s=34, marker=marker, facecolors=RULE_COLOR,
                       edgecolors=RULE_EDGE, linewidths=1.0, zorder=3)
    labels = {}
    for family, rule in TUNED_RULES.items():
        if rule in s.index and pd.notna(s.loc[rule, "mean_fire_time_s"]):
            labels[family] = (s.loc[rule, "mean_fire_time_s"], s.loc[rule, "mean_hp"], MUTED)
    for agent, members in AGENTS.items():
        st = AGENT_STYLE[agent]
        face = st["color"] if st["filled"] else "white"
        runs = s.loc[members]
        if len(members) > 1:
            ax.scatter(runs.mean_fire_time_s, runs.mean_hp, s=14, facecolors=face, edgecolors=st["color"],
                       linewidths=0.9, alpha=0.6, zorder=4)
        x, y = runs.mean_fire_time_s.mean(), runs.mean_hp.mean()
        ax.scatter(x, y, s=70, facecolors=face, edgecolors=st["color"], linewidths=1.6, zorder=5)
        labels[agent] = (x, y, TEXT)
    ox, oy = s.loc[ORACLE, "mean_fire_time_s"], s.loc[ORACLE, "mean_hp"]
    ax.scatter(ox, oy, s=110, marker="*", color=TEXT, zorder=5)
    labels["Oracle"] = (ox, oy, TEXT)
    return labels


def place_labels(ax, labels, positions):
    """Label each point at a position (data coordinates) chosen after rendering, with a leader line."""
    for name, (x, y, color) in labels.items():
        tx, ty = positions[name]
        ax.annotate(SHORT.get(name, name), (x, y), xytext=(tx, ty), textcoords="data", fontsize=7.5,
                    color=color, zorder=6, ha="left", va="center",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "shrinkA": 1, "shrinkB": 4})


SHORT = {"Covariance-gated": "Cov.-gated"}
# Label positions (data coordinates), chosen after rendering to avoid collisions.
POSITIONS = {
    "Test set": {"Ext-steps": (3.0, 0.665), "Ext-sec": (3.0, 0.635), "Base-0": (3.0, 0.565),
                 "Base": (7.8, 0.535), "Plateau": (7.2, 0.665), "Decaying": (10.6, 0.545),
                 "Fixed": (4.9, 0.405), "Covariance-gated": (4.9, 0.44), "Oracle": (10.9, 0.69)},
    "Random set 2": {"Base-0": (3.1, 0.46), "Base": (6.3, 0.465), "Ext-steps": (3.0, 0.31),
                     "Ext-sec": (5.15, 0.405), "Plateau": (5.9, 0.36), "Decaying": (8.2, 0.255),
                     "Fixed": (4.1, 0.29), "Covariance-gated": (4.1, 0.25), "Oracle": (8.05, 0.48)},
}


def main():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for ax, (title, path) in zip(axes, PANELS):
        labels = draw(ax, title, summaries(path))
        place_labels(ax, labels, POSITIONS[title])
        ax.set_title(title, fontsize=9, color=TEXT)
        ax.set_xlabel("Mean firing time (s)")
        ax.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Hit probability")
    handles = [plt.Line2D([], [], marker=m, linestyle="", color=RULE_COLOR, markersize=5, label=lab)
               for m, lab in FAMILIES.values()]
    handles += [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=f, markeredgecolor=c,
                           markersize=6, label=a) for a, (c, f) in
                {a: (st["color"], st["color"] if st["filled"] else "white") for a, st in AGENT_STYLE.items()}.items()]
    handles.append(plt.Line2D([], [], marker="*", linestyle="", color=TEXT, markersize=8, label="Oracle"))
    fig.legend(handles=handles, loc="outside lower center", ncol=9, frameon=False, fontsize=7,
               handletextpad=0.2, columnspacing=0.9)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "hp_vs_time.pdf")
    fig.savefig(OUT_DIR / "hp_vs_time.png", dpi=200)
    print(f"Saved {OUT_DIR / 'hp_vs_time.pdf'}")


if __name__ == "__main__":
    main()
