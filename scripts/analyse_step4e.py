#!/usr/bin/env python3
"""
Rule selection and hypothesis tests on the runs in results/stage4e_eval (Section 4.3 of the paper).

Agent labels: B' = Ext-sec (S2_bp), B = Ext-steps (S2_b), V0r = Base.
1. Select the best rule of each family on the tuning run.
2. H17 (ensemble rule): the ensemble of the five B' networks vs the average single B'
   on the tuning run; the ensemble is reported only if it is better.
3. H16: on random set 2 (heldout2), (a) V0r vs B' and (b) B' and V0r vs the tuned
   rule of each family; (c) on the test set, B' and B vs the tuned rule of each
   family, non-inferiority with delta = 0.02.
In the paper, H16 (a) and (b) are H2, H16 (c) is H1, and H17 is the ensemble rule.
Writes results/stage4e_eval/stage4e_analysis.json and prints a summary.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rl_training.utils.benchmark_metrics import (  # noqa: E402
    agent_group_difference,
    non_inferiority,
    pooled_summary,
    select_best_fixed_threshold,
)

RUN_DIR = ROOT / "results/stage4e_eval"
RAW = "benchmark_traditional_vs_rl_raw.csv"
GROUPS = {
    "Bp": [f"S2_bp_s{s}" for s in range(5)],
    "B": [f"S2_b_s{s}" for s in range(5)],
    "V0r": [f"V0r_s{s}" for s in range(3)],
    "Ens_bp": ["Ens_bp"],
    "DQN": ["DQN"],
}
FAMILIES = {"threshold": "Threshold_", "plateau": "Plateau_", "decaying": "Decaying_", "covgated": "CovGated_"}
DELTA = 0.02


def group_summary(rows, members):
    pooled = pooled_summary([r for r in rows if r["policy"] in members])
    return {"mean_hp": float(pooled["mean_hp"].mean()),
            "min_firing_rate_pct": float(pooled["firing_rate_pct"].min()),
            "mean_fire_time_s": float(pooled["mean_fire_time_s"].mean())}


def compare(rows, a, b, n_boot):
    g = agent_group_difference(rows, a, b, n_boot=n_boot)
    return {k: g[k] for k in ("mean_a", "mean_b", "mean_diff", "lower_bound", "upper_bound",
                              "ci_two_sided", "a_better", "b_better")}


def main(n_boot=10000):
    rows = {run: pd.read_csv(RUN_DIR / run / RAW).to_dict("records")
            for run in ("tuning_run", "heldout2", "evaluation_100")}
    out = {"selected_rules": {}, "summaries": {}}

    for family, prefix in FAMILIES.items():
        name, hp = select_best_fixed_threshold(rows["tuning_run"], prefix=prefix)
        out["selected_rules"][family] = {"policy": name, "tuning_hp": hp}
    rules = [v["policy"] for v in out["selected_rules"].values()]

    for run, r in rows.items():
        out["summaries"][run] = {g: group_summary(r, m) for g, m in GROUPS.items()}
        out["summaries"][run].update({p: group_summary(r, [p]) for p in rules + ["Oracle"]})

    out["H17_tuning"] = compare(rows["tuning_run"], GROUPS["Ens_bp"], GROUPS["Bp"], n_boot)
    out["H16a_heldout2_V0r_minus_Bp"] = compare(rows["heldout2"], GROUPS["V0r"], GROUPS["Bp"], n_boot)
    out["H16b_heldout2"] = {
        agent: {rule: {k: v for k, v in non_inferiority(rows["heldout2"], GROUPS[agent], rule, DELTA,
                                                         n_boot=n_boot).items()
                       if k in ("mean_diff", "lower_bound", "non_inferior")} for rule in rules}
        for agent in ("Bp", "V0r", "Ens_bp")}
    out["H16c_test"] = {
        agent: {rule: {k: v for k, v in non_inferiority(rows["evaluation_100"], GROUPS[agent], rule, DELTA,
                                                         n_boot=n_boot).items()
                       if k in ("mean_diff", "lower_bound", "non_inferior")} for rule in rules}
        for agent in ("Bp", "B", "V0r", "Ens_bp")}
    out["ensemble_vs_Bp"] = {run: compare(r, GROUPS["Ens_bp"], GROUPS["Bp"], n_boot)
                             for run, r in rows.items() if run != "tuning_run"}

    (RUN_DIR / "stage4e_analysis.json").write_text(json.dumps(out, indent=2, default=bool))

    print("Selected rules (tuning run):",
          {f: f"{v['policy']} ({v['tuning_hp']:.3f})" for f, v in out["selected_rules"].items()})
    for run, s in out["summaries"].items():
        print(f"\n{run}: " + " | ".join(f"{k} {v['mean_hp']:.3f} ({v['min_firing_rate_pct']:.0f}%, "
                                         f"{v['mean_fire_time_s']:.1f}s)" for k, v in s.items()))

    def fmt(c):
        verdict = "a better" if c["a_better"] else "b better" if c["b_better"] else "no difference"
        lo, hi = c["ci_two_sided"]
        return f"{c['mean_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}"

    print("\nH17 (tuning) Ens_bp - Bp:", fmt(out["H17_tuning"]))
    print("H16a (heldout2) V0r - Bp:", fmt(out["H16a_heldout2_V0r_minus_Bp"]))
    for key in ("H16b_heldout2", "H16c_test"):
        print(f"{key}:")
        for agent, res in out[key].items():
            print(f"  {agent}: " + ", ".join(f"{rule}: {r['mean_diff']:+.3f} LB {r['lower_bound']:+.3f}"
                                              f"{' NI' if r['non_inferior'] else ''}" for rule, r in res.items()))
    for run, c in out["ensemble_vs_Bp"].items():
        print(f"Ens_bp - Bp on {run}: {fmt(c)}")


if __name__ == "__main__":
    main()
