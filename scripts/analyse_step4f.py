#!/usr/bin/env python3
"""
Pre-registered analysis of step 4f (H18; corrected environment, TODO).

For every distribution shift and held-out set 1: each agent group against the
best tuned rule of each family (rules selected on the corrected tuning run,
step 4e), and on held-out set 1 the paired difference V0r - B'.
Writes results/stage4f_eval/stage4f_analysis.json and prints a summary.
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
)

RUN_DIR = ROOT / "results/stage4f_eval"
RAW = "benchmark_traditional_vs_rl_raw.csv"
RUNS = ("shift_noise1.5", "shift_noise2", "shift_size0.7", "shift_range1.2", "heldout")
GROUPS = {
    "Bp": [f"S2_bp_s{s}" for s in range(5)],
    "B": [f"S2_b_s{s}" for s in range(5)],
    "V0r": [f"V0r_s{s}" for s in range(3)],
    "DQN": ["DQN"],
}
# Selected on the corrected tuning run (results/stage4e_eval/stage4e_analysis.json).
RULES = ("Threshold_0.45", "Plateau_0.40_k3_e0.01", "Decaying_1.00_0.40", "CovGated_0.45_1")


def summary(rows, members):
    pooled = pooled_summary([r for r in rows if r["policy"] in members])
    return {"mean_hp": float(pooled["mean_hp"].mean()),
            "min_firing_rate_pct": float(pooled["firing_rate_pct"].min()),
            "mean_fire_time_s": float(pooled["mean_fire_time_s"].mean())}


def main(n_boot=10000):
    selected = json.loads((ROOT / "results/stage4e_eval/stage4e_analysis.json").read_text())["selected_rules"]
    assert sorted(v["policy"] for v in selected.values()) == sorted(RULES), "rule selection changed"
    out = {}
    for run in RUNS:
        rows = pd.read_csv(RUN_DIR / run / RAW).to_dict("records")
        res = {"summaries": {**{g: summary(rows, m) for g, m in GROUPS.items()},
                             **{p: summary(rows, [p]) for p in (*RULES, "Oracle")}},
               "vs_rules": {}}
        for g, members in GROUPS.items():
            res["vs_rules"][g] = {
                rule: {k: v for k, v in non_inferiority(rows, members, rule, 0.02, n_boot=n_boot).items()
                       if k in ("mean_diff", "lower_bound")} for rule in RULES}
        if run == "heldout":
            g = agent_group_difference(rows, GROUPS["V0r"], GROUPS["Bp"], n_boot=n_boot)
            res["V0r_minus_Bp"] = {k: g[k] for k in ("mean_diff", "lower_bound", "upper_bound",
                                                    "ci_two_sided", "a_better", "b_better")}
        out[run] = res
    (RUN_DIR / "stage4f_analysis.json").write_text(json.dumps(out, indent=2, default=bool))

    for run, res in out.items():
        s = res["summaries"]
        print(f"\n{run}: " + " | ".join(f"{k} {v['mean_hp']:.3f} ({v['min_firing_rate_pct']:.0f}%)"
                                         for k, v in s.items()))
        for g, by_rule in res["vs_rules"].items():
            worst = min(r["lower_bound"] for r in by_rule.values())
            print(f"  {g}: " + ", ".join(f"{rule.split('_')[0]} {r['mean_diff']:+.3f} (LB {r['lower_bound']:+.3f})"
                                          for rule, r in by_rule.items())
                  + f"  -> beats every rule: {'yes' if worst > 0 else 'no'}")
        if "V0r_minus_Bp" in res:
            c = res["V0r_minus_Bp"]
            print(f"  V0r - Bp: {c['mean_diff']:+.3f} CI [{c['ci_two_sided'][0]:+.3f}, {c['ci_two_sided'][1]:+.3f}]")


if __name__ == "__main__":
    main()
