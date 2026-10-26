#!/usr/bin/env python3
"""
Exploratory analyses behind the Discussion (Section 6) of the manuscript.

From the raw per-episode results of step 4e: the length of a decision step in
each scenario set, when each agent fires (step and time, and how long before
the hindsight oracle), and hit probability and firing time per test scenario.
None of these analyses was prespecified. Writes results/discussion_analysis.json.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from paper_numbers import AGENTS, ORACLE, RAW, TUNED_RULES  # noqa: E402

sys.path.insert(0, str(ROOT))
from rl_training.utils.benchmark_metrics import agent_group_difference  # noqa: E402

OUTPUT = ROOT / "results" / "discussion_analysis.json"
SETS = {"tuning": ROOT / "results/stage4e_eval/tuning_run" / RAW,
        "test": ROOT / "results/stage4e_eval/evaluation_100" / RAW,
        "random2": ROOT / "results/stage4e_eval/heldout2" / RAW,
        "random1": ROOT / "results/stage4f_eval/heldout" / RAW}
POLICIES = {**AGENTS, "Plateau": [TUNED_RULES["Plateau"]], "Decaying": [TUNED_RULES["Decaying"]], "Oracle": [ORACLE]}


def timing(df):
    """Step length per set; firing step, time and lead on the oracle per policy (episodes with a shot)."""
    shots = df[df["fired_at_step"] > 0]
    step_s = (shots["fire_time_s"] / shots["fired_at_step"]).groupby(shots["scenario"]).median()
    oracle = df[df["policy"] == ORACLE].set_index(["scenario", "seed"])["fire_time_s"]
    out = {"step_length_s": {"median": float(step_s.median()), "min": float(step_s.min()),
                             "max": float(step_s.max())}, "policies": {}}
    for name, members in POLICIES.items():
        fired = df[df["policy"].isin(members) & df["fired"].astype(bool)]
        keys = pd.MultiIndex.from_frame(fired[["scenario", "seed"]])
        lead = oracle.reindex(keys).to_numpy() - fired["fire_time_s"].to_numpy()
        out["policies"][name] = {"firing_rate_pct": float(100 * df[df["policy"].isin(members)]["fired"].mean()),
                                 "mean_fire_step": float(fired["fired_at_step"].mean()),
                                 "mean_fire_time_s": float(fired["fire_time_s"].mean()),
                                 "mean_lead_on_oracle_s": float(lead.mean())}
    return out


def per_scenario(df):
    """Mean HP (no firing scores 0) and mean firing time per scenario and policy."""
    out = {}
    for scenario, g in df.groupby("scenario"):
        out[scenario] = {name: {"hp": float(g[g["policy"].isin(m)]["hit_probability"].mean()),
                                "fire_time_s": float(g[g["policy"].isin(m)]["fire_time_s"].mean())}
                         for name, m in POLICIES.items()}
    return out


def main():
    data = {name: pd.read_csv(path) for name, path in SETS.items()}
    # Ext-sec and Ext-steps differ only in the time input (Section 3.7).
    time_input = {}
    for name, df in data.items():
        g = agent_group_difference(df.to_dict("records"), AGENTS["Ext-sec"], AGENTS["Ext-steps"], n_boot=10000)
        time_input[name] = {"ext_sec_minus_ext_steps": g["mean_diff"], "ci_two_sided": g["ci_two_sided"]}
    # Across the evaluation sets: how often the tuned Plateau rule fires, and the agents' difference from it.
    numbers = json.loads((ROOT / "results/paper_numbers.json").read_text())
    rule_firing = {}
    for name in ("test", "noise1.5", "noise2", "range1.2", "random1", "random2", "size0.7"):
        sec = numbers[name]
        rule_firing[name] = {"plateau_firing_rate_pct": sec["rules"]["Plateau"]["min_firing_rate_pct"],
                             "oracle_hp": sec["oracle"]["mean_hp"],
                             **{f"{agent}_minus_plateau": sec["tests"][agent]["Plateau"]["mean_diff"]
                                for agent in AGENTS}}
    result = {"note": "exploratory; not prespecified",
              "rule_firing_vs_advantage": rule_firing,
              "time_input_effect": time_input,
              "timing": {name: timing(df) for name, df in data.items()},
              "test_per_scenario": per_scenario(data["test"])}
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
