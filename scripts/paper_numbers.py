#!/usr/bin/env python3
"""
Collect the numbers reported in the Results section of the manuscript.

Every value in the Results text and tables comes from results/paper_numbers.json,
which this script builds from the raw per-episode results in results/stage4e_eval
and results/stage4f_eval with the tests of Section 4.3 (same functions, n_boot and
bootstrap seed as scripts/analyse_step4e.py, so shared values are identical).

Usage: python scripts/paper_numbers.py [--sections test ...]
"""

import argparse
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

RAW = "benchmark_traditional_vs_rl_raw.csv"
OUTPUT = ROOT / "results" / "paper_numbers.json"
N_BOOT = 10000
DELTA = 0.02

# Manuscript names (Section 3.7) -> policy names in the result files.
AGENTS = {
    "Base-0": ["DQN"],
    "Base": [f"V0r_s{s}" for s in range(3)],
    "Ext-steps": [f"S2_b_s{s}" for s in range(5)],
    "Ext-sec": [f"S2_bp_s{s}" for s in range(5)],
}
# Tuned rules (Section 4.2), selected on the tuning set by scripts/analyse_step4e.py.
TUNED_RULES = {
    "Fixed": "Threshold_0.45",
    "Decaying": "Decaying_1.00_0.40",
    "Plateau": "Plateau_0.40_k3_e0.01",
    "Covariance-gated": "CovGated_0.45_1",
}
ORACLE = "Oracle"

RUNS = {
    "test": ROOT / "results/stage4e_eval/evaluation_100" / RAW,
    "random1": ROOT / "results/stage4f_eval/heldout" / RAW,
    "random2": ROOT / "results/stage4e_eval/heldout2" / RAW,
    "noise1.5": ROOT / "results/stage4f_eval/shift_noise1.5" / RAW,
    "noise2": ROOT / "results/stage4f_eval/shift_noise2" / RAW,
    "size0.7": ROOT / "results/stage4f_eval/shift_size0.7" / RAW,
    "range1.2": ROOT / "results/stage4f_eval/shift_range1.2" / RAW,
}
# Agent-versus-agent comparisons (Section 4.3), as (a, b): difference a - b.
AGENT_PAIRS = {"random1": [("Base", "Ext-sec")], "random2": [("Base", "Ext-sec")]}


def policy_summary(rows, members):
    """Mean HP over training seeds (and its SD), lowest firing rate, mean firing time."""
    pooled = pooled_summary([r for r in rows if r["policy"] in members]).set_index("policy").loc[members]
    return {
        "mean_hp": float(pooled["mean_hp"].mean()),
        "sd_hp_over_seeds": float(pooled["mean_hp"].std(ddof=1)) if len(members) > 1 else None,
        "min_firing_rate_pct": float(pooled["firing_rate_pct"].min()),
        "mean_fire_time_s": float(pooled["mean_fire_time_s"].mean()),
        "mean_hp_fired": float(pooled["mean_hp_fired"].mean()),
        "n_seeds": len(members),
    }


def agent_vs_rules(rows, members):
    """Paired difference and one-sided 95% lower bound against each tuned rule (Section 4.3)."""
    out = {}
    for family, rule in TUNED_RULES.items():
        r = non_inferiority(rows, members, rule, DELTA, n_boot=N_BOOT)
        out[family] = {"mean_diff": r["mean_diff"], "lower_bound": r["lower_bound"],
                       "non_inferior": bool(r["lower_bound"] > -DELTA), "better": bool(r["lower_bound"] > 0)}
    return out


def agent_pair(rows, a, b):
    """Difference of mean HP a - b with the two-sided 95% interval (Section 4.3)."""
    g = agent_group_difference(rows, AGENTS[a], AGENTS[b], n_boot=N_BOOT)
    return {"a": a, "b": b, "mean_diff": g["mean_diff"], "ci_two_sided": g["ci_two_sided"]}


def section(path, name):
    rows = pd.read_csv(path).to_dict("records")
    return {
        "source": str(path.relative_to(ROOT)),
        "n_episodes": int(sum(1 for r in rows if r["policy"] == ORACLE)),
        "agents": {name: policy_summary(rows, members) for name, members in AGENTS.items()},
        "rules": {family: {"policy": rule, **policy_summary(rows, [rule])} for family, rule in TUNED_RULES.items()},
        "oracle": policy_summary(rows, [ORACLE]),
        "tests": {agent: agent_vs_rules(rows, members) for agent, members in AGENTS.items()},
        "agent_pairs": [agent_pair(rows, a, b) for a, b in AGENT_PAIRS.get(name, [])],
    }


def hp_validation():
    """HP validation (Section 4.4) from results/hp_validation_4e/, with the NEES acceptance interval."""
    from scipy.stats import chi2

    folder = ROOT / "results/hp_validation_4e"
    summary = json.loads((folder / "hp_validation_summary.json").read_text())
    nees = json.loads((folder / "calibration_diagnosis_summary.json").read_text())
    n, dim = nees["n"], 6  # 6-dimensional target state
    nees["acceptance_95"] = [chi2.ppf(0.025, n * dim) / n, chi2.ppf(0.975, n * dim) / n]
    keep = ("n_shots", "rebuild_ok", "mc_model_hp_minus_base", "full_cov_hp_minus_base",
            "true_hit_rate_minus_base", "by_range", "calibration")
    # Calibration by analytic-HP bin as in Table 9 (the two lowest bins of the summary merged).
    shots = pd.read_csv(folder / "hp_validation_shots.csv")
    table = []
    for interval, g in shots.groupby(pd.cut(shots["base_hp"], [0.0, 0.4, 0.6, 0.8, 1.0]), observed=True):
        table.append({"hp_bin": str(interval), "n": len(g), "analytic": float(g.base_hp.mean()),
                      "monte_carlo": float(g.mc_model_hp.mean()), "full_covariance": float(g.full_cov_hp.mean()),
                      "true_target": float(g.true_hit_rate.mean())})
    return {"source": str(folder.relative_to(ROOT)), **{k: summary[k] for k in keep}, "nees": nees,
            "calibration_table": table}


def decision_cost():
    """Per-decision cost (Section 4.4) from results/decision_cost_benchmark.json, medians."""
    path = ROOT / "results/decision_cost_benchmark.json"
    data = json.loads(path.read_text())
    per = {}
    for scenario, v in data["per_scenario"].items():
        c, d = v["components"], v["per_decision"]
        per[scenario] = {"flight_time_s": c["_inputs"]["impact_time_s"],
                         "rule_hp_chain_ms": c["hp_propagation_diagonal"]["median_us"] / 1000,
                         "kalman_update_us": c["kalman_update"]["median_us"],
                         "forward_pass_us": c["dqn_forward"]["median_us"],
                         "agent_decision_us": d["dqn_decision_us"],
                         "ratio": c["hp_propagation_diagonal"]["median_us"] / d["dqn_decision_us"]}
    return {"source": str(path.relative_to(ROOT)), "hardware": data["hardware"], "per_scenario": per}


EXTRA = {"hp_validation": hp_validation, "decision_cost": decision_cost}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+", default=[*RUNS, *EXTRA], choices=[*RUNS, *EXTRA])
    args = parser.parse_args(argv)
    numbers = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    numbers["config"] = {"n_boot": N_BOOT, "delta": DELTA, "agents": AGENTS, "tuned_rules": TUNED_RULES}
    for name in args.sections:
        numbers[name] = EXTRA[name]() if name in EXTRA else section(RUNS[name], name)
    OUTPUT.write_text(json.dumps(numbers, indent=2))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
