#!/usr/bin/env python3
"""
Non-inferiority report of the final evaluation (TODO step 4, claims T1/T3).

Reads the raw per-episode CSV of every run in results/final_eval/ and tests
the selected agent (variant B, five training seeds) against the
pre-registered rules: Plateau 0.45/k3/eps0.02 (primary, the best rule on the
tuning set) and Decaying 1.00->0.40 (secondary), with delta = 0.02 (primary)
and 0.01 (secondary). Also reports HP, firing rate, firing time and 200 m
blocks of the agent (seed average) and the reference policies.

Writes results/final_eval/noninferiority.json and prints a summary table.

For variant B' (TODO stage 2e) the agent, rules, runs and output are set on
the command line, and --compare_prefix adds a paired comparison of two
multi-seed agents (H11: B' vs B), e.g.
  --results_dir results/stage2e_eval --agent_prefix S2_bp_s --compare_prefix S2_b_s
  --rules DQN Plateau_0.45_k3_e0.02 Decaying_1.00_0.40 --deltas 0.02
  --runs heldout2 evaluation_100 heldout shift_noise2 shift_size0.7
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rl_training.utils.benchmark_metrics import (  # noqa: E402
    agent_group_difference,
    non_inferiority,
    pooled_summary,
)

AGENT_PREFIX = "S2_b_s"
N_TRAINING_SEEDS = 5
AGENTS = tuple(f"{AGENT_PREFIX}{s}" for s in range(N_TRAINING_SEEDS))
RULES = ("Plateau_0.45_k3_e0.02", "Decaying_1.00_0.40")
DELTAS = (0.02, 0.01)
REFERENCES = ("DQN", "Oracle", "Threshold_0.45")
RUNS = ("evaluation_100", "phase4", "heldout",
        "shift_noise1.5", "shift_noise2", "shift_size0.7", "shift_range1.2")
RAW_NAME = "benchmark_traditional_vs_rl_raw.csv"
SUMMARY_COLUMNS = ("mean_hp", "firing_rate_pct", "mean_fire_time_s", "n_blocked_episodes")


def _seed_summary(pooled, agents):
    """Mean over training seeds (min for the firing rate, sum for blocks)."""
    per_agent = pooled.loc[list(agents), list(SUMMARY_COLUMNS)]
    return {
        "mean_hp": float(per_agent["mean_hp"].mean()),
        "sd_hp_over_seeds": float(per_agent["mean_hp"].std(ddof=1)) if len(agents) > 1 else float("nan"),
        "min_firing_rate_pct": float(per_agent["firing_rate_pct"].min()),
        "mean_fire_time_s": float(per_agent["mean_fire_time_s"].mean()),
        "n_blocked_episodes": int(per_agent["n_blocked_episodes"].sum()),
    }


def analyse_run(rows, agents=AGENTS, rules=RULES, deltas=DELTAS, references=REFERENCES,
                n_boot=10000, seed=0, compare_agents=()):
    """Non-inferiority tests and summary metrics for one run.

    rows are per-episode results (all policies). Only agents, rules, the
    compared agents and the references present in the rows are summarised;
    an agent group is summarised over its training seeds. With
    compare_agents, the paired difference agents - compare_agents is added.
    """
    df = pd.DataFrame(rows)
    present = set(df["policy"])
    compare_agents = list(compare_agents)
    missing = [p for p in (*agents, *rules, *compare_agents) if p not in present]
    if missing:
        raise ValueError(f"Policies missing from the run: {missing}")
    singles = [*rules, *(r for r in references if r in present and r not in rules)]
    keep = [*agents, *compare_agents, *singles]
    pooled = pooled_summary(df[df["policy"].isin(keep)].to_dict("records")).set_index("policy")

    summary = {"agent": _seed_summary(pooled, agents)}
    if compare_agents:
        summary["compare_agent"] = _seed_summary(pooled, compare_agents)
    for name in singles:
        summary[name] = {c: float(pooled.loc[name, c]) for c in SUMMARY_COLUMNS}

    tests = []
    for rule in rules:
        for delta in deltas:
            r = non_inferiority(rows, agents, rule, delta, n_boot=n_boot, seed=seed)
            tests.append({k: r[k] for k in ("rule", "delta", "mean_diff", "lower_bound",
                                             "per_agent_mean_diff", "non_inferior",
                                             "n_agents", "n_episodes")})
    result = {"summary": summary, "tests": tests}
    if compare_agents:
        result["group_difference"] = agent_group_difference(rows, agents, compare_agents,
                                                            n_boot=n_boot, seed=seed)
    return result


def format_report(results):
    """Markdown table: one line per (run, rule, delta)."""
    lines = ["| Run | Agent HP | Rule | Rule HP | Diff | Lower bound | delta | Non-inferior |",
             "|---|---|---|---|---|---|---|---|"]
    for run, res in results.items():
        agent_hp = res["summary"]["agent"]["mean_hp"]
        for t in res["tests"]:
            rule_hp = res["summary"][t["rule"]]["mean_hp"]
            lines.append(f"| {run} | {agent_hp:.3f} | {t['rule']} | {rule_hp:.3f} | {t['mean_diff']:+.3f} "
                         f"| {t['lower_bound']:+.3f} | {t['delta']:.2f} | {'yes' if t['non_inferior'] else 'no'} |")
    groups = [(run, res["group_difference"]) for run, res in results.items() if "group_difference" in res]
    if groups:
        lines += ["", "| Run | Agent HP | Compared agent HP | Diff | Lower bound | Two-sided 95% CI | Verdict |",
                  "|---|---|---|---|---|---|---|"]
        for run, g in groups:
            verdict = "agent better" if g["a_better"] else "compared better" if g["b_better"] else "no difference"
            low2, high2 = g["ci_two_sided"]
            lines.append(f"| {run} | {g['mean_a']:.3f} | {g['mean_b']:.3f} | {g['mean_diff']:+.3f} "
                         f"| {g['lower_bound']:+.3f} | [{low2:+.3f}, {high2:+.3f}] | {verdict} |")
    return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=ROOT / "results/final_eval")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--agent_prefix", default=AGENT_PREFIX,
                        help="Agent policy names are <prefix>0 .. <prefix>4")
    parser.add_argument("--compare_prefix", default=None,
                        help="Second multi-seed agent for a paired group comparison (e.g. S2_b_s)")
    parser.add_argument("--rules", nargs="+", default=list(RULES),
                        help="Single policies the agent is tested against (non-inferiority)")
    parser.add_argument("--deltas", nargs="+", type=float, default=list(DELTAS))
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--output", default="noninferiority.json")
    args = parser.parse_args(argv)

    agents = [f"{args.agent_prefix}{s}" for s in range(N_TRAINING_SEEDS)]
    compare = [f"{args.compare_prefix}{s}" for s in range(N_TRAINING_SEEDS)] if args.compare_prefix else []
    results = {}
    for run in args.runs:
        raw = args.results_dir / run / RAW_NAME
        if not raw.exists():
            print(f"skip {run}: {raw} not found")
            continue
        rows = pd.read_csv(raw).to_dict("records")
        results[run] = analyse_run(rows, agents=agents, rules=args.rules, deltas=args.deltas,
                                   n_boot=args.n_boot, compare_agents=compare)
    if not results:
        parser.error(f"No runs found in {args.results_dir}")

    out = args.results_dir / args.output
    out.write_text(json.dumps(results, indent=2, default=_json_default))
    print(format_report(results))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
