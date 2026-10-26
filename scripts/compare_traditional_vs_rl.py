#!/usr/bin/env python3
"""
Benchmark classical FCS firing rules against the trained DQN agent.

Every policy is evaluated on the same five Phase-4 scenarios with the same
per-seed perturbations as evaluate_dqn_trajectories.py, so results are paired
by (scenario, seed). FIRE inside the 200 m dead zone is blocked for all policies.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rl_training.agents.baseline_policies import (
    CovarianceGatedThresholdPolicy,
    DecayingThresholdPolicy,
    PlateauPolicy,
    ThresholdPolicy,
)
from rl_training.utils.evaluation_protocol import (
    MIN_FIRING_RANGE_M,
    make_baseline_decide,
    make_dqn_decide,
    make_oracle_decide,
    perturb_scenario,
    run_policy_episode,
    step_seconds,
)

THRESHOLDS = tuple(round(0.40 + 0.05 * i, 2) for i in range(11))  # 0.40 .. 0.90
TAU_COV_VALUES = (1.0, 2.0, 4.0)  # raw trace; above every scenario's converged level (~0.1-0.7)
DECAYING_GRID = tuple((p_start, p_end) for p_start in (0.70, 0.80, 0.90, 1.00)
                      for p_end in (0.30, 0.35, 0.40, 0.45, 0.50))
PLATEAU_GRID = tuple((min_hp, window, eps) for min_hp in (0.40, 0.45, 0.50)
                     for window in (3, 5) for eps in (0.005, 0.01, 0.02))
RULE_FAMILIES = (("threshold", "Threshold_"), ("decaying", "Decaying_"), ("plateau", "Plateau_"),
                 ("covariance_gated", "CovGated_"))
DQN_POLICY_NAME = "DQN"
ORACLE_POLICY_NAME = "Oracle"


EXTRA_DQN_SUFFIXES = {":cov0": "zero", ":covfrozen": "frozen"}


def parse_extra_dqn(spec):
    """Parse an --extra_dqn value NAME=PATH[:cov0|:covfrozen].

    :cov0 / :covfrozen evaluate the agent with the covariance input zeroed /
    frozen at its first value. Returns (name, Path, cov_ablation).
    """
    name, sep, rest = spec.partition("=")
    if not sep or not name or not rest:
        raise ValueError(f"--extra_dqn must be NAME=PATH[:cov0|:covfrozen], got {spec!r}")
    cov_ablation = None
    for suffix, value in EXTRA_DQN_SUFFIXES.items():
        if rest.endswith(suffix):
            cov_ablation = value
            rest = rest[:-len(suffix)]
            break
    if any(rest.endswith(suffix) for suffix in EXTRA_DQN_SUFFIXES):
        raise ValueError(f"--extra_dqn allows one covariance ablation, got {spec!r}")
    if not rest:
        raise ValueError(f"--extra_dqn has no checkpoint path: {spec!r}")
    if name in (DQN_POLICY_NAME, ORACLE_POLICY_NAME):
        raise ValueError(f"--extra_dqn name {name!r} is reserved")
    return name, Path(rest), cov_ablation


def parse_ensemble(spec):
    """Parse an --ensemble value NAME=PATH1,PATH2,... Returns (name, [Path, ...])."""
    name, sep, rest = spec.partition("=")
    paths = [p for p in rest.split(",") if p]
    if not sep or not name or len(paths) < 2:
        raise ValueError(f"--ensemble must be NAME=PATH1,PATH2[,...] with at least two paths, got {spec!r}")
    if name in (DQN_POLICY_NAME, ORACLE_POLICY_NAME):
        raise ValueError(f"--ensemble name {name!r} is reserved")
    return name, [Path(p) for p in paths]


def build_policies(q_network=None, device=None, thresholds=THRESHOLDS, tau_cov_values=(),
                   decaying_grid=(), plateau_grid=(), oracle_env=None,
                   min_firing_range=MIN_FIRING_RANGE_M,
                   extra_dqns=()):
    """Return a list of (name, decide) pairs for all benchmarked policies.

    The DQN is included when q_network is given, followed by extra_dqns, a
    sequence of (name, q_network, cov_ablation[, cov_feature[, time_feature]]) for
    additional agents (defaults "linear" and "steps", the published inputs). The
    hindsight oracle is included when oracle_env (the env the benchmark runs
    on) is given; min_firing_range must match the one used by run_benchmark.
    The covariance-gated rule is built for every combination of threshold and
    tau_cov; the decaying rule for every (p_start, p_end) in decaying_grid; the
    plateau rule for every (min_hp, window, eps) in plateau_grid.
    """
    policies = []
    if q_network is not None:
        policies.append((DQN_POLICY_NAME, make_dqn_decide(q_network, device)))
    for name, extra_network, cov_ablation, *rest in extra_dqns:
        cov_feature = rest[0] if rest else "linear"
        time_input = rest[1] if len(rest) > 1 else "steps"
        policies.append((name, make_dqn_decide(extra_network, device, cov_ablation=cov_ablation,
                                               cov_feature=cov_feature, time_input=time_input)))
    if oracle_env is not None:
        policies.append((ORACLE_POLICY_NAME, make_oracle_decide(oracle_env, min_firing_range)))
    for p_thr in thresholds:
        policy = ThresholdPolicy(p_thr)
        policies.append((policy.name, make_baseline_decide(policy)))
    for min_hp, window, eps in plateau_grid:
        policy = PlateauPolicy(min_hp, window, eps)
        policies.append((policy.name, make_baseline_decide(policy)))
    for p_start, p_end in decaying_grid:
        policy = DecayingThresholdPolicy(p_start, p_end)
        policies.append((policy.name, make_baseline_decide(policy)))
    for p_thr in thresholds:
        for tau_cov in tau_cov_values:
            policy = CovarianceGatedThresholdPolicy(p_thr, tau_cov)
            policies.append((policy.name, make_baseline_decide(policy)))
    return policies


def run_benchmark(env, scenarios, policies, seeds, min_firing_range=MIN_FIRING_RANGE_M,
                  max_steps=100, progress=None, on_seed_done=None):
    """Run every policy on every perturbed scenario for every seed.

    Scenarios are perturbed once per seed, in order, from
    np.random.default_rng(seed) -- the same draws as evaluate_dqn_trajectories.
    All random state is reset per seed, so a seed's results do not depend on
    which seeds ran before it (this is what makes --resume exact).
    If given, progress(done, total) is called after each (seed, scenario)
    pair has been run for all policies, and on_seed_done(seed, seed_rows)
    after all scenarios of a seed.
    Returns one row (dict) per (seed, scenario, policy) episode.
    """
    seeds = list(seeds)
    total = len(seeds) * len(scenarios)
    done = 0
    rows = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        perturbed = [perturb_scenario(s, rng) for s in scenarios]
        seed_start = len(rows)
        for scenario in perturbed:
            for name, decide in policies:
                result = run_policy_episode(env, scenario, decide, seed=seed,
                                            max_steps=max_steps,
                                            min_firing_range=min_firing_range)
                pos = result["firing_pos"]
                rows.append({
                    "policy": name,
                    "scenario": scenario.name,
                    "seed": seed,
                    "fired": result["fired"],
                    "timed_out": result["timed_out"],
                    "fired_at_step": result["fired_at_step"],
                    "fire_time_s": result["fire_time_s"],
                    "hit_probability": result["hit_probability"],
                    "firing_slant_range": result["firing_slant_range"],
                    "firing_x": pos[0] if pos is not None else None,
                    "steps": result["steps"],
                    "blocked_fire_attempts": result["blocked_fire_attempts"],
                })
            done += 1
            if progress is not None:
                progress(done, total)
        if on_seed_done is not None:
            on_seed_done(seed, rows[seed_start:])
    return rows


PARTIAL_DIR_NAME = "partial"
PARTIAL_CONFIG_NAME = "config.json"


def _seed_file(partial_dir, seed):
    return Path(partial_dir) / f"seed_{seed:04d}.csv"


def init_partial(partial_dir, config, resume):
    """Prepare the per-seed checkpoint directory of a run.

    config (JSON-serialisable: policies, scenarios, scenario set, shift)
    identifies the run. Without resume, existing checkpoints are an error,
    so they are never overwritten silently. With resume, the stored config
    must equal config, so results of a different run are never mixed in.
    """
    import json

    partial_dir = Path(partial_dir)
    config_path = partial_dir / PARTIAL_CONFIG_NAME
    existing = sorted(partial_dir.glob("seed_*.csv")) if partial_dir.exists() else []
    if not resume:
        if existing or config_path.exists():
            raise ValueError(f"{partial_dir} already holds checkpoints; use --resume or remove it")
        partial_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2))
        return
    if not config_path.exists():
        if existing:
            raise ValueError(f"{partial_dir} has seed files but no {PARTIAL_CONFIG_NAME}")
        partial_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2))
        return
    stored = json.loads(config_path.read_text())
    if stored != config:
        diff = sorted(k for k in set(stored) | set(config) if stored.get(k) != config.get(k))
        raise ValueError(f"Cannot resume: run configuration differs from {config_path} in {diff}")


def save_seed_rows(partial_dir, seed, seed_rows):
    """Write one completed seed atomically (temporary file, then rename)."""
    import os

    import pandas as pd

    path = _seed_file(partial_dir, seed)
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(seed_rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_completed_seeds(partial_dir, seeds, n_rows_per_seed):
    """Rows of the seeds already checkpointed, as {seed: rows}.

    Only seeds in seeds are read. A file with the wrong number of rows is an
    error (it cannot come from a complete seed of this run).
    """
    done = {}
    for seed in seeds:
        path = _seed_file(partial_dir, seed)
        if not path.exists():
            continue
        rows = load_raw_rows(path)
        if len(rows) != n_rows_per_seed or any(r["seed"] != seed for r in rows):
            raise ValueError(f"{path} is not a complete checkpoint of seed {seed} "
                             f"({len(rows)} rows, expected {n_rows_per_seed})")
        done[seed] = rows
    return done


def add_fire_times(rows, env, scenarios):
    """Fill fire_time_s for rows produced before it was recorded.

    The step length of an episode depends on its perturbed scenario, so the
    perturbations are regenerated per seed exactly as in run_benchmark and
    converted with step_seconds(). Rows that already have fire_time_s are
    kept as they are. Returns new row dicts.
    """
    step_length = {}
    for seed in sorted({int(r["seed"]) for r in rows}):
        rng = np.random.default_rng(seed)
        for scenario in [perturb_scenario(s, rng) for s in scenarios]:
            step_length[(scenario.name, seed)] = step_seconds(env, scenario)

    out = []
    for r in rows:
        r = dict(r)
        if r.get("fire_time_s") is None:
            fired = bool(r["fired"])
            r["fire_time_s"] = (float(r["fired_at_step"]) * step_length[(r["scenario"], int(r["seed"]))]
                                if fired else None)
        out.append(r)
    return out


def load_raw_rows(path):
    """Read a raw per-episode CSV back into row dicts (NaN -> None)."""
    import pandas as pd

    df = pd.read_csv(path)
    df = df.astype(object).where(df.notna(), None)
    rows = df.to_dict("records")
    for r in rows:
        r["fired"] = bool(r["fired"])
        r["timed_out"] = bool(r["timed_out"])
        r["seed"] = int(r["seed"])
        r["blocked_fire_attempts"] = int(r["blocked_fire_attempts"])
    return rows


TABLE_THRESHOLDS = ("Threshold_0.45", "Threshold_0.55", "Threshold_0.65")
OUTPUT_STEM = "benchmark_traditional_vs_rl"


def policy_label(name, best=()):
    """LaTeX label for a policy name; policies in best get a dagger."""
    if name.startswith("Threshold_"):
        label = rf"$P_{{\mathrm{{thr}}}}={name.split('_')[1]}$"
    elif name.startswith("Plateau_"):
        _, min_hp, window, eps = name.split("_")
        label = rf"Plateau $P_{{\mathrm{{hit}}}}\geq{min_hp}$, $k={window[1:]}$, $\varepsilon={eps[1:]}$"
    elif name.startswith("Decaying_"):
        _, p_start, p_end = name.split("_")
        label = rf"$P_{{\mathrm{{thr}}}}: {p_start} \to {p_end}$"
    elif name.startswith("CovGated_"):
        _, p_thr, tau = name.split("_")
        label = rf"$P_{{\mathrm{{thr}}}}={p_thr}$, $\tau_{{\mathrm{{cov}}}}={tau}$"
    elif name == ORACLE_POLICY_NAME:
        label = r"Hindsight oracle$^{\ddagger}$"
    else:
        label = name.replace("_", r"\_")
    return label + (r"$^{\dagger}$" if name in best else "")


def _format_p(p, is_reference):
    if is_reference:
        return "--"
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


def _format_optional(value, fmt):
    return "--" if value is None or not np.isfinite(value) else format(value, fmt)


def format_latex_table(summary, pooled, comparisons, table_policies, best=(),
                       reference=DQN_POLICY_NAME, caption="", label="tab:comparison-traditional-rl"):
    """Build the comparison table: one block per scenario plus a pooled block.

    summary / pooled / comparisons are the outputs of aggregate_results,
    pooled_summary and paired_comparisons.
    """
    from rl_training.utils.benchmark_metrics import POOLED_LABEL

    p_lookup = {(r.baseline, r.scenario): r.p_holm for r in comparisons.itertuples()}
    blocks = [(s, summary[summary["scenario"] == s]) for s in _unique(summary["scenario"])]
    blocks.append((POOLED_LABEL, pooled))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Policy} & \textbf{Mean HP ($\pm$95\% CI)} & "
        r"\textbf{Fire rate (\%)} & \textbf{Time to fire (s)} & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for i, (scenario, block) in enumerate(blocks):
        rows = [block[block["policy"] == p].iloc[0] for p in table_policies
                if (block["policy"] == p).any()]
        name = "All scenarios" if scenario == POOLED_LABEL else scenario.replace("_", r"\_")
        for j, r in enumerate(rows):
            first = rf"\multirow{{{len(rows)}}}{{*}}{{{name}}}" if j == 0 else ""
            ci = _format_optional(r["ci95_hp"], ".3f")
            lines.append(
                f"{first} & {policy_label(r['policy'], best)} & "
                f"{r['mean_hp']:.3f} $\\pm$ {ci} & {r['firing_rate_pct']:.0f} & "
                f"{_format_optional(r['mean_fire_time_s'], '.1f')} & "
                f"{_format_p(p_lookup.get((r['policy'], scenario)), r['policy'] == reference)} \\\\"
            )
        if i < len(blocks) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _unique(values):
    """Unique values in order of first appearance."""
    return list(dict.fromkeys(values))


def _records(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def export_results(rows, out_dir, table_path, selected_on="evaluation scenarios (oracle tuning)",
                   n_boot=10000, metadata=None, selection_rows=None):
    """Write raw/aggregated CSVs, the JSON summary and the LaTeX table.

    The best rule of each family (fixed, decaying, covariance-gated) is chosen
    on selection_rows if given (e.g. separate tuning scenarios), otherwise on
    the given rows; selected_on records which scenarios that was. All reported
    results, including the chosen rules', come from rows.
    Returns a dict of written paths.
    """
    import json

    import pandas as pd

    from rl_training.utils.benchmark_metrics import (
        aggregate_results,
        paired_comparisons,
        pareto_front,
        pooled_summary,
        select_best_fixed_threshold,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_results(rows)
    pooled = pooled_summary(rows)
    comparisons = paired_comparisons(rows, reference=DQN_POLICY_NAME, n_boot=n_boot)

    select_from = rows if selection_rows is None else selection_rows
    evaluated_hp = dict(zip(pooled["policy"], pooled["mean_hp"]))
    best = {}
    for family, prefix in RULE_FAMILIES:
        if not any(r["policy"].startswith(prefix) for r in select_from):
            continue
        name, selection_hp = select_best_fixed_threshold(select_from, prefix=prefix)
        if name not in evaluated_hp:
            raise ValueError(f"Rule {name} chosen on the selection rows was not evaluated")
        best[family] = {"policy": name, "pooled_mean_hp": float(evaluated_hp[name])}
        if selection_rows is not None:
            best[family]["selection_pooled_mean_hp"] = selection_hp
    best_names = [b["policy"] for b in best.values()]

    present = _unique(r["policy"] for r in rows)
    table_policies = [p for p in _unique([DQN_POLICY_NAME, *TABLE_THRESHOLDS, *best_names,
                                          ORACLE_POLICY_NAME])
                      if p in present]

    paths = {
        "raw_csv": out_dir / f"{OUTPUT_STEM}_raw.csv",
        "summary_csv": out_dir / f"{OUTPUT_STEM}.csv",
        "pooled_csv": out_dir / f"{OUTPUT_STEM}_pooled.csv",
        "paired_csv": out_dir / f"{OUTPUT_STEM}_paired_tests.csv",
        "json": out_dir / f"{OUTPUT_STEM}.json",
        "table": Path(table_path),
    }
    pd.DataFrame(rows).to_csv(paths["raw_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    pooled.to_csv(paths["pooled_csv"], index=False)
    comparisons.to_csv(paths["paired_csv"], index=False)

    seeds = _unique(r["seed"] for r in rows)
    payload = {
        "metadata": {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "scenarios": _unique(r["scenario"] for r in rows),
            "policies": present,
            "hp_definition": "unconditional; timeouts scored as 0",
            "best_fixed_selected_on": selected_on,
            **(metadata or {}),
        },
        "best_fixed": best,
        "pareto_front_hp_vs_time": pareto_front(pooled),
        "summary": _records(summary),
        "pooled": _records(pooled),
        "paired_tests": _records(comparisons),
    }
    paths["json"].write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    caption = (
        "Comparison of the DQN agent with classical threshold firing rules over "
        f"{len(seeds)} perturbed runs per scenario. Mean HP is unconditional (timeouts scored as 0); "
        "$p_{\\mathrm{Holm}}$: paired Wilcoxon signed-rank test against the DQN, Holm-corrected "
        "(n/a: identical outcomes). $^{\\dagger}$Best rule of its family (fixed, decaying, plateau or "
        "covariance-gated threshold), selected on the "
        f"{selected_on}. $^{{\\ddagger}}$Upper bound: highest HP reachable in each episode, "
        "chosen in hindsight from the precomputed trajectory."
    )
    paths["table"].parent.mkdir(parents=True, exist_ok=True)
    paths["table"].write_text(
        format_latex_table(summary, pooled, comparisons, table_policies, best=best_names,
                           caption=caption),
        encoding="utf-8",
    )
    return paths


TUNING_SELECTED_ON = "10 Phase-4 tuning scenarios not used for evaluation"


def scenarios_for(scenario_set, curriculum, shift="none"):
    """Scenarios of a named set with an optional distribution shift.

    Sets: "evaluation" (the 5 test scenarios), "tuning" (the other 10 Phase-4
    scenarios), "phase4" (all 15), "heldout2" (30 random scenarios with the second
    pre-registered seed, stage 2e) and "heldout" (30 random scenarios with the
    pre-registered seed). The shift is applied to the nominal scenarios, before
    the per-seed perturbation.
    """
    from rl_training.utils.evaluation_protocol import (
        apply_shift,
        get_evaluation_scenarios,
        HELDOUT2_SEED,
        get_heldout_scenarios,
        get_tuning_scenarios,
    )

    if scenario_set == "evaluation":
        scenarios = get_evaluation_scenarios(curriculum)
    elif scenario_set == "tuning":
        scenarios = get_tuning_scenarios(curriculum)
    elif scenario_set == "phase4":
        scenarios = curriculum.get_all_scenarios()
    elif scenario_set == "heldout":
        scenarios = get_heldout_scenarios()
    elif scenario_set == "heldout2":
        scenarios = get_heldout_scenarios(seed=HELDOUT2_SEED, prefix="Heldout2")
    else:
        raise ValueError(f"Unknown scenario set {scenario_set!r}")
    return [apply_shift(s, shift) for s in scenarios]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_seeds", type=int, default=30,
                        help="Number of seeds (0..num_seeds-1)")
    parser.add_argument("--tau_cov", type=float, nargs="*", default=list(TAU_COV_VALUES),
                        help="Covariance-trace limits for the gated rule (raw trace units)")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results",
                        help="Directory for CSV and JSON results")
    parser.add_argument("--table", type=Path,
                        default=ROOT / "docs/paper2_rl_methodology/table_comparison_traditional_vs_rl.tex",
                        help="Output path of the LaTeX comparison table")
    parser.add_argument("--n_boot", type=int, default=10000,
                        help="Bootstrap resamples for the median-difference CI")
    parser.add_argument("--from_raw", type=Path, default=None,
                        help="Skip simulation: re-export from an existing raw per-episode CSV "
                             "(fills fire_time_s if missing)")
    parser.add_argument("--scenario_set", choices=("evaluation", "tuning", "phase4", "heldout", "heldout2"),
                        default="evaluation",
                        help="5 evaluation, 10 tuning, all 15 Phase-4, or 30 held-out random scenarios "
                             "(heldout2: second held-out set, stage 2e)")
    parser.add_argument("--shift", choices=("none", "noise1.5", "noise2", "size0.7", "range1.2"),
                        default="none", help="Distribution shift applied to every scenario (claim T3)")
    parser.add_argument("--select_on", type=Path, default=None,
                        help="Raw per-episode CSV (e.g. a tuning run) on which the best rule of "
                             "each family is chosen; results are still reported on this run")
    parser.add_argument("--selected_on", type=str, default=None,
                        help="Description of the selection set for the table caption and JSON")
    parser.add_argument("--extra_dqn", action="append", default=[], metavar="NAME=PATH[:cov0|:covfrozen]",
                        help="Additional DQN checkpoint to benchmark (repeatable); append :cov0 or "
                             ":covfrozen to evaluate it with the covariance input zeroed or frozen")
    parser.add_argument("--ensemble", action="append", default=[], metavar="NAME=PATH1,PATH2,...",
                        help="Agent whose Q-values are averaged over several checkpoints (repeatable)")
    parser.add_argument("--extrapolate_estimate", action="store_true",
                        help="Predict the filter estimate to the decision time (corrected physics, step 4b); "
                             "off reproduces the published environment")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run from the per-seed checkpoints in "
                             "<output_dir>/partial (same arguments required)")
    args = parser.parse_args(argv)
    if args.num_seeds < 1:
        parser.error("--num_seeds must be at least 1")
    try:
        extra_specs = [parse_extra_dqn(spec) for spec in args.extra_dqn]
        ensemble_specs = [parse_ensemble(spec) for spec in args.ensemble]
    except ValueError as e:
        parser.error(str(e))
    names = [spec[0] for spec in extra_specs] + [spec[0] for spec in ensemble_specs]
    if len(set(names)) != len(names):
        parser.error("--extra_dqn and --ensemble names must be unique")

    from evaluate_dqn_trajectories import load_dqn_model
    from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
    from rl_training.environment import ArtilleryFiringEnv
    scenarios = scenarios_for(args.scenario_set, CurriculumScenarios(phase=4), args.shift)
    selection = {}
    if args.select_on is not None:
        selection = {"selection_rows": load_raw_rows(args.select_on),
                     "selected_on": args.selected_on or TUNING_SELECTED_ON}
    elif args.selected_on is not None:
        selection = {"selected_on": args.selected_on}

    if args.from_raw is not None:
        env = ArtilleryFiringEnv(extrapolate_estimate=args.extrapolate_estimate)
        rows = add_fire_times(load_raw_rows(args.from_raw), env, scenarios)
        paths = export_results(rows, args.output_dir, args.table, n_boot=args.n_boot,
                               metadata={"reexported_from": str(args.from_raw),
                                         "scenario_set": args.scenario_set,
                                         "shift": args.shift,
                                         "selection_from": str(args.select_on) if args.select_on else None,
                                         "min_firing_range_m": MIN_FIRING_RANGE_M},
                               **selection)
        print(f"Re-exported {len(rows)} episodes from {args.from_raw}.")
        for kind, path in paths.items():
            print(f"  {kind}: {path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ArtilleryFiringEnv(extrapolate_estimate=args.extrapolate_estimate)
    model_path = ROOT / "rl_training/models/checkpoints/dqn_curriculum_phase4.pth"
    q_network = load_dqn_model(model_path, env.observation_space.shape[0],
                               env.action_space.n, device)

    from rl_training.agents.dqn_components import load_ensemble, load_q_network

    extra_dqns, extra_configs = [], {}
    for name, path, cov_ablation in extra_specs:
        network, config = load_q_network(path, env.observation_space.shape[0], env.action_space.n, device)
        extra_dqns.append((name, network, cov_ablation, config["cov_feature"], config["time_feature"]))
        extra_configs[name] = config
    for name, paths in ensemble_specs:
        try:
            network, config = load_ensemble(paths, env.observation_space.shape[0], env.action_space.n, device)
        except ValueError as e:
            parser.error(f"--ensemble {name}: {e}")
        extra_dqns.append((name, network, None, config["cov_feature"], config["time_feature"]))
        extra_configs[name] = config

    policies = build_policies(q_network, device, tau_cov_values=args.tau_cov,
                              decaying_grid=DECAYING_GRID, plateau_grid=PLATEAU_GRID,
                              oracle_env=env, extra_dqns=extra_dqns)
    print(f"Policies: {', '.join(name for name, _ in policies)}")

    # Every completed seed is checkpointed, so an interrupted run can continue with --resume.
    partial_dir = args.output_dir / PARTIAL_DIR_NAME
    seeds = list(range(args.num_seeds))
    try:
        init_partial(partial_dir, {"policies": [name for name, _ in policies],
                                   "scenarios": [s.name for s in scenarios],
                                   "scenario_set": args.scenario_set, "shift": args.shift,
                                   "extrapolate_estimate": args.extrapolate_estimate},
                     resume=args.resume)
        completed = load_completed_seeds(partial_dir, seeds, len(policies) * len(scenarios))
    except ValueError as e:
        parser.error(str(e))
    remaining = [s for s in seeds if s not in completed]
    if completed:
        print(f"Resuming: {len(completed)} seeds loaded from {partial_dir}, {len(remaining)} to run.")

    from tqdm import tqdm

    # One step per (seed, scenario): each needs a new HP trajectory, which dominates run time.
    with tqdm(total=args.num_seeds * len(scenarios), initial=len(completed) * len(scenarios),
              desc="Scenario x seed", unit="traj", file=sys.stderr) as bar:
        new_rows = run_benchmark(env, scenarios, policies, seeds=remaining,
                                 progress=lambda done, total: bar.update(1),
                                 on_seed_done=lambda seed, seed_rows: save_seed_rows(partial_dir, seed, seed_rows))
    by_seed = dict(completed)
    for r in new_rows:
        by_seed.setdefault(r["seed"], []).append(r)
    rows = [r for s in seeds for r in by_seed[s]]

    paths = export_results(
        rows, args.output_dir, args.table, n_boot=args.n_boot,
        metadata={"checkpoint": str(model_path.relative_to(ROOT)),
                  "min_firing_range_m": MIN_FIRING_RANGE_M,
                  "extrapolate_estimate": args.extrapolate_estimate,
                  "tau_cov_values": list(args.tau_cov),
                  "scenario_set": args.scenario_set,
                  "shift": args.shift,
                  "extra_dqns": [{"name": name, "checkpoint": str(path), "cov_ablation": cov_ablation,
                                  "model_config": extra_configs[name]}
                                 for name, path, cov_ablation in extra_specs],
                  "ensembles": [{"name": name, "checkpoints": [str(p) for p in paths],
                                 "model_config": extra_configs[name]} for name, paths in ensemble_specs]},
        **selection,
    )
    print(f"Saved {len(rows)} episodes.")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
