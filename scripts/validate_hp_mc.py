#!/usr/bin/env python3
"""
Monte Carlo validation of the analytical hit probability (TODO step 4b).

Draws real shots from the evaluation results (stratified by scenario set and
policy group), rebuilds each firing situation exactly, and compares the
analytical (base) HP with (1) a nonlinear Monte Carlo of the same model and
(2) the hit rate on the true target. Writes per-shot results and a summary to
results/hp_validation/.

--source 2e (default) samples the step-2e runs; --source 4e samples the
step-4e runs with the rules selected there. A scenario set may pool several
runs; each shot is rebuilt with the scenario set its run used.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SOURCES = {  # source -> (sets, groups)
    # sets: scenario set -> list of (set the run used, raw results with all policy groups)
    "2e": ({
        "heldout2": [("heldout2", ROOT / "results/stage2e_eval2/heldout2/benchmark_traditional_vs_rl_raw.csv")],
        "phase4": [("phase4", ROOT / "results/stage2e_eval2/phase4/benchmark_traditional_vs_rl_raw.csv")],
    }, {
        "Bprime": lambda p: p.str.startswith("S2_bp_s"),
        "V0r": lambda p: p.str.startswith("V0r_s"),
        "Plateau": lambda p: p == "Plateau_0.45_k3_e0.02",
        "Decaying": lambda p: p == "Decaying_1.00_0.40",
    }),
    "4e": ({
        "heldout2": [("heldout2", ROOT / "results/stage4e_eval/heldout2/benchmark_traditional_vs_rl_raw.csv")],
        "phase4": [("evaluation", ROOT / "results/stage4e_eval/evaluation_100/benchmark_traditional_vs_rl_raw.csv"),
                   ("tuning", ROOT / "results/stage4e_eval/tuning_run/benchmark_traditional_vs_rl_raw.csv")],
    }, {
        "Bprime": lambda p: p.str.startswith("S2_bp_s"),
        "V0r": lambda p: p.str.startswith("V0r_s"),
        "Plateau": lambda p: p == "Plateau_0.40_k3_e0.01",
        "Decaying": lambda p: p == "Decaying_1.00_0.40",
    }),
}
SETS, GROUPS = SOURCES["2e"]
RANGE_BINS = (0, 1500, 2500, 3500, 5000)
HP_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def select_shots(per_group, seed, sets=None, groups=None):
    """Fired episodes, per_group per (scenario set, policy group), sampled with a fixed seed."""
    sets, groups = sets or SETS, groups or GROUPS
    rng = np.random.default_rng(seed)
    picked = []
    for scenario_set, runs in sets.items():
        raw = pd.concat([pd.read_csv(path).assign(rebuild_set=run_set) for run_set, path in runs],
                        ignore_index=True)
        fired = raw[raw["fired"].astype(bool)]
        for group, select in groups.items():
            rows = fired[select(fired["policy"])]
            idx = rng.choice(len(rows), size=min(per_group, len(rows)), replace=False)
            part = rows.iloc[np.sort(idx)].copy()
            part["scenario_set"], part["group"] = scenario_set, group
            picked.append(part)
    return pd.concat(picked, ignore_index=True)


def _perturbed(scenario_set, name, seed):
    from compare_traditional_vs_rl import scenarios_for

    from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
    from rl_training.utils.evaluation_protocol import perturb_scenario

    rng = np.random.default_rng(seed)
    for s in [perturb_scenario(s, rng) for s in scenarios_for(scenario_set, CurriculumScenarios(phase=4))]:
        if s.name == name:
            return s
    raise KeyError(name)


def _validate_shot(job):
    """Worker: rebuild one shot and run both checks."""
    from rl_training.environment import ArtilleryFiringEnv
    from rl_training.utils.hp_validation import firing_situation, validate_situation

    shot, n_samples, sample_seed, extrapolate, strict = job
    env = ArtilleryFiringEnv(extrapolate_estimate=extrapolate)
    scenario = _perturbed(shot.get("rebuild_set", shot["scenario_set"]), shot["scenario"], int(shot["seed"]))
    situation = firing_situation(env, scenario, int(shot["seed"]), int(shot["fired_at_step"]))
    result = validate_situation(situation, n_samples, sample_seed,
                                env.hit_calculator.error_propagation._calculate_projectile_position_advanced)
    # strict: the shot was recorded in the same environment, so the rebuilt HP must match it exactly.
    # Otherwise (2e shots rebuilt with extrapolate) the HP legitimately differs from the recorded one.
    return {**shot, **result, "rebuilt_hp": situation["hp"], "estimate_lag_s": situation["estimate_lag_s"],
            "rebuild_ok": bool(not strict or abs(situation["hp"] - shot["hit_probability"]) < 1e-6)}


def summarise(df):
    """Agreement, calibration and ranking summaries of the per-shot results."""
    out = {"n_shots": len(df), "rebuild_ok": int(df["rebuild_ok"].sum())}
    for name in ("mc_model_hp", "full_cov_hp", "true_hit_rate"):
        d = df[name] - df["base_hp"]
        out[f"{name}_minus_base"] = {"mean": float(d.mean()), "mean_abs": float(d.abs().mean()),
                                     "max_abs": float(d.abs().max()), "sd": float(d.std(ddof=1))}
    df = df.assign(range_bin=pd.cut(df["range_m"], RANGE_BINS), hp_bin=pd.cut(df["base_hp"], HP_BINS))
    out["by_range"] = [
        {"range": str(k), "n": int(len(g)), "mc_minus_base": float((g.mc_model_hp - g.base_hp).mean()),
         "true_minus_base": float((g.true_hit_rate - g.base_hp).mean())}
        for k, g in df.groupby("range_bin", observed=True)]
    out["calibration"] = [
        {"hp_bin": str(k), "n": int(len(g)), "mean_base_hp": float(g.base_hp.mean()),
         "mean_mc_model_hp": float(g.mc_model_hp.mean()), "mean_true_hit_rate": float(g.true_hit_rate.mean())}
        for k, g in df.groupby("hp_bin", observed=True)]
    # Ranking on the sampled shots, as scored (with the ramp): analytical vs Monte Carlo.
    ranking = {}
    for scenario_set, g in df.groupby("scenario_set"):
        means = g.groupby("group").apply(lambda x: pd.Series({
            "scored_hp": (x.base_hp * x.ramp).mean(), "mc_model": (x.mc_model_hp * x.ramp).mean(),
            "true_hits": (x.true_hit_rate * x.ramp).mean()}), include_groups=False)
        ranking[scenario_set] = {
            "means": means.round(4).to_dict(orient="index"),
            "order_scored": list(means.scored_hp.sort_values(ascending=False).index),
            "order_mc_model": list(means.mc_model.sort_values(ascending=False).index),
            "order_true_hits": list(means.true_hits.sort_values(ascending=False).index),
        }
    out["ranking"] = ranking
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_group", type=int, default=25)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results/hp_validation")
    parser.add_argument("--extrapolate_estimate", action="store_true",
                        help="Rebuild the same shots in the corrected environment (estimate predicted to decision time)")
    parser.add_argument("--source", choices=tuple(SOURCES), default="2e",
                        help="Evaluation runs to sample shots from (2e: step-2e runs; 4e: step-4e runs and rules)")
    args = parser.parse_args(argv)
    sets, groups = SOURCES[args.source]

    shots = select_shots(args.per_group, args.seed, sets, groups)
    cols = ["scenario_set", "rebuild_set", "group", "policy", "scenario", "seed", "fired_at_step",
            "hit_probability"]
    # 2e runs used the published environment, 4e runs the estimate prediction.
    strict = args.extrapolate_estimate == (args.source == "4e")
    jobs = [(row, args.n_samples, args.seed + i, args.extrapolate_estimate, strict)
            for i, row in enumerate(shots[cols].to_dict("records"))]
    print(f"{len(jobs)} shots, {args.n_samples} samples each", flush=True)

    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(_validate_shot, jobs), total=len(jobs), file=sys.stderr))
    df = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "hp_validation_shots.csv", index=False)
    summary = {"config": {"per_group": args.per_group, "n_samples": args.n_samples, "seed": args.seed,
                          "extrapolate_estimate": args.extrapolate_estimate, "source": args.source,
                          "sets": {k: {run_set: str(path.relative_to(ROOT)) for run_set, path in runs}
                                   for k, runs in sets.items()}},
               **summarise(df)}
    (args.output_dir / "hp_validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_shots", "rebuild_ok", "mc_model_hp_minus_base",
                                              "full_cov_hp_minus_base", "true_hit_rate_minus_base")}, indent=2))


if __name__ == "__main__":
    main()
