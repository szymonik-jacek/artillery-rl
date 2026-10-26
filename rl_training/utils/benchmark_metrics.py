"""
Aggregate metrics for the traditional-FCS vs. RL benchmark.

Input rows are per-episode results from compare_traditional_vs_rl.run_benchmark.
Conventions follow evaluate_dqn_trajectories: sample std (ddof=1), 95% CI as
the Student-t half-width, CV = std / mean * 100.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def ci95_half_width(values):
    """Half-width of the 95% Student-t CI of the mean (NaN for fewer than 2 values)."""
    v = pd.Series(values, dtype=float).dropna()
    n = len(v)
    if n < 2:
        return float("nan")
    return float(sp_stats.t.ppf(0.975, df=n - 1) * v.std(ddof=1) / np.sqrt(n))


def cv_pct(std, mean):
    """Coefficient of variation in percent (NaN if the mean is 0 or undefined)."""
    if mean is None or not np.isfinite(mean) or mean == 0:
        return float("nan")
    return float(std / mean * 100.0)


def _summarise(group):
    hp = group["hit_probability"].astype(float)
    fired = group[group["fired"].astype(bool)]
    hp_fired = fired["hit_probability"].astype(float)
    step = fired["fired_at_step"].astype(float)
    rng = fired["firing_slant_range"].astype(float)

    mean_hp = float(hp.mean())
    std_hp = float(hp.std(ddof=1)) if len(hp) > 1 else float("nan")
    mean_range = float(rng.mean()) if len(rng) else float("nan")
    std_range = float(rng.std(ddof=1)) if len(rng) > 1 else float("nan")
    # Older raw results have no fire_time_s; report NaN rather than fail.
    fire_time = (fired["fire_time_s"].astype(float) if "fire_time_s" in fired
                 else pd.Series(dtype=float))

    return pd.Series({
        "n_episodes": len(group),
        "firing_rate_pct": 100.0 * len(fired) / len(group),
        "n_timeouts": int(group["timed_out"].astype(bool).sum()),
        "n_blocked_episodes": int((group["blocked_fire_attempts"] > 0).sum()),
        "blocked_fire_attempts": int(group["blocked_fire_attempts"].sum()),
        # Unconditional HP: timeouts count as 0 (primary metric).
        "mean_hp": mean_hp,
        "std_hp": std_hp,
        "ci95_hp": ci95_half_width(hp),
        "cv_hp_pct": cv_pct(std_hp, mean_hp),
        # HP conditional on firing.
        "mean_hp_fired": float(hp_fired.mean()) if len(hp_fired) else float("nan"),
        "std_hp_fired": float(hp_fired.std(ddof=1)) if len(hp_fired) > 1 else float("nan"),
        "ci95_hp_fired": ci95_half_width(hp_fired),
        "mean_fired_step": float(step.mean()) if len(step) else float("nan"),
        "std_fired_step": float(step.std(ddof=1)) if len(step) > 1 else float("nan"),
        "mean_fire_time_s": float(fire_time.mean()) if len(fire_time) else float("nan"),
        "std_fire_time_s": float(fire_time.std(ddof=1)) if len(fire_time) > 1 else float("nan"),
        "mean_firing_range": mean_range,
        "std_firing_range": std_range,
        "cv_range_pct": cv_pct(std_range, mean_range),
        "mean_firing_x": float(fired["firing_x"].astype(float).mean()) if len(fired) else float("nan"),
    })


def aggregate_results(rows, by=("policy", "scenario")):
    """Aggregate per-episode rows into one row per group.

    By default one row per (policy, scenario); by=("policy",) pools all
    scenarios. Groups keep their order of first appearance.
    """
    df = pd.DataFrame(rows)
    by = list(by)
    grouped = df.groupby(by, sort=False)
    summary = grouped[df.columns.drop(by).tolist()].apply(_summarise)
    summary = summary.reset_index()
    summary["n_episodes"] = summary["n_episodes"].astype(int)
    summary["n_timeouts"] = summary["n_timeouts"].astype(int)
    summary["n_blocked_episodes"] = summary["n_blocked_episodes"].astype(int)
    summary["blocked_fire_attempts"] = summary["blocked_fire_attempts"].astype(int)
    return summary


def pooled_summary(rows):
    """Aggregate per-episode rows into one row per policy over all scenarios."""
    return aggregate_results(rows, by=("policy",))


def select_best_fixed_threshold(rows, prefix="Threshold_"):
    """Pick the single threshold policy with the highest pooled unconditional mean HP.

    Only policies whose name starts with prefix are considered. Ties keep the
    first policy in order of appearance. Which rows are passed (tuning or test
    scenarios) is the caller's choice and must be reported.

    Returns (policy_name, pooled_mean_hp).
    """
    pooled = pooled_summary(rows)
    candidates = pooled[pooled["policy"].str.startswith(prefix)]
    if candidates.empty:
        raise ValueError(f"No policies starting with {prefix!r}")
    best = candidates.loc[candidates["mean_hp"].idxmax()]
    return best["policy"], float(best["mean_hp"])


POOLED_LABEL = "ALL"


def holm_correction(p_values):
    """Holm-Bonferroni adjusted p-values. NaN entries are left out of the family."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m = len(valid)
    running_max = 0.0
    for rank, idx in enumerate(valid[np.argsort(p[valid], kind="stable")]):
        running_max = max(running_max, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running_max
    return adjusted


def _paired_differences(df, reference, baseline):
    """HP differences (reference - baseline), matched on (scenario, seed)."""
    keys = ["scenario", "seed"]
    ref = df[df["policy"] == reference].set_index(keys)["hit_probability"]
    base = df[df["policy"] == baseline].set_index(keys)["hit_probability"]
    if not ref.index.is_unique or not base.index.is_unique:
        raise ValueError("Duplicate (scenario, seed) episodes for one policy")
    if set(ref.index) != set(base.index):
        raise ValueError(f"{reference} and {baseline} were not run on the same (scenario, seed) pairs")
    return (ref - base.loc[ref.index]).astype(float)


def _compare(diff, n_boot, rng):
    d = diff.to_numpy()
    n_nonzero = int(np.count_nonzero(d))
    if n_nonzero == 0:
        stat, p = float("nan"), float("nan")
    else:
        res = sp_stats.wilcoxon(d, zero_method="wilcox")
        stat, p = float(res.statistic), float(res.pvalue)
    boot = np.median(d[rng.integers(0, len(d), size=(n_boot, len(d)))], axis=1)
    return {
        "n_pairs": len(d),
        "n_nonzero": n_nonzero,
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "median_diff_ci95_low": float(np.percentile(boot, 2.5)),
        "median_diff_ci95_high": float(np.percentile(boot, 97.5)),
        "wilcoxon_stat": stat,
        "p_value": p,
    }


def paired_comparisons(rows, reference="DQN", n_boot=10000, seed=0, alpha=0.05):
    """Paired comparison of reference against every other policy.

    Differences are reference - baseline unconditional HP (timeouts = 0),
    matched on (scenario, seed). One row per (baseline, scenario) plus a
    pooled row per baseline (scenario = POOLED_LABEL). Wilcoxon signed-rank
    (zero differences dropped; p is NaN if all differences are zero), median
    difference with a percentile bootstrap 95% CI, and Holm correction applied
    separately to the per-scenario family and the pooled family.
    """
    df = pd.DataFrame(rows)
    if reference not in set(df["policy"]):
        raise ValueError(f"Reference policy {reference!r} not in rows")
    rng = np.random.default_rng(seed)
    baselines = [p for p in pd.unique(df["policy"]) if p != reference]

    out = []
    for baseline in baselines:
        diff = _paired_differences(df, reference, baseline)
        for scenario in pd.unique(df["scenario"]):
            d = diff[diff.index.get_level_values("scenario") == scenario]
            out.append({"reference": reference, "baseline": baseline,
                        "scenario": scenario, **_compare(d, n_boot, rng)})
        out.append({"reference": reference, "baseline": baseline,
                    "scenario": POOLED_LABEL, **_compare(diff, n_boot, rng)})

    result = pd.DataFrame(out)
    result["p_holm"] = np.nan
    for pooled in (False, True):
        mask = (result["scenario"] == POOLED_LABEL) == pooled
        result.loc[mask, "p_holm"] = holm_correction(result.loc[mask, "p_value"])
    result["significant"] = result["p_holm"] < alpha
    return result


def pareto_front(summary, require_full_firing=True):
    """Policies not dominated in (mean_hp up, mean_fire_time_s down).

    summary is a pooled_summary (or per-scenario subset). With
    require_full_firing, only policies that fired in every episode are
    candidates, so a high HP from rare shots cannot dominate. A policy is
    dominated if another has HP >= and time <= with at least one strict.
    Returns the policy names on the front, ordered by firing time.
    """
    cand = summary.dropna(subset=["mean_hp", "mean_fire_time_s"])
    if require_full_firing:
        cand = cand[cand["firing_rate_pct"] >= 100.0]
    front = []
    for _, a in cand.iterrows():
        dominated = (
            (cand["mean_hp"] >= a["mean_hp"]) & (cand["mean_fire_time_s"] <= a["mean_fire_time_s"])
            & ((cand["mean_hp"] > a["mean_hp"]) | (cand["mean_fire_time_s"] < a["mean_fire_time_s"]))
        ).any()
        if not dominated:
            front.append((a["mean_fire_time_s"], a["policy"]))
    return [name for _, name in sorted(front)]


def non_inferiority(rows, agents, rule, delta, n_boot=10000, seed=0, alpha=0.05):
    """One-sided non-inferiority test of an agent (several training seeds) vs a rule.

    agents: policy names of the same method trained with different seeds.
    For every agent and every (scenario, seed) episode the paired difference
    agent HP - rule HP is formed (timeouts = 0). The point estimate is the
    mean difference; its uncertainty comes from a two-level bootstrap that
    resamples training seeds and episodes. The method is non-inferior if the
    lower (1 - alpha) confidence bound exceeds -delta.
    """
    df = pd.DataFrame(rows)
    agents = list(agents)
    if not agents:
        raise ValueError("agents must not be empty")
    diffs = np.stack([_paired_differences(df, agent, rule).sort_index().to_numpy() for agent in agents])
    rng = np.random.default_rng(seed)
    n_agents, n_episodes = diffs.shape
    boot = np.empty(n_boot)
    for b in range(n_boot):
        rows_idx = rng.integers(0, n_agents, n_agents)
        cols_idx = rng.integers(0, n_episodes, n_episodes)
        boot[b] = diffs[np.ix_(rows_idx, cols_idx)].mean()
    lower = float(np.percentile(boot, 100 * alpha))
    return {
        "agents": agents, "rule": rule, "delta": delta, "alpha": alpha,
        "n_agents": n_agents, "n_episodes": n_episodes,
        "mean_diff": float(diffs.mean()), "lower_bound": lower,
        "per_agent_mean_diff": [float(x) for x in diffs.mean(axis=1)],
        "non_inferior": lower > -delta,
    }


def agent_group_difference(rows, agents_a, agents_b, n_boot=10000, seed=0, alpha=0.05):
    """Comparison of two methods, each trained with several seeds.

    The difference is mean HP of method a minus method b (timeouts = 0), on
    the same (scenario, seed) episodes. A two-level bootstrap resamples the
    training seeds of each method independently and the episodes jointly
    (paired). Returns the mean difference, the one-sided (1 - alpha) lower and
    upper bounds, the two-sided (1 - alpha) interval and per-method seed
    means; a lower bound > 0 supports a > b, an upper bound < 0 supports b > a.
    """
    df = pd.DataFrame(rows)
    agents_a, agents_b = list(agents_a), list(agents_b)
    if not agents_a or not agents_b:
        raise ValueError("both agent groups must be non-empty")
    keys = ["scenario", "seed"]

    def matrix(agents):
        series = []
        for agent in agents:
            s = df[df["policy"] == agent].set_index(keys)["hit_probability"].astype(float)
            if s.empty:
                raise ValueError(f"Policy {agent!r} not in rows")
            if not s.index.is_unique:
                raise ValueError("Duplicate (scenario, seed) episodes for one policy")
            series.append(s.sort_index())
        index = series[0].index
        if any(set(s.index) != set(index) for s in series):
            raise ValueError("Agents were not run on the same (scenario, seed) pairs")
        return np.stack([s.loc[index].to_numpy() for s in series]), index

    hp_a, index = matrix(agents_a)
    hp_b, index_b = matrix(agents_b)
    if set(index) != set(index_b):
        raise ValueError("The two groups were not run on the same (scenario, seed) pairs")
    hp_b = hp_b[:, index_b.get_indexer(index)]

    rng = np.random.default_rng(seed)
    n_episodes = hp_a.shape[1]
    boot = np.empty(n_boot)
    for b in range(n_boot):
        cols = rng.integers(0, n_episodes, n_episodes)
        ra = rng.integers(0, len(agents_a), len(agents_a))
        rb = rng.integers(0, len(agents_b), len(agents_b))
        boot[b] = hp_a[np.ix_(ra, cols)].mean() - hp_b[np.ix_(rb, cols)].mean()
    return {
        "agents_a": agents_a, "agents_b": agents_b, "alpha": alpha, "n_episodes": n_episodes,
        "mean_a": float(hp_a.mean()), "mean_b": float(hp_b.mean()),
        "mean_diff": float(hp_a.mean() - hp_b.mean()),
        "lower_bound": float(np.percentile(boot, 100 * alpha)),
        "upper_bound": float(np.percentile(boot, 100 * (1 - alpha))),
        # Two-sided (1 - alpha) interval, for comparisons without a pre-set direction.
        "ci_two_sided": [float(np.percentile(boot, 50 * alpha)), float(np.percentile(boot, 100 - 50 * alpha))],
        "per_agent_mean_a": [float(x) for x in hp_a.mean(axis=1)],
        "per_agent_mean_b": [float(x) for x in hp_b.mean(axis=1)],
        "a_better": float(np.percentile(boot, 100 * alpha)) > 0,
        "b_better": float(np.percentile(boot, 100 * (1 - alpha))) < 0,
    }
