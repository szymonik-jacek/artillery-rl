"""
Tests for rl_training/utils/benchmark_metrics.py

Covers: CI and CV helpers and per-(policy, scenario) aggregation, including
timeouts (scored as HP = 0) and blocked dead-zone attempts.
"""

import math
import os
import sys

import numpy as np
import pytest
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.utils.benchmark_metrics import aggregate_results, ci95_half_width, cv_pct


def _row(policy, scenario, seed, hp=None, step=None, rng=None, blocked=0):
    fired = hp is not None
    return {
        "policy": policy,
        "scenario": scenario,
        "seed": seed,
        "fired": fired,
        "timed_out": not fired,
        "fired_at_step": step if fired else None,
        "hit_probability": hp if fired else 0.0,
        "firing_slant_range": rng if fired else None,
        "firing_x": (rng - 10.0) if fired else None,
        "steps": (step + 1) if fired else 20,
        "blocked_fire_attempts": blocked,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestCi95HalfWidth:
    def test_matches_student_t(self):
        v = [0.5, 0.6, 0.7, 0.8]
        expected = sp_stats.t.ppf(0.975, 3) * np.std(v, ddof=1) / 2.0
        assert ci95_half_width(v) == pytest.approx(expected)

    def test_ignores_nan(self):
        assert ci95_half_width([0.5, np.nan, 0.7]) == pytest.approx(ci95_half_width([0.5, 0.7]))

    @pytest.mark.parametrize("values", [[], [0.5], [np.nan, 0.5]])
    def test_nan_for_fewer_than_two(self, values):
        assert math.isnan(ci95_half_width(values))


class TestCvPct:
    def test_basic(self):
        assert cv_pct(0.05, 0.5) == pytest.approx(10.0)

    @pytest.mark.parametrize("mean", [0.0, float("nan"), None])
    def test_nan_for_zero_or_undefined_mean(self, mean):
        assert math.isnan(cv_pct(0.1, mean))


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------

class TestAggregateResults:
    ROWS = [
        # DQN fires every time on S1.
        _row("DQN", "S1", 0, hp=0.6, step=10, rng=1000.0),
        _row("DQN", "S1", 1, hp=0.7, step=12, rng=900.0),
        _row("DQN", "S1", 2, hp=0.8, step=14, rng=800.0),
        # Threshold rule: one timeout, one episode with a blocked attempt.
        _row("Thr", "S1", 0, hp=0.6, step=20, rng=500.0),
        _row("Thr", "S1", 1, blocked=2),
        _row("Thr", "S1", 2, hp=0.9, step=22, rng=300.0, blocked=1),
    ]

    def _get(self, policy, scenario="S1"):
        s = aggregate_results(self.ROWS)
        return s[(s["policy"] == policy) & (s["scenario"] == scenario)].iloc[0]

    def test_one_row_per_policy_and_scenario_in_order(self):
        s = aggregate_results(self.ROWS)
        assert list(zip(s["policy"], s["scenario"])) == [("DQN", "S1"), ("Thr", "S1")]

    def test_counts(self):
        r = self._get("Thr")
        assert r["n_episodes"] == 3
        assert r["firing_rate_pct"] == pytest.approx(200.0 / 3)
        assert r["n_timeouts"] == 1
        assert r["n_blocked_episodes"] == 2
        assert r["blocked_fire_attempts"] == 3

    def test_unconditional_hp_counts_timeout_as_zero(self):
        r = self._get("Thr")
        assert r["mean_hp"] == pytest.approx((0.6 + 0.0 + 0.9) / 3)
        assert r["std_hp"] == pytest.approx(np.std([0.6, 0.0, 0.9], ddof=1))
        assert r["ci95_hp"] == pytest.approx(ci95_half_width([0.6, 0.0, 0.9]))
        assert r["cv_hp_pct"] == pytest.approx(r["std_hp"] / r["mean_hp"] * 100)

    def test_conditional_hp_uses_fired_episodes_only(self):
        r = self._get("Thr")
        assert r["mean_hp_fired"] == pytest.approx(0.75)
        assert r["std_hp_fired"] == pytest.approx(np.std([0.6, 0.9], ddof=1))

    def test_timing_and_range_use_fired_episodes_only(self):
        r = self._get("Thr")
        assert r["mean_fired_step"] == pytest.approx(21.0)
        assert r["mean_firing_range"] == pytest.approx(400.0)
        assert r["cv_range_pct"] == pytest.approx(np.std([500.0, 300.0], ddof=1) / 400.0 * 100)
        assert r["mean_firing_x"] == pytest.approx(390.0)

    def test_full_firing_policy(self):
        r = self._get("DQN")
        assert r["firing_rate_pct"] == 100.0
        assert r["n_timeouts"] == 0
        assert r["mean_hp"] == pytest.approx(r["mean_hp_fired"]) == pytest.approx(0.7)

    def test_all_timeouts(self):
        rows = [_row("Thr", "S2", s) for s in range(3)]
        r = aggregate_results(rows).iloc[0]
        assert r["firing_rate_pct"] == 0.0
        assert r["mean_hp"] == 0.0
        assert math.isnan(r["cv_hp_pct"])
        assert math.isnan(r["mean_hp_fired"])
        assert math.isnan(r["mean_fired_step"])
        assert math.isnan(r["mean_firing_range"])


# ---------------------------------------------------------------------------
# pooled_summary / select_best_fixed_threshold
# ---------------------------------------------------------------------------

from rl_training.utils.benchmark_metrics import pooled_summary, select_best_fixed_threshold


def _two_scenario_rows():
    """Low threshold fires everywhere at modest HP; high threshold is better on
    S1 but times out on S2; DQN is best overall."""
    rows = []
    for seed in range(2):
        rows += [
            _row("DQN", "S1", seed, hp=0.70, step=10, rng=900.0),
            _row("DQN", "S2", seed, hp=0.60, step=8, rng=2000.0),
            _row("Threshold_0.45", "S1", seed, hp=0.50, step=5, rng=1200.0),
            _row("Threshold_0.45", "S2", seed, hp=0.50, step=4, rng=2500.0),
            _row("Threshold_0.65", "S1", seed, hp=0.80, step=15, rng=600.0),
            _row("Threshold_0.65", "S2", seed),  # timeout
        ]
    return rows


class TestPooledSummary:
    def test_one_row_per_policy(self):
        s = pooled_summary(_two_scenario_rows())
        assert list(s["policy"]) == ["DQN", "Threshold_0.45", "Threshold_0.65"]
        assert "scenario" not in s.columns
        assert (s["n_episodes"] == 4).all()

    def test_pools_across_scenarios(self):
        s = pooled_summary(_two_scenario_rows()).set_index("policy")
        assert s.loc["DQN", "mean_hp"] == pytest.approx(0.65)
        assert s.loc["Threshold_0.65", "mean_hp"] == pytest.approx(0.40)
        assert s.loc["Threshold_0.65", "firing_rate_pct"] == pytest.approx(50.0)
        assert s.loc["Threshold_0.65", "mean_hp_fired"] == pytest.approx(0.80)


class TestSelectBestFixedThreshold:
    def test_picks_highest_unconditional_hp_ignoring_dqn(self):
        name, hp = select_best_fixed_threshold(_two_scenario_rows())
        # 0.65 has higher HP when it fires (0.80) but times out on S2.
        assert name == "Threshold_0.45"
        assert hp == pytest.approx(0.50)

    def test_ties_keep_first_in_order(self):
        rows = [_row("Threshold_0.50", "S1", 0, hp=0.6, step=1, rng=1000.0),
                _row("Threshold_0.55", "S1", 0, hp=0.6, step=2, rng=900.0)]
        assert select_best_fixed_threshold(rows)[0] == "Threshold_0.50"

    def test_custom_prefix(self):
        rows = [_row("CovGated_0.50_100", "S1", 0, hp=0.4, step=1, rng=1000.0),
                _row("CovGated_0.55_100", "S1", 0, hp=0.7, step=2, rng=900.0),
                _row("Threshold_0.50", "S1", 0, hp=0.9, step=1, rng=1000.0)]
        assert select_best_fixed_threshold(rows, prefix="CovGated_")[0] == "CovGated_0.55_100"

    def test_raises_without_candidates(self):
        with pytest.raises(ValueError):
            select_best_fixed_threshold([_row("DQN", "S1", 0, hp=0.5, step=1, rng=1000.0)])


# ---------------------------------------------------------------------------
# holm_correction / paired_comparisons
# ---------------------------------------------------------------------------

from rl_training.utils.benchmark_metrics import POOLED_LABEL, holm_correction, paired_comparisons


class TestHolmCorrection:
    def test_known_example(self):
        # Sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone 0.06.
        np.testing.assert_allclose(holm_correction([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_caps_at_one(self):
        np.testing.assert_allclose(holm_correction([0.6, 0.7]), [1.0, 1.0])

    def test_nan_excluded_from_family(self):
        adj = holm_correction([0.01, np.nan, 0.02])
        assert math.isnan(adj[1])
        np.testing.assert_allclose(adj[[0, 2]], [0.02, 0.02])


def _paired_rows(n=12):
    """DQN beats Thr on S1 by varying margins; equal on S2."""
    noise = np.random.default_rng(1)
    rows = []
    for seed in range(n):
        gain = 0.05 + 0.01 * seed
        base = 0.5 + 0.01 * noise.standard_normal()
        rows += [
            _row("DQN", "S1", seed, hp=base + gain, step=10, rng=1000.0),
            _row("Thr", "S1", seed, hp=base, step=12, rng=900.0),
            _row("DQN", "S2", seed, hp=0.6, step=10, rng=1000.0),
            _row("Thr", "S2", seed, hp=0.6, step=11, rng=950.0),
        ]
    return rows


class TestPairedComparisons:
    def _get(self, result, scenario):
        return result[result["scenario"] == scenario].iloc[0]

    def test_rows_per_scenario_and_pooled(self):
        r = paired_comparisons(_paired_rows(), n_boot=200)
        assert list(r["scenario"]) == ["S1", "S2", POOLED_LABEL]
        assert (r["baseline"] == "Thr").all() and (r["reference"] == "DQN").all()
        assert list(r["n_pairs"]) == [12, 12, 24]

    def test_positive_difference_when_reference_better(self):
        r = self._get(paired_comparisons(_paired_rows(), n_boot=500), "S1")
        gains = [0.05 + 0.01 * s for s in range(12)]
        assert r["mean_diff"] == pytest.approx(np.mean(gains))
        assert r["median_diff"] == pytest.approx(np.median(gains))
        assert r["median_diff_ci95_low"] <= r["median_diff"] <= r["median_diff_ci95_high"]
        assert r["significant"]

    def test_matches_scipy_wilcoxon(self):
        r = self._get(paired_comparisons(_paired_rows(), n_boot=200), "S1")
        expected = sp_stats.wilcoxon([0.05 + 0.01 * s for s in range(12)], zero_method="wilcox")
        assert r["p_value"] == pytest.approx(expected.pvalue)

    def test_all_zero_differences_give_no_p_value(self):
        r = self._get(paired_comparisons(_paired_rows(), n_boot=200), "S2")
        assert r["n_nonzero"] == 0
        assert math.isnan(r["p_value"]) and math.isnan(r["p_holm"])
        assert r["median_diff"] == 0.0
        assert r["median_diff_ci95_low"] == r["median_diff_ci95_high"] == 0.0
        assert not r["significant"]

    def test_timeout_counts_as_zero_hp(self):
        rows = [_row("DQN", "S1", s, hp=0.5, step=5, rng=1000.0) for s in range(6)]
        rows += [_row("Thr", "S1", s) for s in range(6)]
        r = self._get(paired_comparisons(rows, n_boot=200), "S1")
        assert r["mean_diff"] == pytest.approx(0.5)

    def test_pairs_by_seed_regardless_of_row_order(self):
        rows = _paired_rows()
        shuffled = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
        a = self._get(paired_comparisons(rows, n_boot=200), "S1")
        b = self._get(paired_comparisons(shuffled, n_boot=200), "S1")
        assert a["mean_diff"] == pytest.approx(b["mean_diff"])
        assert a["p_value"] == pytest.approx(b["p_value"])

    def test_bootstrap_is_reproducible(self):
        a = paired_comparisons(_paired_rows(), n_boot=300, seed=4)
        b = paired_comparisons(_paired_rows(), n_boot=300, seed=4)
        assert a["median_diff_ci95_low"].equals(b["median_diff_ci95_low"])

    def test_holm_applied_per_family(self):
        rows = _paired_rows()
        rows += [dict(r, policy="Thr2") for r in rows if r["policy"] == "Thr"]
        r = paired_comparisons(rows, n_boot=200)
        per_scenario = r[r["scenario"] != POOLED_LABEL]
        pooled = r[r["scenario"] == POOLED_LABEL]
        np.testing.assert_allclose(per_scenario["p_holm"], holm_correction(per_scenario["p_value"]))
        np.testing.assert_allclose(pooled["p_holm"], holm_correction(pooled["p_value"]))

    def test_missing_pair_raises(self):
        rows = [r for r in _paired_rows() if not (r["policy"] == "Thr" and r["seed"] == 3)]
        with pytest.raises(ValueError, match="same"):
            paired_comparisons(rows, n_boot=200)

    def test_missing_reference_raises(self):
        rows = [r for r in _paired_rows() if r["policy"] != "DQN"]
        with pytest.raises(ValueError, match="Reference"):
            paired_comparisons(rows, n_boot=200)


# ---------------------------------------------------------------------------
# Firing time and Pareto front
# ---------------------------------------------------------------------------

from rl_training.utils.benchmark_metrics import pareto_front


def _timed(policy, scenario, seed, hp=None, t=None):
    r = _row(policy, scenario, seed, hp=hp, step=0 if hp is not None else None, rng=1000.0)
    r["fire_time_s"] = t if hp is not None else None
    return r


class TestFireTimeMetrics:
    def test_mean_fire_time_uses_fired_episodes_only(self):
        rows = [_timed("A", "S1", 0, 0.5, 2.0), _timed("A", "S1", 1, 0.6, 4.0), _timed("A", "S1", 2)]
        r = aggregate_results(rows).iloc[0]
        assert r["mean_fire_time_s"] == pytest.approx(3.0)
        assert r["std_fire_time_s"] == pytest.approx(np.std([2.0, 4.0], ddof=1))

    def test_missing_column_gives_nan(self):
        r = aggregate_results([_row("A", "S1", 0, hp=0.5, step=3, rng=900.0)]).iloc[0]
        assert math.isnan(r["mean_fire_time_s"])


class TestParetoFront:
    def _pooled(self, specs):
        rows = []
        for policy, hp, t, fire_all in specs:
            rows.append(_timed(policy, "S1", 0, hp, t))
            rows.append(_timed(policy, "S1", 1, hp, t) if fire_all else _timed(policy, "S1", 1))
        return pooled_summary(rows)

    def test_dominated_policies_removed_and_sorted_by_time(self):
        pooled = self._pooled([
            ("Fast", 0.50, 4.0, True),
            ("DQN", 0.58, 6.0, True),
            ("Slow", 0.60, 10.0, True),
            ("Worse", 0.55, 7.0, True),   # dominated by DQN
        ])
        assert pareto_front(pooled) == ["Fast", "DQN", "Slow"]

    def test_policies_with_timeouts_excluded(self):
        pooled = self._pooled([("DQN", 0.58, 6.0, True), ("Rare", 0.95, 2.0, False)])
        assert pareto_front(pooled) == ["DQN"]
        assert "Rare" in pareto_front(pooled, require_full_firing=False)

    def test_equal_points_both_kept(self):
        pooled = self._pooled([("A", 0.5, 5.0, True), ("B", 0.5, 5.0, True)])
        assert set(pareto_front(pooled)) == {"A", "B"}


# ---------------------------------------------------------------------------
# non_inferiority
# ---------------------------------------------------------------------------

from rl_training.utils.benchmark_metrics import non_inferiority


def _ni_rows(agent_offsets, n=40, noise=0.01, rule_hp=0.6):
    """Rule HP 0.6 everywhere; agent k gets rule HP + offset_k + small noise."""
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(n):
        rows.append(_row("Rule", "S1", seed, hp=rule_hp, step=5, rng=1000.0))
        for k, off in enumerate(agent_offsets):
            rows.append(_row(f"A_s{k}", "S1", seed, hp=rule_hp + off + noise * rng.standard_normal(),
                             step=5, rng=1000.0))
    return rows


class TestNonInferiority:
    def test_clearly_non_inferior(self):
        r = non_inferiority(_ni_rows([0.0, 0.005, -0.005]), ["A_s0", "A_s1", "A_s2"], "Rule",
                            delta=0.02, n_boot=2000)
        assert r["non_inferior"] and r["lower_bound"] > -0.02
        assert r["mean_diff"] == pytest.approx(0.0, abs=0.005)
        assert r["n_agents"] == 3 and r["n_episodes"] == 40

    def test_clearly_inferior(self):
        r = non_inferiority(_ni_rows([-0.05, -0.06]), ["A_s0", "A_s1"], "Rule", delta=0.02, n_boot=2000)
        assert not r["non_inferior"] and r["mean_diff"] == pytest.approx(-0.055, abs=0.005)

    def test_seed_spread_widens_interval(self):
        same = non_inferiority(_ni_rows([0.0, 0.0, 0.0]), ["A_s0", "A_s1", "A_s2"], "Rule", 0.02, n_boot=2000)
        spread = non_inferiority(_ni_rows([-0.03, 0.0, 0.03]), ["A_s0", "A_s1", "A_s2"], "Rule", 0.02, n_boot=2000)
        assert spread["mean_diff"] == pytest.approx(same["mean_diff"], abs=0.005)
        assert spread["lower_bound"] < same["lower_bound"]

    def test_strict_margin(self):
        rows = _ni_rows([-0.012, -0.012])
        loose = non_inferiority(rows, ["A_s0", "A_s1"], "Rule", delta=0.02, n_boot=2000)
        strict = non_inferiority(rows, ["A_s0", "A_s1"], "Rule", delta=0.01, n_boot=2000)
        assert loose["non_inferior"] and not strict["non_inferior"]

    def test_timeouts_count_as_zero(self):
        rows = _ni_rows([0.0])
        rows = [dict(r, hit_probability=0.0, fired=False, timed_out=True) if r["policy"] == "A_s0" and r["seed"] < 4
                else r for r in rows]
        r = non_inferiority(rows, ["A_s0"], "Rule", delta=0.02, n_boot=500)
        assert r["mean_diff"] < -0.05 and not r["non_inferior"]

    def test_requires_agents(self):
        with pytest.raises(ValueError):
            non_inferiority(_ni_rows([0.0]), [], "Rule", 0.02)


from rl_training.utils.benchmark_metrics import agent_group_difference


def _group_rows(offsets_a, offsets_b, n=40, scenarios=("S1", "S2")):
    """Episode HP base_k (shared, varies by episode) + per-agent offset + small noise."""
    rng = np.random.default_rng(3)
    rows = []
    for sc in scenarios:
        for seed in range(n):
            base = 0.3 + 0.4 * rng.random()
            for name, offsets in (("A", offsets_a), ("B", offsets_b)):
                for k, off in enumerate(offsets):
                    rows.append(_row(f"{name}_s{k}", sc, seed, hp=base + off + 0.005 * rng.standard_normal(),
                                     step=5, rng=1000.0))
    return rows


class TestAgentGroupDifference:
    def test_clearly_better(self):
        r = agent_group_difference(_group_rows([0.05, 0.06, 0.04], [0.0, 0.01, -0.01]),
                                   ["A_s0", "A_s1", "A_s2"], ["B_s0", "B_s1", "B_s2"], n_boot=2000)
        assert r["mean_diff"] == pytest.approx(0.05, abs=0.005)
        assert r["a_better"] and 0 < r["lower_bound"] < r["mean_diff"]
        assert r["n_episodes"] == 80 and len(r["per_agent_mean_a"]) == 3

    def test_equal_groups_not_better(self):
        r = agent_group_difference(_group_rows([0.0, 0.01], [0.0, 0.01]), ["A_s0", "A_s1"], ["B_s0", "B_s1"],
                                   n_boot=2000)
        assert not r["a_better"] and r["mean_diff"] == pytest.approx(0.0, abs=0.005)

    def test_pairing_removes_episode_variance(self):
        # Episode HP varies over 0.3-0.7, yet a constant 0.02 gain is detected because episodes are paired.
        r = agent_group_difference(_group_rows([0.02, 0.02], [0.0, 0.0]), ["A_s0", "A_s1"], ["B_s0", "B_s1"],
                                   n_boot=2000)
        assert r["a_better"] and r["lower_bound"] > 0.015

    def test_seed_spread_widens_interval(self):
        same = agent_group_difference(_group_rows([0.02] * 3, [0.0] * 3), ["A_s0", "A_s1", "A_s2"],
                                      ["B_s0", "B_s1", "B_s2"], n_boot=2000)
        spread = agent_group_difference(_group_rows([-0.03, 0.02, 0.07], [0.0] * 3), ["A_s0", "A_s1", "A_s2"],
                                        ["B_s0", "B_s1", "B_s2"], n_boot=2000)
        assert spread["mean_diff"] == pytest.approx(same["mean_diff"], abs=0.005)
        assert spread["lower_bound"] < same["lower_bound"]

    def test_episode_order_does_not_matter(self):
        rows = _group_rows([0.03, 0.02], [0.0, 0.01])
        a = agent_group_difference(rows, ["A_s0", "A_s1"], ["B_s0", "B_s1"], n_boot=500)
        b = agent_group_difference(list(reversed(rows)), ["A_s0", "A_s1"], ["B_s0", "B_s1"], n_boot=500)
        assert a["mean_diff"] == pytest.approx(b["mean_diff"])

    def test_mismatched_episodes_raise(self):
        rows = [r for r in _group_rows([0.0], [0.0]) if not (r["policy"] == "B_s0" and r["seed"] == 3)]
        with pytest.raises(ValueError, match="same"):
            agent_group_difference(rows, ["A_s0"], ["B_s0"], n_boot=10)

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="C_s0"):
            agent_group_difference(_group_rows([0.0], [0.0]), ["A_s0"], ["C_s0"], n_boot=10)


class TestAgentGroupDifferenceBounds:
    def test_bounds_are_ordered_and_bracket_the_mean(self):
        r = agent_group_difference(_group_rows([0.02, 0.03], [0.0, 0.01]), ["A_s0", "A_s1"], ["B_s0", "B_s1"],
                                   n_boot=2000)
        low2, high2 = r["ci_two_sided"]
        assert low2 <= r["lower_bound"] <= r["mean_diff"] <= r["upper_bound"] <= high2

    def test_b_better_when_a_clearly_worse(self):
        r = agent_group_difference(_group_rows([-0.05, -0.06], [0.0, 0.0]), ["A_s0", "A_s1"], ["B_s0", "B_s1"],
                                   n_boot=2000)
        assert r["b_better"] and not r["a_better"] and r["upper_bound"] < 0

    def test_neither_better_when_equal(self):
        r = agent_group_difference(_group_rows([0.0, 0.01], [0.0, 0.01]), ["A_s0", "A_s1"], ["B_s0", "B_s1"],
                                   n_boot=2000)
        assert not r["a_better"] and not r["b_better"]
        assert r["ci_two_sided"][0] < 0 < r["ci_two_sided"][1]

    def test_swapping_groups_mirrors_the_result(self):
        rows = _group_rows([0.03, 0.02], [0.0, 0.01])
        ab = agent_group_difference(rows, ["A_s0", "A_s1"], ["B_s0", "B_s1"], n_boot=500)
        ba = agent_group_difference(rows, ["B_s0", "B_s1"], ["A_s0", "A_s1"], n_boot=500)
        assert ab["mean_diff"] == pytest.approx(-ba["mean_diff"])
        assert ab["a_better"] == ba["b_better"]
