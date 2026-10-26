"""
Tests for rl_training/utils/hp_validation.py (TODO step 4b).

With a linear "ballistic" model and diagonal covariances the hit probability
is known in closed form (product of normal CDF differences), so the Monte
Carlo estimates can be checked against it. The rebuild of a real firing
situation is checked against a recorded evaluation result.
"""

import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from rl_training.utils.hp_validation import (
    MUZZLE_VELOCITY,
    firing_situation,
    full_covariance_hp,
    in_box,
    model_hit_rate,
    projectile_impacts,
    ramp_factor,
    true_hit_rate,
    validate_situation,
)

HALF = np.array([2.5, 1.0, 1.0])
SENS = np.array([0.0, 1000.0, 1000.0])  # metres per radian of elevation (z) and azimuth (y)
AIM = np.array([1000.0, 0.0, 50.0])


def _linear_position(v0, elevation, azimuth, t):
    """Linear stand-in for the ballistics: impact moves with the aiming errors only."""
    return AIM + np.array([0.0, SENS[2] * (azimuth - 0.01), SENS[1] * (elevation - 0.02)])


def _situation(angle_std=1e-3, target_pos_std=0.0, target_vel_std=0.0, offset=(0.0, 0.0, 0.0)):
    target_cov = np.diag([target_pos_std ** 2] * 3 + [target_vel_std ** 2] * 3)
    proj_cov = np.diag([1e-12, angle_std ** 2, angle_std ** 2])
    impact_time = 2.0
    p_cov = np.diag([0.0, (SENS[2] * angle_std) ** 2, (SENS[1] * angle_std) ** 2])
    t_cov = np.diag([target_pos_std ** 2 + (impact_time * target_vel_std) ** 2] * 3)
    return {
        "elevation": 0.02, "azimuth": 0.01, "impact_time": impact_time,
        "projectile_covariance": proj_cov,
        "target_position": AIM + np.array(offset), "target_velocity": np.zeros(3),
        "target_covariance": target_cov,
        "projectile_position_covariance": p_cov, "target_position_covariance": t_cov,
        "miss_mean": -np.array(offset, dtype=float),
        "half_dims": HALF, "true_target_at_impact": AIM.copy(),
        "base_hp": 0.0, "ramp": 1.0, "elapsed_s": 6.0, "range_m": 1000.0,
    }


def _box_probability(mean, stds, half=HALF):
    stds = np.maximum(stds, 1e-12)
    return float(np.prod(norm.cdf((half - mean) / stds) - norm.cdf((-half - mean) / stds)))


class TestHelpers:
    def test_ramp_factor(self):
        assert ramp_factor(0.0) == pytest.approx(0.3)
        assert ramp_factor(2.5) == pytest.approx(0.65)
        assert ramp_factor(5.0) == ramp_factor(9.0) == pytest.approx(1.0)

    def test_in_box(self):
        miss = np.array([[0.0, 0.0, 0.0], [2.5, -1.0, 1.0], [2.6, 0.0, 0.0], [0.0, 0.0, -1.01]])
        assert in_box(miss, HALF).tolist() == [True, True, False, False]

    def test_projectile_impacts_without_errors_hit_the_aim_point(self):
        s = _situation(angle_std=1e-12)
        impacts = projectile_impacts(s, 5, np.random.default_rng(0), _linear_position)
        np.testing.assert_allclose(impacts, np.tile(AIM, (5, 1)), atol=1e-6)

    def test_projectile_impacts_use_the_nominal_muzzle_velocity(self):
        seen = []
        projectile_impacts(_situation(), 3, np.random.default_rng(0),
                           lambda v, e, a, t: seen.append(v) or AIM)
        assert np.allclose(seen, MUZZLE_VELOCITY, atol=1e-3)


class TestHitRates:
    N = 40000

    def test_true_hit_rate_matches_closed_form(self):
        s = _situation(angle_std=1.5e-3)
        impacts = projectile_impacts(s, self.N, np.random.default_rng(1), _linear_position)
        expected = _box_probability(np.zeros(3), np.array([0.0, 1.5, 1.5]))
        assert true_hit_rate(s, impacts) == pytest.approx(expected, abs=0.01)

    def test_true_hit_rate_uses_the_true_target(self):
        s = _situation(angle_std=1e-12)
        s["true_target_at_impact"] = AIM + np.array([0.0, 5.0, 0.0])  # true target 5 m off the aim point
        impacts = projectile_impacts(s, 100, np.random.default_rng(2), _linear_position)
        assert true_hit_rate(s, impacts) == 0.0

    def test_model_hit_rate_matches_closed_form_with_target_uncertainty(self):
        s = _situation(angle_std=1e-3, target_pos_std=0.8, target_vel_std=0.2, offset=(0.5, 0.3, -0.2))
        impacts = projectile_impacts(s, self.N, np.random.default_rng(3), _linear_position)
        target_std = np.sqrt(0.8 ** 2 + (2.0 * 0.2) ** 2)
        stds = np.sqrt(np.array([0.0, 1.0, 1.0]) + target_std ** 2)
        expected = _box_probability(-np.array([0.5, 0.3, -0.2]), stds)
        assert model_hit_rate(s, impacts, np.random.default_rng(4)) == pytest.approx(expected, abs=0.01)

    def test_full_covariance_hp_matches_closed_form_for_diagonal_covariance(self):
        s = _situation(angle_std=1e-3, target_pos_std=0.5, offset=(0.4, 0.0, 0.0))
        stds = np.sqrt(np.diag(s["projectile_position_covariance"] + s["target_position_covariance"]))
        expected = _box_probability(s["miss_mean"], stds)
        assert full_covariance_hp(s, self.N, np.random.default_rng(5)) == pytest.approx(expected, abs=0.01)

    def test_validate_situation_is_reproducible(self):
        s = _situation(angle_std=1e-3, target_pos_std=0.5)
        a = validate_situation(s, 500, seed=7, position_fn=_linear_position)
        b = validate_situation(s, 500, seed=7, position_fn=_linear_position)
        assert a == b and a["n_samples"] == 500


RECORDED = os.path.join(os.path.dirname(__file__), "..", "results", "stage2e_eval2", "heldout2",
                        "benchmark_traditional_vs_rl_raw.csv")


@pytest.mark.skipif(not os.path.exists(RECORDED), reason="recorded evaluation results not available")
def test_firing_situation_reproduces_a_recorded_shot():
    """The rebuilt situation gives exactly the HP recorded in the evaluation."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from compare_traditional_vs_rl import scenarios_for

    from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
    from rl_training.environment import ArtilleryFiringEnv
    from rl_training.utils.evaluation_protocol import perturb_scenario

    raw = pd.read_csv(RECORDED)
    row = raw[(raw.policy == "Decaying_1.00_0.40") & raw.fired].iloc[0]
    rng = np.random.default_rng(int(row.seed))
    scenario = next(s for s in (perturb_scenario(n, rng) for n in scenarios_for("heldout2", CurriculumScenarios(phase=4)))
                    if s.name == row.scenario)
    env = ArtilleryFiringEnv()
    s = firing_situation(env, scenario, int(row.seed), int(row.fired_at_step))
    assert s["hp"] == pytest.approx(row.hit_probability, abs=1e-9)
    assert s["hp"] == pytest.approx(s["base_hp"] * s["ramp"], abs=1e-9)
    # The nominal projectile lands on the computed intersection point.
    nominal = env.hit_calculator.error_propagation._calculate_projectile_position_advanced(
        MUZZLE_VELOCITY, s["elevation"], s["azimuth"], s["impact_time"])
    np.testing.assert_allclose(nominal, s["intersection_point"], atol=0.01)


# ---------------------------------------------------------------------------
# scripts/validate_hp_mc.py: shot selection and summary
# ---------------------------------------------------------------------------

import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import validate_hp_mc as vmc  # noqa: E402


def _raw(path, policies, n_seeds=10):
    rows = [{"policy": p, "scenario": f"S{k % 3}", "seed": k, "fired": k % 4 != 0, "fired_at_step": 10 + k,
             "hit_probability": 0.5} for p in policies for k in range(n_seeds)]
    pd.DataFrame(rows).to_csv(path, index=False)


class TestSelectShots:
    POLICIES = ["S2_bp_s0", "S2_bp_s1", "V0r_s0", "Plateau_0.45_k3_e0.02", "Decaying_1.00_0.40", "Threshold_0.45"]

    def _patch(self, tmp_path, monkeypatch):
        sets = {}
        for name in ("heldout2", "phase4"):
            _raw(tmp_path / f"{name}.csv", self.POLICIES)
            sets[name] = [(name, tmp_path / f"{name}.csv")]
        monkeypatch.setattr(vmc, "SETS", sets)

    def test_stratified_by_set_and_group_fired_only(self, tmp_path, monkeypatch):
        self._patch(tmp_path, monkeypatch)
        shots = vmc.select_shots(per_group=3, seed=1)
        assert len(shots) == 2 * 4 * 3 and shots["fired"].all()
        assert shots.groupby(["scenario_set", "group"]).size().eq(3).all()
        assert not shots["policy"].eq("Threshold_0.45").any()
        assert set(shots[shots.group == "Bprime"].policy) <= {"S2_bp_s0", "S2_bp_s1"}

    def test_reproducible(self, tmp_path, monkeypatch):
        self._patch(tmp_path, monkeypatch)
        a, b = vmc.select_shots(3, seed=5), vmc.select_shots(3, seed=5)
        pd.testing.assert_frame_equal(a, b)

    def test_pooled_set_keeps_the_set_each_run_used(self, tmp_path):
        _raw(tmp_path / "evaluation.csv", self.POLICIES)
        _raw(tmp_path / "tuning.csv", self.POLICIES)
        sets = {"phase4": [("evaluation", tmp_path / "evaluation.csv"), ("tuning", tmp_path / "tuning.csv")]}
        groups = vmc.SOURCES["4e"][1]
        shots = vmc.select_shots(per_group=6, seed=1, sets=sets, groups=groups)
        assert shots["scenario_set"].eq("phase4").all()
        assert set(shots["rebuild_set"]) <= {"evaluation", "tuning"}
        assert shots.groupby("group").size().eq(6).all()

    def test_4e_source_uses_the_rules_selected_in_4e(self):
        groups = vmc.SOURCES["4e"][1]
        assert groups["Plateau"](pd.Series(["Plateau_0.40_k3_e0.01", "Plateau_0.45_k3_e0.02"])).tolist() == [True, False]


class TestSummarise:
    def _df(self):
        rows = []
        for i, (group, base) in enumerate([("Bprime", 0.5), ("V0r", 0.7), ("Plateau", 0.3), ("Decaying", 0.6)] * 2):
            rows.append({"scenario_set": "heldout2" if i < 4 else "phase4", "group": group, "base_hp": base,
                         "mc_model_hp": base + 0.01, "full_cov_hp": base, "true_hit_rate": base - 0.02,
                         "ramp": 1.0, "range_m": 1000.0 + 500 * i, "rebuild_ok": True})
        return pd.DataFrame(rows)

    def test_agreement_and_bias(self):
        s = vmc.summarise(self._df())
        assert s["n_shots"] == 8 and s["rebuild_ok"] == 8
        assert s["mc_model_hp_minus_base"]["mean"] == pytest.approx(0.01)
        assert s["true_hit_rate_minus_base"]["mean_abs"] == pytest.approx(0.02)
        assert s["full_cov_hp_minus_base"]["max_abs"] == pytest.approx(0.0)

    def test_ranking_orders(self):
        r = vmc.summarise(self._df())["ranking"]["heldout2"]
        assert r["order_scored"] == ["V0r", "Decaying", "Bprime", "Plateau"]
        assert r["order_mc_model"] == r["order_scored"] == r["order_true_hits"]

    def test_calibration_bins_cover_all_shots(self):
        s = vmc.summarise(self._df())
        assert sum(b["n"] for b in s["calibration"]) == 8
