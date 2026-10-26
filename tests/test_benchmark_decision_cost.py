"""
Tests for scripts/benchmark_decision_cost.py: timing helpers, inputs built as
in the environment, per-decision totals, and that each component is timed
(slow components replaced by stubs).
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import benchmark_decision_cost as bdc
from rl_training.agents.dqn_components import DQN
from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
from rl_training.utils.evaluation_protocol import get_evaluation_scenarios


def _scenario(i=1):
    return get_evaluation_scenarios(CurriculumScenarios(phase=4))[i]


class TestHelpers:
    def test_summarize(self):
        s = bdc.summarize_us([1.0, 2.0, 3.0, 4.0, 100.0])
        assert s["n"] == 5 and s["median_us"] == 3.0 and s["max_us"] == 100.0 and s["min_us"] == 1.0

    def test_time_call_counts_and_warmup(self):
        calls = []
        times = bdc.time_call(lambda: calls.append(1), repeats=7, warmup=3)
        assert len(times) == 7 and len(calls) == 10 and all(t >= 0 for t in times)

    def test_projectile_covariance_matches_environment_formula(self):
        s = _scenario()
        cov = bdc.projectile_covariance(s)
        angle_std = 2e-4 * (s.range_m / 1000.0) ** 0.2
        np.testing.assert_allclose(np.diag(cov), [6.25, angle_std ** 2, angle_std ** 2])

    def test_per_decision_totals(self):
        comp = {k: {"median_us": v} for k, v in
                {"kalman_update": 20.0, "intercept_ode": 1e6, "hp_propagation_diagonal": 5000.0,
                 "dqn_forward": 25.0}.items()}
        comp["_inputs"] = {}
        t = bdc.per_decision_totals(comp)
        assert t == {"rule_full_chain_us": 1005020.0, "rule_hp_only_us": 5000.0,
                     "dqn_decision_us": 45.0, "dqn_forward_us": 25.0}


class TestDecisionInputs:
    def test_filter_state_after_tracking(self):
        inp = bdc.decision_inputs(_scenario(), np.random.default_rng(0))
        assert inp["position"].shape == (3,) and inp["velocity"].shape == (3,)
        assert inp["covariance"].shape == (6, 6)
        # 5 s of tracking: the velocity estimate is close to the true velocity.
        assert inp["velocity"][0] == pytest.approx(_scenario().target_vx, abs=5.0)

    def test_reproducible(self):
        a = bdc.decision_inputs(_scenario(), np.random.default_rng(3))
        b = bdc.decision_inputs(_scenario(), np.random.default_rng(3))
        np.testing.assert_array_equal(a["covariance"], b["covariance"])
        np.testing.assert_array_equal(a["next_measurement"], b["next_measurement"])


def test_benchmark_scenario_times_every_component(monkeypatch):
    import src.find_optimal_firing_angles as foa
    import src.hit_probability as hp

    monkeypatch.setattr(foa, "find_optimal_firing_angles",
                        lambda *a, **k: (0.1, 0.0, 2.0, np.array([1000.0, 0.0, 50.0]), 0.5))

    class _Calc:
        def __init__(self, projectile_velocity):
            self.target_dimensions = None
            self.error_propagation = self

        def propagate_projectile_covariance(self, cov, v0, elev, azim, t):
            assert cov.shape == (8, 8) and t == 2.0
            return np.eye(3)

        def calculate_hit_probability_analytical(self, **kwargs):
            assert kwargs["target_param_cov"].shape == (6, 6)
            return 0.42

    monkeypatch.setattr(hp, "HitProbabilityCalculator", _Calc)
    repeats = {"fast": 20, "slow": 3, "mc": 2}
    out = bdc.benchmark_scenario(_scenario(), DQN(14, 2), np.zeros(14, dtype=np.float32),
                                 np.random.default_rng(0), repeats)
    expected = {"kalman_update", "intercept_ode", "projectile_error_propagation_ode", "hp_propagation_diagonal",
                "hp_propagation_full_covariance", "hp_propagation_mc1000", "dqn_forward"}
    assert expected <= set(out)
    assert out["intercept_ode"]["n"] == 3 and out["hp_propagation_mc1000"]["n"] == 2
    assert out["dqn_forward"]["n"] == 20 and out["kalman_update"]["min_us"] >= 0.0
    assert out["_inputs"]["hp_diagonal"] == 0.42
