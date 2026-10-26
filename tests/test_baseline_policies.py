"""
Tests for rl_training/agents/baseline_policies.py

Covers: ThresholdPolicy, DecayingThresholdPolicy and
CovarianceGatedThresholdPolicy decisions,
parameter validation and naming.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.agents.baseline_policies import (
    CovarianceGatedThresholdPolicy,
    DecayingThresholdPolicy,
    ThresholdPolicy,
)
from rl_training.environment import Action


def _state(hp, cov=100.0):
    return SimpleNamespace(current_hit_probability=hp, covariance_trace=cov)


# ---------------------------------------------------------------------------
# ThresholdPolicy
# ---------------------------------------------------------------------------

class TestThresholdPolicy:
    def test_holds_below_threshold(self):
        assert ThresholdPolicy(0.55).act(_state(0.54)) == Action.HOLD

    def test_fires_at_threshold(self):
        assert ThresholdPolicy(0.55).act(_state(0.55)) == Action.FIRE

    def test_fires_above_threshold(self):
        assert ThresholdPolicy(0.55).act(_state(0.80)) == Action.FIRE

    def test_ignores_covariance(self):
        assert ThresholdPolicy(0.45).act(_state(0.50, cov=1e6)) == Action.FIRE

    @pytest.mark.parametrize("p_thr", [-0.01, 1.01])
    def test_rejects_out_of_range_threshold(self, p_thr):
        with pytest.raises(ValueError):
            ThresholdPolicy(p_thr)

    def test_name(self):
        assert ThresholdPolicy(0.5).name == "Threshold_0.50"


# ---------------------------------------------------------------------------
# CovarianceGatedThresholdPolicy
# ---------------------------------------------------------------------------

class TestCovarianceGatedThresholdPolicy:
    def test_fires_when_both_conditions_met(self):
        policy = CovarianceGatedThresholdPolicy(0.5, tau_cov=200.0)
        assert policy.act(_state(0.6, cov=150.0)) == Action.FIRE

    def test_fires_at_covariance_limit(self):
        policy = CovarianceGatedThresholdPolicy(0.5, tau_cov=200.0)
        assert policy.act(_state(0.6, cov=200.0)) == Action.FIRE

    def test_holds_when_covariance_not_converged(self):
        policy = CovarianceGatedThresholdPolicy(0.5, tau_cov=200.0)
        assert policy.act(_state(0.9, cov=200.1)) == Action.HOLD

    def test_holds_when_hp_below_threshold(self):
        policy = CovarianceGatedThresholdPolicy(0.5, tau_cov=200.0)
        assert policy.act(_state(0.4, cov=10.0)) == Action.HOLD

    @pytest.mark.parametrize("tau_cov", [0.0, -1.0])
    def test_rejects_non_positive_tau(self, tau_cov):
        with pytest.raises(ValueError):
            CovarianceGatedThresholdPolicy(0.5, tau_cov=tau_cov)

    def test_rejects_out_of_range_threshold(self):
        with pytest.raises(ValueError):
            CovarianceGatedThresholdPolicy(1.5, tau_cov=100.0)

    def test_name(self):
        assert CovarianceGatedThresholdPolicy(0.55, tau_cov=250.0).name == "CovGated_0.55_250"


# ---------------------------------------------------------------------------
# DecayingThresholdPolicy
# ---------------------------------------------------------------------------

def _timed_state(hp, time_remaining, duration=10.0):
    return SimpleNamespace(current_hit_probability=hp, time_remaining=time_remaining,
                           scenario=SimpleNamespace(tracking_duration=duration))


class TestDecayingThresholdPolicy:
    def test_threshold_at_start_middle_end(self):
        policy = DecayingThresholdPolicy(0.9, 0.5)
        assert policy.threshold(_timed_state(0.0, 10.0)) == pytest.approx(0.9)
        assert policy.threshold(_timed_state(0.0, 5.0)) == pytest.approx(0.7)
        assert policy.threshold(_timed_state(0.0, 0.0)) == pytest.approx(0.5)

    def test_elapsed_fraction_is_clipped(self):
        policy = DecayingThresholdPolicy(0.9, 0.5)
        assert policy.threshold(_timed_state(0.0, 12.0)) == pytest.approx(0.9)
        assert policy.threshold(_timed_state(0.0, -1.0)) == pytest.approx(0.5)

    def test_same_hp_holds_early_fires_late(self):
        policy = DecayingThresholdPolicy(0.9, 0.5)
        assert policy.act(_timed_state(0.6, 9.0)) == Action.HOLD   # threshold 0.86
        assert policy.act(_timed_state(0.6, 2.0)) == Action.FIRE   # threshold 0.58

    def test_fires_at_threshold(self):
        policy = DecayingThresholdPolicy(0.9, 0.5)
        assert policy.act(_timed_state(0.7, 5.0)) == Action.FIRE

    def test_equal_start_and_end_is_fixed_threshold(self):
        policy = DecayingThresholdPolicy(0.55, 0.55)
        for t in (10.0, 5.0, 0.0):
            assert policy.threshold(_timed_state(0.0, t)) == pytest.approx(0.55)

    @pytest.mark.parametrize("p_start, p_end", [(0.4, 0.5), (1.1, 0.5), (0.9, -0.1)])
    def test_rejects_invalid_parameters(self, p_start, p_end):
        with pytest.raises(ValueError):
            DecayingThresholdPolicy(p_start, p_end)

    def test_name(self):
        assert DecayingThresholdPolicy(0.9, 0.45).name == "Decaying_0.90_0.45"


# ---------------------------------------------------------------------------
# PlateauPolicy
# ---------------------------------------------------------------------------

from rl_training.agents.baseline_policies import PlateauPolicy


def _run(policy, hps):
    """Feed an HP sequence step by step; return the step of the first FIRE or None."""
    for step, hp in enumerate(hps):
        if policy.act(SimpleNamespace(current_hit_probability=hp, episode_step=step)) == Action.FIRE:
            return step
    return None


class TestPlateauPolicy:
    def test_fires_when_rise_over_window_drops_below_eps(self):
        # Rises 0.05/step, then flattens after step 6 (0.60).
        hps = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.605, 0.607, 0.608]
        # k=2, eps=0.01: step 8 compares 0.607 - 0.60 = 0.007 < 0.01.
        assert _run(PlateauPolicy(0.45, window=2, eps=0.01), hps) == 8

    def test_holds_below_min_hp_even_when_flat(self):
        flat_low = [0.30] * 10
        assert _run(PlateauPolicy(0.45, window=2, eps=0.01), flat_low) is None

    def test_needs_full_window_of_history(self):
        flat_high = [0.70] * 5
        assert _run(PlateauPolicy(0.45, window=3, eps=0.01), flat_high) == 3

    def test_fires_when_hp_starts_falling(self):
        hps = [0.40, 0.50, 0.55, 0.56, 0.54]
        assert _run(PlateauPolicy(0.50, window=1, eps=0.005), hps) == 4

    def test_history_resets_each_episode(self):
        policy = PlateauPolicy(0.45, window=2, eps=0.01)
        assert _run(policy, [0.70] * 4) == 2
        # New episode: must not reuse the previous episode's history.
        assert policy.act(SimpleNamespace(current_hit_probability=0.70, episode_step=0)) == Action.HOLD

    @pytest.mark.parametrize("kwargs", [{"min_hp": 1.5, "window": 2, "eps": 0.01},
                                        {"min_hp": 0.5, "window": 0, "eps": 0.01},
                                        {"min_hp": 0.5, "window": 2, "eps": 0.0}])
    def test_rejects_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            PlateauPolicy(**kwargs)

    def test_name(self):
        assert PlateauPolicy(0.45, window=3, eps=0.01).name == "Plateau_0.45_k3_e0.01"
