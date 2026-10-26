"""
Tests for the opt-in reward_mode="aligned" option of ArtilleryFiringEnv
(P_hit - time_cost * t/T, used only as a training signal) and for the
guarantee that hit probability is never part of the agent's observation.

HP trajectories are injected through the environment cache so reset() is
instant.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.curriculum.scenario_generator import ScenarioParameters
from rl_training.environment import Action, ArtilleryFiringEnv, EnvironmentState


def _scenario():
    s = ScenarioParameters(
        range_m=800.0,
        target_length=10.0,
        target_width=8.0,
        target_height=4.0,
        target_vx=0.0,
        target_vy=0.0,
        target_vz=0.0,
        tracking_duration=10.0,
        measurement_noise_std=np.array([0.5, 0.5, 0.5]),
        process_noise_std=np.array([0.5, 0.5, 0.5]),
    )
    s.name = "Test_Static"
    return s


def _env_with_trajectory(hp_of_step, seed=0, **kwargs):
    """Env whose cache holds a known HP trajectory for _scenario() and seed."""
    env = ArtilleryFiringEnv(**kwargs)
    scenario = _scenario()
    steps = range(env._calculate_max_steps(scenario) + 1)
    env.hp_cache[env._get_cache_key(scenario, seed=seed)] = (
        {k: hp_of_step(k) for k in steps},
        {k: 5.0 for k in steps},
    )
    return env, scenario


def _ramp(k):
    return min(0.9, 0.2 + 0.05 * k)


# ---------------------------------------------------------------------------
# Hit probability is not an input at decision time
# ---------------------------------------------------------------------------

class TestObservationExcludesHitProbability:
    def test_state_vector_ignores_hit_probability_and_its_history(self):
        def state(hp, history):
            s = EnvironmentState(scenario=_scenario(), time_remaining=5.0, current_hit_probability=hp,
                                 episode_step=3, initial_hit_probability=hp, covariance_trace=300.0)
            s.hp_history = history
            return s

        a = state(0.05, [0.01, 0.05]).to_array()
        b = state(0.95, [0.90, 0.95]).to_array()
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("kwargs", [{}, {"reward_mode": "aligned", "time_cost": 0.15}])
    def test_env_observations_identical_for_different_hp_trajectories(self, kwargs):
        low, s = _env_with_trajectory(lambda k: 0.05, **kwargs)
        high, _ = _env_with_trajectory(lambda k: 0.95, **kwargs)
        assert low.observation_space.shape == high.observation_space.shape == (14,)
        obs_low, _ = low.reset(scenario_override=s, seed=0)
        obs_high, _ = high.reset(scenario_override=s, seed=0)
        for _ in range(5):
            np.testing.assert_array_equal(obs_low, obs_high)
            obs_low, *_ = low.step(Action.HOLD)
            obs_high, *_ = high.step(Action.HOLD)


# ---------------------------------------------------------------------------
# Defaults are unchanged
# ---------------------------------------------------------------------------

class TestDefaultsUnchanged:
    def test_default_observation_is_original_14_features(self):
        env, s = _env_with_trajectory(_ramp)
        assert env.observation_space.shape == (14,)
        obs, _ = env.reset(scenario_override=s, seed=0)
        np.testing.assert_array_equal(obs, env.current_state.to_array())
        obs, *_ = env.step(Action.HOLD)
        np.testing.assert_array_equal(obs, env.current_state.to_array())

    def test_aligned_mode_does_not_change_observations(self):
        actions = [Action.HOLD] * 6 + [Action.FIRE]
        observations = []
        for kwargs in ({}, {"reward_mode": "aligned", "time_cost": 0.15}):
            env, s = _env_with_trajectory(_ramp, **kwargs)
            obs, _ = env.reset(scenario_override=s, seed=0)
            seq = [obs] + [env.step(a)[0] for a in actions]
            observations.append(np.stack(seq))
        np.testing.assert_array_equal(observations[0], observations[1])

    @pytest.mark.parametrize("kwargs", [{"reward_mode": "other"}, {"time_cost": -0.1}])
    def test_rejects_invalid_options(self, kwargs):
        with pytest.raises(ValueError):
            ArtilleryFiringEnv(**kwargs)


# ---------------------------------------------------------------------------
# reward_mode="aligned" (training signal only)
# ---------------------------------------------------------------------------

class TestAlignedReward:
    def test_hold_gives_zero(self):
        env, s = _env_with_trajectory(_ramp, reward_mode="aligned", time_cost=0.15)
        env.reset(scenario_override=s, seed=0)
        assert env.step(Action.HOLD)[1] == 0.0

    def test_fire_gives_hp_minus_time_cost(self):
        env, s = _env_with_trajectory(_ramp, reward_mode="aligned", time_cost=0.15)
        env.reset(scenario_override=s, seed=0)
        for _ in range(4):
            env.step(Action.HOLD)
        _, reward, terminated, _, _ = env.step(Action.FIRE)
        assert terminated
        assert reward == pytest.approx(_ramp(4) - 0.15 * 4 / env.max_episode_steps)

    def test_zero_time_cost_rewards_hp(self):
        env, s = _env_with_trajectory(_ramp, reward_mode="aligned", time_cost=0.0)
        env.reset(scenario_override=s, seed=0)
        assert env.step(Action.FIRE)[1] == pytest.approx(_ramp(0))

    def test_timeout_gives_minus_time_cost(self):
        env, s = _env_with_trajectory(_ramp, reward_mode="aligned", time_cost=0.3)
        env.reset(scenario_override=s, seed=0)
        rewards, done = [], False
        while not done:
            _, r, terminated, truncated, _ = env.step(Action.HOLD)
            rewards.append(r)
            done = terminated or truncated
        assert rewards[-1] == pytest.approx(-0.3)
        assert all(r == 0.0 for r in rewards[:-1])

    def test_episode_reward_accumulates_aligned_reward(self):
        env, s = _env_with_trajectory(_ramp, reward_mode="aligned", time_cost=0.15)
        env.reset(scenario_override=s, seed=0)
        env.step(Action.HOLD)
        _, r, *_ = env.step(Action.FIRE)
        assert env.episode_reward == pytest.approx(r)



# ---------------------------------------------------------------------------
# cov_feature: scaling of the covariance-trace input (filter information)
# ---------------------------------------------------------------------------

from rl_training.environment import COVARIANCE_FEATURE_INDEX, covariance_feature


class TestCovarianceFeature:
    def test_linear_is_published_scaling(self):
        assert covariance_feature(7.0) == pytest.approx(0.007)
        assert covariance_feature(5000.0) == 1.0

    def test_log_spreads_practical_range(self):
        assert covariance_feature(0.1, "log") == pytest.approx(0.2)
        assert covariance_feature(1.0, "log") == pytest.approx(0.4)
        assert covariance_feature(1000.0, "log") == pytest.approx(1.0)
        assert covariance_feature(1e-6, "log") == 0.0 and covariance_feature(1e9, "log") == 1.0

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            covariance_feature(1.0, "other")
        with pytest.raises(ValueError):
            ArtilleryFiringEnv(cov_feature="other")

    def test_log_changes_only_the_covariance_input(self):
        lin, s = _env_with_trajectory(_ramp)
        log, _ = _env_with_trajectory(_ramp, cov_feature="log")
        obs_lin, _ = lin.reset(scenario_override=s, seed=0)
        obs_log, _ = log.reset(scenario_override=s, seed=0)
        for _ in range(4):
            others = [i for i in range(14) if i != COVARIANCE_FEATURE_INDEX]
            np.testing.assert_array_equal(obs_lin[others], obs_log[others])
            assert obs_log[COVARIANCE_FEATURE_INDEX] == pytest.approx(
                covariance_feature(log.current_state.covariance_trace, "log"))
            obs_lin, *_ = lin.step(Action.HOLD)
            obs_log, *_ = log.step(Action.HOLD)

    def test_log_observation_independent_of_hp(self):
        low, s = _env_with_trajectory(lambda k: 0.05, cov_feature="log")
        high, _ = _env_with_trajectory(lambda k: 0.95, cov_feature="log")
        obs_low, _ = low.reset(scenario_override=s, seed=0)
        obs_high, _ = high.reset(scenario_override=s, seed=0)
        for _ in range(5):
            np.testing.assert_array_equal(obs_low, obs_high)
            obs_low, *_ = low.step(Action.HOLD)
            obs_high, *_ = high.step(Action.HOLD)


def test_adapter_reproduces_training_observation_with_log_covariance():
    """make_dqn_decide(cov_feature="log") in the default env feeds the network
    exactly what ArtilleryFiringEnv(cov_feature="log") produced in training."""
    import torch

    from rl_training.utils.evaluation_protocol import make_dqn_decide

    seen = []

    def net(x):
        seen.append(x.squeeze(0).numpy().copy())
        return torch.tensor([[1.0, 0.0]])  # always HOLD

    decide = make_dqn_decide(net, torch.device("cpu"), cov_feature="log")
    train_env, s = _env_with_trajectory(_ramp, cov_feature="log")
    bench_env, _ = _env_with_trajectory(_ramp)
    train_obs, _ = train_env.reset(scenario_override=s, seed=0)
    bench_obs, _ = bench_env.reset(scenario_override=s, seed=0)
    for _ in range(6):
        decide(bench_obs, bench_env.current_state)
        np.testing.assert_allclose(seen[-1], train_obs, atol=1e-6)
        train_obs, *_ = train_env.step(Action.HOLD)
        bench_obs, *_ = bench_env.step(Action.HOLD)


# ---------------------------------------------------------------------------
# time_feature: step count or elapsed seconds (stage 2e, variant B')
# ---------------------------------------------------------------------------

from rl_training.environment import ELAPSED_SECONDS_SCALE, TIME_FEATURE_INDEX, time_feature


def _state(step, time_remaining, window=10.0):
    s = _scenario()
    s.tracking_duration = window
    return EnvironmentState(scenario=s, time_remaining=time_remaining, current_hit_probability=0.5,
                            episode_step=step, initial_hit_probability=0.5, covariance_trace=5.0)


class TestTimeFeature:
    def test_steps_is_the_published_feature(self):
        state = _state(step=7, time_remaining=6.0)
        assert TIME_FEATURE_INDEX == 10
        assert time_feature(state) == pytest.approx(0.07)
        assert state.to_array()[TIME_FEATURE_INDEX] == pytest.approx(time_feature(state, "steps"))

    def test_seconds_is_elapsed_tracking_time(self):
        assert ELAPSED_SECONDS_SCALE == 30.0
        assert time_feature(_state(step=7, time_remaining=4.0), "seconds") == pytest.approx(6.0 / 30.0)
        assert time_feature(_state(step=0, time_remaining=10.0), "seconds") == 0.0

    def test_seconds_is_clipped(self):
        assert time_feature(_state(step=99, time_remaining=-0.5), "seconds") == pytest.approx(10.5 / 30.0)
        assert time_feature(_state(step=1, time_remaining=0.0, window=45.0), "seconds") == 1.0
        assert time_feature(_state(step=0, time_remaining=12.0), "seconds") == 0.0

    def test_same_step_different_seconds_across_step_lengths(self):
        # The point of the option: step 16 is 2.6 s in one scenario and 4.8 s in another.
        short = time_feature(_state(step=16, time_remaining=13.5 - 16 * 13.5 / 90, window=13.5), "seconds")
        long_ = time_feature(_state(step=16, time_remaining=12.0 - 16 * 12.0 / 40, window=12.0), "seconds")
        assert short * 30 == pytest.approx(2.4) and long_ * 30 == pytest.approx(4.8)

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            time_feature(_state(1, 9.0), "minutes")
        with pytest.raises(ValueError):
            ArtilleryFiringEnv(time_feature="minutes")

    def test_seconds_changes_only_the_time_input(self):
        steps, s = _env_with_trajectory(_ramp, cov_feature="log")
        secs, _ = _env_with_trajectory(_ramp, cov_feature="log", time_feature="seconds")
        assert secs.observation_space.shape == (14,)
        obs_steps, _ = steps.reset(scenario_override=s, seed=0)
        obs_secs, _ = secs.reset(scenario_override=s, seed=0)
        step_length = s.tracking_duration / secs.max_episode_steps
        for k in range(5):
            others = [i for i in range(14) if i != TIME_FEATURE_INDEX]
            np.testing.assert_array_equal(obs_steps[others], obs_secs[others])
            assert obs_secs[TIME_FEATURE_INDEX] == pytest.approx(k * step_length / ELAPSED_SECONDS_SCALE, abs=1e-6)
            obs_steps, *_ = steps.step(Action.HOLD)
            obs_secs, *_ = secs.step(Action.HOLD)

    def test_seconds_observation_independent_of_hp(self):
        low, s = _env_with_trajectory(lambda k: 0.05, time_feature="seconds")
        high, _ = _env_with_trajectory(lambda k: 0.95, time_feature="seconds")
        obs_low, _ = low.reset(scenario_override=s, seed=0)
        obs_high, _ = high.reset(scenario_override=s, seed=0)
        for _ in range(5):
            np.testing.assert_array_equal(obs_low, obs_high)
            obs_low, *_ = low.step(Action.HOLD)
            obs_high, *_ = high.step(Action.HOLD)

    def test_default_is_steps(self):
        assert ArtilleryFiringEnv().time_feature == "steps"


def test_adapter_reproduces_training_observation_with_seconds_and_log_covariance():
    """make_dqn_decide(cov_feature="log", time_input="seconds") in the default env
    feeds the network exactly what the B' training environment produced."""
    import torch

    from rl_training.utils.evaluation_protocol import make_dqn_decide

    seen = []

    def net(x):
        seen.append(x.squeeze(0).numpy().copy())
        return torch.tensor([[1.0, 0.0]])  # always HOLD

    decide = make_dqn_decide(net, torch.device("cpu"), cov_feature="log", time_input="seconds")
    train_env, s = _env_with_trajectory(_ramp, cov_feature="log", time_feature="seconds")
    bench_env, _ = _env_with_trajectory(_ramp)
    train_obs, _ = train_env.reset(scenario_override=s, seed=0)
    bench_obs, _ = bench_env.reset(scenario_override=s, seed=0)
    for _ in range(6):
        decide(bench_obs, bench_env.current_state)
        np.testing.assert_allclose(seen[-1], train_obs, atol=1e-6)
        train_obs, *_ = train_env.step(Action.HOLD)
        bench_obs, *_ = bench_env.step(Action.HOLD)


def test_adapter_rejects_unknown_time_input():
    import torch

    from rl_training.utils.evaluation_protocol import make_dqn_decide

    with pytest.raises(ValueError):
        make_dqn_decide(lambda x: x, torch.device("cpu"), time_input="minutes")


# ---------------------------------------------------------------------------
# extrapolate_estimate: filter estimate predicted to the decision time (step 4b)
# ---------------------------------------------------------------------------

from rl_training.environment import extrapolate_filter  # noqa: E402


class TestExtrapolateEstimate:
    def _kf(self):
        from src.kalman_filter import TargetKalmanFilter

        kf = TargetKalmanFilter(initial_state=np.array([1000.0, 5.0, 50.0, -20.0, 3.0, 0.0]),
                                initial_covariance=np.diag([0.04, 0.04, 0.04, 0.01, 0.01, 0.01]),
                                process_noise_std=np.array([0.5, 0.5, 0.5]),
                                measurement_noise_std=np.array([0.5, 0.5, 0.5]), dt=0.5)
        return kf

    def test_predicts_position_with_velocity_and_grows_covariance(self):
        kf = self._kf()
        out = extrapolate_filter(kf, 0.3, _scenario())
        np.testing.assert_allclose(out.state[:3], kf.state[:3] + 0.3 * kf.state[3:])
        np.testing.assert_allclose(out.state[3:], kf.state[3:])
        assert np.trace(out.covariance) > np.trace(kf.covariance)
        np.testing.assert_allclose(kf.state, [1000.0, 5.0, 50.0, -20.0, 3.0, 0.0])  # input untouched

    def test_default_is_off_and_keeps_cache_keys(self):
        off, on = ArtilleryFiringEnv(), ArtilleryFiringEnv(extrapolate_estimate=True)
        assert off.extrapolate_estimate is False
        s = _scenario()
        assert off._get_cache_key(s, seed=0) == ArtilleryFiringEnv()._get_cache_key(s, seed=0)
        assert on._get_cache_key(s, seed=0) != off._get_cache_key(s, seed=0)

    def test_trajectory_pool_is_refused_when_on(self):
        from rl_training.utils.trajectory_pool import install_pool

        with pytest.raises(ValueError, match="extrapolate_estimate"):
            install_pool(ArtilleryFiringEnv(extrapolate_estimate=True), {})
        install_pool(ArtilleryFiringEnv(), {})  # published setting unaffected


def _rebuilt(extrapolate, scenario_set="heldout2", policy="Decaying_1.00_0.40", n=3):
    """Rebuild n recorded shots with and without the extrapolation (skips if results are missing)."""
    import sys

    path = os.path.join(os.path.dirname(__file__), "..", "results", "stage2e_eval2", scenario_set,
                        "benchmark_traditional_vs_rl_raw.csv")
    if not os.path.exists(path):
        pytest.skip("recorded evaluation results not available")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import pandas as pd
    from validate_hp_mc import _perturbed

    from rl_training.utils.hp_validation import firing_situation

    raw = pd.read_csv(path)
    rows = raw[(raw.policy == policy) & raw.fired].head(n)
    env = ArtilleryFiringEnv(extrapolate_estimate=extrapolate)
    return [(r, firing_situation(env, _perturbed(scenario_set, r.scenario, int(r.seed)), int(r.seed),
                                 int(r.fired_at_step))) for r in rows.itertuples()]


def test_extrapolation_moves_the_aim_onto_the_true_target():
    """Off: the aim point misses the true target by the motion during the lag. On: it is much closer."""
    off, on = _rebuilt(False), _rebuilt(True)
    for (row, a), (_, b) in zip(off, on):
        assert a["hp"] == pytest.approx(row.hit_probability, abs=1e-9)  # off reproduces the evaluation
        assert a["estimate_lag_s"] == 0.0 and b["estimate_lag_s"] >= 0.0
        miss_off = np.linalg.norm(a["intersection_point"] - a["true_target_at_impact"])
        miss_on = np.linalg.norm(b["intersection_point"] - b["true_target_at_impact"])
        if b["estimate_lag_s"] > 0.05:
            assert miss_on < miss_off
