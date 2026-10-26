"""
Tests for rl_training/utils/evaluation_protocol.py

Covers: evaluation scenario selection and the per-seed scenario perturbation
shared by the DQN evaluation and the baseline benchmark.
"""

import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
from rl_training.curriculum.scenario_generator import ScenarioParameters
from rl_training.utils.evaluation_protocol import (
    EVALUATION_SCENARIO_INDICES,
    get_evaluation_scenarios,
    perturb_scenario,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scenario(range_m=2000.0, vx=-100.0, vy=0.0, vz=5.0, noise=0.5):
    s = ScenarioParameters(
        range_m=range_m,
        target_length=5.0,
        target_width=2.0,
        target_height=2.0,
        target_vx=vx,
        target_vy=vy,
        target_vz=vz,
        tracking_duration=10.0,
        measurement_noise_std=noise,
        process_noise_std=noise,
    )
    s.name = "Test_Scenario"
    return s


def _legacy_perturb(scenario, rng):
    """Inline perturbation previously defined inside evaluate_dqn_trajectories.main."""
    s = copy.copy(scenario)
    s.range_m           = scenario.range_m           * rng.uniform(0.95, 1.05)
    s.target_vx         = scenario.target_vx         * rng.uniform(0.90, 1.10) if scenario.target_vx != 0 else 0.0
    s.target_vy         = scenario.target_vy         * rng.uniform(0.90, 1.10) if scenario.target_vy != 0 else 0.0
    s.target_vz         = scenario.target_vz         * rng.uniform(0.90, 1.10) if scenario.target_vz != 0 else 0.0
    s.measurement_noise_std = scenario.measurement_noise_std * rng.uniform(0.80, 1.20)
    return s


# ---------------------------------------------------------------------------
# get_evaluation_scenarios
# ---------------------------------------------------------------------------

class TestGetEvaluationScenarios:
    def test_returns_five_scenarios_in_order(self):
        scenarios = get_evaluation_scenarios(CurriculumScenarios(phase=4))
        assert [s.name for s in scenarios] == [
            "Static_Close",
            "Fast_Approach_Long",
            "Medium_Approach",
            "Complex_3D_Motion",
            "ExtremeLong_Large_Lateral",
        ]

    def test_uses_declared_indices(self):
        curriculum = CurriculumScenarios(phase=4)
        scenarios = get_evaluation_scenarios(curriculum)
        expected = [curriculum.get_scenario_by_index(i).name for i in EVALUATION_SCENARIO_INDICES]
        assert [s.name for s in scenarios] == expected


# ---------------------------------------------------------------------------
# perturb_scenario
# ---------------------------------------------------------------------------

class TestPerturbScenario:
    def test_matches_legacy_inline_implementation(self):
        scenario = _make_scenario()
        new = perturb_scenario(scenario, np.random.default_rng(7))
        old = _legacy_perturb(scenario, np.random.default_rng(7))
        for attr in ("range_m", "target_vx", "target_vy", "target_vz", "measurement_noise_std"):
            assert getattr(new, attr) == getattr(old, attr)

    def test_does_not_mutate_original(self):
        scenario = _make_scenario()
        perturb_scenario(scenario, np.random.default_rng(0))
        assert scenario.range_m == 2000.0
        assert scenario.target_vx == -100.0
        assert scenario.target_vz == 5.0
        assert scenario.measurement_noise_std == 0.5

    def test_same_seed_gives_same_result(self):
        scenario = _make_scenario()
        a = perturb_scenario(scenario, np.random.default_rng(3))
        b = perturb_scenario(scenario, np.random.default_rng(3))
        assert a.range_m == b.range_m
        assert a.target_vx == b.target_vx
        assert a.measurement_noise_std == b.measurement_noise_std

    @pytest.mark.parametrize("seed", range(30))
    def test_perturbations_within_bounds(self, seed):
        scenario = _make_scenario()
        s = perturb_scenario(scenario, np.random.default_rng(seed))
        assert 0.95 * 2000.0 <= s.range_m <= 1.05 * 2000.0
        assert 1.10 * -100.0 <= s.target_vx <= 0.90 * -100.0
        assert 0.90 * 5.0 <= s.target_vz <= 1.10 * 5.0
        assert 0.80 * 0.5 <= s.measurement_noise_std <= 1.20 * 0.5

    def test_zero_velocity_stays_zero_and_consumes_no_draw(self):
        scenario = _make_scenario(vx=0.0, vy=0.0, vz=0.0)
        rng = np.random.default_rng(11)
        s = perturb_scenario(scenario, rng)
        assert s.target_vx == 0.0 and s.target_vy == 0.0 and s.target_vz == 0.0

        # Only range and noise are drawn: two draws in total.
        reference = np.random.default_rng(11)
        reference.uniform(0.95, 1.05)
        reference.uniform(0.80, 1.20)
        assert rng.uniform() == reference.uniform()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

from types import SimpleNamespace

import torch

from rl_training.agents.baseline_policies import ThresholdPolicy
from rl_training.agents.dqn_components import DQN
from rl_training.utils.evaluation_protocol import (
    make_baseline_decide,
    make_dqn_decide,
    run_policy_episode,
    target_position_at_step,
)


class _FakeEnv:
    """Minimal env: precomputed HP per step, FIRE terminates, HOLD advances."""

    def __init__(self, hp_values):
        self.hp_values = hp_values
        self.max_episode_steps = len(hp_values) - 1
        self.reset_calls = []

    def reset(self, scenario_override=None, seed=None):
        self.reset_calls.append((scenario_override, seed))
        self.step_idx = 0
        self.hp_trajectory = dict(enumerate(self.hp_values))
        self.current_state = SimpleNamespace(
            current_hit_probability=self.hp_values[0], covariance_trace=100.0,
            episode_step=0, scenario=scenario_override,
        )
        return np.full(14, 0.0), {}

    def step(self, action):
        if action == 1:
            hp = self.hp_values[self.step_idx]
            terminated = True
        else:
            self.step_idx += 1
            hp = self.hp_values[self.step_idx]
            self.current_state.current_hit_probability = hp
            self.current_state.episode_step = self.step_idx
            terminated = self.step_idx >= self.max_episode_steps
        info = {"hit_probability": hp, "covariance_trace": 100.0}
        return np.full(14, float(self.step_idx)), 0.0, terminated, False, info


class TestTargetPositionAtStep:
    def test_initial_position(self):
        s = _make_scenario(range_m=2000.0, vx=-100.0, vz=5.0)
        assert target_position_at_step(s, 0, 50) == (2000.0, 0.0, 50.0)

    def test_constant_velocity_motion(self):
        s = _make_scenario(range_m=2000.0, vx=-100.0, vz=5.0)  # dt = 10 s / 50 = 0.2 s
        x, y, z = target_position_at_step(s, 10, 50)
        assert x == pytest.approx(2000.0 - 100.0 * 2.0)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(50.0 + 5.0 * 2.0)


class TestRunPolicyEpisode:
    HP = [0.20, 0.35, 0.50, 0.60, 0.55]

    def test_fires_at_first_step_meeting_threshold(self):
        env = _FakeEnv(self.HP)
        s = _make_scenario()
        r = run_policy_episode(env, s, make_baseline_decide(ThresholdPolicy(0.5)), seed=3)
        assert r["fired"] and not r["timed_out"]
        assert r["fired_at_step"] == 2
        assert r["hit_probability"] == 0.50
        assert r["steps"] == 3

    def test_firing_range_matches_geometry(self):
        env = _FakeEnv(self.HP)
        s = _make_scenario()
        r = run_policy_episode(env, s, make_baseline_decide(ThresholdPolicy(0.5)))
        expected = target_position_at_step(s, 2, env.max_episode_steps)
        assert r["firing_pos"] == expected
        assert r["firing_slant_range"] == pytest.approx(np.linalg.norm(expected))

    def test_timeout_scores_zero(self):
        env = _FakeEnv(self.HP)
        r = run_policy_episode(env, _make_scenario(), make_baseline_decide(ThresholdPolicy(0.65)))
        assert not r["fired"] and r["timed_out"]
        assert r["hit_probability"] == 0.0
        assert r["fired_at_step"] is None and r["firing_slant_range"] is None
        assert r["steps"] == env.max_episode_steps

    def test_max_steps_cap_counts_as_timeout(self):
        env = _FakeEnv(self.HP)
        r = run_policy_episode(env, _make_scenario(), lambda o, s: 0, max_steps=2)
        assert r["timed_out"] and r["steps"] == 2

    def test_passes_scenario_and_seed_to_reset(self):
        env = _FakeEnv(self.HP)
        s = _make_scenario()
        run_policy_episode(env, s, lambda o, st: 1, seed=17)
        assert env.reset_calls == [(s, 17)]

    def test_decide_receives_observation_and_state(self):
        env = _FakeEnv(self.HP)
        seen = []

        def decide(obs, state):
            seen.append((obs[0], state.current_hit_probability))
            return 1 if len(seen) == 2 else 0

        run_policy_episode(env, _make_scenario(), decide)
        assert seen == [(0.0, 0.20), (1.0, 0.35)]


class TestMakeDqnDecide:
    def test_returns_greedy_action(self):
        net = DQN(14, 2)
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
            net.network[-1].bias[1] = 1.0  # FIRE has the larger Q-value
        decide = make_dqn_decide(net, torch.device("cpu"))
        assert decide(np.zeros(14), None) == 1


# ---------------------------------------------------------------------------
# Integration: runner reproduces evaluate_trajectory on the real env
# ---------------------------------------------------------------------------

CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "rl_training", "models", "checkpoints", "dqn_curriculum_phase4.pth"
)


@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="phase-4 checkpoint not available")
def test_runner_matches_evaluate_trajectory_for_dqn():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from evaluate_dqn_trajectories import evaluate_trajectory
    from rl_training.environment import ArtilleryFiringEnv

    device = torch.device("cpu")
    net = DQN(14, 2)
    net.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    net.eval()

    scenario = perturb_scenario(
        get_evaluation_scenarios(CurriculumScenarios(phase=4))[0], np.random.default_rng(0)
    )
    reference = evaluate_trajectory(ArtilleryFiringEnv(), net, scenario, device, seed=0)
    result = run_policy_episode(ArtilleryFiringEnv(), scenario, make_dqn_decide(net, device), seed=0)

    assert result["fired_at_step"] == reference["fired_at_step"]
    assert result["hit_probability"] == pytest.approx(reference["final_hp"])
    assert result["firing_slant_range"] == pytest.approx(reference["firing_range"])


# ---------------------------------------------------------------------------
# Dead-zone interlock (min_firing_range)
# ---------------------------------------------------------------------------

from rl_training.utils.evaluation_protocol import MIN_FIRING_RANGE_M


class TestMinFiringRange:
    def test_default_constant_is_200_m(self):
        assert MIN_FIRING_RANGE_M == 200.0

    def _closing_scenario(self):
        # FakeEnv with 5 HP values: 4 steps of 2.5 s. Slant ranges per step:
        # 0: 602 m, 1: 354 m, 2: 112 m, 3: 158 m, 4: 403 m.
        return _make_scenario(range_m=600.0, vx=-100.0, vz=0.0)

    def test_no_limit_by_default(self):
        env = _FakeEnv([0.2, 0.3, 0.6, 0.6, 0.6])
        r = run_policy_episode(env, self._closing_scenario(),
                               make_baseline_decide(ThresholdPolicy(0.5)))
        assert r["fired_at_step"] == 2
        assert r["firing_slant_range"] < MIN_FIRING_RANGE_M
        assert r["blocked_fire_attempts"] == 0

    def test_fire_inside_zone_is_blocked_until_timeout(self):
        env = _FakeEnv([0.2, 0.3, 0.6, 0.6, 0.6])
        r = run_policy_episode(env, self._closing_scenario(),
                               make_baseline_decide(ThresholdPolicy(0.5)),
                               min_firing_range=MIN_FIRING_RANGE_M)
        assert r["timed_out"] and r["hit_probability"] == 0.0
        assert r["blocked_fire_attempts"] == 2

    def test_fires_once_target_leaves_zone(self):
        # 5 steps of 2 s. Slant ranges: 0: 403 m, 1: 112 m, 2: 206 m.
        env = _FakeEnv([0.2, 0.6, 0.6, 0.6, 0.6, 0.6])
        s = _make_scenario(range_m=400.0, vx=-150.0, vz=0.0)
        r = run_policy_episode(env, s, make_baseline_decide(ThresholdPolicy(0.5)),
                               min_firing_range=MIN_FIRING_RANGE_M)
        assert r["fired_at_step"] == 2
        assert r["firing_slant_range"] >= MIN_FIRING_RANGE_M
        assert r["blocked_fire_attempts"] == 1

    def test_no_effect_outside_zone(self):
        env = _FakeEnv(TestRunPolicyEpisode.HP)
        r = run_policy_episode(env, _make_scenario(), make_baseline_decide(ThresholdPolicy(0.5)),
                               min_firing_range=MIN_FIRING_RANGE_M)
        assert r["fired_at_step"] == 2
        assert r["blocked_fire_attempts"] == 0


# ---------------------------------------------------------------------------
# Hindsight oracle
# ---------------------------------------------------------------------------

from rl_training.utils.evaluation_protocol import best_reachable_step, make_oracle_decide


class TestBestReachableStep:
    def test_picks_highest_hp(self):
        hp = {0: 0.1, 1: 0.5, 2: 0.7, 3: 0.4, 4: 0.2}
        assert best_reachable_step(hp, _make_scenario(), 4) == 2

    def test_ties_go_to_earliest_step(self):
        hp = {0: 0.1, 1: 0.6, 2: 0.6, 3: 0.2, 4: 0.0}
        assert best_reachable_step(hp, _make_scenario(), 4) == 1

    def test_last_step_is_not_reachable(self):
        # Reaching max_episode_steps ends the episode, so step 4 cannot be fired at.
        hp = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.9}
        assert best_reachable_step(hp, _make_scenario(), 4) == 3

    def test_respects_min_firing_range(self):
        # Slant ranges per step (2.5 s steps): 602, 354, 112, 158 m.
        s = _make_scenario(range_m=600.0, vx=-100.0, vz=0.0)
        hp = {0: 0.1, 1: 0.3, 2: 0.9, 3: 0.8, 4: 0.0}
        assert best_reachable_step(hp, s, 4) == 2
        assert best_reachable_step(hp, s, 4, min_firing_range=200.0) == 1

    def test_none_when_no_step_qualifies(self):
        s = _make_scenario(range_m=100.0, vx=0.0, vz=0.0)
        assert best_reachable_step({0: 0.5, 1: 0.6, 2: 0.0}, s, 2, min_firing_range=200.0) is None


class TestOracleDecide:
    def test_fires_at_best_step(self):
        env = _FakeEnv([0.2, 0.5, 0.8, 0.6, 0.3])
        r = run_policy_episode(env, _make_scenario(), make_oracle_decide(env))
        assert r["fired_at_step"] == 2 and r["hit_probability"] == 0.8

    def test_at_least_as_good_as_any_threshold(self):
        hp = [0.1, 0.45, 0.62, 0.58, 0.7, 0.3, 0.2]
        oracle_env = _FakeEnv(hp)
        best = run_policy_episode(oracle_env, _make_scenario(), make_oracle_decide(oracle_env))
        for p_thr in np.arange(0.0, 1.0, 0.05):
            r = run_policy_episode(_FakeEnv(hp), _make_scenario(),
                                   make_baseline_decide(ThresholdPolicy(round(float(p_thr), 2))))
            assert best["hit_probability"] >= r["hit_probability"]

    def test_respects_min_firing_range(self):
        env = _FakeEnv([0.1, 0.3, 0.9, 0.8, 0.0])
        s = _make_scenario(range_m=600.0, vx=-100.0, vz=0.0)
        r = run_policy_episode(env, s, make_oracle_decide(env, MIN_FIRING_RANGE_M),
                               min_firing_range=MIN_FIRING_RANGE_M)
        assert r["fired_at_step"] == 1
        assert r["blocked_fire_attempts"] == 0

    def test_times_out_when_nothing_reachable(self):
        env = _FakeEnv([0.5, 0.6, 0.0])
        s = _make_scenario(range_m=100.0, vx=0.0, vz=0.0)
        r = run_policy_episode(env, s, make_oracle_decide(env, MIN_FIRING_RANGE_M),
                               min_firing_range=MIN_FIRING_RANGE_M)
        assert r["timed_out"]


@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="phase-4 checkpoint not available")
def test_oracle_bounds_dqn_on_real_env():
    from rl_training.environment import ArtilleryFiringEnv

    device = torch.device("cpu")
    net = DQN(14, 2)
    net.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    net.eval()
    scenario = perturb_scenario(
        get_evaluation_scenarios(CurriculumScenarios(phase=4))[0], np.random.default_rng(0)
    )

    env = ArtilleryFiringEnv()
    dqn = run_policy_episode(env, scenario, make_dqn_decide(net, device), seed=0,
                             min_firing_range=MIN_FIRING_RANGE_M)
    oracle = run_policy_episode(env, scenario, make_oracle_decide(env, MIN_FIRING_RANGE_M), seed=0,
                                min_firing_range=MIN_FIRING_RANGE_M)

    reachable = [env.hp_trajectory[k] for k in range(env.max_episode_steps)]
    assert oracle["hit_probability"] == pytest.approx(max(reachable))
    assert oracle["hit_probability"] >= dqn["hit_probability"]


class TestMakeDqnDecideObserveHp:
    def _capture(self):
        seen = []

        def net(x):
            seen.append(x.squeeze(0).numpy().copy())
            return torch.tensor([[0.0, 1.0]])

        return net, seen

    def test_default_passes_observation_unchanged(self):
        net, seen = self._capture()
        decide = make_dqn_decide(net, torch.device("cpu"))
        obs = np.arange(14, dtype=np.float32) / 14
        assert decide(obs, SimpleNamespace(current_hit_probability=0.9, hp_history=[0.9])) == 1
        np.testing.assert_allclose(seen[0], obs)


# ---------------------------------------------------------------------------
# Firing time in seconds
# ---------------------------------------------------------------------------

from rl_training.utils.evaluation_protocol import step_seconds


class TestFireTime:
    def test_fired_episode_reports_seconds(self):
        env = _FakeEnv([0.2, 0.35, 0.50, 0.60, 0.55])  # 4 steps over 10 s -> 2.5 s/step
        r = run_policy_episode(env, _make_scenario(), make_baseline_decide(ThresholdPolicy(0.5)))
        assert r["fired_at_step"] == 2
        assert r["fire_time_s"] == pytest.approx(5.0)

    def test_timeout_has_no_fire_time(self):
        env = _FakeEnv([0.2, 0.3, 0.4])
        r = run_policy_episode(env, _make_scenario(), make_baseline_decide(ThresholdPolicy(0.9)))
        assert r["fire_time_s"] is None

    def test_step_seconds_follows_perturbed_range(self):
        from rl_training.environment import ArtilleryFiringEnv
        env = ArtilleryFiringEnv()
        # ExtremeLong-like target: the 3000 m boundary changes the episode length.
        near = _make_scenario(range_m=2950.0, vx=0.0, vz=0.0)
        far = _make_scenario(range_m=3050.0, vx=0.0, vz=0.0)
        near.tracking_duration = far.tracking_duration = 20.0
        assert step_seconds(env, near) == pytest.approx(20.0 / 90)
        assert step_seconds(env, far) == pytest.approx(20.0 / 100)


# ---------------------------------------------------------------------------
# Covariance-input ablation
# ---------------------------------------------------------------------------

from rl_training.utils.evaluation_protocol import COVARIANCE_FEATURE_INDEX


class TestCovarianceAblation:
    def _capture(self):
        seen = []

        def net(x):
            seen.append(x.squeeze(0).numpy().copy())
            return torch.tensor([[1.0, 0.0]])

        return net, seen

    def test_index_matches_state_vector(self):
        from rl_training.environment import EnvironmentState
        s = EnvironmentState(scenario=_make_scenario(), time_remaining=5.0,
                             current_hit_probability=0.5, episode_step=0, covariance_trace=420.0)
        assert s.to_array()[COVARIANCE_FEATURE_INDEX] == pytest.approx(0.42)

    def test_zero_replaces_only_covariance(self):
        net, seen = self._capture()
        decide = make_dqn_decide(net, torch.device("cpu"), cov_ablation="zero")
        obs = np.full(14, 0.5, dtype=np.float32)
        decide(obs, SimpleNamespace(episode_step=3))
        expected = obs.copy()
        expected[COVARIANCE_FEATURE_INDEX] = 0.0
        np.testing.assert_allclose(seen[0], expected)
        assert obs[COVARIANCE_FEATURE_INDEX] == 0.5  # caller's array untouched

    def test_frozen_holds_first_step_value_per_episode(self):
        net, seen = self._capture()
        decide = make_dqn_decide(net, torch.device("cpu"), cov_ablation="frozen")
        for episode_first in (0.007, 0.004):
            for step, cov in enumerate([episode_first, 0.002, 0.0005]):
                obs = np.zeros(14, dtype=np.float32)
                obs[COVARIANCE_FEATURE_INDEX] = cov
                decide(obs, SimpleNamespace(episode_step=step))
        values = [v[COVARIANCE_FEATURE_INDEX] for v in seen]
        np.testing.assert_allclose(values, [0.007] * 3 + [0.004] * 3)

    def test_rejects_unknown_ablation(self):
        with pytest.raises(ValueError):
            make_dqn_decide(lambda x: None, torch.device("cpu"), cov_ablation="other")


class TestGetTuningScenarios:
    def test_ten_scenarios_disjoint_from_evaluation(self):
        from rl_training.utils.evaluation_protocol import get_tuning_scenarios
        curriculum = CurriculumScenarios(phase=4)
        tuning = [s.name for s in get_tuning_scenarios(curriculum)]
        evaluation = [s.name for s in get_evaluation_scenarios(curriculum)]
        assert len(tuning) == 10
        assert not set(tuning) & set(evaluation)
        assert set(tuning) | set(evaluation) == {s.name for s in curriculum.get_all_scenarios()}


# ---------------------------------------------------------------------------
# Guard: the DQN adapter never reads hit probability at decision time
# ---------------------------------------------------------------------------

class _NoHitProbabilityState:
    """Environment state whose HP fields raise if anything reads them."""

    episode_step = 0

    @property
    def current_hit_probability(self):
        raise AssertionError("hit probability read at decision time")

    @property
    def hp_history(self):
        raise AssertionError("hit probability history read at decision time")

    @property
    def initial_hit_probability(self):
        raise AssertionError("initial hit probability read at decision time")


@pytest.mark.parametrize("cov_ablation", [None, "zero", "frozen"])
def test_dqn_decision_never_reads_hit_probability(cov_ablation):
    decide = make_dqn_decide(DQN(14, 2), torch.device("cpu"), cov_ablation=cov_ablation)
    action = decide(np.zeros(14, dtype=np.float32), _NoHitProbabilityState())
    assert action in (0, 1)


def test_baseline_dqn_inputs_have_14_features():
    if not os.path.exists(CHECKPOINT):
        pytest.skip("phase-4 checkpoint not available")
    state = torch.load(CHECKPOINT, map_location="cpu")
    assert state["network.0.weight"].shape[1] == 14



class _NoHitProbabilityStateWithFilter(_NoHitProbabilityState):
    covariance_trace = 0.5


@pytest.mark.parametrize("cov_ablation", [None, "zero", "frozen"])
def test_dqn_decision_with_log_covariance_never_reads_hit_probability(cov_ablation):
    decide = make_dqn_decide(DQN(14, 2), torch.device("cpu"), cov_ablation=cov_ablation, cov_feature="log")
    assert decide(np.zeros(14, dtype=np.float32), _NoHitProbabilityStateWithFilter()) in (0, 1)


def test_adapter_rejects_unknown_cov_feature():
    with pytest.raises(ValueError):
        make_dqn_decide(DQN(14, 2), torch.device("cpu"), cov_feature="other")


# ---------------------------------------------------------------------------
# Distribution shifts and the random scenario sets
# ---------------------------------------------------------------------------

from rl_training.utils.evaluation_protocol import (
    HELDOUT_COUNT,
    HELDOUT_SEED,
    SHIFTS,
    apply_shift,
    get_heldout_scenarios,
)


class TestShifts:
    def _base(self):
        s = _make_scenario(range_m=2000.0, vx=-50.0, noise=np.array([0.5, 0.5, 0.5]))
        return s

    def test_none_is_identity_copy(self):
        s = self._base()
        out = apply_shift(s, "none")
        assert out is not s and out.range_m == s.range_m
        np.testing.assert_array_equal(out.measurement_noise_std, s.measurement_noise_std)

    @pytest.mark.parametrize("shift, factor", [("noise1.5", 1.5), ("noise2", 2.0)])
    def test_noise(self, shift, factor):
        out = apply_shift(self._base(), shift)
        np.testing.assert_allclose(out.measurement_noise_std, 0.5 * factor)
        assert out.range_m == 2000.0 and out.target_length == 5.0

    def test_size(self):
        out = apply_shift(self._base(), "size0.7")
        assert (out.target_length, out.target_width, out.target_height) == pytest.approx((3.5, 1.4, 1.4))

    def test_range(self):
        out = apply_shift(self._base(), "range1.2")
        assert out.range_m == pytest.approx(2400.0) and out.target_vx == -50.0

    def test_original_untouched_and_name_kept(self):
        s = self._base()
        for shift in SHIFTS:
            out = apply_shift(s, shift)
            assert out.name == s.name
        assert s.range_m == 2000.0 and s.target_length == 5.0
        np.testing.assert_array_equal(s.measurement_noise_std, [0.5, 0.5, 0.5])

    def test_unknown_shift_raises(self):
        with pytest.raises(ValueError):
            apply_shift(self._base(), "noise3")


class TestHeldoutScenarios:
    def test_fixed_seed_and_names(self):
        a, b = get_heldout_scenarios(), get_heldout_scenarios()
        assert HELDOUT_SEED == 20261014 and len(a) == HELDOUT_COUNT == 30
        assert [s.name for s in a] == [f"Heldout_{i:02d}" for i in range(30)]
        assert [s.range_m for s in a] == [s.range_m for s in b]

    def test_not_the_curriculum_scenarios(self):
        curriculum = {(s.range_m, s.target_vx, s.target_vy) for s in CurriculumScenarios(phase=4).get_all_scenarios()}
        heldout = get_heldout_scenarios()
        assert not {(s.range_m, s.target_vx, s.target_vy) for s in heldout} & curriculum
        assert all(800.0 <= s.range_m <= 4200.0 for s in heldout)


class TestSecondHeldoutSet:
    """Random set 2: a second random set, fixed before any run. Only properties are checked."""

    def test_seed_and_names(self):
        from rl_training.utils.evaluation_protocol import HELDOUT2_SEED

        a = get_heldout_scenarios(seed=HELDOUT2_SEED, prefix="Heldout2")
        b = get_heldout_scenarios(seed=HELDOUT2_SEED, prefix="Heldout2")
        assert HELDOUT2_SEED == 20261015 and len(a) == 30
        assert [s.name for s in a] == [f"Heldout2_{i:02d}" for i in range(30)]
        assert [s.range_m for s in a] == [s.range_m for s in b]

    def test_differs_from_first_heldout_and_curriculum(self):
        from rl_training.utils.evaluation_protocol import HELDOUT2_SEED

        key = lambda s: (s.range_m, s.target_vx, s.target_vy)
        second = {key(s) for s in get_heldout_scenarios(seed=HELDOUT2_SEED, prefix="Heldout2")}
        first = {key(s) for s in get_heldout_scenarios()}
        curriculum = {key(s) for s in CurriculumScenarios(phase=4).get_all_scenarios()}
        assert not second & first and not second & curriculum

    def test_first_heldout_set_unchanged(self):
        # The default arguments must keep producing the set used in the evaluation runs.
        assert [s.name for s in get_heldout_scenarios()][:2] == ["Heldout_00", "Heldout_01"]


def test_benchmark_scenario_set_heldout2():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from compare_traditional_vs_rl import scenarios_for

    scenarios = scenarios_for("heldout2", CurriculumScenarios(phase=4))
    assert len(scenarios) == 30 and scenarios[0].name == "Heldout2_00"
    shifted = scenarios_for("heldout2", CurriculumScenarios(phase=4), shift="range1.2")
    assert shifted[0].range_m == pytest.approx(1.2 * scenarios[0].range_m)
