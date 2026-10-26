"""
Tests for the training options used by the HP-observation retraining
experiment: checkpoint naming, the overwrite guard, CLI parsing and passing
environment options through to ArtilleryFiringEnv.

Training itself runs against a small fake environment so the tests are fast.
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rl_training.train.train_dqn_curriculum as train_mod
from rl_training.train.train_dqn_curriculum import (
    CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_PREFIX,
    checkpoint_path,
    env_options_from_args,
    parse_args,
    train_curriculum_phase,
)


class _FakeEnv:
    """Gym-like env: episodes end after 3 steps; records constructor kwargs."""

    instances = []

    def __init__(self, reward_mode="legacy", time_cost=0.0, cov_feature="linear", time_feature="steps", **kwargs):
        self.options = {"reward_mode": reward_mode, "time_cost": time_cost, "cov_feature": cov_feature,
                        "time_feature": time_feature}
        dim = 14
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        _FakeEnv.instances.append(self)

    def reset(self, scenario_override=None, seed=None):
        self.t = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, self.t >= 3, False, {"final_hp": 0.5}


@pytest.fixture
def fake_training(monkeypatch):
    _FakeEnv.instances = []
    monkeypatch.setattr(train_mod, "ArtilleryFiringEnv", _FakeEnv)
    monkeypatch.setattr(train_mod, "evaluate_curriculum", lambda *a, **k: {})
    monkeypatch.setattr(train_mod, "check_phase_completion", lambda *a, **k: False)


# ---------------------------------------------------------------------------
# checkpoint_path
# ---------------------------------------------------------------------------

class TestCheckpointPath:
    def test_default_matches_base0_file_name(self):
        assert checkpoint_path(phase=4) == CHECKPOINT_DIR / "dqn_curriculum_phase4.pth"
        assert DEFAULT_CHECKPOINT_PREFIX == "dqn_curriculum"

    def test_custom_prefix_and_dir(self, tmp_path):
        assert checkpoint_path("dqn_hp_v1_seed0", 2, tmp_path) == tmp_path / "dqn_hp_v1_seed0_phase2.pth"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults_reproduce_original_training(self):
        args = parse_args([])
        assert (args.start_phase, args.end_phase, args.timesteps) == (1, 3, 10000)
        assert args.checkpoint_prefix == DEFAULT_CHECKPOINT_PREFIX and not args.overwrite
        assert env_options_from_args(args) == {"reward_mode": "legacy", "time_cost": 0.0, "cov_feature": "linear", "time_feature": "steps"}

    def test_variant_options(self):
        args = parse_args(["--reward_mode", "aligned", "--time_cost", "0.15",
                           "--seed", "2", "--checkpoint_prefix", "dqn_hp_v2_l0.15_seed2"])
        assert env_options_from_args(args) == {"reward_mode": "aligned", "time_cost": 0.15, "cov_feature": "linear", "time_feature": "steps"}
        assert args.seed == 2 and args.checkpoint_prefix == "dqn_hp_v2_l0.15_seed2"

    def test_rejects_unknown_reward_mode(self):
        with pytest.raises(SystemExit):
            parse_args(["--reward_mode", "other"])


# ---------------------------------------------------------------------------
# train_curriculum_phase
# ---------------------------------------------------------------------------

class TestTrainCurriculumPhase:
    def test_refuses_to_overwrite_before_training(self, tmp_path, monkeypatch):
        existing = checkpoint_path("x", 1, tmp_path)
        existing.write_bytes(b"keep me")
        monkeypatch.setattr(train_mod, "ArtilleryFiringEnv",
                            lambda **k: pytest.fail("training started despite existing checkpoint"))
        with pytest.raises(FileExistsError):
            train_curriculum_phase(phase=1, total_timesteps=10, checkpoint_prefix="x", save_dir=tmp_path)
        assert existing.read_bytes() == b"keep me"

    def test_overwrite_replaces_existing(self, tmp_path, fake_training):
        existing = checkpoint_path("x", 1, tmp_path)
        existing.write_bytes(b"old")
        train_curriculum_phase(phase=1, total_timesteps=10, checkpoint_prefix="x",
                               save_dir=tmp_path, overwrite=True)
        assert existing.read_bytes() != b"old"

    def test_passes_env_options_and_saves_14_input_network(self, tmp_path, fake_training):
        options = {"reward_mode": "aligned", "time_cost": 0.15, "cov_feature": "linear", "time_feature": "steps"}
        train_curriculum_phase(phase=1, total_timesteps=10, env_options=options,
                               checkpoint_prefix="dqn_hp_test", save_dir=tmp_path)
        assert _FakeEnv.instances[0].options == options
        state = torch.load(checkpoint_path("dqn_hp_test", 1, tmp_path))
        assert state["network.0.weight"].shape == (128, 14)

    def test_default_env_options_give_14_inputs(self, tmp_path, fake_training):
        train_curriculum_phase(phase=1, total_timesteps=10, checkpoint_prefix="legacy",
                               save_dir=tmp_path)
        assert _FakeEnv.instances[0].options == {"reward_mode": "legacy", "time_cost": 0.0, "cov_feature": "linear", "time_feature": "steps"}
        state = torch.load(checkpoint_path("legacy", 1, tmp_path))
        assert state["network.0.weight"].shape == (128, 14)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_refuses_to_touch_base0_checkpoint(self, monkeypatch):
        base0 = checkpoint_path(phase=4)
        if not base0.exists():
            pytest.skip("Base-0 checkpoint not available")
        before = base0.read_bytes()
        monkeypatch.setattr(train_mod, "train_curriculum_phase",
                            lambda **k: pytest.fail("training started"))
        with pytest.raises(FileExistsError):
            train_mod.main(["--start_phase", "4", "--end_phase", "4"])
        assert base0.read_bytes() == before

    def test_passes_options_and_chains_phases(self, monkeypatch):
        calls = []
        monkeypatch.setattr(train_mod, "train_curriculum_phase", lambda **k: calls.append(k))
        train_mod.main(["--start_phase", "1", "--end_phase", "2", "--timesteps", "5",
                        "--reward_mode", "aligned", "--time_cost", "0.3",
                        "--checkpoint_prefix", "dqn_hp_unit_test_never_saved"])
        assert [c["phase"] for c in calls] == [1, 2]
        assert all(c["env_options"] == {"reward_mode": "aligned", "time_cost": 0.3, "cov_feature": "linear", "time_feature": "steps"} for c in calls)
        assert calls[0]["load_from"] is None
        assert calls[1]["load_from"] == str(checkpoint_path("dqn_hp_unit_test_never_saved", 1))
        assert all(c["checkpoint_prefix"] == "dqn_hp_unit_test_never_saved" for c in calls)


# ---------------------------------------------------------------------------
# Training on a trajectory pool
# ---------------------------------------------------------------------------

import copy

from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
from rl_training.train.train_dqn_curriculum import reset_episode
from rl_training.utils.trajectory_pool import PoolSampler


class _PoolFakeEnv(_FakeEnv):
    """_FakeEnv with the cache interface install_pool uses; records resets."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hp_cache = {}
        self.resets = []

    def _get_cache_key(self, scenario, seed=None):
        return (scenario.name, scenario.range_m, seed)

    def reset(self, scenario_override=None, seed=None):
        self.resets.append((scenario_override, seed))
        return super().reset(scenario_override, seed)


def _fake_pool(phase=4, variants=3):
    pool = {}
    for i, s in enumerate(CurriculumScenarios(phase=phase).get_all_scenarios()):
        for k in range(variants):
            v = copy.copy(s)
            v.range_m = s.range_m + k + 1
            pool.setdefault(s.name, []).append(
                {"name": s.name, "scenario": v, "seed": 1000 + 50 * i + k,
                 "trajectories": ({0: 0.1}, {0: 1.0}, {0: 0}, {0: 0})})
    return pool


@pytest.fixture
def pool_training(monkeypatch):
    _PoolFakeEnv.instances = []
    _FakeEnv.instances = []
    monkeypatch.setattr(train_mod, "ArtilleryFiringEnv", _PoolFakeEnv)
    monkeypatch.setattr(train_mod, "evaluate_curriculum", lambda *a, **k: {})
    monkeypatch.setattr(train_mod, "check_phase_completion", lambda *a, **k: False)


class TestResetEpisode:
    def test_without_sampler_uses_nominal_scenario(self):
        env = _PoolFakeEnv()
        s = CurriculumScenarios(phase=1).get_scenario_by_index(0)
        reset_episode(env, s)
        assert env.resets == [(s, None)]

    def test_with_sampler_uses_pool_variant_and_seed(self):
        env = _PoolFakeEnv()
        pool = _fake_pool()
        s = CurriculumScenarios(phase=1).get_scenario_by_index(0)
        reset_episode(env, s, PoolSampler(pool, seed=0))
        scenario, seed = env.resets[0]
        assert scenario.name == s.name and scenario.range_m != s.range_m
        assert seed in {e["seed"] for e in pool[s.name]}


class TestTrainOnPool:
    def _train(self, tmp_path, seed, prefix):
        train_curriculum_phase(phase=2, total_timesteps=60, checkpoint_prefix=prefix,
                               save_dir=tmp_path, seed=seed, trajectory_pool=_fake_pool())
        return _FakeEnv.instances[-1]

    def test_every_training_episode_uses_a_pool_variant(self, tmp_path, pool_training):
        env = self._train(tmp_path, seed=0, prefix="a")
        assert len(env.resets) > 5
        assert all(seed is not None and seed >= 1000 for _, seed in env.resets)
        assert len(env.hp_cache) == 15 * 3  # whole pool installed

    def test_variant_sequence_reproducible_from_seed(self, tmp_path, pool_training):
        a = [seed for _, seed in self._train(tmp_path, seed=1, prefix="a").resets]
        b = [seed for _, seed in self._train(tmp_path, seed=1, prefix="b").resets]
        c = [seed for _, seed in self._train(tmp_path, seed=2, prefix="c").resets]
        assert a == b and a != c

    def test_no_pool_keeps_nominal_resets(self, tmp_path, pool_training):
        train_curriculum_phase(phase=1, total_timesteps=30, checkpoint_prefix="n", save_dir=tmp_path)
        env = _FakeEnv.instances[-1]
        assert all(seed is None for _, seed in env.resets)

    def test_main_loads_and_passes_pool(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(train_mod, "train_curriculum_phase", lambda **k: calls.append(k))
        import rl_training.utils.trajectory_pool as tp
        monkeypatch.setattr(tp, "load_pool", lambda path: {"loaded_from": path})
        train_mod.main(["--start_phase", "1", "--end_phase", "1", "--checkpoint_prefix",
                        "dqn_pool_unit_test_never_saved", "--trajectory_pool", str(tmp_path)])
        assert calls[0]["trajectory_pool"] == {"loaded_from": str(tmp_path)}


def test_cov_feature_option_reaches_environment_options():
    args = parse_args(["--cov_feature", "log"])
    assert env_options_from_args(args)["cov_feature"] == "log"
    with pytest.raises(SystemExit):
        parse_args(["--cov_feature", "other"])


def test_time_feature_option_reaches_environment_options():
    assert env_options_from_args(parse_args([]))["time_feature"] == "steps"
    assert env_options_from_args(parse_args(["--time_feature", "seconds"]))["time_feature"] == "seconds"
    with pytest.raises(SystemExit):
        parse_args(["--time_feature", "minutes"])
