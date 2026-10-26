"""
Tests for the extended training path: n-step replay with HP labels, the
Double DQN target, the auxiliary HP loss (training only) and the model-config
sidecar. With default options training must stay on the original path.
"""

import json
import os
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rl_training.train.train_dqn_curriculum as train_mod
from rl_training.agents.dqn_components import DuelingDQN, NStepReplayBuffer, model_config_path
from rl_training.train.train_dqn_curriculum import (
    _extended_update_step,
    checkpoint_path,
    model_options_from_args,
    parse_args,
    train_curriculum_phase,
)


# ---------------------------------------------------------------------------
# NStepReplayBuffer
# ---------------------------------------------------------------------------

class TestNStepReplayBuffer:
    def _push_episode(self, buf, rewards, hps):
        for t, (r, hp) in enumerate(zip(rewards, hps)):
            buf.push(np.array([t]), t % 2, r, np.array([t + 1]), t == len(rewards) - 1, hp)

    def test_n_step_returns_and_discounts(self):
        buf = NStepReplayBuffer(capacity=100, n_step=3, gamma=0.5)
        self._push_episode(buf, [1.0, 2.0, 4.0, 8.0], [0.1, 0.2, 0.3, 0.4])
        stored = list(buf.buffer)
        assert len(stored) == 4
        s, a, ret, nxt, done, disc, hp = stored[0]
        assert ret == pytest.approx(1 + 0.5 * 2 + 0.25 * 4) and disc == pytest.approx(0.125)
        assert nxt[0] == 3 and done == 0.0 and hp == 0.1
        # Truncated at the episode end: the last transitions bootstrap from the terminal state.
        s, a, ret, nxt, done, disc, hp = stored[2]
        assert ret == pytest.approx(4 + 0.5 * 8) and done == 1.0 and nxt[0] == 4 and hp == 0.3
        assert stored[3][2] == pytest.approx(8.0) and stored[3][4] == 1.0

    def test_one_step_matches_plain_transitions(self):
        buf = NStepReplayBuffer(capacity=100, n_step=1, gamma=0.99)
        self._push_episode(buf, [1.0, 2.0], [0.5, 0.6])
        assert [(t[2], t[5]) for t in buf.buffer] == [(1.0, pytest.approx(0.99)), (2.0, pytest.approx(0.99))]

    def test_nothing_stored_until_n_steps_known(self):
        buf = NStepReplayBuffer(capacity=100, n_step=3, gamma=0.9)
        buf.push(np.zeros(1), 0, 1.0, np.ones(1), False, 0.2)
        buf.push(np.zeros(1), 0, 1.0, np.ones(1), False, 0.2)
        assert len(buf) == 0

    def test_rejects_invalid_n(self):
        with pytest.raises(ValueError):
            NStepReplayBuffer(10, n_step=0, gamma=0.9)


# ---------------------------------------------------------------------------
# _extended_update_step
# ---------------------------------------------------------------------------

class _Projection(nn.Module):
    """Q(x) = scale * x[:, cols]; one trainable scale so the optimiser has a parameter."""

    def __init__(self, cols):
        super().__init__()
        self.cols = cols
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return self.scale * x[:, self.cols]


class _OneTransition:
    def __init__(self, state, action, ret, nxt, done, disc, hp):
        self.batch = ([state], [action], [ret], [nxt], [done], [disc], [hp])

    def __len__(self):
        return 1

    def sample(self, n):
        return self.batch


@pytest.mark.parametrize("double, expected_next", [(False, 5.0), (True, 1.0)])
def test_double_dqn_target(double, expected_next):
    # q(next) = [1, 5] -> argmax 1; target(next) = [5, 1] -> max 5, but target at argmax-of-q is 1.
    q_net, target_net = _Projection([0, 1]), _Projection([1, 0])
    state, nxt = np.array([2.0, 3.0]), np.array([1.0, 5.0])
    buf = _OneTransition(state, 0, 0.5, nxt, 0.0, 0.9, 0.3)
    loss = _extended_update_step(q_net, target_net, torch.optim.SGD(q_net.parameters(), lr=0.0),
                                 buf, 1, torch.device("cpu"), double_dqn=double)
    assert loss == pytest.approx((2.0 - (0.5 + 0.9 * expected_next)) ** 2)


def test_auxiliary_loss_trains_aux_head_only_when_enabled():
    torch.manual_seed(0)
    states = [np.random.rand(14).astype(np.float32) for _ in range(8)]
    batch = (states, [0] * 8, [1.0] * 8, states, [1.0] * 8, [0.9] * 8, [0.8] * 8)

    class _Buf:
        def __len__(self):
            return 8

        def sample(self, n):
            return batch

    for weight, changes in ((0.0, False), (1.0, True)):
        net = DuelingDQN(14, 2, aux_head=True)
        before = net.aux.weight.detach().clone()
        _extended_update_step(net, DuelingDQN(14, 2, aux_head=True), torch.optim.SGD(net.parameters(), lr=0.1),
                              _Buf(), 8, torch.device("cpu"), aux_hp_weight=weight)
        assert (not torch.equal(before, net.aux.weight.detach())) == changes


# ---------------------------------------------------------------------------
# train_curriculum_phase on the extended path (fake environment)
# ---------------------------------------------------------------------------

class _FakeEnv:
    """Episodes of 3 steps; HP of the current state is 0.1 * step."""

    instances = []

    def __init__(self, reward_mode="legacy", time_cost=0.0, cov_feature="linear", **kwargs):
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.current_state = SimpleNamespace(current_hit_probability=0.0)
        _FakeEnv.instances.append(self)

    def reset(self, scenario_override=None, seed=None):
        self.t = 0
        self.current_state.current_hit_probability = 0.0
        return np.zeros(14, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        self.current_state.current_hit_probability = 0.1 * self.t
        return np.full(14, self.t, dtype=np.float32), 0.0, self.t >= 3, False, {"final_hp": 0.5}


@pytest.fixture
def fake_training(monkeypatch):
    _FakeEnv.instances = []
    monkeypatch.setattr(train_mod, "ArtilleryFiringEnv", _FakeEnv)
    monkeypatch.setattr(train_mod, "evaluate_curriculum", lambda *a, **k: {})
    monkeypatch.setattr(train_mod, "check_phase_completion", lambda *a, **k: False)


class TestTrainExtended:
    def test_saves_dueling_checkpoint_and_sidecar(self, tmp_path, fake_training):
        train_curriculum_phase(phase=1, total_timesteps=20, checkpoint_prefix="ext", save_dir=tmp_path,
                               env_options={"cov_feature": "log"},
                               model_config={"architecture": "dueling", "aux_head": True},
                               double_dqn=True, n_step=3, aux_hp_weight=1.0)
        state = torch.load(checkpoint_path("ext", 1, tmp_path))
        assert "trunk.0.weight" in state and "aux.weight" in state
        config = json.loads(model_config_path(checkpoint_path("ext", 1, tmp_path)).read_text())
        assert config == {"architecture": "dueling", "aux_head": True, "cov_feature": "log",
                          "time_feature": "steps"}

    def test_sidecar_records_seconds_time_feature(self, tmp_path, fake_training):
        train_curriculum_phase(phase=1, total_timesteps=20, checkpoint_prefix="bp", save_dir=tmp_path,
                               env_options={"cov_feature": "log", "time_feature": "seconds"},
                               model_config={"architecture": "dueling", "aux_head": True},
                               double_dqn=True, n_step=3, aux_hp_weight=1.0)
        config = json.loads(model_config_path(checkpoint_path("bp", 1, tmp_path)).read_text())
        assert config["time_feature"] == "seconds"

    def test_hp_labels_are_the_states_hit_probability(self, tmp_path, fake_training, monkeypatch):
        pushes = []

        class _Recording(NStepReplayBuffer):
            def push(self, state, action, reward, next_state, done, hp_label):
                pushes.append((float(state[0]), hp_label))
                super().push(state, action, reward, next_state, done, hp_label)

        monkeypatch.setattr(train_mod, "NStepReplayBuffer", _Recording)
        train_curriculum_phase(phase=1, total_timesteps=9, checkpoint_prefix="lbl", save_dir=tmp_path, n_step=3)
        # State with first feature t has HP 0.1 * t.
        assert all(hp == pytest.approx(0.1 * t) for t, hp in pushes)

    def test_default_options_keep_original_path(self, tmp_path, fake_training):
        train_curriculum_phase(phase=1, total_timesteps=10, checkpoint_prefix="legacy", save_dir=tmp_path)
        state = torch.load(checkpoint_path("legacy", 1, tmp_path))
        assert "network.0.weight" in state
        assert not model_config_path(checkpoint_path("legacy", 1, tmp_path)).exists()

    def test_aux_weight_requires_aux_head(self, tmp_path, fake_training):
        with pytest.raises(ValueError):
            train_curriculum_phase(phase=1, total_timesteps=5, checkpoint_prefix="bad", save_dir=tmp_path,
                                   aux_hp_weight=1.0)


class TestCliOptions:
    def test_defaults_select_original_path(self):
        assert model_options_from_args(parse_args([])) == {
            "model_config": None, "double_dqn": False, "n_step": 1, "aux_hp_weight": 0.0}

    def test_preregistered_variant_a(self):
        args = parse_args(["--architecture", "dueling", "--aux_hp_weight", "1.0", "--double_dqn",
                           "--n_step", "3", "--cov_feature", "log"])
        assert model_options_from_args(args) == {
            "model_config": {"architecture": "dueling", "aux_head": True},
            "double_dqn": True, "n_step": 3, "aux_hp_weight": 1.0}
