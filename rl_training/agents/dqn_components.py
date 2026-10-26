"""
DQN Components for Artillery Firing Training

Contains the DQN network and ReplayBuffer classes.
"""

import torch
import torch.nn as nn
from collections import deque
import random
from typing import Tuple


class DQN(nn.Module):
    """Deep Q-Network for artillery firing decisions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class DuelingDQN(nn.Module):
    """Dueling Q-network (Wang et al., 2016) with an optional auxiliary head.

    forward(x) returns the Q-values used for decisions. With aux_head=True,
    predict_aux(x) returns a prediction in [0, 1] from the same features; it
    is trained on hit probability as an auxiliary task and never used to
    decide, so HP is not needed at decision time.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, aux_head: bool = False):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)
        self.aux = nn.Linear(hidden_dim, 1) if aux_head else None

    def forward(self, x):
        features = self.trunk(x)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=-1, keepdim=True)

    def predict_aux(self, x):
        if self.aux is None:
            raise RuntimeError("network was built without an auxiliary head")
        return torch.sigmoid(self.aux(self.trunk(x))).squeeze(-1)


DEFAULT_MODEL_CONFIG = {"architecture": "dqn", "aux_head": False, "cov_feature": "linear",
                        "time_feature": "steps"}


def model_config_path(checkpoint_path):
    """Sidecar file describing a checkpoint's network and inputs: <checkpoint>.json."""
    from pathlib import Path

    return Path(str(checkpoint_path) + ".json")


def build_q_network(state_dim: int, action_dim: int, config=None):
    """Network for a model config (see DEFAULT_MODEL_CONFIG)."""
    config = {**DEFAULT_MODEL_CONFIG, **(config or {})}
    if config["architecture"] == "dqn":
        if config["aux_head"]:
            raise ValueError("the auxiliary head requires architecture='dueling'")
        return DQN(state_dim, action_dim)
    if config["architecture"] == "dueling":
        return DuelingDQN(state_dim, action_dim, aux_head=config["aux_head"])
    raise ValueError(f"unknown architecture {config['architecture']!r}")


def save_model_config(checkpoint_path, config):
    """Write the sidecar config for a checkpoint."""
    import json

    config = {**DEFAULT_MODEL_CONFIG, **config}
    model_config_path(checkpoint_path).write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_model_config(checkpoint_path):
    """Config of a checkpoint; DEFAULT_MODEL_CONFIG if it has no sidecar (e.g. the Base-0 model)."""
    import json

    path = model_config_path(checkpoint_path)
    if not path.exists():
        return dict(DEFAULT_MODEL_CONFIG)
    return {**DEFAULT_MODEL_CONFIG, **json.loads(path.read_text(encoding="utf-8"))}


class QEnsemble(nn.Module):
    """Q-values averaged over several trained networks (e.g. the training seeds of one method)."""

    def __init__(self, networks):
        super().__init__()
        if not networks:
            raise ValueError("an ensemble needs at least one network")
        self.members = nn.ModuleList(networks)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(dim=0)


def load_ensemble(checkpoint_paths, state_dim: int, action_dim: int, device):
    """Load networks with identical input configs as one QEnsemble. Returns (ensemble, config)."""
    networks, configs = zip(*(load_q_network(p, state_dim, action_dim, device) for p in checkpoint_paths))
    inputs = {(c["cov_feature"], c["time_feature"]) for c in configs}
    if len(inputs) != 1:
        raise ValueError(f"ensemble members use different inputs: {sorted(inputs)}")
    ensemble = QEnsemble(list(networks)).to(device)
    ensemble.eval()
    return ensemble, {**configs[0], "ensemble_size": len(networks)}


def load_q_network(checkpoint_path, state_dim: int, action_dim: int, device):
    """Load a checkpoint with the architecture from its sidecar. Returns (network, config)."""
    config = load_model_config(checkpoint_path)
    network = build_q_network(state_dim, action_dim, config).to(device)
    network.load_state_dict(torch.load(checkpoint_path, map_location=device))
    network.eval()
    return network, config


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        return len(self.buffer)

class NStepReplayBuffer:
    """Replay buffer of n-step transitions with a hit-probability label per state.

    push() receives single steps; a transition is stored once n steps (or the
    episode end) are known: return R = sum_i gamma^i r_i, the state reached
    after those steps, whether the episode ended, and the bootstrap discount
    gamma^k. The HP label of the starting state is a training-only target for
    an auxiliary head; it is never part of the network input.
    """

    def __init__(self, capacity: int, n_step: int, gamma: float):
        if n_step < 1:
            raise ValueError(f"n_step must be at least 1, got {n_step}")
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self._pending = deque()

    def push(self, state, action, reward, next_state, done, hp_label):
        self._pending.append((state, action, reward, next_state, done, hp_label))
        if done:
            while self._pending:
                self._emit()
        elif len(self._pending) >= self.n_step:
            self._emit()

    def _emit(self):
        ret, discount = 0.0, 1.0
        last_next, last_done = None, False
        for k, (_, _, reward, next_state, done, _) in enumerate(self._pending):
            if k >= self.n_step:
                break
            ret += discount * reward
            discount *= self.gamma
            last_next, last_done = next_state, done
            if done:
                break
        state, action, _, _, _, hp_label = self._pending.popleft()
        self.buffer.append((state, action, ret, last_next, float(last_done), discount, hp_label))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.buffer)
