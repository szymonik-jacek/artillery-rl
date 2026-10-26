"""
Tests for the imitation warm start (rl_training/train/imitation.py,
scripts/pretrain_imitation.py and train_dqn_curriculum.py --init_checkpoint).
The student network never receives hit probability as input.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from rl_training.agents.dqn_components import DuelingDQN, load_model_config
from rl_training.curriculum.scenario_generator import ScenarioParameters
from rl_training.environment import COVARIANCE_FEATURE_INDEX, Action, ArtilleryFiringEnv, covariance_feature
from rl_training.train.imitation import (
    class_weights,
    collect_teacher_dataset,
    large_margin_loss,
    pretrain,
)


# ---------------------------------------------------------------------------
# Loss and weights
# ---------------------------------------------------------------------------

class TestLargeMarginLoss:
    def test_hand_computed(self):
        q = torch.tensor([[1.0, 0.95], [0.2, 0.5]])
        actions = torch.tensor([0, 1])
        # Sample 1: max(1.0, 0.95 + 0.1) - 1.0 = 0.05; sample 2: max(0.2 + 0.1, 0.5) - 0.5 = 0.
        assert float(large_margin_loss(q, actions, margin=0.1)) == pytest.approx(0.025)

    def test_zero_when_teacher_action_leads_by_margin(self):
        q = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        assert float(large_margin_loss(q, torch.tensor([0, 1]), margin=0.5)) == 0.0

    def test_weighted_mean(self):
        q = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        actions = torch.tensor([0, 0])
        w = torch.tensor([3.0, 1.0])
        # Per-sample: [max(0, 1 + 0.1) - 0, max(0, 0 + 0.1) - 0] = [1.1, 0.1]
        # -> weighted mean (3 * 1.1 + 0.1) / 4
        assert float(large_margin_loss(q, actions, 0.1, w)) == pytest.approx(3.4 / 4)


def test_class_weights_balance_fire_and_hold():
    actions = np.array([0, 0, 0, 0, 1])
    w = class_weights(actions)
    assert list(w) == [1.0, 1.0, 1.0, 1.0, 4.0]
    assert w[actions == 1].sum() == w[actions == 0].sum()
    assert list(class_weights(np.zeros(3, dtype=int))) == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Teacher dataset on the real environment (trajectory injected via the pool)
# ---------------------------------------------------------------------------

def _scenario():
    s = ScenarioParameters(range_m=800.0, target_length=10.0, target_width=8.0, target_height=4.0,
                           target_vx=0.0, target_vy=0.0, target_vz=0.0, tracking_duration=10.0,
                           measurement_noise_std=np.array([0.5, 0.5, 0.5]),
                           process_noise_std=np.array([0.5, 0.5, 0.5]))
    s.name = "Test_Static"
    return s


def _ramp(k):
    return min(0.9, 0.2 + 0.05 * k)


def _pool():
    s = _scenario()
    steps = range(ArtilleryFiringEnv()._calculate_max_steps(s) + 1)
    trajectories = ({k: _ramp(k) for k in steps}, {k: 5.0 / (1 + k) for k in steps},
                    {k: np.array([800.0, 0.0, 50.0]) for k in steps}, {k: np.zeros(3) for k in steps})
    return {s.name: [{"name": s.name, "scenario": s, "seed": 1000, "trajectories": trajectories}]}


def test_dataset_matches_teacher_and_student_inputs():
    env = ArtilleryFiringEnv(cov_feature="log")
    features, actions, hps = collect_teacher_dataset(env, _pool())
    # Decaying 0.90 -> 0.40 over 40 steps of 0.25 s: threshold(k) = 0.9 - 0.5 * k / 40.
    fire_step = next(k for k in range(40) if _ramp(k) >= 0.9 - 0.5 * k / 40)
    assert len(actions) == fire_step + 1
    assert list(actions[:-1]) == [Action.HOLD] * fire_step and actions[-1] == Action.FIRE
    np.testing.assert_allclose(hps, [_ramp(k) for k in range(fire_step + 1)])
    # Student inputs: 14 features with the log covariance, no HP.
    assert features.shape == (fire_step + 1, 14)
    assert features[3, COVARIANCE_FEATURE_INDEX] == pytest.approx(covariance_feature(5.0 / 4, "log"))


def test_dataset_features_do_not_depend_on_hp():
    env = ArtilleryFiringEnv(cov_feature="log")
    pool_low, pool_high = _pool(), _pool()
    entry = pool_high["Test_Static"][0]
    entry["trajectories"] = ({k: min(0.95, v + 0.05) for k, v in entry["trajectories"][0].items()},) \
        + entry["trajectories"][1:]
    f_low, _, hp_low = collect_teacher_dataset(env, pool_low)
    f_high, _, hp_high = collect_teacher_dataset(ArtilleryFiringEnv(cov_feature="log"), pool_high)
    n = min(len(f_low), len(f_high))
    assert not np.allclose(hp_low[:n], hp_high[:n])          # the HP really differs ...
    np.testing.assert_array_equal(f_low[:n], f_high[:n])     # ... but the student's inputs do not


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def test_pretrain_learns_separable_teacher():
    rng = np.random.default_rng(0)
    features = rng.random((600, 14)).astype(np.float32)
    actions = (features[:, 0] > 0.85).astype(np.int64)            # ~15% FIRE: imbalanced
    hps = (0.5 * features[:, 0] + 0.2).astype(np.float32)
    torch.manual_seed(0)
    net = DuelingDQN(14, 2, aux_head=True)
    history = pretrain(net, features, actions, hps, epochs=60, batch_size=64, lr=3e-3, margin=0.1)
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["agreement_fire"] > 0.9 and history[-1]["agreement_hold"] > 0.9
    with torch.no_grad():
        pred = net.predict_aux(torch.as_tensor(features)).numpy()
    assert np.corrcoef(pred, hps)[0, 1] > 0.9


# ---------------------------------------------------------------------------
# Script and training option
# ---------------------------------------------------------------------------

def test_pretrain_script_saves_checkpoint_sidecar_and_history(tmp_path, monkeypatch):
    import pretrain_imitation
    import rl_training.train.imitation as imitation
    import rl_training.utils.trajectory_pool as tp

    rng = np.random.default_rng(1)
    feats = rng.random((50, 14)).astype(np.float32)
    acts = (feats[:, 0] > 0.8).astype(np.int64)
    monkeypatch.setattr(tp, "load_pool", lambda path: {})
    monkeypatch.setattr(imitation, "collect_teacher_dataset", lambda env, pool: (feats, acts, feats[:, 1]))
    argv = ["--prefix", "unit", "--epochs", "2", "--checkpoint_dir", str(tmp_path),
            "--history_dir", str(tmp_path / "hist")]
    pretrain_imitation.main(argv)
    ckpt = tmp_path / "unit_pretrain.pth"
    assert "aux.weight" in torch.load(ckpt)
    assert load_model_config(ckpt) == {"architecture": "dueling", "aux_head": True, "cov_feature": "log",
                                       "time_feature": "steps"}
    history = json.loads((tmp_path / "hist" / "unit_pretrain_history.json").read_text())
    assert history["teacher"] == "Decaying_0.90_0.40" and len(history["history"]) == 2
    with pytest.raises(SystemExit):  # refuses to overwrite
        pretrain_imitation.main(argv)


def test_init_checkpoint_is_first_phase_starting_point(tmp_path, monkeypatch):
    import rl_training.train.train_dqn_curriculum as train_mod

    init = tmp_path / "init.pth"
    init.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(train_mod, "train_curriculum_phase", lambda **k: calls.append(k))
    train_mod.main(["--start_phase", "1", "--end_phase", "2", "--checkpoint_prefix",
                    "dqn_init_unit_test_never_saved", "--init_checkpoint", str(init)])
    assert calls[0]["load_from"] == str(init)
    assert calls[1]["load_from"].endswith("dqn_init_unit_test_never_saved_phase1.pth")

    with pytest.raises(FileNotFoundError):
        train_mod.main(["--start_phase", "1", "--end_phase", "1", "--checkpoint_prefix",
                        "dqn_init_unit_test_never_saved", "--init_checkpoint", str(tmp_path / "missing.pth")])


def test_pretrain_script_uses_seconds_time_feature(tmp_path, monkeypatch):
    """B' pre-training: the student's dataset comes from an env with time_feature="seconds"."""
    import pretrain_imitation
    import rl_training.train.imitation as imitation
    import rl_training.utils.trajectory_pool as tp

    rng = np.random.default_rng(2)
    feats = rng.random((40, 14)).astype(np.float32)
    acts = (feats[:, 0] > 0.8).astype(np.int64)
    envs = []
    monkeypatch.setattr(tp, "load_pool", lambda path: {})
    monkeypatch.setattr(imitation, "collect_teacher_dataset",
                        lambda env, pool: envs.append(env) or (feats, acts, feats[:, 1]))
    pretrain_imitation.main(["--prefix", "bp", "--epochs", "1", "--time_feature", "seconds",
                             "--checkpoint_dir", str(tmp_path), "--history_dir", str(tmp_path / "hist")])
    assert envs[0].time_feature == "seconds" and envs[0].cov_feature == "log"
    assert load_model_config(tmp_path / "bp_pretrain.pth")["time_feature"] == "seconds"
