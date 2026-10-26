"""
Tests for the dueling Q-network with optional auxiliary head and for the
model-config sidecar used to load checkpoints with the right architecture and
inputs.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.agents.dqn_components import (
    DEFAULT_MODEL_CONFIG,
    DQN,
    DuelingDQN,
    build_q_network,
    load_model_config,
    load_q_network,
    model_config_path,
    save_model_config,
)


class TestDuelingDQN:
    def test_q_values_shape(self):
        net = DuelingDQN(14, 2)
        assert net(torch.zeros(5, 14)).shape == (5, 2)

    def test_dueling_combination(self):
        net = DuelingDQN(14, 2)
        x = torch.randn(3, 14)
        f = net.trunk(x)
        a = net.advantage(f)
        expected = net.value(f) + a - a.mean(dim=-1, keepdim=True)
        torch.testing.assert_close(net(x), expected)

    def test_aux_head_does_not_change_decisions(self):
        torch.manual_seed(0)
        net = DuelingDQN(14, 2, aux_head=True)
        x = torch.randn(4, 14)
        q_before = net(x).detach().clone()
        with torch.no_grad():
            net.aux.weight.add_(5.0)
            net.aux.bias.add_(5.0)
        torch.testing.assert_close(net(x), q_before)

    def test_aux_prediction_in_unit_interval(self):
        net = DuelingDQN(14, 2, aux_head=True)
        p = net.predict_aux(torch.randn(6, 14) * 10)
        assert p.shape == (6,) and torch.all((p >= 0) & (p <= 1))

    def test_predict_aux_without_head_raises(self):
        with pytest.raises(RuntimeError):
            DuelingDQN(14, 2).predict_aux(torch.zeros(1, 14))


class TestBuildQNetwork:
    def test_default_is_published_dqn(self):
        assert type(build_q_network(14, 2)) is DQN

    def test_dueling_with_aux(self):
        net = build_q_network(14, 2, {"architecture": "dueling", "aux_head": True})
        assert isinstance(net, DuelingDQN) and net.aux is not None

    @pytest.mark.parametrize("config", [{"architecture": "dqn", "aux_head": True}, {"architecture": "other"}])
    def test_invalid_configs(self, config):
        with pytest.raises(ValueError):
            build_q_network(14, 2, config)


class TestModelConfigSidecar:
    def test_round_trip(self, tmp_path):
        ckpt = tmp_path / "m_phase4.pth"
        config = {"architecture": "dueling", "aux_head": True, "cov_feature": "log"}
        save_model_config(ckpt, config)
        assert model_config_path(ckpt) == tmp_path / "m_phase4.pth.json"
        assert load_model_config(ckpt) == {**config, "time_feature": "steps"}

    def test_old_sidecar_without_time_feature_reads_as_steps(self, tmp_path):
        # Stage-2 sidecars were written before time_feature existed.
        ckpt = tmp_path / "old_phase4.pth"
        model_config_path(ckpt).write_text('{"architecture": "dueling", "aux_head": true, "cov_feature": "log"}')
        assert load_model_config(ckpt)["time_feature"] == "steps"

    def test_time_feature_round_trip(self, tmp_path):
        ckpt = tmp_path / "bp_phase4.pth"
        save_model_config(ckpt, {"architecture": "dueling", "aux_head": True, "cov_feature": "log",
                                 "time_feature": "seconds"})
        assert load_model_config(ckpt)["time_feature"] == "seconds"

    def test_missing_sidecar_gives_default(self, tmp_path):
        assert load_model_config(tmp_path / "published.pth") == DEFAULT_MODEL_CONFIG

    def test_load_q_network_uses_sidecar(self, tmp_path):
        ckpt = tmp_path / "m_phase4.pth"
        net = DuelingDQN(14, 2, aux_head=True)
        torch.save(net.state_dict(), ckpt)
        save_model_config(ckpt, {"architecture": "dueling", "aux_head": True, "cov_feature": "log"})
        loaded, config = load_q_network(ckpt, 14, 2, torch.device("cpu"))
        x = torch.randn(3, 14)
        torch.testing.assert_close(loaded(x), net(x))
        assert config["cov_feature"] == "log" and not loaded.training

    def test_published_checkpoint_loads_unchanged(self):
        ckpt = os.path.join(os.path.dirname(__file__), "..", "rl_training", "models", "checkpoints",
                            "dqn_curriculum_phase4.pth")
        if not os.path.exists(ckpt):
            pytest.skip("published checkpoint not available")
        loaded, config = load_q_network(ckpt, 14, 2, torch.device("cpu"))
        assert type(loaded) is DQN and config == DEFAULT_MODEL_CONFIG


def test_benchmark_passes_cov_feature_from_extra_dqn_tuple():
    """build_policies feeds a log-covariance agent the log-scaled input from the state."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from types import SimpleNamespace

    from compare_traditional_vs_rl import build_policies
    from rl_training.environment import COVARIANCE_FEATURE_INDEX, covariance_feature

    seen = []

    def net(x):
        seen.append(x.squeeze(0).numpy().copy())
        return torch.tensor([[1.0, 0.0]])

    policies = dict(build_policies(thresholds=(), extra_dqns=[("A", net, None, "log"), ("B", net, None)]))
    state = SimpleNamespace(episode_step=0, covariance_trace=1.0)
    policies["A"](np.zeros(14, dtype=np.float32), state)
    policies["B"](np.zeros(14, dtype=np.float32), state)
    assert seen[0][COVARIANCE_FEATURE_INDEX] == pytest.approx(covariance_feature(1.0, "log"))
    assert seen[1][COVARIANCE_FEATURE_INDEX] == 0.0


def test_benchmark_passes_time_feature_from_extra_dqn_tuple():
    """build_policies feeds a seconds agent the elapsed-time input; others keep the step count."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from types import SimpleNamespace

    from compare_traditional_vs_rl import build_policies
    from rl_training.environment import TIME_FEATURE_INDEX

    seen = []

    def net(x):
        seen.append(x.squeeze(0).numpy().copy())
        return torch.tensor([[1.0, 0.0]])

    policies = dict(build_policies(thresholds=(), extra_dqns=[("S", net, None, "log", "seconds"),
                                                              ("L", net, None, "log")]))
    state = SimpleNamespace(episode_step=16, covariance_trace=1.0, time_remaining=9.0,
                            scenario=SimpleNamespace(tracking_duration=12.0))
    obs = np.zeros(14, dtype=np.float32)
    obs[TIME_FEATURE_INDEX] = 0.16
    policies["S"](obs, state)
    policies["L"](obs, state)
    assert seen[0][TIME_FEATURE_INDEX] == pytest.approx(3.0 / 30.0)
    assert seen[1][TIME_FEATURE_INDEX] == pytest.approx(0.16)


# ---------------------------------------------------------------------------
# QEnsemble: Q-values averaged over training seeds
# ---------------------------------------------------------------------------

from rl_training.agents.dqn_components import QEnsemble, load_ensemble  # noqa: E402


class TestQEnsemble:
    def test_forward_is_the_mean_of_members(self):
        nets = [DuelingDQN(14, 2, aux_head=True) for _ in range(3)]
        x = torch.randn(4, 14)
        expected = torch.stack([n(x) for n in nets]).mean(dim=0)
        torch.testing.assert_close(QEnsemble(nets)(x), expected)

    def test_single_member_equals_the_member(self):
        net = DQN(14, 2)
        x = torch.randn(2, 14)
        torch.testing.assert_close(QEnsemble([net])(x), net(x))

    def test_empty_is_an_error(self):
        with pytest.raises(ValueError):
            QEnsemble([])

    def test_load_ensemble_checks_member_inputs(self, tmp_path):
        paths = []
        for i, time_input in enumerate(("seconds", "seconds", "steps")):
            p = tmp_path / f"m{i}_phase4.pth"
            torch.save(DuelingDQN(14, 2, aux_head=True).state_dict(), p)
            save_model_config(p, {"architecture": "dueling", "aux_head": True, "cov_feature": "log",
                                  "time_feature": time_input})
            paths.append(p)
        ens, config = load_ensemble(paths[:2], 14, 2, torch.device("cpu"))
        assert config["time_feature"] == "seconds" and config["ensemble_size"] == 2 and not ens.training
        with pytest.raises(ValueError, match="different inputs"):
            load_ensemble(paths, 14, 2, torch.device("cpu"))


def test_bprime_ensemble_loads_and_decides():
    ckpt = os.path.join(os.path.dirname(__file__), "..", "rl_training", "models", "checkpoints")
    paths = [os.path.join(ckpt, f"dqn_s2_bp_seed{s}_phase4.pth") for s in range(5)]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("B' checkpoints not available")
    ens, config = load_ensemble(paths, 14, 2, torch.device("cpu"))
    assert config["time_feature"] == "seconds" and config["cov_feature"] == "log" and config["ensemble_size"] == 5
    q = ens(torch.rand(3, 14))
    assert q.shape == (3, 2)


def test_parse_ensemble():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from compare_traditional_vs_rl import parse_ensemble

    name, paths = parse_ensemble("Ens=a.pth,b.pth,c.pth")
    assert name == "Ens" and [str(p) for p in paths] == ["a.pth", "b.pth", "c.pth"]
    for bad in ("Ens=a.pth", "=a.pth,b.pth", "Ens", "DQN=a.pth,b.pth", "Oracle=a.pth,b.pth"):
        with pytest.raises(ValueError):
            parse_ensemble(bad)
