"""
Tests for evaluate_dqn_trajectories.py

Covers: evaluate_trajectory, load_dqn_model (error path),
        print_trajectory_summary, and the duplicate-loop dead-code block
        in plot_trajectory_results.
"""

import io
import contextlib
import sys
import os
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")

from evaluate_dqn_trajectories import (
    evaluate_trajectory,
    load_dqn_model,
    print_trajectory_summary,
    plot_trajectory_results,
    plot_individual_scenarios,
)
from rl_training.agents.dqn_components import DQN
from rl_training.curriculum.scenario_generator import ScenarioParameters
import evaluate_dqn_trajectories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scenario(range_m=1000.0, vx=0.0) -> ScenarioParameters:
    noise = np.array([1.0, 1.0, 1.0])
    s = ScenarioParameters(
        range_m=range_m,
        target_length=5.0,
        target_width=2.0,
        target_height=2.0,
        target_vx=vx,
        target_vy=0.0,
        target_vz=0.0,
        tracking_duration=10.0,
        measurement_noise_std=noise,
        process_noise_std=noise,
    )
    # evaluate_trajectory accesses scenario.name
    s.name = "Test_Scenario"
    return s


def _make_env_mock(hp_values, max_episode_steps=5, cov_value=500.0):
    """Return a mock env that yields hp_values on successive steps."""
    env = MagicMock()
    env.max_episode_steps = max_episode_steps

    # Build step returns: each call gives (state, reward, terminated, truncated, info)
    step_returns = []
    for i, hp in enumerate(hp_values):
        terminated = i == len(hp_values) - 1
        step_returns.append((
            np.zeros(14),          # next_state
            1.0,                   # reward
            terminated,            # terminated
            False,                 # truncated
            {"hit_probability": hp, "covariance_trace": cov_value},
        ))
    env.step.side_effect = step_returns

    # reset returns (initial_state, {})
    initial_state = np.zeros(14)
    env.reset.return_value = (initial_state, {})

    # current_state
    state = MagicMock()
    state.current_hit_probability = 0.4
    state.covariance_trace = cov_value
    env.current_state = state

    return env


def _make_q_network(action_sequence):
    """Return a mock Q-network that returns actions from action_sequence in order."""
    import torch

    net = MagicMock()
    outputs = []
    for a in action_sequence:
        q_vals = MagicMock()
        q_vals.argmax.return_value = MagicMock()
        q_vals.argmax.return_value.item.return_value = a
        outputs.append(q_vals)

    net.return_value = MagicMock()
    # __call__ returns successive outputs
    net.side_effect = outputs
    return net


# ---------------------------------------------------------------------------
# evaluate_trajectory
# ---------------------------------------------------------------------------

class TestEvaluateTrajectory:

    def test_returns_required_keys(self):
        """evaluate_trajectory result dict has all expected keys."""
        scenario = _make_scenario()
        env = _make_env_mock(hp_values=[0.5, 0.6, 0.7])
        q_net = _make_q_network([0, 0, 0])  # all HOLD

        import torch
        with patch("torch.no_grad", return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: False)):
            result = evaluate_trajectory(env, q_net, scenario, device="cpu", max_steps=10)

        required = {"hp_trace", "actions", "rewards", "ranges", "target_positions",
                    "cov_traces", "fired_at_step", "final_hp", "scenario_name",
                    "initial_range", "steps"}
        assert required.issubset(result.keys())

    def test_fire_action_recorded(self):
        """fired_at_step and firing_range are set on the first FIRE action (action==1)."""
        scenario = _make_scenario()
        env = _make_env_mock(hp_values=[0.5, 0.8, 0.9])
        q_net = _make_q_network([0, 1, 0])  # FIRE at step 1

        import torch
        with patch("torch.no_grad", return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: False)):
            result = evaluate_trajectory(env, q_net, scenario, device="cpu", max_steps=10)

        assert result["fired_at_step"] == 1
        assert result["firing_range"] is not None
        assert result["firing_pos"] is not None

    def test_no_fire_action(self):
        """fired_at_step and firing_range are None when agent never fires."""
        scenario = _make_scenario()
        env = _make_env_mock(hp_values=[0.3, 0.4, 0.5])
        q_net = _make_q_network([0, 0, 0])  # all HOLD

        import torch
        with patch("torch.no_grad", return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: False)):
            result = evaluate_trajectory(env, q_net, scenario, device="cpu", max_steps=10)

        assert result["fired_at_step"] is None
        assert result["firing_range"] is None
        assert result["firing_pos"] is None

    def test_final_hp_matches_last_step(self):
        """final_hp equals the last hit_probability returned by env.step."""
        scenario = _make_scenario()
        hp_values = [0.3, 0.55, 0.72]
        env = _make_env_mock(hp_values=hp_values)
        q_net = _make_q_network([0, 0, 0])

        import torch
        with patch("torch.no_grad", return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: False)):
            result = evaluate_trajectory(env, q_net, scenario, device="cpu", max_steps=10)

        assert abs(result["final_hp"] - hp_values[-1]) < 1e-9

    def test_ranges_computed_from_velocity(self):
        """ranges are computed using 3D slant range from artillery origin."""
        vx = -20.0  # approaching
        scenario = _make_scenario(range_m=2000.0, vx=vx)
        env = _make_env_mock(hp_values=[0.5, 0.6])
        env.max_episode_steps = 10
        q_net = _make_q_network([0, 0])

        import torch
        with patch("torch.no_grad", return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: False)):
            result = evaluate_trajectory(env, q_net, scenario, device="cpu", max_steps=10)

        # Initial 3D slant range is sqrt(2000^2 + 50^2)
        expected_slant = np.sqrt(2000.0**2 + 50.0**2)
        assert abs(result["ranges"][0] - expected_slant) < 1e-6


# ---------------------------------------------------------------------------
# load_dqn_model — error path
# ---------------------------------------------------------------------------

class TestLoadDqnModel:

    def test_raises_file_not_found(self):
        """load_dqn_model raises FileNotFoundError for a non-existent path."""
        import torch
        device = torch.device("cpu")
        missing = Path("/nonexistent/path/model.pth")
        with pytest.raises(FileNotFoundError):
            load_dqn_model(missing, state_dim=14, action_dim=2, device=device)


# ---------------------------------------------------------------------------
# print_trajectory_summary
# ---------------------------------------------------------------------------

class TestPrintTrajectorySummary:

    def _make_results(self):
        return [
            {
                "scenario_name": "Scenario_A",
                "initial_range": 1500.0,
                "steps": 10,
                "fired_at_step": 5,
                "final_hp": 0.72,
                "ranges": [1500.0 - i * 10 for i in range(12)],
                "target_positions": [(1500.0 - i * 10, 0.0, 50.0) for i in range(12)],
            },
            {
                "scenario_name": "Scenario_B",
                "initial_range": 3000.0,
                "steps": 15,
                "fired_at_step": None,
                "final_hp": 0.31,
                "ranges": [3000.0] * 16,
                "target_positions": [(3000.0, 0.0, 50.0)] * 16,
            },
        ]

    def test_output_contains_scenario_names(self):
        """print_trajectory_summary prints each scenario name."""
        results = self._make_results()
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            print_trajectory_summary(results)
        output = captured.getvalue()
        assert "Scenario_A" in output
        assert "Scenario_B" in output

    def test_output_contains_statistics(self):
        """print_trajectory_summary prints total count and average HP."""
        results = self._make_results()
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            print_trajectory_summary(results)
        output = captured.getvalue()
        assert "2" in output          # total trajectories
        assert "1" in output          # trajectories with firing


def _full_result(name="S", n=5, fired=2):
    actions = [0] * (n - 1) + [1] if fired is not None else [0] * n
    return {
        "scenario_name": name,
        "initial_range": 1500.0,
        "steps": n,
        "fired_at_step": fired,
        "final_hp": 0.7,
        "hp_trace": [0.1 * i for i in range(n + 1)],
        "ranges": [1500.0 - 10 * i for i in range(n + 1)],
        "target_positions": [(1500.0 - 10 * i, 0.0, 50.0) for i in range(n + 1)],
        "cov_traces": [500.0 - 10 * i for i in range(n + 1)],
        "actions": actions,
        "rewards": [1.0] * n,
    }


class TestLoadDqnModelSuccess:

    def test_loads_saved_state_dict(self, tmp_path):
        import torch
        net = DQN(14, 2)
        p = tmp_path / "model.pth"
        torch.save(net.state_dict(), p)
        loaded = load_dqn_model(p, state_dim=14, action_dim=2, device=torch.device("cpu"))
        assert isinstance(loaded, DQN)
        assert loaded.training is False  # eval() was called


class TestPlots:

    def test_plot_trajectory_results_saves(self, tmp_path):
        results = [_full_result("A"), _full_result("B", fired=None)]
        out = tmp_path / "traj.png"
        plot_trajectory_results(results, save_path=out)
        assert out.exists()

    def test_plot_individual_scenarios_saves(self, tmp_path):
        plot_individual_scenarios([_full_result("Scn A")], plots_dir=str(tmp_path))
        assert (tmp_path / "Scn_A.png").exists()


def _canned(name="S", rng=1500.0):
    n = 6
    return {
        "scenario_name": name,
        "initial_range": rng,
        "steps": n,
        "fired_at_step": 2,
        "final_hp": 0.7,
        "ranges": [rng - 10 * i for i in range(n + 1)],
        "target_positions": [(rng - 10 * i, 0.0, 50.0) for i in range(n + 1)],
    }


class TestMain:

    def test_main_model_missing(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog"])
        with patch("evaluate_dqn_trajectories.load_dqn_model", side_effect=FileNotFoundError):
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_dqn_trajectories.main()
        assert result is None

    def test_main_happy(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["prog", "--num_runs", "1"])
        # main() writes under Path(__file__).parent.parent; point it at tmp_path so the
        # test never writes to the repository's results/ folder.
        monkeypatch.setattr(evaluate_dqn_trajectories, "__file__", str(tmp_path / "scripts" / "x.py"))
        with patch("evaluate_dqn_trajectories.load_dqn_model", return_value=MagicMock()), \
             patch("evaluate_dqn_trajectories.evaluate_trajectory",
                   side_effect=lambda env, q, sc, dev, seed=None, **k: _canned(sc.name, sc.range_m)) as ev, \
             patch("evaluate_dqn_trajectories.plot_trajectory_results"), \
             patch("evaluate_dqn_trajectories.plot_individual_scenarios"), \
             patch("pandas.DataFrame.to_csv"):
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_dqn_trajectories.main()
        assert ev.call_count == 5
        assert (tmp_path / "results" / "evaluation_details.json").exists()
