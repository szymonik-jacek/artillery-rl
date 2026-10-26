"""
Tests for scripts/compare_traditional_vs_rl.py

Covers: policy list construction, the paired benchmark loop (shared
perturbations, row contents) and, on the real environment, that the
precomputed HP trajectory does not depend on the policy that ran before.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compare_traditional_vs_rl import (
    DQN_POLICY_NAME,
    THRESHOLDS,
    build_policies,
    run_benchmark,
)
from rl_training.agents.dqn_components import DQN
from rl_training.curriculum.scenario_generator import ScenarioParameters
from rl_training.utils.evaluation_protocol import perturb_scenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scenario(name, range_m=2000.0, vx=-100.0):
    s = ScenarioParameters(
        range_m=range_m,
        target_length=5.0,
        target_width=2.0,
        target_height=2.0,
        target_vx=vx,
        target_vy=0.0,
        target_vz=0.0,
        tracking_duration=10.0,
        measurement_noise_std=0.5,
        process_noise_std=0.5,
    )
    s.name = name
    return s


class _FakeEnv:
    """HP rises 0.1 per step from 0.2; FIRE terminates, HOLD advances."""

    max_episode_steps = 6

    def __init__(self):
        self.reset_calls = []

    def reset(self, scenario_override=None, seed=None):
        self.reset_calls.append((scenario_override, seed))
        self.step_idx = 0
        self.current_state = SimpleNamespace(current_hit_probability=0.2, covariance_trace=100.0)
        return np.zeros(14), {}

    def step(self, action):
        if action != 1:
            self.step_idx += 1
        hp = 0.2 + 0.1 * self.step_idx
        self.current_state.current_hit_probability = hp
        terminated = action == 1 or self.step_idx >= self.max_episode_steps
        return np.zeros(14), 0.0, terminated, False, {"hit_probability": hp}


# ---------------------------------------------------------------------------
# build_policies
# ---------------------------------------------------------------------------

class TestBuildPolicies:
    def test_thresholds_only(self):
        names = [n for n, _ in build_policies()]
        assert names == [f"Threshold_{t:.2f}" for t in THRESHOLDS]

    def test_dqn_comes_first(self):
        names = [n for n, _ in build_policies(DQN(14, 2), torch.device("cpu"))]
        assert names[0] == DQN_POLICY_NAME
        assert len(names) == 1 + len(THRESHOLDS)

    def test_threshold_grid_spans_040_to_090(self):
        assert THRESHOLDS[0] == 0.40 and THRESHOLDS[-1] == 0.90 and len(THRESHOLDS) == 11
        assert np.allclose(np.diff(THRESHOLDS), 0.05)

    def test_oracle_follows_dqn_when_env_given(self):
        names = [n for n, _ in build_policies(DQN(14, 2), torch.device("cpu"),
                                              thresholds=(0.5,), oracle_env=_FakeEnv())]
        assert names == [DQN_POLICY_NAME, "Oracle", "Threshold_0.50"]

    def test_extra_dqns_follow_dqn(self):
        names = [n for n, _ in build_policies(DQN(14, 2), torch.device("cpu"), thresholds=(0.5,),
                                              extra_dqns=[("V0r_s0", DQN(14, 2), None),
                                                          ("V0r_s1", DQN(14, 2), "zero")],
                                              oracle_env=_FakeEnv())]
        assert names == [DQN_POLICY_NAME, "V0r_s0", "V0r_s1", "Oracle", "Threshold_0.50"]

    def test_no_oracle_by_default(self):
        assert "Oracle" not in [n for n, _ in build_policies()]

    def test_decaying_rules_after_fixed_thresholds(self):
        names = [n for n, _ in build_policies(thresholds=(0.5,), tau_cov_values=(1.0,),
                                              decaying_grid=((0.9, 0.45), (0.8, 0.4)))]
        assert names == ["Threshold_0.50", "Decaying_0.90_0.45", "Decaying_0.80_0.40",
                         "CovGated_0.50_1"]

    def test_default_decaying_grid(self):
        from compare_traditional_vs_rl import DECAYING_GRID
        assert len(DECAYING_GRID) == 20
        assert all(p_end < p_start for p_start, p_end in DECAYING_GRID)
        # The previous best rule (0.90 -> 0.40) is now inside the grid, not at its edge.
        starts = {p for p, _ in DECAYING_GRID}
        ends = {e for _, e in DECAYING_GRID}
        assert max(starts) > 0.90 and min(ends) < 0.40

    def test_default_plateau_grid(self):
        from compare_traditional_vs_rl import PLATEAU_GRID
        assert len(PLATEAU_GRID) == 18 and len(set(PLATEAU_GRID)) == 18

    def test_plateau_rules_follow_fixed_thresholds(self):
        names = [n for n, _ in build_policies(thresholds=(0.5,), plateau_grid=((0.45, 3, 0.01),),
                                              decaying_grid=((0.9, 0.4),))]
        assert names == ["Threshold_0.50", "Plateau_0.45_k3_e0.01", "Decaying_0.90_0.40"]

    def test_gated_rule_for_every_threshold_and_tau(self):
        names = [n for n, _ in build_policies(thresholds=(0.5, 0.6), tau_cov_values=(100.0, 200.0))]
        assert names == [
            "Threshold_0.50", "Threshold_0.60",
            "CovGated_0.50_100", "CovGated_0.50_200",
            "CovGated_0.60_100", "CovGated_0.60_200",
        ]


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def _run(self, seeds=(0, 1)):
        env = _FakeEnv()
        scenarios = [_make_scenario("A"), _make_scenario("B", range_m=3000.0)]
        policies = build_policies(thresholds=(0.4, 0.9))
        rows = run_benchmark(env, scenarios, policies, seeds=seeds)
        return env, scenarios, rows

    def test_progress_called_once_per_seed_and_scenario(self):
        calls = []
        scenarios = [_make_scenario("A"), _make_scenario("B", range_m=3000.0)]
        run_benchmark(_FakeEnv(), scenarios, build_policies(thresholds=(0.4, 0.9)),
                      seeds=range(3), progress=lambda done, total: calls.append((done, total)))
        assert calls == [(i, 6) for i in range(1, 7)]

    def test_one_row_per_seed_scenario_policy(self):
        _, _, rows = self._run()
        assert len(rows) == 2 * 2 * 2
        keys = {(r["seed"], r["scenario"], r["policy"]) for r in rows}
        assert len(keys) == 8

    def test_all_policies_see_identical_perturbed_scenario(self):
        env, _, _ = self._run()
        # Calls come in pairs: two policies per (seed, scenario).
        for i in range(0, len(env.reset_calls), 2):
            (s1, seed1), (s2, seed2) = env.reset_calls[i], env.reset_calls[i + 1]
            assert s1 is s2 and seed1 == seed2

    def test_perturbations_match_shared_protocol(self):
        env, scenarios, _ = self._run(seeds=(5,))
        rng = np.random.default_rng(5)
        expected = [perturb_scenario(s, rng) for s in scenarios]
        used = [env.reset_calls[0][0], env.reset_calls[2][0]]
        for e, u in zip(expected, used):
            assert u.range_m == e.range_m
            assert u.target_vx == e.target_vx
            assert u.measurement_noise_std == e.measurement_noise_std

    def test_row_contents(self):
        _, _, rows = self._run(seeds=(0,))
        fired = next(r for r in rows if r["policy"] == "Threshold_0.40" and r["scenario"] == "A")
        assert fired["fired"] and not fired["timed_out"]
        assert fired["fired_at_step"] == 2
        assert fired["hit_probability"] == pytest.approx(0.4)
        assert fired["firing_x"] == pytest.approx(fired["firing_slant_range"], rel=0.01)
        assert fired["blocked_fire_attempts"] == 0

        timeout = next(r for r in rows if r["policy"] == "Threshold_0.90" and r["scenario"] == "A")
        assert timeout["timed_out"] and timeout["hit_probability"] == 0.0
        assert timeout["firing_x"] is None and timeout["firing_slant_range"] is None


# ---------------------------------------------------------------------------
# Integration: HP trajectory is independent of the policy (real env)
# ---------------------------------------------------------------------------

def test_hp_trajectory_independent_of_policy():
    from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
    from rl_training.environment import ArtilleryFiringEnv
    from rl_training.utils.evaluation_protocol import get_evaluation_scenarios

    scenario = perturb_scenario(
        get_evaluation_scenarios(CurriculumScenarios(phase=4))[0], np.random.default_rng(0)
    )

    # Fresh env: trajectory computed from scratch.
    fresh = ArtilleryFiringEnv()
    fresh.reset(scenario_override=scenario, seed=0)
    reference_hp = dict(fresh.hp_trajectory)
    reference_cov = dict(fresh.cov_trajectory)

    # Second env: run two different policies on another seed first, then the same one.
    env = ArtilleryFiringEnv()
    policies = build_policies(thresholds=(0.3, 0.9))
    run_benchmark(env, [get_evaluation_scenarios(CurriculumScenarios(phase=4))[0]],
                  policies, seeds=[1])
    rows = run_benchmark(env, [get_evaluation_scenarios(CurriculumScenarios(phase=4))[0]],
                         policies, seeds=[0])

    assert env.hp_trajectory == reference_hp
    assert env.cov_trajectory == reference_cov
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Export: labels, LaTeX table, files
# ---------------------------------------------------------------------------

import json
import shutil
import subprocess

from compare_traditional_vs_rl import _format_p, export_results, policy_label


def _episode(policy, scenario, seed, hp=None, step=None):
    fired = hp is not None
    return {
        "policy": policy, "scenario": scenario, "seed": seed,
        "fired": fired, "timed_out": not fired,
        "fired_at_step": step if fired else None,
        "hit_probability": hp if fired else 0.0,
        "firing_slant_range": 1000.0 - 10 * seed if fired else None,
        "firing_x": 990.0 - 10 * seed if fired else None,
        "steps": step + 1 if fired else 30,
        "blocked_fire_attempts": 0,
    }


def _benchmark_rows(n=6):
    """Pooled mean HP: DQN 0.66, 0.45 -> 0.50, 0.50 -> 0.555 (best), 0.55 -> 0.40, 0.65 -> 0.35."""
    rows = []
    for seed in range(n):
        e = 0.001 * seed
        rows += [
            _episode("DQN", "Static_Close", seed, 0.70 + e, 12),
            _episode("DQN", "Fast_Approach", seed, 0.62 + e, 9),
            _episode("Threshold_0.45", "Static_Close", seed, 0.50, 3),
            _episode("Threshold_0.45", "Fast_Approach", seed, 0.50, 2),
            _episode("Threshold_0.50", "Static_Close", seed, 0.56, 5),
            _episode("Threshold_0.50", "Fast_Approach", seed, 0.55, 4),
            _episode("Threshold_0.55", "Static_Close", seed, 0.80, 8),
            _episode("Threshold_0.55", "Fast_Approach", seed),            # timeout
            _episode("Threshold_0.60", "Static_Close", seed, 0.70, 7),
            _episode("Threshold_0.60", "Fast_Approach", seed),            # timeout
            _episode("Threshold_0.65", "Static_Close", seed, 0.70, 10),
            _episode("Threshold_0.65", "Fast_Approach", seed),            # timeout
        ]
    return rows


class TestPolicyLabel:
    def test_threshold(self):
        assert policy_label("Threshold_0.45") == r"$P_{\mathrm{thr}}=0.45$"

    def test_covariance_gated(self):
        assert policy_label("CovGated_0.50_250") == r"$P_{\mathrm{thr}}=0.50$, $\tau_{\mathrm{cov}}=250$"

    def test_plain_name_escaped(self):
        assert policy_label("My_Policy") == r"My\_Policy"

    def test_best_gets_dagger(self):
        assert policy_label("Threshold_0.50", best=["Threshold_0.50"]).endswith(r"$^{\dagger}$")

    def test_oracle(self):
        assert policy_label("Oracle") == r"Hindsight oracle$^{\ddagger}$"

    def test_decaying(self):
        assert policy_label("Decaying_0.90_0.45") == r"$P_{\mathrm{thr}}: 0.90 \to 0.45$"


class TestFormatP:
    @pytest.mark.parametrize("p, reference, expected", [
        (0.01, True, "--"),
        (float("nan"), False, "n/a"),
        (None, False, "n/a"),
        (0.0004, False, "$<0.001$"),
        (0.01234, False, "0.012"),
    ])
    def test_formatting(self, p, reference, expected):
        assert _format_p(p, reference) == expected


class TestExportResults:
    @pytest.fixture
    def exported(self, tmp_path):
        paths = export_results(_benchmark_rows(), tmp_path / "results", tmp_path / "tab.tex",
                               n_boot=200, metadata={"checkpoint": "x.pth"})
        return paths

    def test_writes_all_files(self, exported):
        assert set(exported) == {"raw_csv", "summary_csv", "pooled_csv", "paired_csv", "json", "table"}
        for path in exported.values():
            assert path.exists() and path.stat().st_size > 0

    def test_json_contents(self, exported):
        data = json.loads(exported["json"].read_text(encoding="utf-8"))
        assert data["best_fixed"]["threshold"]["policy"] == "Threshold_0.50"
        assert data["best_fixed"]["threshold"]["pooled_mean_hp"] == pytest.approx(0.555)
        assert "covariance_gated" not in data["best_fixed"]
        meta = data["metadata"]
        assert meta["n_seeds"] == 6 and meta["checkpoint"] == "x.pth"
        assert meta["scenarios"] == ["Static_Close", "Fast_Approach"]
        assert "oracle" in meta["best_fixed_selected_on"]
        # NaN (e.g. mean step of an all-timeout group) is written as null.
        timeout_group = next(r for r in data["summary"]
                             if r["policy"] == "Threshold_0.55" and r["scenario"] == "Fast_Approach")
        assert timeout_group["mean_fired_step"] is None
        assert len(data["paired_tests"]) == 5 * 3

    def test_table_policies_and_blocks(self, exported):
        tex = exported["table"].read_text(encoding="utf-8")
        assert tex.count(r"\multirow{5}{*}") == 3          # 2 scenarios + pooled block
        assert r"Static\_Close" in tex and "All scenarios" in tex
        assert r"$P_{\mathrm{thr}}=0.60$" not in tex         # not in table, not best
        assert r"$P_{\mathrm{thr}}=0.50$$^{\dagger}$" in tex  # best fixed threshold

    def test_table_rows(self, exported):
        lines = exported["table"].read_text(encoding="utf-8").splitlines()
        dqn = next(l for l in lines if "Static" in l and "& DQN &" in l)
        assert dqn.rstrip().endswith(r"-- \\")
        timeout = [l for l in lines if r"$P_{\mathrm{thr}}=0.55$ &" in l]
        fast = timeout[1]                                   # Fast_Approach block
        assert r"0.000 $\pm$ 0.000 & 0 & -- &" in fast

    @pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
    def test_oracle_row_is_last_in_each_block(self, tmp_path):
        rows = _benchmark_rows()
        rows += [dict(r, policy="Oracle", hit_probability=r["hit_probability"] + 0.1)
                 for r in rows if r["policy"] == "DQN"]
        paths = export_results(rows, tmp_path / "r", tmp_path / "t.tex", n_boot=200)
        lines = paths["table"].read_text(encoding="utf-8").splitlines()
        oracle_idx = [i for i, l in enumerate(lines) if "Hindsight oracle" in l]
        assert len(oracle_idx) == 3
        for i in oracle_idx:
            assert lines[i + 1].startswith((r"\midrule", r"\bottomrule"))
        assert r"$^{\ddagger}$Upper bound" in "\n".join(lines)
        # The oracle never becomes the best fixed rule.
        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert data["best_fixed"]["threshold"]["policy"] == "Threshold_0.50"

    def test_best_decaying_rule_selected_and_shown(self, tmp_path):
        rows = _benchmark_rows()
        # Decaying_0.90_0.45 beats DQN pooled; Decaying_0.80_0.40 is worse.
        rows += [dict(r, policy="Decaying_0.90_0.45", hit_probability=r["hit_probability"] + 0.02)
                 for r in rows if r["policy"] == "DQN"]
        rows += [dict(r, policy="Decaying_0.80_0.40", hit_probability=r["hit_probability"] - 0.1)
                 for r in rows if r["policy"] == "DQN"]
        paths = export_results(rows, tmp_path / "r", tmp_path / "t.tex", n_boot=200)
        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert data["best_fixed"]["decaying"]["policy"] == "Decaying_0.90_0.45"
        assert data["best_fixed"]["threshold"]["policy"] == "Threshold_0.50"
        tex = paths["table"].read_text(encoding="utf-8")
        assert r"$P_{\mathrm{thr}}: 0.90 \to 0.45$$^{\dagger}$" in tex
        assert r"0.80 \to 0.40" not in tex
        assert tex.count(r"\multirow{6}{*}") == 3

    @pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
    def test_table_compiles(self, exported, tmp_path):
        doc = tmp_path / "doc.tex"
        doc.write_text(
            "\\documentclass{article}\n\\usepackage{booktabs,multirow}\n"
            "\\begin{document}\n\\input{tab.tex}\n\\end{document}\n", encoding="utf-8")
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
                                cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout[-2000:]


# ---------------------------------------------------------------------------
# parse_extra_dqn
# ---------------------------------------------------------------------------

from pathlib import Path

from compare_traditional_vs_rl import parse_extra_dqn


class TestParseExtraDqn:
    def test_plain_checkpoint(self):
        assert parse_extra_dqn("V0r=ckpt/a_phase4.pth") == ("V0r", Path("ckpt/a_phase4.pth"), None)

    def test_windows_path(self):
        assert parse_extra_dqn("A=C:/m/x_phase4.pth") == ("A", Path("C:/m/x_phase4.pth"), None)

    def test_covariance_ablation_suffixes(self):
        assert parse_extra_dqn("A=C:/m/x.pth:cov0") == ("A", Path("C:/m/x.pth"), "zero")
        assert parse_extra_dqn("A=x.pth:covfrozen") == ("A", Path("x.pth"), "frozen")

    @pytest.mark.parametrize("spec", ["nopath", "=x.pth", "V1=", "DQN=x.pth", "Oracle=x.pth",
                                      "A=:cov0", "A=x.pth:cov0:covfrozen"])
    def test_rejects_invalid(self, spec):
        with pytest.raises(ValueError):
            parse_extra_dqn(spec)


# ---------------------------------------------------------------------------
# Firing time: rows, re-export from raw CSV, table and JSON
# ---------------------------------------------------------------------------

from compare_traditional_vs_rl import add_fire_times, load_raw_rows


class TestFireTimeExport:
    def test_run_benchmark_rows_have_fire_time(self):
        env = _FakeEnv()  # 6 steps over 10 s
        rows = run_benchmark(env, [_make_scenario("A")], build_policies(thresholds=(0.4, 0.9)), seeds=(0,))
        fired = next(r for r in rows if r["policy"] == "Threshold_0.40")
        assert fired["fire_time_s"] == pytest.approx(fired["fired_at_step"] * 10.0 / 6)
        assert next(r for r in rows if r["policy"] == "Threshold_0.90")["fire_time_s"] is None

    def test_add_fire_times_uses_each_perturbed_episode(self):
        from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
        from rl_training.environment import ArtilleryFiringEnv
        from rl_training.utils.evaluation_protocol import get_evaluation_scenarios, step_seconds

        env = ArtilleryFiringEnv()
        scenarios = get_evaluation_scenarios(CurriculumScenarios(phase=4))
        rows = [dict(_episode("DQN", "ExtremeLong_Large_Lateral", seed, 0.5, 10)) for seed in range(30)]
        rows.append(_episode("DQN", "ExtremeLong_Large_Lateral", 30))         # timeout
        rows.append(dict(_episode("DQN", "Static_Close", 0, 0.7, 4), fire_time_s=99.0))  # kept
        out = add_fire_times(rows, env, scenarios)

        for seed in range(30):
            rng = np.random.default_rng(seed)
            perturbed = [perturb_scenario(s, rng) for s in scenarios][4]
            assert out[seed]["fire_time_s"] == pytest.approx(10 * step_seconds(env, perturbed))
        # The 3000 m boundary gives both 90- and 100-step episodes across seeds.
        assert len({round(out[s]["fire_time_s"], 6) for s in range(30)}) == 2
        assert out[30]["fire_time_s"] is None
        assert out[31]["fire_time_s"] == 99.0
        assert "fire_time_s" not in rows[0]  # input not modified

    def test_load_raw_rows_round_trip(self, tmp_path):
        import pandas as pd
        rows = _benchmark_rows(n=2)
        path = tmp_path / "raw.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        loaded = load_raw_rows(path)
        assert len(loaded) == len(rows)
        timeout = next(r for r in loaded if r["policy"] == "Threshold_0.55" and r["scenario"] == "Fast_Approach")
        assert timeout["fired"] is False and timeout["fired_at_step"] is None
        assert isinstance(loaded[0]["seed"], int)

    def test_table_and_json_report_time_and_front(self, tmp_path):
        rows = [dict(r, fire_time_s=(r["fired_at_step"] * 0.5 if r["fired"] else None))
                for r in _benchmark_rows()]
        paths = export_results(rows, tmp_path / "r", tmp_path / "t.tex", n_boot=200)
        tex = paths["table"].read_text(encoding="utf-8")
        assert r"\textbf{Time to fire (s)}" in tex and "Mean step" not in tex
        dqn = next(l for l in tex.splitlines() if "Static" in l and "& DQN &" in l)
        assert "& 6.0 &" in dqn  # step 12 x 0.5 s
        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert data["pareto_front_hp_vs_time"][-1] == "DQN"


# ---------------------------------------------------------------------------
# Selecting rules on a separate (tuning) set
# ---------------------------------------------------------------------------

from compare_traditional_vs_rl import scenarios_for


class TestScenariosFor:
    def test_sets(self):
        from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
        c = CurriculumScenarios(phase=4)
        assert len(scenarios_for("evaluation", c)) == 5
        assert len(scenarios_for("tuning", c)) == 10
        with pytest.raises(ValueError):
            scenarios_for("other", c)


class TestSelectionRows:
    def _selection_rows(self):
        """Tuning set where Threshold_0.45 is best (unlike the test set, where 0.50 wins)."""
        rows = []
        for seed in range(3):
            rows += [_episode("Threshold_0.45", "Tune_A", seed, 0.70, 3),
                     _episode("Threshold_0.50", "Tune_A", seed, 0.60, 4),
                     _episode("Decaying_0.90_0.40", "Tune_A", seed, 0.65, 5)]
        return rows

    def _test_rows(self):
        rows = _benchmark_rows()
        rows += [dict(r, policy="Decaying_0.90_0.40", hit_probability=r["hit_probability"] + 0.02)
                 for r in rows if r["policy"] == "DQN"]
        return rows

    def test_rule_chosen_on_selection_rows_reported_on_test_rows(self, tmp_path):
        paths = export_results(self._test_rows(), tmp_path / "r", tmp_path / "t.tex", n_boot=200,
                               selection_rows=self._selection_rows(), selected_on="tuning scenarios")
        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        best = data["best_fixed"]["threshold"]
        assert best["policy"] == "Threshold_0.45"
        assert best["pooled_mean_hp"] == pytest.approx(0.50)          # its test-set HP
        assert best["selection_pooled_mean_hp"] == pytest.approx(0.70)
        assert data["metadata"]["best_fixed_selected_on"] == "tuning scenarios"
        tex = paths["table"].read_text(encoding="utf-8")
        assert r"$P_{\mathrm{thr}}=0.45$$^{\dagger}$" in tex
        assert "selected on the tuning scenarios" in tex

    def test_without_selection_rows_behaviour_unchanged(self, tmp_path):
        paths = export_results(self._test_rows(), tmp_path / "r", tmp_path / "t.tex", n_boot=200)
        best = json.loads(paths["json"].read_text(encoding="utf-8"))["best_fixed"]["threshold"]
        assert best["policy"] == "Threshold_0.50" and "selection_pooled_mean_hp" not in best

    def test_rule_missing_from_test_rows_raises(self, tmp_path):
        selection = self._selection_rows() + [_episode("Threshold_0.85", "Tune_A", 0, 0.99, 1)]
        with pytest.raises(ValueError, match="not evaluated"):
            export_results(self._test_rows(), tmp_path / "r", tmp_path / "t.tex", n_boot=200,
                           selection_rows=selection)


# ---------------------------------------------------------------------------
# Plateau rule family in the export
# ---------------------------------------------------------------------------

class TestPlateauExport:
    def test_label(self):
        assert policy_label("Plateau_0.45_k3_e0.01") == \
            r"Plateau $P_{\mathrm{hit}}\geq0.45$, $k=3$, $\varepsilon=0.01$"

    def test_best_plateau_rule_selected_and_tabled(self, tmp_path):
        rows = _benchmark_rows()
        rows += [dict(r, policy="Plateau_0.45_k3_e0.01", hit_probability=r["hit_probability"] - 0.01)
                 for r in rows if r["policy"] == "DQN"]
        rows += [dict(r, policy="Plateau_0.50_k5_e0.02", hit_probability=r["hit_probability"] - 0.2)
                 for r in rows if r["policy"] == "DQN"]
        paths = export_results(rows, tmp_path / "r", tmp_path / "t.tex", n_boot=200)
        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert data["best_fixed"]["plateau"]["policy"] == "Plateau_0.45_k3_e0.01"
        tex = paths["table"].read_text(encoding="utf-8")
        assert r"Plateau $P_{\mathrm{hit}}\geq0.45$" in tex and "0.50, $k=5$" not in tex

    @pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
    def test_table_with_plateau_compiles(self, tmp_path):
        rows = _benchmark_rows()
        rows += [dict(r, policy="Plateau_0.45_k3_e0.01") for r in rows if r["policy"] == "DQN"]
        export_results(rows, tmp_path / "r", tmp_path / "tab.tex", n_boot=200)
        (tmp_path / "doc.tex").write_text(
            "\\documentclass{article}\n\\usepackage{booktabs,multirow}\n"
            "\\begin{document}\n\\input{tab.tex}\n\\end{document}\n", encoding="utf-8")
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
                                cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout[-2000:]


class TestScenarioSetsAndShifts:
    def test_phase4_and_heldout_sets(self):
        from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
        c = CurriculumScenarios(phase=4)
        assert len(scenarios_for("phase4", c)) == 15
        assert len(scenarios_for("heldout", c)) == 30

    def test_shift_applied_to_every_scenario(self):
        from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
        c = CurriculumScenarios(phase=4)
        base = scenarios_for("evaluation", c)
        shifted = scenarios_for("evaluation", c, shift="range1.2")
        assert [s.name for s in shifted] == [s.name for s in base]
        assert all(b.range_m * 1.2 == pytest.approx(s.range_m) for b, s in zip(base, shifted))


# ---------------------------------------------------------------------------
# Per-seed checkpoints and --resume
# ---------------------------------------------------------------------------

import compare_traditional_vs_rl as ctr  # noqa: E402

_CONFIG = {"policies": ["Threshold_0.40", "Threshold_0.90"], "scenarios": ["A", "B"],
           "scenario_set": "evaluation", "shift": "none"}


def _fake_rows(seeds=(0, 1)):
    scenarios = [_make_scenario("A"), _make_scenario("B", range_m=3000.0)]
    return run_benchmark(_FakeEnv(), scenarios, build_policies(thresholds=(0.4, 0.9)), seeds=seeds)


class TestSeedCheckpoints:
    def test_on_seed_done_gets_each_seeds_rows(self):
        calls = []
        scenarios = [_make_scenario("A"), _make_scenario("B", range_m=3000.0)]
        rows = run_benchmark(_FakeEnv(), scenarios, build_policies(thresholds=(0.4, 0.9)), seeds=(3, 7),
                             on_seed_done=lambda seed, seed_rows: calls.append((seed, list(seed_rows))))
        assert [seed for seed, _ in calls] == [3, 7]
        assert all(len(r) == 4 and {x["seed"] for x in r} == {seed} for seed, r in calls)
        assert calls[0][1] + calls[1][1] == rows

    def test_save_and_load_round_trip(self, tmp_path):
        rows = _fake_rows(seeds=(0, 1))
        for seed in (0, 1):
            ctr.save_seed_rows(tmp_path, seed, [r for r in rows if r["seed"] == seed])
        assert not list(tmp_path.glob("*.tmp"))
        done = ctr.load_completed_seeds(tmp_path, [0, 1, 2], n_rows_per_seed=4)
        assert sorted(done) == [0, 1]
        for seed in (0, 1):
            orig = [r for r in rows if r["seed"] == seed]
            assert [(r["policy"], r["scenario"]) for r in done[seed]] == [(r["policy"], r["scenario"]) for r in orig]
            assert [r["hit_probability"] for r in done[seed]] == pytest.approx([r["hit_probability"] for r in orig])
            assert [r["fired"] for r in done[seed]] == [r["fired"] for r in orig]

    def test_only_requested_seeds_are_loaded(self, tmp_path):
        rows = _fake_rows(seeds=(0, 1))
        for seed in (0, 1):
            ctr.save_seed_rows(tmp_path, seed, [r for r in rows if r["seed"] == seed])
        assert sorted(ctr.load_completed_seeds(tmp_path, [1], n_rows_per_seed=4)) == [1]

    def test_incomplete_seed_file_is_an_error(self, tmp_path):
        rows = [r for r in _fake_rows(seeds=(0,))][:3]
        ctr.save_seed_rows(tmp_path, 0, rows)
        with pytest.raises(ValueError, match="not a complete checkpoint"):
            ctr.load_completed_seeds(tmp_path, [0], n_rows_per_seed=4)

    def test_seed_mismatch_is_an_error(self, tmp_path):
        ctr.save_seed_rows(tmp_path, 1, _fake_rows(seeds=(0,)))
        with pytest.raises(ValueError, match="seed 1"):
            ctr.load_completed_seeds(tmp_path, [1], n_rows_per_seed=4)


class TestInitPartial:
    def test_fresh_run_writes_config(self, tmp_path):
        ctr.init_partial(tmp_path / "partial", _CONFIG, resume=False)
        assert (tmp_path / "partial" / ctr.PARTIAL_CONFIG_NAME).exists()

    def test_fresh_run_refuses_existing_checkpoints(self, tmp_path):
        ctr.init_partial(tmp_path, _CONFIG, resume=False)
        with pytest.raises(ValueError, match="--resume"):
            ctr.init_partial(tmp_path, _CONFIG, resume=False)

    def test_resume_with_same_config(self, tmp_path):
        ctr.init_partial(tmp_path, _CONFIG, resume=False)
        ctr.init_partial(tmp_path, dict(_CONFIG), resume=True)

    def test_resume_refuses_different_config(self, tmp_path):
        ctr.init_partial(tmp_path, _CONFIG, resume=False)
        with pytest.raises(ValueError, match="shift"):
            ctr.init_partial(tmp_path, {**_CONFIG, "shift": "noise2"}, resume=True)
        with pytest.raises(ValueError, match="policies"):
            ctr.init_partial(tmp_path, {**_CONFIG, "policies": ["Threshold_0.40"]}, resume=True)

    def test_resume_without_anything_starts_fresh(self, tmp_path):
        ctr.init_partial(tmp_path / "p", _CONFIG, resume=True)
        assert (tmp_path / "p" / ctr.PARTIAL_CONFIG_NAME).exists()

    def test_resume_refuses_seed_files_without_config(self, tmp_path):
        ctr.save_seed_rows(tmp_path, 0, _fake_rows(seeds=(0,)))
        with pytest.raises(ValueError, match="no config"):
            ctr.init_partial(tmp_path, _CONFIG, resume=True)


class _SpacedFakeEnv(_FakeEnv):
    observation_space = SimpleNamespace(shape=(14,))
    action_space = SimpleNamespace(n=2)

    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs


class _Crash(Exception):
    pass


def _patch_main(monkeypatch, crash_at_seed=None):
    """Run main() on the fake env with two threshold rules; returns the captured rows list."""
    import evaluate_dqn_trajectories
    import rl_training.environment

    monkeypatch.setattr(rl_training.environment, "ArtilleryFiringEnv", _SpacedFakeEnv)
    monkeypatch.setattr(evaluate_dqn_trajectories, "load_dqn_model", lambda *a, **k: None)
    monkeypatch.setattr(ctr, "scenarios_for",
                        lambda *a, **k: [_make_scenario("A"), _make_scenario("B", range_m=3000.0)])
    real_build = build_policies

    def fake_build(q_network, device, oracle_env=None, **kwargs):
        policies = real_build(thresholds=(0.4, 0.9))
        name, decide = policies[0]

        def crashing(observation, env_state):
            if crash_at_seed is not None and oracle_env.reset_calls[-1][1] == crash_at_seed:
                raise _Crash()
            return decide(observation, env_state)
        return [(name, crashing)] + policies[1:]

    monkeypatch.setattr(ctr, "build_policies", fake_build)
    captured = []
    monkeypatch.setattr(ctr, "export_results", lambda rows, *a, **k: captured.append(rows) or {})
    return captured


class TestMainResume:
    def _args(self, tmp_path, *extra):
        return ["--num_seeds", "4", "--output_dir", str(tmp_path), "--table", str(tmp_path / "t.tex"), *extra]

    @staticmethod
    def _key(rows):
        return [(r["seed"], r["scenario"], r["policy"], bool(r["fired"]), round(float(r["hit_probability"]), 12))
                for r in rows]

    def test_interrupted_then_resumed_equals_uninterrupted(self, tmp_path, monkeypatch):
        captured = _patch_main(monkeypatch)
        ctr.main(self._args(tmp_path / "full"))
        reference = captured[0]

        _patch_main(monkeypatch, crash_at_seed=2)
        with pytest.raises(_Crash):
            ctr.main(self._args(tmp_path / "run"))
        partial = tmp_path / "run" / ctr.PARTIAL_DIR_NAME
        assert sorted(p.name for p in partial.glob("seed_*.csv")) == ["seed_0000.csv", "seed_0001.csv"]

        captured = _patch_main(monkeypatch)
        ctr.main(self._args(tmp_path / "run", "--resume"))
        assert len(captured[0]) == len(reference) == 4 * 2 * 2
        assert self._key(captured[0]) == self._key(reference)

    def test_rerun_without_resume_is_refused(self, tmp_path, monkeypatch):
        _patch_main(monkeypatch)
        ctr.main(self._args(tmp_path))
        _patch_main(monkeypatch)
        with pytest.raises(SystemExit):
            ctr.main(self._args(tmp_path))

    def test_resume_with_different_shift_is_refused(self, tmp_path, monkeypatch):
        _patch_main(monkeypatch, crash_at_seed=1)
        with pytest.raises(_Crash):
            ctr.main(self._args(tmp_path))
        _patch_main(monkeypatch)
        with pytest.raises(SystemExit):
            ctr.main(self._args(tmp_path, "--resume", "--shift", "noise2"))
