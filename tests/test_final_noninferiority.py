"""
Tests for scripts/final_noninferiority.py: per-run analysis on synthetic
episodes, the report table and the command line (missing runs skipped).
"""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import final_noninferiority as fni

AGENTS = ("Ag_s0", "Ag_s1")
RULES = ("RuleA",)


def _rows(hp_by_policy, n_scen=2, n_seeds=10, fire_time=5.0):
    rows = []
    for policy, hp in hp_by_policy.items():
        for s in range(n_scen):
            for seed in range(n_seeds):
                h = hp(s, seed) if callable(hp) else hp
                fired = h is not None
                rows.append({"policy": policy, "scenario": f"S{s}", "seed": seed,
                             "hit_probability": h if fired else 0.0, "fired": fired,
                             "timed_out": not fired, "fired_at_step": 10 if fired else None,
                             "firing_slant_range": 1000.0 if fired else None,
                             "firing_x": 1.0 if fired else None,
                             "fire_time_s": fire_time if fired else None,
                             "blocked_fire_attempts": 0})
    return rows


class TestAnalyseRun:
    def test_equal_policies_are_non_inferior(self):
        rows = _rows({"Ag_s0": 0.6, "Ag_s1": 0.6, "RuleA": 0.6, "DQN": 0.5})
        res = fni.analyse_run(rows, AGENTS, RULES, (0.02,), n_boot=200)
        t = res["tests"][0]
        assert t["rule"] == "RuleA" and t["delta"] == 0.02
        assert t["mean_diff"] == pytest.approx(0.0) and t["non_inferior"]
        assert res["summary"]["agent"]["mean_hp"] == pytest.approx(0.6)
        assert res["summary"]["DQN"]["mean_hp"] == pytest.approx(0.5)
        assert "Oracle" not in res["summary"]

    def test_worse_agent_is_not_non_inferior(self):
        rows = _rows({"Ag_s0": 0.55, "Ag_s1": 0.55, "RuleA": 0.6})
        t = fni.analyse_run(rows, AGENTS, RULES, (0.02,), n_boot=200)["tests"][0]
        assert t["mean_diff"] == pytest.approx(-0.05) and not t["non_inferior"]

    def test_one_test_per_rule_and_delta(self):
        rows = _rows({"Ag_s0": 0.6, "Ag_s1": 0.6, "RuleA": 0.6, "RuleB": 0.6})
        res = fni.analyse_run(rows, AGENTS, ("RuleA", "RuleB"), (0.02, 0.01), n_boot=50)
        assert [(t["rule"], t["delta"]) for t in res["tests"]] == [
            ("RuleA", 0.02), ("RuleA", 0.01), ("RuleB", 0.02), ("RuleB", 0.01)]

    def test_agent_summary_over_seeds(self):
        # Seed 1 times out on one scenario: HP 0 there, firing rate 50%.
        rows = _rows({"Ag_s0": 0.6, "Ag_s1": lambda s, seed: 0.6 if s == 0 else None, "RuleA": 0.6})
        agent = fni.analyse_run(rows, AGENTS, RULES, (0.02,), n_boot=50)["summary"]["agent"]
        assert agent["mean_hp"] == pytest.approx((0.6 + 0.3) / 2)
        assert agent["min_firing_rate_pct"] == pytest.approx(50.0)
        assert agent["mean_fire_time_s"] == pytest.approx(5.0)

    def test_missing_policy_raises(self):
        rows = _rows({"Ag_s0": 0.6, "RuleA": 0.6})
        with pytest.raises(ValueError, match="Ag_s1"):
            fni.analyse_run(rows, AGENTS, RULES, (0.02,), n_boot=10)


class TestReport:
    def test_format_report(self):
        rows = _rows({"Ag_s0": 0.6, "Ag_s1": 0.6, "RuleA": 0.6})
        res = {"run1": fni.analyse_run(rows, AGENTS, RULES, (0.02,), n_boot=50)}
        lines = fni.format_report(res).splitlines()
        assert len(lines) == 3
        assert lines[2].startswith("| run1 | 0.600 | RuleA | 0.600 |") and lines[2].endswith("| yes |")


class TestMain:
    def test_main_skips_missing_runs(self, tmp_path, monkeypatch, capsys):
        run_dir = tmp_path / "heldout"
        run_dir.mkdir()
        pd.DataFrame(_rows({"Ag_s0": 0.6, "Ag_s1": 0.6, "RuleA": 0.59})).to_csv(
            run_dir / fni.RAW_NAME, index=False)
        monkeypatch.setattr(fni, "N_TRAINING_SEEDS", 2)
        fni.main(["--results_dir", str(tmp_path), "--n_boot", "50", "--agent_prefix", "Ag_s",
                  "--rules", "RuleA", "--deltas", "0.02"])
        out = capsys.readouterr().out
        assert "skip evaluation_100" in out and "| heldout |" in out
        saved = json.loads((tmp_path / "noninferiority.json").read_text())
        assert list(saved) == ["heldout"] and saved["heldout"]["tests"][0]["non_inferior"] is True

    def test_main_no_runs_errors(self, tmp_path):
        with pytest.raises(SystemExit):
            fni.main(["--results_dir", str(tmp_path)])


class TestGroupComparison:
    def test_analyse_run_adds_group_difference(self):
        rows = _rows({"Ag_s0": 0.65, "Ag_s1": 0.66, "Old_s0": 0.60, "Old_s1": 0.61, "RuleA": 0.6, "DQN": 0.62})
        res = fni.analyse_run(rows, AGENTS, ("DQN", "RuleA"), (0.02,), n_boot=200,
                              compare_agents=("Old_s0", "Old_s1"))
        g = res["group_difference"]
        assert g["mean_diff"] == pytest.approx(0.05) and g["a_better"]
        assert res["summary"]["compare_agent"]["mean_hp"] == pytest.approx(0.605)
        assert [t["rule"] for t in res["tests"]] == ["DQN", "RuleA"]
        assert "DQN" in res["summary"] and "Oracle" not in res["summary"]

    def test_no_group_difference_by_default(self):
        res = fni.analyse_run(_rows({"Ag_s0": 0.6, "Ag_s1": 0.6, "RuleA": 0.6}), AGENTS, RULES, (0.02,), n_boot=50)
        assert "group_difference" not in res and "compare_agent" not in res["summary"]

    def test_missing_compared_agent_raises(self):
        with pytest.raises(ValueError, match="Old_s1"):
            fni.analyse_run(_rows({"Ag_s0": 0.6, "Ag_s1": 0.6, "RuleA": 0.6, "Old_s0": 0.6}), AGENTS, RULES,
                            (0.02,), n_boot=10, compare_agents=("Old_s0", "Old_s1"))

    def test_main_bprime_options(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(fni, "N_TRAINING_SEEDS", 2)
        run_dir = tmp_path / "heldout2"
        run_dir.mkdir()
        pd.DataFrame(_rows({"New_s0": 0.66, "New_s1": 0.64, "Old_s0": 0.6, "Old_s1": 0.6, "DQN": 0.63,
                            "RuleA": 0.6})).to_csv(run_dir / fni.RAW_NAME, index=False)
        fni.main(["--results_dir", str(tmp_path), "--n_boot", "50", "--agent_prefix", "New_s",
                  "--compare_prefix", "Old_s", "--rules", "DQN", "RuleA", "--deltas", "0.02",
                  "--runs", "heldout2", "--output", "bprime.json"])
        out = capsys.readouterr().out
        assert "| heldout2 | 0.650 | 0.600 | +0.050 |" in out
        saved = json.loads((tmp_path / "bprime.json").read_text())
        assert saved["heldout2"]["group_difference"]["a_better"] is True
        assert not (tmp_path / "noninferiority.json").exists()
