"""
Tests for scripts/plot_benchmark.py: policy families, labels, and the
HP-vs-time figure (files written, hidden families, front drawn through
visible points only).
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import plot_benchmark
from plot_benchmark import display_name, plot_pareto, policy_family


def _p(policy, hp, t, rate=100.0):
    return {"policy": policy, "mean_hp": hp, "mean_fire_time_s": t, "firing_rate_pct": rate}


POOLED = [
    _p("DQN", 0.58, 5.8),
    _p("Oracle", 0.68, 10.5),
    _p("Threshold_0.50", 0.52, 4.4),
    _p("Threshold_0.65", 0.30, 7.3, rate=70.0),
    _p("Decaying_0.90_0.40", 0.60, 10.1),
    _p("CovGated_0.50_1", 0.53, 4.6),
    _p("Plateau_0.45_k3_e0.01", 0.56, 7.0),
    _p("V2_l0.15_s2", 0.65, 9.6),
]
FRONT = ["Threshold_0.50", "CovGated_0.50_1", "DQN", "Decaying_0.90_0.40", "Oracle"]


def _scatter_points(fig):
    ax = fig.axes[0]
    return {tuple(np.round(o, 3)) for c in ax.collections for o in c.get_offsets()}


class TestNames:
    def test_policy_family(self):
        assert policy_family("Threshold_0.50") == 0
        assert policy_family("Decaying_0.90_0.40") == 1
        assert policy_family("Plateau_0.45_k3_e0.01") == 2
        assert policy_family("CovGated_0.50_1") == 3
        assert policy_family("DQN") is None and policy_family("V1_s0") is None

    def test_display_name(self):
        assert display_name("Threshold_0.50") == "Fixed 0.50"
        assert display_name("Decaying_0.90_0.40") == "Decaying 0.90→0.40"
        assert display_name("CovGated_0.50_1") == "Cov-gated 0.50, τ=1"
        assert display_name("Plateau_0.45_k3_e0.01") == "Plateau ≥0.45, k=3"
        assert display_name("Oracle") == "Hindsight oracle"
        assert display_name("V2_l0.15_s2") == "V2 l0.15 s2"


class TestPlotPareto:
    def test_writes_pdf_and_png(self, tmp_path):
        paths = [tmp_path / "f.pdf", tmp_path / "f.png"]
        plot_pareto(POOLED, FRONT, paths, labelled=["Threshold_0.50", "Decaying_0.90_0.40"])
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)

    def test_gated_hidden_by_default(self, tmp_path):
        fig = plot_pareto(POOLED, FRONT, [tmp_path / "f.png"])
        points = _scatter_points(fig)
        assert (4.6, 0.53) not in points
        assert {(4.4, 0.52), (5.8, 0.58), (10.5, 0.68), (9.6, 0.65), (7.0, 0.56)} <= points

    def test_gated_shown_on_request(self, tmp_path):
        fig = plot_pareto(POOLED, FRONT, [tmp_path / "f.png"], include_gated=True)
        assert (4.6, 0.53) in _scatter_points(fig)

    def test_front_line_skips_hidden_points(self, tmp_path):
        fig = plot_pareto(POOLED, FRONT, [tmp_path / "f.png"])
        line = fig.axes[0].lines[0]
        assert list(line.get_xdata()) == [4.4, 5.8, 10.1, 10.5]

    def test_labels(self, tmp_path):
        fig = plot_pareto(POOLED, FRONT, [tmp_path / "f.png"], labelled=["Threshold_0.50", "CovGated_0.50_1"])
        texts = {t.get_text() for t in fig.axes[0].texts}
        assert {"DQN", "Hindsight oracle", "Fixed 0.50"} <= texts
        assert not any("Cov-gated" in t for t in texts)  # hidden family is not labelled


def test_main_reads_benchmark_json(tmp_path):
    data = {"pooled": POOLED, "pareto_front_hp_vs_time": FRONT,
            "best_fixed": {"threshold": {"policy": "Threshold_0.50"},
                           "decaying": {"policy": "Decaying_0.90_0.40"}}}
    src = tmp_path / "bench.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    plot_benchmark.main(["--results_json", str(src), "--output_stem", str(tmp_path / "fig")])
    assert (tmp_path / "fig.pdf").exists() and (tmp_path / "fig.png").exists()
