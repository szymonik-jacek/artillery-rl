"""
Tests for rl_training/utils/trajectory_pool.py: pool seeds, building and
resuming, save/load, installation into the environment cache and sampling.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.curriculum.scenario_generator import ScenarioParameters
from rl_training.utils.evaluation_protocol import perturb_scenario
from rl_training.utils.trajectory_pool import (
    PoolSampler,
    build_pool,
    install_pool,
    load_pool,
    pool_seed,
)


def _scenario(name, range_m=2000.0, vx=-50.0):
    s = ScenarioParameters(
        range_m=range_m, target_length=5.0, target_width=2.0, target_height=2.0,
        target_vx=vx, target_vy=0.0, target_vz=0.0, tracking_duration=10.0,
        measurement_noise_std=np.array([0.5, 0.5, 0.5]), process_noise_std=np.array([0.5, 0.5, 0.5]),
    )
    s.name = name
    return s


class _FakeEnv:
    """reset() 'computes' trajectories that depend on the scenario range and seed."""

    def __init__(self):
        self.resets = []

    def reset(self, scenario_override=None, seed=None):
        self.resets.append((scenario_override.name, seed))
        hp = {k: round(scenario_override.range_m / 1e4 + seed / 1e5 + k / 100, 6) for k in range(4)}
        self.hp_trajectory = hp
        self.cov_trajectory = {k: 1.0 for k in hp}
        self.pos_trajectory = {k: np.zeros(3) for k in hp}
        self.vel_trajectory = {k: np.zeros(3) for k in hp}
        return np.zeros(14), {}


# ---------------------------------------------------------------------------
# pool_seed
# ---------------------------------------------------------------------------

class TestPoolSeed:
    def test_unique_and_disjoint_from_benchmark(self):
        seeds = [pool_seed(i, k) for i in range(15) for k in range(50)]
        assert len(set(seeds)) == 750
        assert min(seeds) == 1000 and max(seeds) == 1749
        assert not set(seeds) & set(range(30))

    def test_rejects_out_of_range_variant(self):
        with pytest.raises(ValueError):
            pool_seed(0, 50)


# ---------------------------------------------------------------------------
# build / load
# ---------------------------------------------------------------------------

class TestBuildAndLoad:
    def test_entries_use_pool_seeds_and_benchmark_perturbation(self, tmp_path):
        env = _FakeEnv()
        nominal = [_scenario("A"), _scenario("B", range_m=3000.0)]
        build_pool(env, nominal, tmp_path, variants=3)
        pool = load_pool(tmp_path)
        assert set(pool) == {"A", "B"} and all(len(v) == 3 for v in pool.values())
        entry = pool["B"][2]
        assert entry["seed"] == pool_seed(1, 2, variants=3)
        expected = perturb_scenario(nominal[1], np.random.default_rng(entry["seed"]))
        assert entry["scenario"].range_m == expected.range_m
        assert entry["scenario"].target_vx == expected.target_vx
        np.testing.assert_allclose(entry["scenario"].measurement_noise_std, expected.measurement_noise_std)
        assert entry["trajectories"][0][0] == pytest.approx(expected.range_m / 1e4 + entry["seed"] / 1e5, abs=1e-6)

    def test_resume_skips_finished_scenarios(self, tmp_path):
        nominal = [_scenario("A"), _scenario("B")]
        build_pool(_FakeEnv(), nominal[:1], tmp_path, variants=2)
        env = _FakeEnv()
        progress = []
        build_pool(env, nominal, tmp_path, variants=2, progress=lambda d, t: progress.append((d, t)))
        assert {name for name, _ in env.resets} == {"B"}
        assert progress[-1] == (4, 4)

    def test_no_partial_files(self, tmp_path):
        class _Failing(_FakeEnv):
            def reset(self, scenario_override=None, seed=None):
                if len(self.resets) == 1:
                    raise RuntimeError("interrupted")
                return super().reset(scenario_override, seed)

        with pytest.raises(RuntimeError):
            build_pool(_Failing(), [_scenario("A")], tmp_path, variants=3)
        assert not list(tmp_path.glob("*.pkl"))

    def test_load_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pool(tmp_path)


# ---------------------------------------------------------------------------
# PoolSampler
# ---------------------------------------------------------------------------

class TestPoolSampler:
    POOL = {"A": [{"scenario": f"A{k}", "seed": 1000 + k} for k in range(5)],
            "B": [{"scenario": "B0", "seed": 2000}]}

    def test_samples_only_variants_of_the_scenario(self):
        sampler = PoolSampler(self.POOL, seed=0)
        for _ in range(20):
            scenario, seed = sampler.sample(SimpleNamespace(name="A"))
            assert scenario.startswith("A") and 1000 <= seed < 1005
        assert sampler.sample(SimpleNamespace(name="B")) == ("B0", 2000)

    def test_reproducible_and_covers_all_variants(self):
        a = [PoolSampler(self.POOL, seed=7).sample(SimpleNamespace(name="A")) for _ in range(1)]
        b = [PoolSampler(self.POOL, seed=7).sample(SimpleNamespace(name="A")) for _ in range(1)]
        assert a == b
        sampler = PoolSampler(self.POOL, seed=1)
        seen = {sampler.sample(SimpleNamespace(name="A"))[1] for _ in range(200)}
        assert seen == set(range(1000, 1005))

    def test_unknown_scenario_raises(self):
        with pytest.raises(KeyError):
            PoolSampler(self.POOL).sample(SimpleNamespace(name="C"))


# ---------------------------------------------------------------------------
# Integration: an installed pool makes reset() a cache hit (real env)
# ---------------------------------------------------------------------------

def test_installed_pool_reproduces_trajectories_without_computation(tmp_path, monkeypatch):
    import rl_training.environment as environment
    from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
    from rl_training.environment import ArtilleryFiringEnv

    static_close = CurriculumScenarios(phase=4).get_scenario_by_index(0)
    build_pool(ArtilleryFiringEnv(), [static_close], tmp_path, variants=2)
    pool = load_pool(tmp_path)

    env = ArtilleryFiringEnv()
    install_pool(env, pool)

    class _NoCompute:
        def __init__(self, *a, **k):
            raise AssertionError("trajectory was recomputed")

    monkeypatch.setattr(environment.concurrent.futures, "ProcessPoolExecutor", _NoCompute)
    for entry in pool[static_close.name]:
        env.reset(scenario_override=entry["scenario"], seed=entry["seed"])
        assert env.hp_trajectory == entry["trajectories"][0]
    # The two variants really differ.
    hp0, hp1 = (e["trajectories"][0] for e in pool[static_close.name])
    assert hp0 != hp1


# ---------------------------------------------------------------------------
# scripts/build_trajectory_pool.py
# ---------------------------------------------------------------------------

def test_build_script_uses_all_phase4_scenarios(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import build_trajectory_pool

    calls = []

    def fake_build(env, scenarios, pool_dir, variants, progress):
        calls.append(([s.name for s in scenarios], pool_dir, variants))
        progress(variants * len(scenarios), variants * len(scenarios))
        return []

    monkeypatch.setattr(build_trajectory_pool, "build_pool", fake_build)
    build_trajectory_pool.main(["--pool_dir", str(tmp_path), "--variants", "3"])
    names, pool_dir, variants = calls[0]
    assert len(names) == 15 and len(set(names)) == 15
    assert pool_dir == tmp_path and variants == 3
