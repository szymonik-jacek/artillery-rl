"""
Pool of pre-computed, perturbed HP trajectories for training the extended agents.

Without a pool, every training episode of a scenario replays the same cached
HP trajectory. The pool stores, per Phase-4 scenario, a number of perturbed
variants (same perturbation ranges as the benchmark, disjoint seeds) with
their environment trajectories. Training installs the pool into the
environment's cache and draws a random variant each episode, so reset() is
as fast as before while the agent sees a different HP curve every time.
"""

import pickle
from pathlib import Path

import numpy as np

from rl_training.utils.evaluation_protocol import perturb_scenario

POOL_SEED_BASE = 1000
VARIANTS_PER_SCENARIO = 50
POOL_FORMAT_VERSION = 1


def pool_seed(scenario_index, variant, base=POOL_SEED_BASE, variants=VARIANTS_PER_SCENARIO):
    """Seed of a pool variant: base + variants * scenario_index + variant.

    Used both for the perturbation and for the environment's measurement
    noise. With the defaults, seeds are 1000-1749 for 15 scenarios, disjoint
    from the benchmark seeds 0-29.
    """
    if not 0 <= variant < variants:
        raise ValueError(f"variant must be in [0, {variants}), got {variant}")
    return base + variants * scenario_index + variant


def _scenario_file(pool_dir, scenario_index, name):
    return Path(pool_dir) / f"{scenario_index:02d}_{name}.pkl"


def build_pool(env, scenarios, pool_dir, variants=VARIANTS_PER_SCENARIO, progress=None):
    """Compute and save the pool, one file per scenario; existing files are kept.

    For each scenario (index i) and variant k, the scenario is perturbed with
    np.random.default_rng(pool_seed(i, k)) and the environment is reset with
    that seed; the resulting trajectories are stored. progress(done, total)
    is called after every variant. Returns the list of scenario files.
    """
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    total = len(scenarios) * variants
    done = 0
    files = []
    for i, nominal in enumerate(scenarios):
        path = _scenario_file(pool_dir, i, nominal.name)
        files.append(path)
        if path.exists():
            done += variants
            if progress is not None:
                progress(done, total)
            continue
        entries = []
        for k in range(variants):
            seed = pool_seed(i, k, variants=variants)
            scenario = perturb_scenario(nominal, np.random.default_rng(seed))
            env.reset(scenario_override=scenario, seed=seed)
            entries.append({
                "name": nominal.name,
                "scenario_index": i,
                "variant": k,
                "seed": seed,
                "scenario": scenario,
                "trajectories": (dict(env.hp_trajectory), dict(env.cov_trajectory),
                                 dict(env.pos_trajectory), dict(env.vel_trajectory)),
            })
            done += 1
            if progress is not None:
                progress(done, total)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"version": POOL_FORMAT_VERSION, "entries": entries}, f)
        tmp.replace(path)  # only complete scenario files ever exist under the final name
    return files


def load_pool(pool_dir):
    """Load a pool directory into {scenario name: [entries]}."""
    pool = {}
    for path in sorted(Path(pool_dir).glob("*.pkl")):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != POOL_FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported pool format {data.get('version')!r}")
        for entry in data["entries"]:
            pool.setdefault(entry["name"], []).append(entry)
    if not pool:
        raise FileNotFoundError(f"No pool files in {pool_dir}")
    return pool


def install_pool(env, pool):
    """Put every pool trajectory into env.hp_cache so reset() needs no computation."""
    if getattr(env, "extrapolate_estimate", False):
        raise ValueError("The trajectory pool was computed without extrapolate_estimate; rebuild it for this env")
    for entries in pool.values():
        for entry in entries:
            env.hp_cache[env._get_cache_key(entry["scenario"], seed=entry["seed"])] = entry["trajectories"]


class PoolSampler:
    """Draws a random pool variant of a nominal scenario (uniform, seeded)."""

    def __init__(self, pool, seed=None):
        self.pool = pool
        self.rng = np.random.default_rng(seed)

    def sample(self, nominal_scenario):
        """Return (perturbed scenario, seed) of a random variant of nominal_scenario."""
        entries = self.pool.get(nominal_scenario.name)
        if not entries:
            raise KeyError(f"No pool variants for scenario {nominal_scenario.name!r}")
        entry = entries[int(self.rng.integers(len(entries)))]
        return entry["scenario"], entry["seed"]
