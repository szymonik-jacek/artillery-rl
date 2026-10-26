#!/usr/bin/env python3
"""
Build the pool of perturbed HP trajectories used to train the extended agents.

15 Phase-4 scenarios x 50 variants (seeds 1000-1749, disjoint from the
benchmark's 0-29). One file per scenario is written when that scenario is
complete, so an interrupted run can simply be restarted.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rl_training.utils.trajectory_pool import VARIANTS_PER_SCENARIO, build_pool

DEFAULT_POOL_DIR = ROOT / "rl_training" / "models" / "trajectory_pool"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--variants", type=int, default=VARIANTS_PER_SCENARIO,
                        help="Variants per scenario (50 in the paper)")
    args = parser.parse_args(argv)
    if args.variants < 1:
        parser.error("--variants must be at least 1")

    from tqdm import tqdm

    from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
    from rl_training.environment import ArtilleryFiringEnv

    scenarios = CurriculumScenarios(phase=4).get_all_scenarios()
    env = ArtilleryFiringEnv()
    with tqdm(total=len(scenarios) * args.variants, desc="Pool trajectories", unit="traj",
              file=sys.stderr) as bar:
        def progress(done, total):
            bar.n = done
            bar.refresh()

        files = build_pool(env, scenarios, args.pool_dir, variants=args.variants, progress=progress)
    print(f"Pool complete: {len(files)} scenario files in {args.pool_dir}")


if __name__ == "__main__":
    main()
