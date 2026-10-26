#!/usr/bin/env python3
"""
Train pre-registered agent variants (all without hit probability in the state).

--set v0r (default): the published setup retrained with seeds 0-2
(reproducibility control).
--set pool: the same setup trained on a pre-computed pool of perturbed
trajectories (a random variant each episode).
--set stage2: the pre-registered stage-2 variants (TODO, commit ca6e1ac), seeds
0-4: A0 (log covariance input, dueling, Double DQN, 3-step returns, aligned
reward with lambda 0.15, pool), A (A0 + auxiliary HP head) and B (A started
from an imitation pre-training of the decaying rule).
--set stage2e: variant B' (TODO stage 2e), seeds 0-4: B with the step-count
input replaced by the elapsed tracking time in seconds, in pre-training and
training.

Each variant is trained with seeds 0-2 through train_dqn_curriculum.py using
the published agent's config. Runs whose phase-4 checkpoint already exists
are skipped, so the script can be restarted after an interruption.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rl_training.train.train_dqn_curriculum import CHECKPOINT_DIR, checkpoint_path

TRAIN_SCRIPT = ROOT / "rl_training" / "train" / "train_dqn_curriculum.py"
CONFIG = ROOT / "config" / "paper2_experiment.json"
SEEDS = (0, 1, 2)
STAGE2_SEEDS = (0, 1, 2, 3, 4)
PRETRAIN_SCRIPT = ROOT / "scripts" / "pretrain_imitation.py"
DEFAULT_POOL_DIR = ROOT / "rl_training" / "models" / "trajectory_pool"
POOL_SCENARIOS = 15  # one pool file per Phase-4 scenario
TIMESTEPS = 10000
FINAL_PHASE = 4


def v0r_runs(seeds=SEEDS):
    """Published setup retrained, as (checkpoint_prefix, extra CLI arguments)."""
    return [(f"dqn_v0r_seed{seed}", ["--seed", str(seed)]) for seed in seeds]


def pool_runs(pool_dir=DEFAULT_POOL_DIR, seeds=SEEDS):
    """Published setup trained on the trajectory pool, as (prefix, extra CLI arguments)."""
    return [(f"dqn_pool_v0r_seed{seed}", ["--seed", str(seed), "--trajectory_pool", str(pool_dir)])
            for seed in seeds]


def stage2_runs(pool_dir=DEFAULT_POOL_DIR, seeds=STAGE2_SEEDS, checkpoint_dir=CHECKPOINT_DIR):
    """Pre-registered stage-2 runs as (prefix, training args, pre-training args or None)."""
    runs = []
    for variant in ("a0", "a", "b"):
        for seed in seeds:
            prefix = f"dqn_s2_{variant}_seed{seed}"
            args = ["--seed", str(seed), "--architecture", "dueling", "--double_dqn", "--n_step", "3",
                    "--reward_mode", "aligned", "--time_cost", "0.15", "--cov_feature", "log",
                    "--trajectory_pool", str(pool_dir)]
            pretrain = None
            if variant in ("a", "b"):
                args += ["--aux_hp_weight", "1.0"]
            if variant == "b":
                init = Path(checkpoint_dir) / f"{prefix}_pretrain.pth"
                args += ["--init_checkpoint", str(init)]
                pretrain = ["--prefix", prefix, "--seed", str(seed), "--pool_dir", str(pool_dir),
                            "--cov_feature", "log", "--checkpoint_dir", str(checkpoint_dir)]
            runs.append((prefix, args, pretrain))
    return runs


def stage2e_runs(pool_dir=DEFAULT_POOL_DIR, seeds=STAGE2_SEEDS, checkpoint_dir=CHECKPOINT_DIR):
    """Variant B' (stage 2e): variant B with time_feature="seconds", prefix dqn_s2_bp_seed<s>."""
    runs = []
    for prefix, args, pretrain in stage2_runs(pool_dir, seeds, checkpoint_dir):
        if not prefix.startswith("dqn_s2_b_"):
            continue
        new_prefix = prefix.replace("dqn_s2_b_", "dqn_s2_bp_")
        args = [a.replace(prefix, new_prefix) for a in args] + ["--time_feature", "seconds"]
        pretrain = [a.replace(prefix, new_prefix) for a in pretrain] + ["--time_feature", "seconds"]
        runs.append((new_prefix, args, pretrain))
    return runs


def build_pretrain_command(pretrain_args, python=sys.executable):
    """Command line for an imitation pre-training."""
    return [python, "-u", str(PRETRAIN_SCRIPT), *pretrain_args]


def build_command(prefix, extra_args, python=sys.executable):
    """Command line for one training run (phases 1-4, published config)."""
    return [python, "-u", str(TRAIN_SCRIPT),
            "--start_phase", "1", "--end_phase", str(FINAL_PHASE),
            "--timesteps", str(TIMESTEPS), "--config", str(CONFIG),
            "--checkpoint_prefix", prefix, *extra_args]


def run_status(prefix, checkpoint_dir=CHECKPOINT_DIR):
    """'done' if the final checkpoint exists, 'partial' if only earlier phases do, else 'pending'."""
    if checkpoint_path(prefix, FINAL_PHASE, checkpoint_dir).exists():
        return "done"
    if any(checkpoint_path(prefix, p, checkpoint_dir).exists() for p in range(1, FINAL_PHASE)):
        return "partial"
    return "pending"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("v0r", "pool", "stage2", "stage2e"), default="v0r")
    parser.add_argument("--pool_dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--log_dir", type=Path, default=None,
                        help="Default: results/<set>_retraining_logs")
    parser.add_argument("--dry_run", action="store_true", help="Print the runs without training")
    args = parser.parse_args(argv)

    if args.set in ("pool", "stage2", "stage2e"):
        n_files = len(list(Path(args.pool_dir).glob("*.pkl")))
        if not args.dry_run and n_files < POOL_SCENARIOS:
            parser.error(f"Trajectory pool in {args.pool_dir} is incomplete ({n_files}/{POOL_SCENARIOS} "
                         "scenario files); run scripts/build_trajectory_pool.py first")
    if args.set == "stage2":
        runs = stage2_runs(args.pool_dir)
    elif args.set == "stage2e":
        runs = stage2e_runs(args.pool_dir)
    elif args.set == "pool":
        runs = [(prefix, extra, None) for prefix, extra in pool_runs(args.pool_dir)]
    else:
        runs = [(prefix, extra, None) for prefix, extra in v0r_runs()]
    log_dir = args.log_dir or ROOT / "results" / f"{args.set}_retraining_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, (prefix, extra_args, pretrain_args) in enumerate(runs, 1):
        status = run_status(prefix)
        label = f"[{i}/{len(runs)}] {prefix}"
        if status == "done":
            print(f"{label}: already trained, skipping", flush=True)
            continue
        if status == "partial":
            print(f"{label}: partial checkpoints exist, skipping (remove them to retrain)", flush=True)
            failed.append(prefix)
            continue
        command = build_command(prefix, extra_args)
        pre_command = build_pretrain_command(pretrain_args) if pretrain_args else None
        if args.dry_run:
            if pre_command:
                print(f"{label} (pre-training): {' '.join(pre_command)}", flush=True)
            print(f"{label}: {' '.join(command)}", flush=True)
            continue
        start = time.time()
        with open(log_dir / f"{prefix}.log", "w", encoding="utf-8") as log:
            init = Path(extra_args[extra_args.index("--init_checkpoint") + 1]) if pre_command else None
            if pre_command and not init.exists():
                result = subprocess.run(pre_command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                if result.returncode != 0:
                    print(f"{label}: pre-training FAILED (exit {result.returncode})", flush=True)
                    failed.append(prefix)
                    continue
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        minutes = (time.time() - start) / 60
        outcome = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"{label}: {outcome} in {minutes:.1f} min", flush=True)
        if result.returncode != 0:
            failed.append(prefix)

    if failed:
        print(f"Not completed: {', '.join(failed)}", flush=True)
        return 1
    if not args.dry_run:
        print("All runs completed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
