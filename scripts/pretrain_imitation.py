#!/usr/bin/env python3
"""
Pre-train the HP-free agent by imitating the decaying threshold rule (TODO 2c).

Collects (14 target features, teacher action, HP) from every trajectory of
the pool, pre-trains a dueling network with an auxiliary HP head using the
DQfD large-margin loss, and saves <prefix>_pretrain.pth with its model-config
sidecar. Training then continues with RL via
train_dqn_curriculum.py --init_checkpoint <that file>.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rl_training.train.train_dqn_curriculum import CHECKPOINT_DIR

DEFAULT_POOL_DIR = ROOT / "rl_training" / "models" / "trajectory_pool"
MODEL_CONFIG = {"architecture": "dueling", "aux_head": True}


def pretrain_checkpoint_path(prefix, checkpoint_dir=CHECKPOINT_DIR):
    """Pre-trained network for a run: <checkpoint_dir>/<prefix>_pretrain.pth."""
    return Path(checkpoint_dir) / f"{prefix}_pretrain.pth"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Run prefix; writes <prefix>_pretrain.pth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool_dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--cov_feature", choices=("linear", "log"), default="log")
    parser.add_argument("--time_feature", choices=("steps", "seconds"), default="steps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--aux_weight", type=float, default=1.0)
    parser.add_argument("--checkpoint_dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--history_dir", type=Path, default=ROOT / "results" / "imitation")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out = pretrain_checkpoint_path(args.prefix, args.checkpoint_dir)
    if out.exists() and not args.overwrite:
        parser.error(f"{out} already exists; use --overwrite to replace it")

    import torch

    from rl_training.agents.dqn_components import build_q_network, save_model_config
    from rl_training.environment import ArtilleryFiringEnv
    from rl_training.train.imitation import collect_teacher_dataset, pretrain
    from rl_training.utils.trajectory_pool import load_pool

    torch.manual_seed(args.seed)
    env = ArtilleryFiringEnv(cov_feature=args.cov_feature, time_feature=args.time_feature)
    features, actions, hps = collect_teacher_dataset(env, load_pool(args.pool_dir))
    print(f"Teacher dataset: {len(actions)} samples, {int(actions.sum())} FIRE", flush=True)

    network = build_q_network(env.observation_space.shape[0], env.action_space.n, MODEL_CONFIG)
    history = pretrain(network, features, actions, hps, epochs=args.epochs, margin=args.margin,
                       aux_weight=args.aux_weight, seed=args.seed)
    last = history[-1]
    print(f"Agreement with teacher: {last['agreement']:.3f} "
          f"(FIRE {last['agreement_fire']:.3f}, HOLD {last['agreement_hold']:.3f})", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), out)
    save_model_config(out, {**MODEL_CONFIG, "cov_feature": args.cov_feature,
                            "time_feature": args.time_feature})
    args.history_dir.mkdir(parents=True, exist_ok=True)
    (args.history_dir / f"{args.prefix}_pretrain_history.json").write_text(json.dumps(
        {"seed": args.seed, "samples": int(len(actions)), "fire_samples": int(actions.sum()),
         "epochs": args.epochs, "margin": args.margin, "aux_weight": args.aux_weight,
         "teacher": "Decaying_0.90_0.40", "history": history}, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
