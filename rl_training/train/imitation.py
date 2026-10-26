"""
Warm start of the HP-free agent by imitating a threshold rule (TODO step 2c).

The teacher (by default the decaying threshold 0.90 -> 0.40) decides from
the hit probability; the student network sees only the agent's 14 target
features. A dataset of (features, teacher action, HP) is collected from the
trajectory pool, and the network is pre-trained with the large-margin loss of
DQfD (Hester et al., 2018) so that the teacher's action gets the highest
Q-value; an auxiliary head, if present, learns HP at the same time. HP is a
training target only, never a network input.
"""

import numpy as np
import torch

from rl_training.agents.baseline_policies import DecayingThresholdPolicy
from rl_training.environment import Action
from rl_training.utils.trajectory_pool import install_pool


def default_teacher():
    """The pre-registered teacher: decaying threshold 0.90 -> 0.40."""
    return DecayingThresholdPolicy(0.90, 0.40)


def collect_teacher_dataset(env, pool, teacher=None, max_steps=100):
    """Roll out the teacher on every pool trajectory.

    env provides the student's observation (create it with the student's
    cov_feature). Each step until the teacher fires (or the episode ends)
    gives one sample. Returns (features [N, 14], actions [N], hp [N]).
    """
    teacher = teacher or default_teacher()
    install_pool(env, pool)
    features, actions, hps = [], [], []
    for entries in pool.values():
        for entry in entries:
            observation, _ = env.reset(scenario_override=entry["scenario"], seed=entry["seed"])
            for _ in range(max_steps):
                action = int(teacher.act(env.current_state))
                features.append(np.asarray(observation, dtype=np.float32))
                actions.append(action)
                hps.append(float(env.current_state.current_hit_probability))
                if action == Action.FIRE:
                    break
                observation, _, terminated, truncated, _ = env.step(Action.HOLD)
                if terminated or truncated:
                    break
    return np.stack(features), np.asarray(actions, dtype=np.int64), np.asarray(hps, dtype=np.float32)


def large_margin_loss(q_values, actions, margin, weights=None):
    """DQfD large-margin loss: max_a [Q(s,a) + margin * (a != a_E)] - Q(s, a_E)."""
    penalty = torch.full_like(q_values, margin)
    penalty.scatter_(1, actions.unsqueeze(1), 0.0)
    per_sample = (q_values + penalty).max(dim=1)[0] - q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    if weights is None:
        return per_sample.mean()
    return (per_sample * weights).sum() / weights.sum()


def class_weights(actions):
    """Per-sample weights that balance FIRE and HOLD (FIRE weighted by #HOLD / #FIRE)."""
    n_fire = int((actions == Action.FIRE).sum())
    n_hold = len(actions) - n_fire
    if n_fire == 0 or n_hold == 0:
        return np.ones(len(actions), dtype=np.float32)
    return np.where(actions == Action.FIRE, n_hold / n_fire, 1.0).astype(np.float32)


def pretrain(network, features, actions, hps, epochs=30, batch_size=256, lr=1e-3, margin=0.1,
             aux_weight=1.0, seed=0, device="cpu"):
    """Pre-train network on the teacher dataset. Returns per-epoch history dicts."""
    generator = torch.Generator().manual_seed(seed)
    device = torch.device(device)
    network.to(device).train()
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    a = torch.as_tensor(actions, dtype=torch.long, device=device)
    hp = torch.as_tensor(hps, dtype=torch.float32, device=device)
    w = torch.as_tensor(class_weights(actions), dtype=torch.float32, device=device)
    use_aux = aux_weight > 0 and getattr(network, "aux", None) is not None
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    history = []
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(x), batch_size):
            idx = order[start:start + batch_size]
            loss = large_margin_loss(network(x[idx]), a[idx], margin, w[idx])
            if use_aux:
                loss = loss + aux_weight * torch.nn.functional.mse_loss(network.predict_aux(x[idx]), hp[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(idx)
        history.append({"loss": total / len(x), **teacher_agreement(network, features, actions, device)})
    network.eval()
    return history


def teacher_agreement(network, features, actions, device="cpu"):
    """Fraction of samples where the network's greedy action equals the teacher's, per class."""
    with torch.no_grad():
        pred = network(torch.as_tensor(features, dtype=torch.float32, device=torch.device(device)))
        pred = pred.argmax(dim=1).cpu().numpy()
    fire = actions == Action.FIRE
    return {
        "agreement": float((pred == actions).mean()),
        "agreement_fire": float((pred[fire] == actions[fire]).mean()) if fire.any() else float("nan"),
        "agreement_hold": float((pred[~fire] == actions[~fire]).mean()) if (~fire).any() else float("nan"),
    }
