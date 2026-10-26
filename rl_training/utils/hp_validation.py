"""
Monte Carlo validation of the analytical hit probability (Sections 4.4 and 5.4 of the paper).

The environment scores a shot with HP = analytical HP x synthetic 5 s ramp.
The ramp is not physics, so the analytical part ("base HP") is validated:

1. Model check: sample the projectile launch errors (v0, elevation, azimuth)
   and the filter's target estimate from their covariances, fly every
   projectile through the full nonlinear ballistics to the impact time, move
   the target at constant velocity and count hits in the target box. This
   tests the Gaussian / linearised / diagonal approximations of the formula.
2. Calibration against truth: sample only the launch errors and count hits on
   the true target position at impact (known in simulation). This tests the
   whole chain, including whether the filter covariance is realistic.

Firing situations are rebuilt exactly from (scenario, seed, firing step): the
environment draws a step's measurements from its seeded generator starting at
t = 0, so re-seeding and recomputing that step reproduces the evaluated shot.
"""

import gymnasium as gym
import numpy as np

from rl_training.environment import EnvironmentState

HP_RAMP_SECONDS = 5.0
MUZZLE_VELOCITY = 800.0


def ramp_factor(elapsed_s):
    """Synthetic HP ramp of the environment: 0.3 + 0.7 * min(1, t / 5 s)."""
    return 0.3 + 0.7 * min(1.0, elapsed_s / HP_RAMP_SECONDS)


def firing_situation(env, scenario, seed, step):
    """Rebuild the state of a shot fired at `step` of the episode (scenario, seed).

    Returns a dict with the scored HP, the analytical (base) HP and everything
    needed to sample it: target estimate and covariance, firing angles, impact
    time, projectile covariance, target half-dimensions and the true target
    position at impact.
    """
    gym.Env.reset(env, seed=seed)
    env.measurement_history = []
    env.last_measurement_time = -0.5
    env.max_episode_steps = env._calculate_max_steps(scenario)
    env.current_state = EnvironmentState(
        scenario=scenario,
        time_remaining=scenario.tracking_duration * (1.0 - step / env.max_episode_steps),
        current_hit_probability=0.0, episode_step=step, initial_hit_probability=0.0,
        covariance_trace=1000.0)
    hp, info = env._calculate_hit_probability(scenario, debug=True)
    if info.get("firing_angles") is None or "hp_details" not in info:
        raise ValueError(f"No firing solution for {scenario.name} seed {seed} step {step}")
    elev, azim, impact_time, intersection_point, _ = info["firing_angles"]
    elapsed = float(info["elapsed_time"])
    true_start = np.array([scenario.range_m, 0.0, 50.0])
    true_velocity = np.array([scenario.target_vx, scenario.target_vy, scenario.target_vz])
    return {
        "hp": float(hp),
        "base_hp": float(info["hp_details"]["hit_probability"]),
        "ramp": ramp_factor(elapsed),
        "elapsed_s": elapsed,
        "estimate_lag_s": float(info.get("estimate_lag_s", 0.0)),
        "target_position": np.asarray(info["target_position"], dtype=float),
        "target_velocity": np.asarray(info["target_velocity"], dtype=float),
        "target_covariance": np.asarray(info["target_covariance"], dtype=float),
        "elevation": float(elev),
        "azimuth": float(azim),
        "impact_time": float(impact_time),
        "intersection_point": np.asarray(intersection_point, dtype=float),
        "projectile_covariance": np.asarray(info["projectile_covariance"], dtype=float),
        "projectile_position_covariance": np.asarray(info["hp_details"]["projectile_position_covariance"]),
        "target_position_covariance": np.asarray(info["hp_details"]["target_position_covariance"]),
        "miss_mean": np.asarray(info["hp_details"]["miss_mean"], dtype=float),
        "half_dims": np.array([scenario.target_length, scenario.target_width, scenario.target_height]) / 2.0,
        "true_target_at_impact": true_start + true_velocity * (elapsed + float(impact_time)),
        "range_m": float(scenario.range_m),
    }


def projectile_impacts(situation, n, rng, position_fn):
    """Positions at the impact time of n projectiles with sampled launch errors.

    position_fn(v0, elevation, azimuth, t) -> [x, y, z] is the nonlinear
    ballistic model (ErrorPropagation._calculate_projectile_position_advanced).
    """
    mean = np.array([MUZZLE_VELOCITY, situation["elevation"], situation["azimuth"]])
    params = rng.multivariate_normal(mean, situation["projectile_covariance"], size=n)
    return np.array([position_fn(v, e, a, situation["impact_time"]) for v, e, a in params])


def in_box(miss, half_dims):
    """Hit test of the model: |miss_i| <= half_dim_i on every axis."""
    return np.all(np.abs(np.atleast_2d(miss)) <= half_dims, axis=1)


def model_hit_rate(situation, impacts, rng):
    """Check 1: hits with the target state drawn from the filter's estimate and covariance."""
    state = np.concatenate([situation["target_position"], situation["target_velocity"]])
    targets = rng.multivariate_normal(state, situation["target_covariance"], size=len(impacts))
    target_at_impact = targets[:, :3] + targets[:, 3:] * situation["impact_time"]
    return float(in_box(impacts - target_at_impact, situation["half_dims"]).mean())


def true_hit_rate(situation, impacts):
    """Check 2: hits on the true target position at impact (launch errors only)."""
    return float(in_box(impacts - situation["true_target_at_impact"], situation["half_dims"]).mean())


def full_covariance_hp(situation, n, rng):
    """Linearised Gaussian HP without the diagonal approximation (sampled)."""
    cov = situation["projectile_position_covariance"] + situation["target_position_covariance"]
    misses = rng.multivariate_normal(situation["miss_mean"], cov, size=n)
    return float(in_box(misses, situation["half_dims"]).mean())


def validate_situation(situation, n, seed, position_fn):
    """All checks for one shot; the same projectile samples serve both checks."""
    rng = np.random.default_rng(seed)
    impacts = projectile_impacts(situation, n, rng, position_fn)
    return {
        "base_hp": situation["base_hp"],
        "full_cov_hp": full_covariance_hp(situation, max(n, 20000), rng),
        "mc_model_hp": model_hit_rate(situation, impacts, rng),
        "true_hit_rate": true_hit_rate(situation, impacts),
        "ramp": situation["ramp"],
        "elapsed_s": situation["elapsed_s"],
        "impact_time": situation["impact_time"],
        "range_m": situation["range_m"],
        "n_samples": n,
    }
