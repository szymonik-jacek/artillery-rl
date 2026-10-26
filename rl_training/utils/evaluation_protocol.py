"""
Shared evaluation protocol for Paper 2 experiments.

Defines the five Phase-4 evaluation scenarios and the per-seed perturbation
applied to them, so that the DQN evaluation and the baseline benchmark draw
identical random perturbations for the same seed.
"""

import copy

import numpy as np

from rl_training.environment import (
    COVARIANCE_FEATURE_INDEX,
    TIME_FEATURE_INDEX,
    covariance_feature,
    time_feature,
)

# Target start height used by the evaluation geometry (metres).
INITIAL_TARGET_Z = 50.0

# Inner dead-zone boundary for firing (metres), see the manuscript's safety criterion.
MIN_FIRING_RANGE_M = 200.0

# Indices into CurriculumScenarios(phase=4) used for evaluation.
EVALUATION_SCENARIO_INDICES = (
    0,   # Static_Close (800m)
    4,   # Fast_Approach_Long (3000m)
    6,   # Medium_Approach (2000m)
    11,  # Complex_3D_Motion (1800m)
    13,  # ExtremeLong_Large_Lateral (3000m)
)


def get_evaluation_scenarios(curriculum):
    """Return the evaluation scenarios from a Phase-4 curriculum."""
    return [curriculum.get_scenario_by_index(i) for i in EVALUATION_SCENARIO_INDICES]


def get_tuning_scenarios(curriculum):
    """Phase-4 scenarios not used for evaluation, for tuning baseline rules."""
    return [curriculum.get_scenario_by_index(i) for i in range(curriculum.num_scenarios())
            if i not in EVALUATION_SCENARIO_INDICES]


def perturb_scenario(scenario, rng):
    """Return a copy of scenario with small random perturbations.

    Perturbations (±5–10 %) simulate real-world uncertainty in initial
    conditions, giving each run a genuinely different physics trajectory.
    Zero velocity components stay zero and consume no random draw.
    """
    s = copy.copy(scenario)
    s.range_m           = scenario.range_m           * rng.uniform(0.95, 1.05)
    s.target_vx         = scenario.target_vx         * rng.uniform(0.90, 1.10) if scenario.target_vx != 0 else 0.0
    s.target_vy         = scenario.target_vy         * rng.uniform(0.90, 1.10) if scenario.target_vy != 0 else 0.0
    s.target_vz         = scenario.target_vz         * rng.uniform(0.90, 1.10) if scenario.target_vz != 0 else 0.0
    s.measurement_noise_std = scenario.measurement_noise_std * rng.uniform(0.80, 1.20)
    return s


def target_position_at_step(scenario, step, max_episode_steps):
    """Nominal target position (x, y, z) at a decision step.

    Uses the same constant-velocity geometry as evaluate_dqn_trajectories:
    the target starts at (range_m, 0, INITIAL_TARGET_Z).
    """
    t = step * (scenario.tracking_duration / max_episode_steps)
    return (
        float(scenario.range_m) + scenario.target_vx * t,
        scenario.target_vy * t,
        INITIAL_TARGET_Z + scenario.target_vz * t,
    )


def step_seconds(env, scenario):
    """Length of one decision step (s) for a scenario, as env.reset() would set it."""
    return scenario.tracking_duration / env._calculate_max_steps(scenario)


def run_policy_episode(env, scenario, decide, seed=None, max_steps=100,
                       min_firing_range=None):
    """Run one episode with a FIRE/HOLD decision function.

    decide(observation, env_state) returns an action (0 = HOLD, 1 = FIRE).
    An episode that ends without FIRE is a timeout and scores HP = 0.

    If min_firing_range is set, FIRE is blocked (executed as HOLD) while the
    target's slant range is below it, as a dead-zone interlock would do.
    Blocked attempts are counted in blocked_fire_attempts.

    Returns a dict with fired, timed_out, fired_at_step, fire_time_s,
    hit_probability, firing_pos, firing_slant_range, steps and
    blocked_fire_attempts. fire_time_s is the time into the engagement window
    at which FIRE was given (step length depends on the episode's
    max_episode_steps, so steps are not comparable across scenarios).
    """
    observation, _ = env.reset(scenario_override=scenario, seed=seed)
    done = False
    step = 0
    blocked = 0
    while not done and step < max_steps:
        action = int(decide(observation, env.current_state))
        pos = target_position_at_step(scenario, step, env.max_episode_steps)
        slant_range = float(np.linalg.norm(pos))
        if action == 1 and min_firing_range is not None and slant_range < min_firing_range:
            action = 0
            blocked += 1
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if action == 1:
            return {
                "fired": True,
                "timed_out": False,
                "fired_at_step": step,
                "fire_time_s": step * scenario.tracking_duration / env.max_episode_steps,
                "hit_probability": float(info["hit_probability"]),
                "firing_pos": pos,
                "firing_slant_range": slant_range,
                "steps": step + 1,
                "blocked_fire_attempts": blocked,
            }
        step += 1

    return {
        "fired": False,
        "timed_out": True,
        "fired_at_step": None,
        "fire_time_s": None,
        "hit_probability": 0.0,
        "firing_pos": None,
        "firing_slant_range": None,
        "steps": step,
        "blocked_fire_attempts": blocked,
    }


COV_ABLATIONS = ("zero", "frozen")


def make_dqn_decide(q_network, device, cov_ablation=None, cov_feature="linear", time_input="steps"):
    """Wrap a Q-network as a decide(observation, env_state) function (greedy).

    The network sees only the 14-feature observation; hit probability is never
    an input at decision time. env_state is read only for episode_step (to
    reset the frozen covariance ablation) and, with cov_feature="log", for the
    Kalman covariance trace, rescaled with the same covariance_feature() the
    environment uses in training; with time_input="seconds", for the elapsed
    tracking time, via the same time_feature().

    cov_ablation tests whether the agent uses the covariance-trace input:
    "zero" replaces it with 0; "frozen" holds it at its value from the
    episode's first step (keeps the scale, removes the convergence signal).
    """
    import torch

    if cov_ablation not in (None, *COV_ABLATIONS):
        raise ValueError(f"cov_ablation must be None or one of {COV_ABLATIONS}, got {cov_ablation!r}")
    covariance_feature(1.0, cov_feature)  # validates the mode
    if time_input not in ("steps", "seconds"):
        raise ValueError(f"time_input must be 'steps' or 'seconds', got {time_input!r}")
    frozen = {}

    def decide(observation, env_state):
        if cov_feature != "linear" or cov_ablation is not None or time_input != "steps":
            observation = np.array(observation, dtype=np.float32, copy=True)
        if cov_feature != "linear":
            observation[COVARIANCE_FEATURE_INDEX] = covariance_feature(env_state.covariance_trace, cov_feature)
        if time_input != "steps":
            observation[TIME_FEATURE_INDEX] = time_feature(env_state, time_input)
        if cov_ablation is not None:
            if cov_ablation == "zero":
                observation[COVARIANCE_FEATURE_INDEX] = 0.0
            else:
                if env_state.episode_step == 0:
                    frozen["value"] = observation[COVARIANCE_FEATURE_INDEX]
                observation[COVARIANCE_FEATURE_INDEX] = frozen["value"]
        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
            return q_network(state_tensor).argmax().item()

    return decide


def best_reachable_step(hp_trajectory, scenario, max_episode_steps, min_firing_range=None):
    """Decision step with the highest HP that a policy could still fire at.

    FIRE is possible at steps 0 .. max_episode_steps - 1 (the episode times out
    on reaching max_episode_steps) and, if min_firing_range is set, only where
    the target is at least that far away. Ties go to the earliest step.
    Returns None if no step qualifies.
    """
    candidates = [
        k for k in range(max_episode_steps)
        if k in hp_trajectory and (
            min_firing_range is None
            or np.linalg.norm(target_position_at_step(scenario, k, max_episode_steps)) >= min_firing_range
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda k: (hp_trajectory[k], -k))


def make_oracle_decide(env, min_firing_range=None):
    """Hindsight oracle: fires at the best reachable step of the current episode.

    Reads the environment's precomputed HP trajectory, which no real policy
    can see, so it gives an upper bound on achievable HP, not a baseline.
    """
    def decide(observation, env_state):
        best = best_reachable_step(env.hp_trajectory, env_state.scenario,
                                   env.max_episode_steps, min_firing_range)
        return 1 if best is not None and env_state.episode_step == best else 0

    return decide


def make_baseline_decide(policy):
    """Wrap a baseline policy (act(env_state)) as a decide function."""
    def decide(observation, env_state):
        return policy.act(env_state)

    return decide


# ---------------------------------------------------------------------------
# Distribution shifts and the held-out scenario set (claim T3, TODO step 3a)
# ---------------------------------------------------------------------------

# Pre-registered: shifts applied to the nominal scenarios before the per-seed
# perturbation; neither the rules nor the agent are re-tuned or retrained.
SHIFTS = {
    "none": {},
    "noise1.5": {"measurement_noise": 1.5},
    "noise2": {"measurement_noise": 2.0},
    "size0.7": {"target_size": 0.7},
    "range1.2": {"range": 1.2},
}
HELDOUT_SEED = 20261014
HELDOUT_COUNT = 30
# Second held-out set (TODO stage 2e): fixed before any run; used for variant B'.
HELDOUT2_SEED = 20261015


def apply_shift(scenario, shift):
    """Copy of scenario with a named distribution shift applied (see SHIFTS)."""
    if shift not in SHIFTS:
        raise ValueError(f"shift must be one of {sorted(SHIFTS)}, got {shift!r}")
    factors = SHIFTS[shift]
    s = copy.copy(scenario)
    if "measurement_noise" in factors:
        s.measurement_noise_std = scenario.measurement_noise_std * factors["measurement_noise"]
    if "target_size" in factors:
        f = factors["target_size"]
        s.target_length = scenario.target_length * f
        s.target_width = scenario.target_width * f
        s.target_height = scenario.target_height * f
    if "range" in factors:
        s.range_m = scenario.range_m * factors["range"]
    return s


def get_heldout_scenarios(count=HELDOUT_COUNT, seed=HELDOUT_SEED, prefix="Heldout"):
    """Random scenarios from ScenarioGenerator with a fixed seed, named <prefix>_00, <prefix>_01, ..."""
    from rl_training.curriculum.scenario_generator import ScenarioGenerator

    generator = ScenarioGenerator(seed=seed)
    scenarios = []
    for i in range(count):
        s = generator.generate_scenario()
        s.name = f"{prefix}_{i:02d}"
        scenarios.append(s)
    return scenarios
