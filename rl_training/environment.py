"""
Reinforcement Learning Environment for Artillery Firing Decisions

This module implements an OpenAI Gym-compatible environment for training
autonomous artillery systems to make optimal firing decisions based on
hit probability analysis.
"""

import numpy as np
import concurrent.futures
import gymnasium as gym
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import IntEnum

try:
    from .curriculum.scenario_generator import ScenarioGenerator, ScenarioParameters
    from src.hit_probability import HitProbabilityCalculator
    from src.find_optimal_firing_angles import find_optimal_firing_angles
    from src.kalman_filter import TargetKalmanFilter
    from .infrastructure.training_config import TrainingConstants, RewardConfig
except ImportError:
    # Fallback for standalone execution
    import sys
    import os

    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    # Import from the rl_training package
    from rl_training.curriculum.scenario_generator import ScenarioGenerator, ScenarioParameters
    from src.hit_probability import HitProbabilityCalculator
    from src.find_optimal_firing_angles import find_optimal_firing_angles
    from src.kalman_filter import TargetKalmanFilter
    from rl_training.infrastructure.training_config import TrainingConstants, RewardConfig


class Action(IntEnum):
    """Available actions for the artillery agent."""

    HOLD = 0  # Wait for better conditions
    FIRE = 1  # Fire immediately


@dataclass
class EnvironmentState:
    """Current state of the RL environment."""

    scenario: ScenarioParameters
    time_remaining: float  # seconds
    current_hit_probability: float
    episode_step: int
    initial_hit_probability: float = 0.0  # Track initial hit probability for reward shaping
    covariance_trace: float = 1000.0  # Kalman filter convergence indicator
    hp_history: list = None  # Track HP history for plateau detection
    target_position_est: np.ndarray = None  # Estimated 3D position from Kalman filter
    target_velocity_est: np.ndarray = None  # Estimated 3D velocity from Kalman filter
    initial_range_est: float = None  # Estimated range at episode start

    def __post_init__(self):
        """Initialize hp_history and estimated state if not provided."""
        if self.hp_history is None:
            self.hp_history = []
        if self.target_position_est is None:
            x0 = getattr(self.scenario, 'target_x0', self.scenario.range_m)
            y0 = getattr(self.scenario, 'target_y0', 0.0)
            z0 = getattr(self.scenario, 'target_z0', 50.0)
            self.target_position_est = np.array([x0, y0, z0], dtype=np.float32)
        if self.target_velocity_est is None:
            self.target_velocity_est = np.array([
                self.scenario.target_vx,
                self.scenario.target_vy,
                self.scenario.target_vz
            ], dtype=np.float32)
        if self.initial_range_est is None:
            self.initial_range_est = float(np.linalg.norm(self.target_position_est))

    def to_array(self) -> np.ndarray:
        """Convert current state to neural network input array using Kalman filter estimates."""
        # Use Kalman filter estimated position and velocity
        current_pos = np.asarray(self.target_position_est, dtype=np.float32)
        current_vel = np.asarray(self.target_velocity_est, dtype=np.float32)

        current_range = float(np.linalg.norm(current_pos))
        current_vx = float(current_vel[0])
        current_vy = float(current_vel[1])
        current_vz = float(current_vel[2])

        # Range change from initial estimated position
        range_change = current_range - self.initial_range_est

        # Speed magnitude from estimated velocity
        speed_magnitude = float(np.linalg.norm(current_vel))

        # Engagement geometry from estimated state
        heading_angle = float(np.arctan2(current_vy, current_vx) / np.pi)  # Normalized to [-1, 1]

        # Closure rate: line-of-sight radial velocity component from estimated state
        shooter_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        r_vec = current_pos - shooter_pos
        r_unit = r_vec / (np.linalg.norm(r_vec) + 1e-8)
        radial_velocity = float(np.dot(current_vel, r_unit))
        closure_rate = abs(radial_velocity) / 100.0  # Normalized closure indicator

        # Agent observes estimated state and filter uncertainty to learn timing decisions:
        # - Kalman convergence (when to fire after filter stabilizes)
        # - Range optimization (wait for better engagement geometry)  
        # - Time pressure (don't wait too long)
        return np.array([
            # 1. Range information (from Kalman filter state)
            current_range / 5000.0,  # Current estimated range, normalized [0-5km]
            range_change / 1000.0,    # Range change from initial estimate, normalized

            # 2. Target characteristics (static)
            self.scenario.target_length / 20.0,
            self.scenario.target_width / 20.0,
            self.scenario.target_height / 20.0,

            # 3. Motion state (from Kalman filter estimated velocity)
            (current_vx + 50.0) / 100.0,  # Estimated radial/x velocity
            (current_vy + 50.0) / 100.0,  # Estimated lateral/y velocity
            (current_vz + 50.0) / 100.0,  # Estimated vertical/z velocity
            speed_magnitude / 100.0,      # Estimated speed magnitude

            # 4. Time and convergence
            self.time_remaining / self.scenario.tracking_duration,  # Time remaining
            self.episode_step / 100.0,                             # Step count
            min(1.0, self.covariance_trace / 1000.0),              # Kalman convergence

            # 5. Engagement geometry (from Kalman filter state)
            heading_angle,                    # Estimated target heading angle
            min(1.0, closure_rate),          # Estimated closure rate indicator
        ], dtype=np.float32)


REWARD_MODES = ("legacy", "aligned")
COV_FEATURES = ("linear", "log")
# Position of the covariance feature in EnvironmentState.to_array().
COVARIANCE_FEATURE_INDEX = 11


def covariance_feature(covariance_trace, mode="linear"):
    """Covariance-trace input of the agent (Kalman filter information, not HP).

    "linear" is the published scaling min(1, Tr/1000), which maps the traces
    seen in practice (~0.1-7) to ~0.0001-0.007. "log" maps log10(Tr) from
    [-2, 3] to [0, 1] (clipped), so the same traces span ~0.2-0.57.
    """
    if mode == "linear":
        return float(min(1.0, covariance_trace / 1000.0))
    if mode == "log":
        return float(np.clip((np.log10(max(covariance_trace, 1e-12)) + 2.0) / 5.0, 0.0, 1.0))
    raise ValueError(f"cov_feature must be one of {COV_FEATURES}, got {mode!r}")


TIME_FEATURES = ("steps", "seconds")
# Position of the step-count feature in EnvironmentState.to_array().
TIME_FEATURE_INDEX = 10
# Longest tracking window of the scenario generators (s); scales the "seconds" feature.
ELAPSED_SECONDS_SCALE = 30.0


def time_feature(env_state, mode="steps"):
    """Time input of the agent (no HP).

    "steps" is the published step count episode_step / 100; a step lasts
    window / max_episode_steps, which differs between scenarios. "seconds" is
    the time since tracking started, (window - time_remaining) / 30 s,
    clipped to [0, 1] -- known to any fire-control system.
    """
    if mode == "steps":
        return float(env_state.episode_step / 100.0)
    if mode == "seconds":
        elapsed = env_state.scenario.tracking_duration - env_state.time_remaining
        return float(np.clip(elapsed / ELAPSED_SECONDS_SCALE, 0.0, 1.0))
    raise ValueError(f"time_feature must be one of {TIME_FEATURES}, got {mode!r}")


def extrapolate_filter(kf, lag, scenario):
    """Filter predicted by `lag` seconds with the same constant-velocity and noise model."""
    predicted = TargetKalmanFilter(
        initial_state=kf.state.copy(),
        initial_covariance=kf.covariance.copy(),
        process_noise_std=scenario.process_noise_std,
        measurement_noise_std=scenario.measurement_noise_std,
        dt=lag,
    )
    predicted.predict()
    return predicted


def get_static_scenario():
    """Return a static scenario for overfitting/debugging."""
    from rl_training.curriculum.scenario_generator import ScenarioParameters
    import numpy as np
    return ScenarioParameters(
        range_m=800.0,
        target_length=10.0,
        target_width=3.0,
        target_height=2.5,
        target_vx=0.0,
        target_vy=0.0,
        target_vz=0.0,
        tracking_duration=10.0,
        measurement_noise_std=np.array([1.0, 1.0, 1.0]),
        process_noise_std=np.array([0.1, 0.1, 0.1])
    )

def hp_worker(args):
    env_state, scenario, step, max_episode_steps, calculate_hp_func = args
    # Simulate state at this step
    time_remaining = scenario.tracking_duration * (1.0 - step / max_episode_steps)
    # Temporarily set state for HP calculation
    old_time = env_state.time_remaining
    old_step = env_state.episode_step
    env_state.time_remaining = time_remaining
    env_state.episode_step = step
    hp_value = calculate_hp_func(scenario)
    cov_trace = getattr(env_state, 'covariance_trace', 1000.0)
    pos_est = getattr(env_state, 'target_position_est', None)
    vel_est = getattr(env_state, 'target_velocity_est', None)
    if pos_est is None:
        pos_est = np.array([scenario.range_m, 0.0, 50.0], dtype=np.float32)
    if vel_est is None:
        vel_est = np.array([scenario.target_vx, scenario.target_vy, scenario.target_vz], dtype=np.float32)
    # Restore state
    env_state.time_remaining = old_time
    env_state.episode_step = old_step
    return (step, hp_value, cov_trace, pos_est, vel_est)


class ArtilleryFiringEnv(gym.Env):
    """
    OpenAI Gym environment for artillery firing decision training.

    The agent must decide whether to fire immediately or hold for better
    conditions, maximizing expected reward based on hit probability.
    """

    def __init__(
        self,
        scenario_generator: Optional[ScenarioGenerator] = None,
        hit_probability_calculator: Optional[HitProbabilityCalculator] = None,
        hit_threshold: float = 0.2,
        max_episode_steps: int = 50,
        reward_scale: float = 100.0,
        # Reward thresholds (based on military doctrine)
        hp_excellent_threshold: float = 0.7,  # Excellent firing opportunity
        hp_good_threshold: float = 0.6,       # Good firing opportunity
        hp_minimum_threshold: float = 0.5,    # Below this is poor shooting
        min_engagement_range_m: float = 200.0,  # Minimum effective range
        force_static: bool = False,  # If True, always use static scenario for overfitting/debugging
        debug: bool = False,  # If True, print debugging info during training
        firing_reward_high: float = 100.0,  # Reward for firing at high HP
        time_penalty_factor: float = -100.0,  # Penalty factor for late firing
        hold_penalty_high: float = -10.0,  # Penalty for holding when HP is excellent
        # Additional configurable reward parameters
        reward_excellent: float = 100.0,     # Base reward for excellent HP firing
        reward_good: float = 80.0,           # Base reward for good HP firing
        reward_minimum: float = 60.0,        # Base reward for minimum HP firing
        reward_fair: float = 40.0,           # Base reward for fair HP firing
        reward_poor: float = 20.0,           # Base reward for poor HP firing
        reward_failure: float = -30.0,       # Base reward for failure HP firing
        hold_penalty_base: float = -10.0,    # Base penalty for holding
        opportunity_cost_excellent: float = -30.0,  # Opportunity cost for holding at excellent HP
        opportunity_cost_good: float = -15.0,       # Opportunity cost for holding at good HP
        opportunity_cost_minimum: float = -8.0,     # Opportunity cost for holding at minimum HP
        reward_mode: str = "legacy",  # "legacy" or "aligned" (P_hit - time_cost * t/T)
        time_cost: float = 0.0,      # lambda for reward_mode="aligned"
        cov_feature: str = "linear",  # scaling of the covariance-trace input ("linear" or "log")
        time_feature: str = "steps",  # time input: step count ("steps") or elapsed seconds ("seconds")
        extrapolate_estimate: bool = False,  # predict the filter estimate to the decision time
    ):
        """
        Initialize the artillery firing environment.

        Args:
            scenario_generator: Generator for training scenarios
            hit_probability_calculator: Calculator for hit probabilities
            hit_threshold: Minimum hit probability for successful engagement
            max_episode_steps: Maximum steps per episode
            reward_scale: Scaling factor for rewards
            reward_mode: "legacy" keeps the original shaped reward. "aligned"
                rewards FIRE with P_hit - time_cost * step / max_episode_steps,
                HOLD with 0 and a timeout with -time_cost.
            time_cost: Cost of waiting the whole window under reward_mode="aligned".
            cov_feature: "linear" keeps the published covariance input; "log"
                uses covariance_feature(..., "log"). Neither involves HP.
            time_feature: "steps" keeps the published step-count input;
                "seconds" uses time_feature(..., "seconds"). Neither involves HP.
            extrapolate_estimate: the filter estimate refers to the last
                measurement (every 0.5 s), but the decision is taken up to
                0.5 s later. False (published) aims with the estimate as it
                is; True predicts it to the decision time with the filter's
                own motion and noise model, as a fire-control system would.
        """
        super().__init__()

        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        if time_cost < 0.0:
            raise ValueError(f"time_cost must be non-negative, got {time_cost}")
        self.reward_mode = reward_mode
        self.time_cost = time_cost
        if cov_feature not in COV_FEATURES:
            raise ValueError(f"cov_feature must be one of {COV_FEATURES}, got {cov_feature!r}")
        self.cov_feature = cov_feature
        if time_feature not in TIME_FEATURES:
            raise ValueError(f"time_feature must be one of {TIME_FEATURES}, got {time_feature!r}")
        self.time_feature = time_feature
        self.extrapolate_estimate = bool(extrapolate_estimate)

        # Initialize components
        self.scenario_generator = scenario_generator or ScenarioGenerator()
        self.hit_calculator = hit_probability_calculator or HitProbabilityCalculator(
            projectile_velocity=800.0
        )

        # Environment parameters
        self.hit_threshold = hit_threshold
        self.base_max_episode_steps = max_episode_steps  # Base value, can be adjusted per scenario
        self.max_episode_steps = max_episode_steps  # Will be set dynamically per scenario
        self.reward_scale = reward_scale

        # Reward structure parameters (configurable for tuning)
        self.hp_excellent_threshold = hp_excellent_threshold
        self.hp_good_threshold = hp_good_threshold
        self.hp_minimum_threshold = hp_minimum_threshold
        self.min_engagement_range_m = min_engagement_range_m

        # Configurable reward parameters
        self.firing_reward_high = firing_reward_high
        self.time_penalty_factor = time_penalty_factor
        self.hold_penalty_high = hold_penalty_high
        
        # Additional configurable reward values
        self.reward_excellent = reward_excellent
        self.reward_good = reward_good
        self.reward_minimum = reward_minimum
        self.reward_fair = reward_fair
        self.reward_poor = reward_poor
        self.reward_failure = reward_failure
        self.hold_penalty_base = hold_penalty_base
        self.opportunity_cost_excellent = opportunity_cost_excellent
        self.opportunity_cost_good = opportunity_cost_good
        self.opportunity_cost_minimum = opportunity_cost_minimum

        # Option to force static scenario for overfitting/debugging
        self.force_static = force_static
        self.debug = debug

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(Action))

        # Enhanced observation space - 14 features for better decision making
        # Agent now observes current state, not just initial conditions
        self.observation_space = gym.spaces.Box(
            low=np.zeros(14, dtype=np.float32),
            high=np.ones(14, dtype=np.float32),
            dtype=np.float32,
        )

        # Initialize episode state
        self.current_state: Optional[EnvironmentState] = None
        self.episode_reward = 0.0
        self.episode_length = 0
        self.previous_hit_probability = 0.0  # Track for immediate improvement rewards
        self.hp_cache = {}  # Global cache for HP trajectories
        
        # HP plateau detection parameters
        self.hp_improvement_threshold = 0.002  # HP improvement rate threshold (per step)
        self.hp_history_length = 10  # Number of steps to track for plateau detection
        self.plateau_fire_bonus = 50.0  # Bonus for firing when HP plateaued
        self.plateau_hold_multiplier = 5.0  # Multiply HOLD penalty when plateaued

    def _get_cache_key(self, scenario, seed=None):
        """Generate a deterministic cache key for HP trajectory caching."""
        import hashlib
        import json
        key_data = {
            "range_m": scenario.range_m,
            "vx": scenario.target_vx, "vy": scenario.target_vy, "vz": scenario.target_vz,
            "size": [scenario.target_length, scenario.target_width, scenario.target_height],
            "tracking_duration": scenario.tracking_duration,
            "meas_noise": scenario.measurement_noise_std.tolist(),
            "proc_noise": scenario.process_noise_std.tolist(),
            "seed": seed,  # Include seed so different runs get different trajectories
        }
        if self.extrapolate_estimate:  # keys of the published setting stay unchanged
            key_data["extrapolate_estimate"] = True
        key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        return key

    def _calculate_max_steps(self, scenario) -> int:
        """
        Calculate dynamic max_episode_steps based on scenario characteristics.
        Longer episodes for longer ranges to allow HP convergence.
        Args:
            scenario: ScenarioParameters object
        Returns:
            Maximum episode steps for this scenario
        """
        # Force mini-episodes for Fast_Approach_Long, but give more time for Very_Fast_Approach
        # Fast_Approach_Long: range=3000m, velocity=(-50, 0, 0)
        # Very_Fast_Approach: range=4000m, velocity=(-80, 0, 0)
        if (abs(getattr(scenario, 'range_m', 0) - 3000.0) < 200 and getattr(scenario, 'target_vx', 0) < -40):
            return 25  # Mini-episode for Fast_Approach_Long
        elif (abs(getattr(scenario, 'range_m', 0) - 4000.0) < 200 and getattr(scenario, 'target_vx', 0) < -70):
            return 40  # Give more time for Very_Fast_Approach
        elif scenario.range_m > 3000:
            return 100
        elif scenario.range_m > 2500:
            return 90
        else:
            return 40

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None, scenario_override=None, force_static: Optional[bool] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to start a new episode.
        If force_static is True (either from argument or instance), always use static scenario for overfitting/debugging.
        Otherwise, use scenario_override or scenario_generator.
        Args:
            seed: Random seed
            options: Additional options
            scenario_override: If provided, use this scenario instead of generating a new one
            force_static: If True, always use static scenario (overrides instance default)
        Returns:
            observation: Initial state as numpy array
            info: Additional information
        """
        super().reset(seed=seed)
        # Always reset measurement history and filter state for Kalman filter
        self.measurement_history = []
        self.last_measurement_time = -0.5  # dt is 0.5 in _calculate_hit_probability
        # If you add more filter state variables, reset them here as well
        # Determine static scenario usage
        use_static = force_static if force_static is not None else self.force_static
        if use_static:
            scenario = get_static_scenario()
        elif scenario_override is not None:
            scenario = scenario_override
        else:
            scenario = self.scenario_generator.generate_scenario()
        # Set dynamic max_episode_steps based on scenario characteristics
        self.max_episode_steps = self._calculate_max_steps(scenario)
        # Initialize state with default covariance_trace (will be updated during HP calc)
        self.current_state = EnvironmentState(
            scenario=scenario,
            time_remaining=scenario.tracking_duration,
            current_hit_probability=0.0,  # Will be updated
            episode_step=0,
            initial_hit_probability=0.0,  # Will be updated
            covariance_trace=1000.0,  # Default high uncertainty
        )
        # PRE-CALCULATE HP TRAJECTORY - sparse sampling to reduce computation
        # Calculate HP every 5 steps instead of every step (10x speedup!)
        # Cache trajectories for identical scenarios to avoid recomputation
        cache_key = self._get_cache_key(scenario, seed=seed)
        if cache_key in self.hp_cache:
            cache_entry = self.hp_cache[cache_key]
            if len(cache_entry) == 4:
                self.hp_trajectory, self.cov_trajectory, self.pos_trajectory, self.vel_trajectory = cache_entry
            else:
                self.hp_trajectory, self.cov_trajectory = cache_entry[:2]
                self.pos_trajectory = {s: np.array([scenario.range_m, 0.0, 50.0], dtype=np.float32) for s in self.hp_trajectory}
                self.vel_trajectory = {s: np.array([scenario.target_vx, scenario.target_vy, scenario.target_vz], dtype=np.float32) for s in self.hp_trajectory}
        else:
            # Calculate new trajectory in parallel
            self.hp_trajectory = {}  # {step: hp_value}
            self.cov_trajectory = {}  # {step: cov_trace}
            self.pos_trajectory = {}  # {step: target_position_est}
            self.vel_trajectory = {}  # {step: target_velocity_est}
            hp_sample_interval = 1  # Calculate every step for accuracy
            steps = list(range(0, self.max_episode_steps + 1, hp_sample_interval))

            # Prepare arguments for top-level hp_worker
            worker_args = [
                (self.current_state, scenario, step, self.max_episode_steps, self._calculate_hit_probability)
                for step in steps
            ]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(hp_worker, worker_args))
            for step, hp_value, cov_trace, pos_est, vel_est in results:
                self.hp_trajectory[step] = hp_value
                self.cov_trajectory[step] = cov_trace
                self.pos_trajectory[step] = pos_est
                self.vel_trajectory[step] = vel_est
            # Cache for future use
            self.hp_cache[cache_key] = (
                self.hp_trajectory.copy(),
                self.cov_trajectory.copy(),
                self.pos_trajectory.copy(),
                self.vel_trajectory.copy(),
            )
            print(f"Cache MISS -> computed {scenario.name}")
        # Set initial values
        initial_hit_prob = self.hp_trajectory[0]
        self.current_state.current_hit_probability = initial_hit_prob
        self.current_state.initial_hit_probability = initial_hit_prob
        self.current_state.covariance_trace = self.cov_trajectory[0]
        if hasattr(self, 'pos_trajectory') and 0 in self.pos_trajectory:
            self.current_state.target_position_est = self.pos_trajectory[0]
            self.current_state.initial_range_est = float(np.linalg.norm(self.pos_trajectory[0]))
        if hasattr(self, 'vel_trajectory') and 0 in self.vel_trajectory:
            self.current_state.target_velocity_est = self.vel_trajectory[0]
        self.current_state.hp_history = [initial_hit_prob]  # Initialize HP history
        self.episode_reward = 0.0
        self.episode_length = 0
        self.previous_hit_probability = initial_hit_prob  # Track for improvement rewards

        # Log HP trajectory for debugging
        if self.debug:
            print("[DEBUG] HP trajectory for episode:")
            for step in sorted(self.hp_trajectory.keys()):
                print(f"  Step {step}: HP={self.hp_trajectory[step]:.3f}")

        observation = self._observe()
        info = {
            "scenario": scenario,
            "initial_hit_probability": initial_hit_prob,
        }
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Reward shaping for better learning:
        - HOLD: Reward for HP improvement, small penalty for waiting.
        - FIRE: Reward is HP, bonus for high HP, penalty for firing too early.
        """
        if self.current_state is None:
            raise RuntimeError("Environment must be reset before calling step()")

        terminated = False
        truncated = False
        reward = 0.0

        # Track previous HP for improvement reward
        prev_hp = getattr(self, "_prev_hp", None)
        if prev_hp is None:
            prev_hp = self.current_state.current_hit_probability

        # Track if HP was ever high during episode
        if not hasattr(self, "_hp_high_reached"):
            self._hp_high_reached = False
        if self.current_state.current_hit_probability >= 0.6:
            self._hp_high_reached = True

        if self.debug:
            print(f"\n[DEBUG] Step {self.current_state.episode_step} | Action: {action} | Prev HP: {prev_hp:.3f}")

        if action == Action.FIRE:
            reward, terminated = self._handle_fire()
        elif action == Action.HOLD:
            reward, terminated = self._handle_hold(prev_hp)
        else:
            raise ValueError(f"Invalid action: {action}")

        self._prev_hp = self.current_state.current_hit_probability
        self.episode_length += 1
        if self.episode_length >= self.max_episode_steps:
            truncated = True
        if self.reward_mode == "aligned":
            reward = self._aligned_reward(action, terminated or truncated)
        self.episode_reward += reward

        observation = self._observe()
        info = {
            "hit_probability": self.current_state.current_hit_probability,
            "time_remaining": self.current_state.time_remaining,
            "episode_step": self.current_state.episode_step,
            "covariance_trace": getattr(self.current_state, 'covariance_trace', None),
            "reward": reward,
        }
        if self.debug:
            print(f"[DEBUG] Step {self.current_state.episode_step}: Action={action}, HP={self.current_state.current_hit_probability:.3f}, Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}")
            print(f"[DEBUG] Episode cumulative reward: {self.episode_reward:.2f}, Episode length: {self.episode_length}")
            if terminated:
                print(f"[DEBUG] Episode Summary: Step={self.current_state.episode_step}, HP={self.current_state.current_hit_probability:.3f}, Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}")
        return observation, reward, terminated, truncated, info

    def _observe(self) -> np.ndarray:
        """Agent observation: the 14 target features, with the chosen covariance and time inputs."""
        observation = self.current_state.to_array()
        if self.cov_feature != "linear":
            observation[COVARIANCE_FEATURE_INDEX] = covariance_feature(
                self.current_state.covariance_trace, self.cov_feature)
        if self.time_feature != "steps":
            observation[TIME_FEATURE_INDEX] = time_feature(self.current_state, self.time_feature)
        return observation

    def _aligned_reward(self, action: int, done: bool) -> float:
        """Reward for reward_mode="aligned".

        FIRE: P_hit at the firing step minus time_cost * step / max_episode_steps.
        HOLD: 0, or -time_cost if the episode ends without firing (P_hit 0 at t = T).
        """
        if action == Action.FIRE:
            elapsed = self.current_state.episode_step / self.max_episode_steps
            return float(self.current_state.current_hit_probability - self.time_cost * elapsed)
        return -self.time_cost if done else 0.0

    def _get_hp_for_step(self, step: int) -> float:
        """Get hit probability for given step using pre-calculated trajectory."""
        if not hasattr(self, 'hp_trajectory'):
            return 0.0

        if step in self.hp_trajectory:
            return self.hp_trajectory[step]

        # Step not in trajectory — apply Kalman convergence factor
        if self.current_state is not None:
            scenario = self.current_state.scenario
            elapsed_time = (step / self.max_episode_steps) * scenario.tracking_duration
        else:
            elapsed_time = step * 0.2

        base_hp = self.hp_trajectory.get(0, 0.05)
        convergence_factor = min(1.0, elapsed_time / 5.0)
        return base_hp * (0.3 + 0.7 * convergence_factor)
    
    def _handle_fire(self) -> Tuple[float, bool]:
        """Execute FIRE action: update state, compute reward, terminate episode."""
        hp = self.hp_trajectory[self.current_state.episode_step]
        self.current_state.current_hit_probability = hp
        self.current_state.covariance_trace = self.cov_trajectory[self.current_state.episode_step]
        if hasattr(self, 'pos_trajectory') and self.current_state.episode_step in self.pos_trajectory:
            self.current_state.target_position_est = self.pos_trajectory[self.current_state.episode_step]
        if hasattr(self, 'vel_trajectory') and self.current_state.episode_step in self.vel_trajectory:
            self.current_state.target_velocity_est = self.vel_trajectory[self.current_state.episode_step]
        self.current_state.hp_history.append(hp)
        reward = self._calculate_firing_reward(hp, self.current_state.episode_step)
        if self.debug:
            print(f"[DEBUG] FIRE at step {self.current_state.episode_step}: HP={hp:.3f}, Reward={reward:.2f}")
            print(f"[DEBUG] Episode terminated by FIRE action.")
        return reward, True

    def _handle_hold(self, prev_hp: float) -> Tuple[float, bool]:
        """Execute HOLD action: advance time, update state, compute reward."""
        time_step = self.current_state.scenario.tracking_duration / self.max_episode_steps
        self.current_state.time_remaining -= time_step
        self.current_state.episode_step += 1
        hp = self.hp_trajectory[self.current_state.episode_step]
        self.current_state.current_hit_probability = hp
        self.current_state.covariance_trace = self.cov_trajectory[self.current_state.episode_step]
        if hasattr(self, 'pos_trajectory') and self.current_state.episode_step in self.pos_trajectory:
            self.current_state.target_position_est = self.pos_trajectory[self.current_state.episode_step]
        if hasattr(self, 'vel_trajectory') and self.current_state.episode_step in self.vel_trajectory:
            self.current_state.target_velocity_est = self.vel_trajectory[self.current_state.episode_step]
        self.current_state.hp_history.append(hp)
        if len(self.current_state.hp_history) > self.hp_history_length:
            self.current_state.hp_history.pop(0)

        hp_improvement = hp - prev_hp
        if hp < 0.5:
            reward = hp_improvement * 5.0 - 0.1
        elif hp < 0.6:
            reward = -0.5
        elif hp < 0.7:
            reward = -2.0
        else:
            reward = self.hold_penalty_high * 0.5

        if self._is_hp_plateau() and hp >= self.hp_minimum_threshold:
            reward *= self.plateau_hold_multiplier
            if self.debug:
                print(f"[DEBUG] HP plateau detected at step {self.current_state.episode_step}, HOLD penalty multiplied by {self.plateau_hold_multiplier}")
        if self.debug:
            print(f"[DEBUG] HOLD: Step={self.current_state.episode_step}, HP={hp:.3f}, HP_Improvement={hp_improvement:.3f}, Reward={reward:.2f}")

        terminated = False
        if self.current_state.time_remaining <= 0 or self.current_state.episode_step >= self.max_episode_steps:
            terminated = True
            if self.debug:
                print(f"[DEBUG] Episode ended by timeout (no FIRE). time_remaining={self.current_state.time_remaining:.2f}, episode_step={self.current_state.episode_step}")
            if self._hp_high_reached:
                reward -= 40.0
                if self.debug:
                    print(f"[DEBUG] Timeout penalty applied: HP was high but agent never fired.")
        return reward, terminated

    def _calculate_hit_probability(self, scenario: ScenarioParameters, debug: bool = False):
        """
        Calculate current hit probability for a scenario using full pipeline:
        1. Generate noisy trajectory measurements
        2. Apply Kalman filtering to estimate target state
        3. Find optimal firing angles
        4. Error propagation with full Jacobian
        5. Analytical hit probability calculation

        Args:
            scenario: Current scenario parameters

        Returns:
            Hit probability [0, 1]
        """
        # Ground truth target state
        shooter_pos = np.array([0.0, 0.0, 0.0])
        target_position_true = np.array(
            [scenario.range_m, 0.0, 50.0]
        )
        target_velocity_true = np.array(
            [scenario.target_vx, scenario.target_vy, scenario.target_vz]
        )

        # STEP 1 & 2: Accumulate noisy measurements over time for smoother HP evolution
        elapsed_time = scenario.tracking_duration - self.current_state.time_remaining if self.current_state else 0.0
        dt = 0.5  # 2.0 Hz measurement rate

        # Accumulate measurements as time progresses
        if not hasattr(self, 'measurement_history') or self.current_state.episode_step == 0:
            self.measurement_history = []
            self.last_measurement_time = -dt

        current_measurement_time = self.last_measurement_time + dt
        while current_measurement_time <= elapsed_time:
            pos_true = target_position_true + target_velocity_true * current_measurement_time
            noise = self.np_random.multivariate_normal(
                np.zeros(3), np.diag(scenario.measurement_noise_std ** 2)
            ) if hasattr(self, 'np_random') else np.random.multivariate_normal(
                np.zeros(3), np.diag(scenario.measurement_noise_std ** 2)
            )
            measurement = pos_true + noise
            self.measurement_history.append((current_measurement_time, measurement))
            self.last_measurement_time = current_measurement_time
            current_measurement_time += dt

        measurements = [m[1] for m in self.measurement_history]
        if self.debug:
            print(f"[DEBUG] Step {self.current_state.episode_step}: Num measurements = {len(measurements)}, Covariance trace (pre-KF) = {getattr(self.current_state, 'covariance_trace', None)}")
        if len(measurements) < 2:
            # Always require at least 2 measurements for Kalman filter
            measurements = [target_position_true, target_position_true]

        # Initialize Kalman filter with first measurement
        initial_state = np.concatenate([measurements[0], np.zeros(3)])  # [x,y,z,vx,vy,vz]
        initial_covariance = np.diag([25.0, 25.0, 25.0, 2.0, 2.0, 2.0])  # Reduced initial uncertainty

        kf = TargetKalmanFilter(
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            process_noise_std=scenario.process_noise_std,
            measurement_noise_std=scenario.measurement_noise_std,
            dt=dt
        )

        # Process remaining measurements
        for i in range(1, len(measurements)):
            kf.predict()
            kf.update(measurements[i])

        # The estimate refers to the last measurement; optionally predict it to the decision time.
        estimate_lag = 0.0
        if self.extrapolate_estimate and len(self.measurement_history) >= 2:
            estimate_lag = max(0.0, elapsed_time - self.measurement_history[-1][0])
            if estimate_lag > 1e-9:
                kf = extrapolate_filter(kf, estimate_lag, scenario)

        # Use Kalman-filtered estimates
        target_position = kf.state[:3].copy()
        target_velocity = kf.state[3:].copy()
        target_cov = kf.covariance.copy()  # 6x6 covariance from Kalman filter
        
        # Calculate covariance trace as convergence indicator
        # Trace = sum of diagonal elements = total variance
        # Decreases as filter converges with more measurements
        covariance_trace = np.trace(kf.covariance)
        if self.debug:
            print(f"[DEBUG] Step {self.current_state.episode_step}: Covariance trace (post-KF) = {covariance_trace}")
        
        # Store for observation (will be added to state)
        if self.current_state is not None:
            self.current_state.covariance_trace = covariance_trace
            self.current_state.target_position_est = target_position.copy()
            self.current_state.target_velocity_est = target_velocity.copy()

        # Collect debug info
        debug_info = {
            "measurements": np.array(measurements),
            "kalman_state": np.array(kf.state),
            "kalman_covariance": np.array(kf.covariance),
            "covariance_trace": covariance_trace,
            "target_position": np.array(target_position),
            "target_velocity": np.array(target_velocity),
            "target_covariance": np.array(target_cov),
            "elapsed_time": elapsed_time,
            "num_measurements": len(measurements),
            "dt": dt,
            "estimate_lag_s": estimate_lag,
        }
        
        # STEP 3: Find optimal firing angles with robust error handling
        try:
            elev, azim, impact_time, intersection_point, min_dist = (
                find_optimal_firing_angles(
                    shooter_pos,
                    target_position,
                    target_velocity,
                    800.0,
                    "tpt",
                    30.0,
                )
            )

            # Check for invalid results
            if np.isnan(elev) or np.isnan(azim) or np.isnan(impact_time):
                if debug:
                    debug_info["firing_angles"] = None
                    debug_info["firing_error"] = "Invalid angles"
                    return (0.05, debug_info)
                return 0.05
            if impact_time <= 0 or min_dist > 100:  # Unrealistic impact time or distance
                if debug:
                    debug_info["firing_angles"] = None
                    debug_info["firing_error"] = "Unrealistic impact/min_dist"
                    return (0.05, debug_info)
                return 0.05

            debug_info["firing_angles"] = (elev, azim, impact_time, intersection_point, min_dist)

        except (ValueError, RuntimeError, FloatingPointError) as e:
            if debug:
                debug_info["firing_angles"] = None
                debug_info["firing_error"] = str(e)
                return (0.05, debug_info)
            return 0.05
        except Exception as e:
            if debug:
                debug_info["firing_angles"] = None
                debug_info["firing_error"] = str(e)
                return (0.05, debug_info)
            return 0.05

        # STEP 4: Projectile covariances for error propagation
        projectile_velocity_std = 2.5  # m/s
        base_angle_std = 2e-4  # radians (optimistic for modern precision artillery)
        reference_range = 1000.0
        angle_std = base_angle_std * (scenario.range_m / reference_range) ** 0.2

        projectile_cov = np.diag(
            [
                projectile_velocity_std**2,
                angle_std**2,
                angle_std**2,
            ]
        )

        debug_info["projectile_covariance"] = np.array(projectile_cov)

        # Update hit calculator with scenario-specific target dimensions
        self.hit_calculator.target_dimensions = (
            scenario.target_length,
            scenario.target_width,
            scenario.target_height,
        )

        # STEP 5: Analytical hit probability with full Jacobian error propagation
        try:
            details = self.hit_calculator.calculate_hit_probability_analytical(
                target_position=target_position,
                target_velocity=target_velocity,
                projectile_param_cov=projectile_cov,
                target_param_cov=target_cov,
                elevation_angle=elev,
                azimuth_angle=azim,
                use_diagonal_approx=True,
                mc_refine_samples=0,
                impact_time=impact_time,
                intersection_point=intersection_point,
                return_details=True,
            )
            base_hp = float(details["hit_probability"])

            # CRITICAL: Make HP depend on elapsed time (Kalman filter convergence)
            convergence_factor = min(1.0, elapsed_time / 5.0)  # HP improves over first 5 seconds
            hit_prob = base_hp * (0.3 + 0.7 * convergence_factor)

            debug_info["hp_details"] = details
            debug_info["hit_probability"] = hit_prob

            # Sanity check the result
            if np.isnan(hit_prob) or hit_prob < 0 or hit_prob > 1:
                if debug:
                    debug_info["hp_error"] = "NaN or out of bounds"
                    return (0.05, debug_info)
                return 0.05

            if debug:
                return (hit_prob, debug_info)
            return hit_prob
        except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError) as e:
            if debug:
                debug_info["hp_error"] = str(e)
                return (0.05, debug_info)
            return 0.05
        except Exception as e:
            if debug:
                debug_info["hp_error"] = str(e)
                return (0.05, debug_info)
            return 0.05

    def _get_expected_max_hp(self, range_m: float) -> float:
        """
        Estimate maximum achievable HP at given range based on ballistic physics.
        Error propagation increases with range, limiting achievable HP.
        
        Args:
            range_m: Engagement range in meters
            
        Returns:
            Expected maximum achievable HP at this range
        """
        if range_m < 1000:
            return 0.95  # Close range: excellent HP possible
        elif range_m < 2000:
            return 0.75  # Medium-long range: good HP possible
        elif range_m < 3000:
            return 0.55  # Long range: moderate HP achievable
        elif range_m < 4000:
            return 0.35  # Very long range: limited HP
        else:
            return 0.20  # Extreme range: minimal HP
    
    def _calculate_firing_reward(self, hit_probability: float, episode_step: int) -> float:
        """
        Calculate reward for firing decision with STRONG time-based penalties.
        Includes scenario-specific modifiers for problematic fast-approaching scenarios.
        Also includes HP plateau bonus to encourage firing when HP won't improve further.
        """
        # HP-based reward (using configurable values)
        if hit_probability >= self.hp_excellent_threshold:
            base_reward = self.reward_excellent
        elif hit_probability >= self.hp_good_threshold:
            base_reward = self.reward_good
        elif hit_probability >= self.hp_minimum_threshold:
            base_reward = self.reward_minimum
        elif hit_probability >= TrainingConstants.HP_FAIR_THRESHOLD:
            base_reward = self.reward_fair
        elif hit_probability >= TrainingConstants.HP_ACCEPTABLE_THRESHOLD:
            base_reward = self.reward_poor
        else:
            base_reward = self.reward_failure

        # CRITICAL FIX: Strong progressive time penalty that makes waiting VERY expensive
        max_steps = self.max_episode_steps
        time_penalty = self.time_penalty_factor * (episode_step / max_steps)  # Strong penalty: -50 at final step

        # Time efficiency bonus (reward for firing early)
        time_efficiency = TrainingConstants.TIME_EFFICIENCY_BASE * (1.0 - episode_step / max_steps)  # +30 for firing immediately

        # Late firing penalty (extra penalty for firing after step 30)
        late_penalty = 0.0
        if episode_step > 30:
            late_penalty = -20.0 * ((episode_step - 30) / (max_steps - 30))  # Extra penalty for late firing

        # SCENARIO-SPECIFIC MODIFIERS for problematic fast-approaching scenarios
        scenario_bonus = 0.0
        if self._is_fast_approaching_scenario():
            # Extreme incentives for Fast_Approach_Long and Very_Fast_Approach
            if hit_probability >= 0.3:
                # Huge bonus for firing with any HP above 0.3
                scenario_bonus = 500.0 * (hit_probability - 0.3)  # Up to +350 bonus for HP=1.0
            if episode_step > 40:
                # Massive penalty for firing late
                scenario_bonus -= 500.0 * ((episode_step - 40) / (max_steps - 40))  # Up to -500 penalty
        
        # HP PLATEAU BONUS: Encourage firing when HP has plateaued and is acceptable
        plateau_bonus = 0.0
        if self._is_hp_plateau() and hit_probability >= self.hp_minimum_threshold:
            plateau_bonus = self.plateau_fire_bonus
            if self.debug:
                print(f"[DEBUG] HP plateau detected, adding bonus {plateau_bonus} to FIRE reward")

        total_reward = base_reward + time_efficiency + time_penalty + late_penalty + scenario_bonus + plateau_bonus

        # Suppress FIRE_REWARD_TIMING printout to avoid unnecessary console output
        # (Uncomment for debugging if needed)
        # if TrainingConstants.DEBUG:
        #     print(f"FIRE_REWARD_TIMING: HP={hit_probability:.3f}, Step={episode_step}, "
        #           f"Base={base_reward:.1f}, TimeEff={time_efficiency:.1f}, "
        #           f"TimePenalty={time_penalty:.1f}, LatePenalty={late_penalty:.1f}, "
        #           f"ScenarioBonus={scenario_bonus:.1f}, Total={total_reward:.1f}")

        return total_reward

    def _is_hp_plateau(self) -> bool:
        """
        Detect if hit probability has plateaued (not improving significantly).
        
        Returns True if:
        1. Have enough history (at least hp_history_length steps)
        2. Recent HP improvement rate is below threshold
        3. HP is above minimum threshold (otherwise still waiting for convergence)
        
        Returns:
            bool: True if HP has plateaued
        """
        if self.current_state is None:
            return False
        
        hp_history = self.current_state.hp_history
        if len(hp_history) < self.hp_history_length:
            return False
        
        # Calculate improvement rate over last hp_history_length steps
        recent_hp = np.mean(hp_history[-5:])  # Average of last 5 steps
        older_hp = np.mean(hp_history[-self.hp_history_length:-5])  # Average of older 5 steps
        improvement_rate = (recent_hp - older_hp) / 5.0  # Per-step improvement
        
        current_hp = self.current_state.current_hit_probability
        
        # Plateau = low improvement AND reasonable HP
        is_plateau = (improvement_rate < self.hp_improvement_threshold and 
                     current_hp >= self.hp_minimum_threshold * 0.8)  # 80% of minimum
        
        if self.debug and is_plateau:
            print(f"[DEBUG] HP plateau detected: improvement_rate={improvement_rate:.4f}, threshold={self.hp_improvement_threshold}")
        
        return is_plateau
    
    def _is_fast_approaching_scenario(self) -> bool:
        """
        Identify scenarios that need special firing incentives.
        Specifically targets Fast_Approach_Long and Very_Fast_Approach scenarios.
        """
        if self.current_state is None:
            return False

        scenario = self.current_state.scenario


        # Fast_Approach_Long: range=3000m, velocity=(-50, 0, 0)
        if (abs(scenario.range_m - 3000.0) < 200 and  # Range around 3000m (±200m to handle perturbations)
            scenario.target_vx < -40 and  # Fast approaching (negative velocity > 40 m/s)
            abs(scenario.target_vy) < 5 and  # Minimal lateral movement
            np.all(scenario.measurement_noise_std <= 0.3)):
            return True

        # Very_Fast_Approach: range=4000m, velocity=(-80, 0, 0)
        if (abs(scenario.range_m - 4000.0) < 200 and  # Range around 4000m (±200m to handle perturbations)
            scenario.target_vx < -70 and  # Very fast approaching (negative velocity > 70 m/s)
            abs(scenario.target_vy) < 5 and  # Minimal lateral movement
            np.all(scenario.measurement_noise_std <= 0.3)):
            return True

        return False

    def render(self, mode: str = "human") -> None:
        """Render the current environment state."""
        if self.current_state is None:
            print("Environment not initialized")
            return

        state = self.current_state
        print(f"Range: {state.scenario.range_m:.1f}m")
        print(
            f"Target size: {state.scenario.target_length:.1f} x {state.scenario.target_width:.1f} x {state.scenario.target_height:.1f}m"
        )
        print(
            f"Target velocity: ({state.scenario.target_vx:.1f}, {state.scenario.target_vy:.1f}, {state.scenario.target_vz:.1f}) m/s"
        )
        print(f"Time remaining: {state.time_remaining:.1f}s")
        print(f"Hit probability: {state.current_hit_probability:.3f}")
        print(f"Episode step: {state.episode_step}")

    def close(self) -> None:
        """Clean up environment resources."""
        pass
