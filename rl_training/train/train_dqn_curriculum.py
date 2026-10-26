import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
from typing import Tuple, Dict, List
import time
from pathlib import Path

try:
    from ..environment import ArtilleryFiringEnv, Action
    from ..curriculum.curriculum_scenarios import CurriculumScenarios
    from ..infrastructure.training_config import TrainingConstants, RewardConfig
    from ..agents.dqn_components import DQN, ReplayBuffer, NStepReplayBuffer, build_q_network, save_model_config
    from ..utils.evaluation_utils import evaluate_on_scenario, evaluate_curriculum, check_phase_completion
    from ..utils.training_monitor import TrainingMonitor
except ImportError:
    # Fallback for standalone execution
    import sys
    import os

    # Add project root directory to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    # Import from the rl_training package
    from rl_training.environment import ArtilleryFiringEnv, Action
    from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
    from rl_training.infrastructure.training_config import TrainingConstants, RewardConfig
    from rl_training.agents.dqn_components import DQN, ReplayBuffer, NStepReplayBuffer, build_q_network, save_model_config
    from rl_training.utils.evaluation_utils import evaluate_on_scenario, evaluate_curriculum, check_phase_completion
    from rl_training.utils.training_monitor import TrainingMonitor


def set_seed(seed: int = 42):
    """Set global random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_action(state, q_network, epsilon, action_dim, device):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()


def _dqn_update_step(q_network, target_network, optimizer, replay_buffer, batch_size, gamma, device):
    """Sample a minibatch and perform one DQN gradient update. No-op if buffer is too small."""
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_t = torch.FloatTensor(rewards).to(device)
    next_states_t = torch.FloatTensor(next_states).to(device)
    dones_t = torch.FloatTensor(dones).to(device)

    current_q = q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_network(next_states_t).max(1)[0]
        target_q = rewards_t + gamma * next_q * (1 - dones_t)

    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _extended_update_step(q_network, target_network, optimizer, replay_buffer, batch_size, device,
                          double_dqn=False, aux_hp_weight=0.0):
    """One update from an NStepReplayBuffer: n-step targets, optional Double DQN
    target and optional auxiliary HP-prediction loss. Returns the loss, or None
    if the buffer is too small."""
    if len(replay_buffer) < batch_size:
        return None
    states, actions, returns, next_states, dones, discounts, hp_labels = replay_buffer.sample(batch_size)

    states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    next_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)
    discounts_t = torch.as_tensor(discounts, dtype=torch.float32, device=device)

    current_q = q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if double_dqn:
            best = q_network(next_t).argmax(1, keepdim=True)
            next_q = target_network(next_t).gather(1, best).squeeze(1)
        else:
            next_q = target_network(next_t).max(1)[0]
        target_q = returns_t + discounts_t * next_q * (1 - dones_t)

    loss = nn.MSELoss()(current_q, target_q)
    if aux_hp_weight > 0:
        hp_t = torch.as_tensor(hp_labels, dtype=torch.float32, device=device)
        loss = loss + aux_hp_weight * nn.MSELoss()(q_network.predict_aux(states_t), hp_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


CHECKPOINT_DIR = Path(__file__).parent.parent / "models" / "checkpoints"
DEFAULT_CHECKPOINT_PREFIX = "dqn_curriculum"


def checkpoint_path(prefix: str = DEFAULT_CHECKPOINT_PREFIX, phase: int = 1,
                    save_dir=CHECKPOINT_DIR) -> Path:
    """Checkpoint file for a curriculum phase: <save_dir>/<prefix>_phase<phase>.pth."""
    return Path(save_dir) / f"{prefix}_phase{phase}.pth"


def reset_episode(env, scenario, sampler=None):
    """Reset env for a curriculum scenario; with a pool sampler, use a random pool variant."""
    if sampler is None:
        return env.reset(scenario_override=scenario)
    variant, seed = sampler.sample(scenario)
    return env.reset(scenario_override=variant, seed=seed)


def train_curriculum_phase(
    phase: int,
    total_timesteps: int = 10000,
    eval_interval: int = TrainingConstants.EVAL_INTERVAL,
    learning_rate: float = TrainingConstants.LEARNING_RATE,
    gamma: float = TrainingConstants.GAMMA,
    epsilon_start: float = TrainingConstants.EPSILON_START,
    epsilon_end: float = TrainingConstants.EPSILON_END,
    epsilon_decay_steps: int = TrainingConstants.EPSILON_DECAY_STEPS,
    batch_size: int = TrainingConstants.BATCH_SIZE,
    buffer_size: int = TrainingConstants.BUFFER_SIZE,
    target_update_freq: int = TrainingConstants.TARGET_UPDATE_FREQ,
    learning_starts: int = TrainingConstants.LEARNING_STARTS,
    device: str = "cpu",
    load_from: str = None,
    reward_config: RewardConfig = None,
    seed: int = None,
    # Individual reward params kept for backward compatibility
    firing_reward_high: float = None,
    time_penalty_factor: float = None,
    hold_penalty_high: float = None,
    reward_excellent: float = None,
    reward_good: float = None,
    reward_minimum: float = None,
    reward_fair: float = None,
    reward_poor: float = None,
    reward_failure: float = None,
    hold_penalty_base: float = None,
    opportunity_cost_excellent: float = None,
    opportunity_cost_good: float = None,
    opportunity_cost_minimum: float = None,
    env_options: dict = None,
    checkpoint_prefix: str = DEFAULT_CHECKPOINT_PREFIX,
    save_dir=CHECKPOINT_DIR,
    overwrite: bool = False,
    trajectory_pool: dict = None,
    model_config: dict = None,
    double_dqn: bool = False,
    n_step: int = 1,
    aux_hp_weight: float = 0.0,
):
    """Train DQN on a curriculum phase.

    env_options are passed to ArtilleryFiringEnv (e.g. reward_mode, time_cost). The model is saved to checkpoint_path(checkpoint_prefix, phase,
    save_dir); an existing file is only replaced when overwrite=True, and the
    check happens before training starts.

    trajectory_pool (from trajectory_pool.load_pool) makes every training
    episode use a random pre-computed perturbed variant of the curriculum
    scenario; the variant is drawn with a generator seeded from (seed, phase).
    Phase-mastery evaluation keeps using the nominal scenarios.

    model_config (architecture, aux_head), double_dqn, n_step and
    aux_hp_weight select the extended training path (n-step replay with HP
    labels, optional Double DQN target and auxiliary HP loss). With their
    defaults the original DQN, replay buffer and update are used unchanged.
    HP is only a training target; the network input never contains it.
    """
    extended = bool(model_config) or double_dqn or n_step > 1 or aux_hp_weight > 0
    if aux_hp_weight > 0 and not (model_config or {}).get("aux_head"):
        raise ValueError("aux_hp_weight > 0 requires model_config with aux_head=True")
    save_path = checkpoint_path(checkpoint_prefix, phase, save_dir)
    if save_path.exists() and not overwrite:
        raise FileExistsError(
            f"{save_path} already exists; use a different checkpoint_prefix or overwrite=True"
        )

    if seed is not None:
        set_seed(seed)

    device = torch.device(device)

    # Build RewardConfig from individual params if not provided as object
    if reward_config is None:
        rc = RewardConfig()
        if firing_reward_high is not None: rc.firing_reward_high = firing_reward_high
        if time_penalty_factor is not None: rc.time_penalty_factor = time_penalty_factor
        if hold_penalty_high is not None: rc.hold_penalty_high = hold_penalty_high
        if reward_excellent is not None: rc.reward_excellent = reward_excellent
        if reward_good is not None: rc.reward_good = reward_good
        if reward_minimum is not None: rc.reward_minimum = reward_minimum
        if reward_fair is not None: rc.reward_fair = reward_fair
        if reward_poor is not None: rc.reward_poor = reward_poor
        if reward_failure is not None: rc.reward_failure = reward_failure
        if hold_penalty_base is not None: rc.hold_penalty_base = hold_penalty_base
        if opportunity_cost_excellent is not None: rc.opportunity_cost_excellent = opportunity_cost_excellent
        if opportunity_cost_good is not None: rc.opportunity_cost_good = opportunity_cost_good
        if opportunity_cost_minimum is not None: rc.opportunity_cost_minimum = opportunity_cost_minimum
        reward_config = rc

    # Initialize curriculum and environment
    curriculum = CurriculumScenarios(phase=phase)
    curriculum.print_curriculum_info()

    # Initialize training monitor
    monitor = TrainingMonitor()

    # Create environment without fixed max_episode_steps to enable dynamic step calculation
    env = ArtilleryFiringEnv(
        firing_reward_high=reward_config.firing_reward_high,
        time_penalty_factor=reward_config.time_penalty_factor,
        hold_penalty_high=reward_config.hold_penalty_high,
        reward_excellent=reward_config.reward_excellent,
        reward_good=reward_config.reward_good,
        reward_minimum=reward_config.reward_minimum,
        reward_fair=reward_config.reward_fair,
        reward_poor=reward_config.reward_poor,
        reward_failure=reward_config.reward_failure,
        hold_penalty_base=reward_config.hold_penalty_base,
        opportunity_cost_excellent=reward_config.opportunity_cost_excellent,
        opportunity_cost_good=reward_config.opportunity_cost_good,
        opportunity_cost_minimum=reward_config.opportunity_cost_minimum,
        **(env_options or {}),
    )
    sampler = None
    if trajectory_pool is not None:
        from rl_training.utils.trajectory_pool import PoolSampler, install_pool

        install_pool(env, trajectory_pool)
        sampler = PoolSampler(trajectory_pool, seed=None if seed is None else [seed, phase])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize networks
    if extended:
        q_network = build_q_network(state_dim, action_dim, model_config).to(device)
        target_network = build_q_network(state_dim, action_dim, model_config).to(device)
    else:
        q_network = DQN(state_dim, action_dim).to(device)
        target_network = DQN(state_dim, action_dim).to(device)
    
    # Load previous phase if available
    if load_from and Path(load_from).exists():
        print(f"\nLoading weights from {load_from}")
        q_network.load_state_dict(torch.load(load_from))
    
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    if extended:
        replay_buffer = NStepReplayBuffer(capacity=buffer_size, n_step=n_step, gamma=gamma)
    else:
        replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    print(f"\nTraining Phase {phase}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Epsilon: {epsilon_start} -> {epsilon_end} over {epsilon_decay_steps} steps")
    print(f"Evaluation every {eval_interval} episodes")
    print(f"Starting training...\n")
    
    # Training loop
    scenario = curriculum.get_next_scenario()
    state, _ = reset_episode(env, scenario, sampler)
    
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    episode_rewards = []
    
    # Per-scenario tracking
    scenario_stats = defaultdict(lambda: {'rewards': [], 'hps': [], 'steps': []})
    
    start_time = time.time()
    
    final_results = None
    for step in range(total_timesteps):
        # Epsilon decay
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * step / epsilon_decay_steps)

        # Select and execute action
        action = select_action(state, q_network, epsilon, action_dim, device)
        hp_label = env.current_state.current_hit_probability if extended else None  # training target only
        next_state, reward, terminated, truncated, info = env.step(Action(action))
        done = terminated or truncated

        # Log per-step details
        # ...existing code...

        # Store transition
        if extended:
            replay_buffer.push(state, action, reward, next_state, done, hp_label)
        else:
            replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward
        episode_length += 1

        # Training step
        if len(replay_buffer) >= learning_starts and step % 4 == 0:
            if extended:
                _extended_update_step(q_network, target_network, optimizer, replay_buffer, batch_size,
                                      device, double_dqn=double_dqn, aux_hp_weight=aux_hp_weight)
            else:
                _dqn_update_step(q_network, target_network, optimizer, replay_buffer, batch_size, gamma, device)

        # Target network update
        if step % target_update_freq == 0 and step > 0:
            target_network.load_state_dict(q_network.state_dict())

        # Episode end
        if done:
            final_hp = info.get('final_hp', 0.0)
            episode_rewards.append(episode_reward)
            # Track per-scenario stats
            scenario_name = scenario.name
            scenario_stats[scenario_name]['rewards'].append(episode_reward)
            scenario_stats[scenario_name]['hps'].append(final_hp)
            scenario_stats[scenario_name]['steps'].append(episode_length)
            
            # Record episode in monitor
            fired = terminated and not truncated  # Fire terminates, hold may timeout
            monitor.record_episode(scenario_name, episode_reward, final_hp, episode_length, fired, phase)
            # ...existing code...
            episode_count += 1
            # Progress reporting
            if episode_count % 20 == 0:
                elapsed = time.time() - start_time
                avg_reward = np.mean(episode_rewards[-20:])
                print(f"Step {step:5d} | Ep {episode_count:4d} | eps={epsilon:.3f} | "
                      f"Reward={avg_reward:6.1f} | Len={episode_length:2d} | Time={elapsed:.0f}s")
                
                # Monitor summary every 100 episodes
                if episode_count % 100 == 0:
                    summary = monitor.get_recent_summary(window=100)
                    print(f"Monitor: Avg Reward={summary['avg_reward']:.1f}, Avg HP={summary['avg_hp']:.3f}, Avg Steps={summary['avg_steps']:.1f}")
            # Curriculum evaluation
            if episode_count % eval_interval == 0:
                eval_results = evaluate_curriculum(env, q_network, curriculum, device, verbose=True)
                final_results = eval_results
                # Check if phase complete
                if check_phase_completion(curriculum, eval_results):
                    print(f"\n[OK] Phase {phase} mastered after {episode_count} episodes!")
                    break
            # Reset for next episode
            scenario = curriculum.get_next_scenario()
            state, _ = reset_episode(env, scenario, sampler)
            episode_reward = 0
            episode_length = 0
        else:
            state = next_state
    
    # Save model to checkpoints directory
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(q_network.state_dict(), save_path)
    cov_feature = (env_options or {}).get("cov_feature", "linear")
    time_input = (env_options or {}).get("time_feature", "steps")
    if extended or cov_feature != "linear" or time_input != "steps":
        save_model_config(save_path, {**(model_config or {}), "cov_feature": cov_feature,
                                      "time_feature": time_input})
    print(f"\nModel saved to {save_path}")
    
    # If phase was never mastered, run final evaluation
    if final_results is None:
        final_results = evaluate_curriculum(env, q_network, curriculum, device, verbose=True)
    return q_network, final_results


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="DQN Curriculum Training")
    parser.add_argument("--start_phase", type=int, default=1, help="First curriculum phase to train (1-4)")
    parser.add_argument("--end_phase", type=int, default=3, help="Last curriculum phase to train (1-4)")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps per phase")
    parser.add_argument("--config", type=str, help="JSON config file for parameter overrides")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible training")
    parser.add_argument("--reward_mode", choices=("legacy", "aligned"), default="legacy",
                        help="legacy: original shaped reward; aligned: P_hit - time_cost * t/T")
    parser.add_argument("--time_cost", type=float, default=0.0,
                        help="Time cost lambda for --reward_mode aligned")
    parser.add_argument("--cov_feature", choices=("linear", "log"), default="linear",
                        help="Scaling of the covariance-trace input (filter information, not HP)")
    parser.add_argument("--time_feature", choices=("steps", "seconds"), default="steps",
                        help="Time input: step count (published) or elapsed tracking time in seconds")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Weights to start the first trained phase from (e.g. an imitation pre-training)")
    parser.add_argument("--architecture", choices=("dqn", "dueling"), default="dqn")
    parser.add_argument("--aux_hp_weight", type=float, default=0.0,
                        help="Weight of the auxiliary HP-prediction loss (training only; needs dueling)")
    parser.add_argument("--double_dqn", action="store_true")
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--checkpoint_prefix", type=str, default=DEFAULT_CHECKPOINT_PREFIX,
                        help="Checkpoints are saved as <prefix>_phase<N>.pth")
    parser.add_argument("--trajectory_pool", type=str, default=None,
                        help="Directory of a pre-computed trajectory pool; each training episode "
                             "uses a random perturbed variant of the curriculum scenario")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing checkpoints with the same prefix")
    return parser.parse_args(argv)


def model_options_from_args(args) -> dict:
    """Network and update options; empty values keep the original training path."""
    model_config = None
    if args.architecture != "dqn" or args.aux_hp_weight > 0:
        model_config = {"architecture": args.architecture, "aux_head": args.aux_hp_weight > 0}
    return {"model_config": model_config, "double_dqn": args.double_dqn,
            "n_step": args.n_step, "aux_hp_weight": args.aux_hp_weight}


def env_options_from_args(args) -> dict:
    """Environment options: aligned reward (training only), covariance-input scaling and time input."""
    return {"reward_mode": args.reward_mode, "time_cost": args.time_cost, "cov_feature": args.cov_feature,
            "time_feature": args.time_feature}


def load_config(path):
    """Read a JSON config; the paper2_experiment.json format is flattened."""
    import json

    with open(path, 'r') as f:
        raw_config = json.load(f)
    # If standard paper2_experiment.json format is used, flatten parameters
    if "reward_function" not in raw_config:
        return raw_config
    rf = raw_config["reward_function"]
    fir = rf.get("firing_rewards", {})
    hld = rf.get("hold_penalties", {})
    ts = rf.get("time_and_safety", {})
    ct = raw_config.get("curriculum_training", {})
    return {
        "firing_reward_high": fir.get("excellent", {}).get("reward", 100.0),
        "reward_excellent": fir.get("excellent", {}).get("reward", 100.0),
        "reward_good": fir.get("good", {}).get("reward", 80.0),
        "reward_minimum": fir.get("acceptable", {}).get("reward", 60.0),
        "reward_fair": fir.get("fair", {}).get("reward", 20.0),
        "reward_poor": 20.0,
        "reward_failure": fir.get("failure", {}).get("reward", -30.0),
        "time_penalty_factor": ts.get("progressive_time_penalty_factor", -100.0),
        "hold_penalty_high": hld.get("excellent", {}).get("penalty", -10.0),
        "eval_interval": ct.get("eval_interval", 100),
        "epsilon_decay_steps": ct.get("epsilon_decay_steps", 2000),
    }


def main(argv=None):
    print("=" * 80)
    print("DQN CURRICULUM LEARNING")
    print("=" * 80)

    args = parse_args(argv)
    if args.seed is not None:
        set_seed(args.seed)

    config = load_config(args.config) if args.config else {}
    env_options = env_options_from_args(args)
    trajectory_pool = None
    if args.trajectory_pool:
        from rl_training.utils.trajectory_pool import load_pool

        trajectory_pool = load_pool(args.trajectory_pool)

    # Fail before any training if a phase would replace an existing checkpoint.
    if not args.overwrite:
        for phase in range(args.start_phase, args.end_phase + 1):
            path = checkpoint_path(args.checkpoint_prefix, phase)
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; use a different --checkpoint_prefix or --overwrite"
                )

    prev_checkpoint = args.init_checkpoint
    if prev_checkpoint is not None and not Path(prev_checkpoint).exists():
        raise FileNotFoundError(f"--init_checkpoint {prev_checkpoint} does not exist")
    for phase in range(args.start_phase, args.end_phase + 1):
        print(f"\n\nSTARTING PHASE {phase}: Curriculum Training")
        print("=" * 80)
        rc = RewardConfig(
            firing_reward_high=config.get("firing_reward_high", 100.0),
            time_penalty_factor=config.get("time_penalty_factor", -100.0),
            hold_penalty_high=config.get("hold_penalty_high", -10.0),
            reward_excellent=config.get("reward_excellent", 100.0),
            reward_good=config.get("reward_good", 80.0),
            reward_minimum=config.get("reward_minimum", 60.0),
            reward_fair=config.get("reward_fair", 40.0),
            reward_poor=config.get("reward_poor", 20.0),
            reward_failure=config.get("reward_failure", -30.0),
            hold_penalty_base=config.get("hold_penalty_base", -10.0),
            opportunity_cost_excellent=config.get("opportunity_cost_excellent", -30.0),
            opportunity_cost_good=config.get("opportunity_cost_good", -15.0),
            opportunity_cost_minimum=config.get("opportunity_cost_minimum", -8.0),
        )
        train_curriculum_phase(
            phase=phase,
            total_timesteps=args.timesteps,
            eval_interval=config.get("eval_interval", 100),
            epsilon_decay_steps=config.get("epsilon_decay_steps", 2000),
            reward_config=rc,
            load_from=prev_checkpoint,
            seed=args.seed,
            env_options=env_options,
            checkpoint_prefix=args.checkpoint_prefix,
            overwrite=args.overwrite,
            trajectory_pool=trajectory_pool,
            **model_options_from_args(args),
        )
        prev_checkpoint = str(checkpoint_path(args.checkpoint_prefix, phase))

    print("\n" + "=" * 80)
    print("CURRICULUM LEARNING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
