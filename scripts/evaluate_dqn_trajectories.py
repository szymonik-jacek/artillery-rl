#!/usr/bin/env python3
"""
Evaluate the trained DQN model on artillery firing trajectories.

This script loads the trained DQN model and evaluates its performance
on various trajectory scenarios, showing hit probability traces and firing decisions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_training.environment import ArtilleryFiringEnv
from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
from rl_training.agents.dqn_components import DQN
from rl_training.utils.evaluation_utils import evaluate_curriculum
from rl_training.utils.evaluation_protocol import get_evaluation_scenarios, perturb_scenario


def load_dqn_model(model_path, state_dim, action_dim, device):
    """Load the trained DQN model."""
    model = DQN(state_dim, action_dim).to(device)
    if model_path.exists():
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")


def evaluate_trajectory(env, q_network, scenario, device, max_steps=100, seed=None):
    """Evaluate a single trajectory and return detailed results."""
    state, _ = env.reset(scenario_override=scenario, seed=seed)
    hp_trace = []
    actions = []
    rewards = []
    ranges = []
    target_positions = []  # Track full target position (x,y,z)
    cov_traces = []  # Track covariance trace
    done = False
    step_count = 0
    fired_at_step = None
    firing_range = None
    firing_pos = None

    initial_x = float(scenario.range_m)
    initial_y = 0.0  # Target starts at y=0
    initial_z = 50.0  # Target starts at z=50m
    initial_slant_range = float(np.sqrt(initial_x**2 + initial_y**2 + initial_z**2))

    # Record initial HP, covariance, slant range, and position (step 0 before step)
    hp_trace.append(env.current_state.current_hit_probability)
    cov_traces.append(env.current_state.covariance_trace)
    ranges.append(initial_slant_range)
    target_positions.append((initial_x, initial_y, initial_z))

    while not done and step_count < max_steps:
        # Calculate target position and 3D slant range at decision step before action
        elapsed_time = step_count * (scenario.tracking_duration / env.max_episode_steps)
        current_x = initial_x + scenario.target_vx * elapsed_time
        current_y = initial_y + scenario.target_vy * elapsed_time
        current_z = initial_z + scenario.target_vz * elapsed_time
        current_slant_range = float(np.sqrt(current_x**2 + current_y**2 + current_z**2))
        current_pos = (current_x, current_y, current_z)

        # Get action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = q_network(state_tensor).argmax().item()

        # Record firing decision and exact 3D slant range at decision moment
        if action == 1 and fired_at_step is None:  # FIRE action
            fired_at_step = step_count
            firing_range = current_slant_range
            firing_pos = current_pos

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record trajectory data
        actions.append(action)
        rewards.append(reward)

        # Calculate target position and 3D slant range after step
        next_time = (step_count + 1) * (scenario.tracking_duration / env.max_episode_steps)
        next_x = initial_x + scenario.target_vx * next_time
        next_y = initial_y + scenario.target_vy * next_time
        next_z = initial_z + scenario.target_vz * next_time
        next_slant_range = float(np.sqrt(next_x**2 + next_y**2 + next_z**2))
        ranges.append(next_slant_range)
        target_positions.append((next_x, next_y, next_z))

        # Record HP and covariance after step
        hp_trace.append(info["hit_probability"])
        cov_traces.append(info["covariance_trace"])

        state = next_state
        step_count += 1

    return {
        'hp_trace': hp_trace,
        'actions': actions,
        'rewards': rewards,
        'ranges': ranges,
        'target_positions': target_positions,
        'cov_traces': cov_traces,
        'fired_at_step': fired_at_step,
        'firing_range': firing_range,
        'firing_pos': firing_pos,
        'final_hp': hp_trace[-1] if hp_trace else 0.0,
        'scenario_name': scenario.name,
        'initial_range': initial_x,
        'initial_slant_range': initial_slant_range,
        'steps': step_count
    }


def plot_trajectory_results(results, save_path=None, seed=None):
    """Plot hit probability traces with firing decisions and additional analyses."""
    fig = plt.figure(figsize=(16, 12))
    seed_label = f" (seed {seed})" if seed is not None else ""
    fig.suptitle(f'DQN Model Evaluation on Artillery Trajectories{seed_label}', fontsize=16)

    # Create subplots: 2x3 grid (HP traces + analyses)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # HP trace plots (first 4 scenarios)
    hp_axes = []
    for i in range(4):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        hp_axes.append(ax)

    colors = ['blue', 'red', 'green', 'orange']

    for i, result in enumerate(results[:4]):  # Plot first 4 trajectories
        ax = hp_axes[i]

        # Plot HP trace
        steps = range(len(result['hp_trace']))
        ax.plot(steps, result['hp_trace'], 'b-', linewidth=2, label='Hit Probability')

        # Mark firing decision
        if result['fired_at_step'] is not None:
            fire_step = result['fired_at_step']
            fire_hp = result['hp_trace'][fire_step]
            ax.scatter(fire_step, fire_hp, color='red', s=100, marker='x',
                      linewidth=3, label=f'FIRE (HP={fire_hp:.3f})')

        # Add scenario info
        ax.set_title(f"{result['scenario_name']}\nRange: {result['initial_range']:.0f}m")
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Hit Probability')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set reasonable y-axis limits
        ax.set_ylim(0, 1.0)

    # Covariance Trace vs Time
    ax_cov_time = fig.add_subplot(gs[:, 2])
    for i, result in enumerate(results):
        if 'cov_traces' in result and result['cov_traces']:
            steps = range(len(result['cov_traces']))
            ax_cov_time.scatter(steps, result['cov_traces'], label=result['scenario_name'], alpha=0.7, marker='o', s=10)
    ax_cov_time.set_xlabel('Time Step')
    ax_cov_time.set_ylabel('Covariance Trace')
    ax_cov_time.set_title('Covariance Trace vs Time')
    ax_cov_time.legend()
    ax_cov_time.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plots to {save_path}")
    else:
        plt.show()


def plot_individual_scenarios(results, plots_dir="scenario_plots", seed=None):
    """Create individual plots for each scenario and save to a directory."""
    import os
    os.makedirs(plots_dir, exist_ok=True)
    
    for result in results:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        seed_label = f" (seed {seed})" if seed is not None else ""
        fig.suptitle(f"DQN Evaluation: {result['scenario_name']}{seed_label}", fontsize=14)
        
        scenario_name = result['scenario_name']
        steps = range(len(result['hp_trace']))
        
        # HP Trace
        ax1.plot(steps, result['hp_trace'], 'b-', linewidth=2, label='Hit Probability')
        if result['fired_at_step'] is not None:
            fire_step = result['fired_at_step']
            fire_hp = result['hp_trace'][fire_step]
            ax1.scatter(fire_step, fire_hp, color='red', s=100, marker='x',
                       linewidth=3, label=f'FIRE (HP={fire_hp:.3f})')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Hit Probability')
        ax1.set_title('Hit Probability Trace')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1.0)
        
        # Range vs Time
        if result['ranges']:
            ax2.plot(steps, result['ranges'], 'g-', linewidth=2)
            if result['fired_at_step'] is not None:
                fire_step = result['fired_at_step']
                fire_range = result['ranges'][fire_step] if fire_step < len(result['ranges']) else result['ranges'][-1]
                ax2.scatter(fire_step, fire_range, color='red', s=100, marker='x', linewidth=3)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Range (m)')
            ax2.set_title('Target Range vs Time')
            ax2.grid(True, alpha=0.3)
        
        # Covariance Trace
        if result['cov_traces'] and any(c is not None for c in result['cov_traces']):
            valid_cov = [(i, c) for i, c in enumerate(result['cov_traces']) if c is not None]
            if valid_cov:
                cov_steps, cov_values = zip(*valid_cov)
                ax3.plot(cov_steps, cov_values, 'purple', linewidth=2, marker='o', markersize=3)
                if result['fired_at_step'] is not None:
                    fire_step = result['fired_at_step']
                    if fire_step < len(result['cov_traces']) and result['cov_traces'][fire_step] is not None:
                        ax3.scatter(fire_step, result['cov_traces'][fire_step], color='red', s=100, marker='x', linewidth=3)
                ax3.set_xlabel('Time Step')
                ax3.set_ylabel('Covariance Trace')
                ax3.set_title('Kalman Filter Covariance Trace')
                ax3.grid(True, alpha=0.3)
        
        # Actions and Rewards
        ax4_twin = ax4.twinx()
        # Actions are recorded per step, so they have one less element than hp_trace
        action_steps = range(len(result['actions']))
        ax4.step(action_steps, result['actions'], 'orange', linewidth=2, where='post', label='Action')
        ax4_twin.plot(action_steps, result['rewards'], 'tab:blue', linewidth=2, marker='s', markersize=3, label='Reward')
        
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Action (0=HOLD, 1=FIRE)', color='orange')
        ax4_twin.set_ylabel('Reward', color='tab:blue')
        ax4.set_title('Actions and Rewards')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['HOLD', 'FIRE'])
        ax4.grid(True, alpha=0.3)
        
        # Add summary text
        summary_text = f"Initial Range: {result['initial_range']:.0f}m\n"
        summary_text += f"Steps: {result['steps']}\n"
        summary_text += f"Final HP: {result['final_hp']:.3f}\n"
        if result['fired_at_step'] is not None:
            summary_text += f"Fired at step: {result['fired_at_step']}\n"
        else:
            summary_text += "No firing decision\n"
        
        fig.text(0.02, 0.98, summary_text, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        safe_name = scenario_name.replace(' ', '_').replace('/', '_')
        plot_path = os.path.join(plots_dir, f"{safe_name}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Individual scenario plots saved to {plots_dir}/ directory")


def print_trajectory_summary(results):
    """Print detailed summary of trajectory evaluations."""
    print("\n" + "="*100)
    print("TRAJECTORY EVALUATION SUMMARY")
    print("="*100)
    print(f"{'Scenario':<25} {'Initial Range':<12} {'Steps':<6} {'Fired At':<9} {'Final HP':<9} {'Range@Fire':<12} {'Position@Fire (x,y,z)':<20}")
    print("-"*100)

    for result in results:
        fired_step = result['fired_at_step']
        fire_range = result.get('firing_range')
        fire_pos = result.get('firing_pos')
        if fire_range is None:
            if fired_step is not None and fired_step < len(result['ranges']):
                fire_range = result['ranges'][fired_step]
                fire_pos = result['target_positions'][fired_step]
            else:
                fire_range = result['ranges'][-1] if result['ranges'] else 0.0
                fire_pos = result['target_positions'][-1] if result['target_positions'] else (0.0, 0.0, 0.0)
        pos_str = f"({fire_pos[0]:.0f},{fire_pos[1]:.0f},{fire_pos[2]:.0f})"

        print(f"{result['scenario_name']:<25} "
              f"{result['initial_range']:<12.0f} "
              f"{result['steps']:<6} "
              f"{fired_step if fired_step is not None else '-':<9} "
              f"{result['final_hp']:<9.3f} "
              f"{fire_range:<12.0f} "
              f"{pos_str:<20}")

    # Summary statistics
    fired_count = sum(1 for r in results if r['fired_at_step'] is not None)
    avg_final_hp = np.mean([r['final_hp'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])

    print("-"*100)
    print(f"Total trajectories: {len(results)}")
    print(f"Trajectories with firing: {fired_count}")
    print(f"Average final HP:         {avg_final_hp:.3f}")
    print(f"Average steps:            {avg_steps:.1f}")
    print("="*100)


def main(num_runs: int = 1):
    """Main evaluation function."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=num_runs,
                        help="Number of independent evaluation runs (each with a different seed)")
    parser.add_argument("--plot_run", type=int, default=0,
                        help="Zero-based run/seed used for detailed traces and plots (default: 0)")
    args = parser.parse_args()
    num_runs = args.num_runs
    if num_runs < 1:
        parser.error("--num_runs must be at least 1")
    if not 0 <= args.plot_run < num_runs:
        parser.error("--plot_run must be between 0 and num_runs - 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment and curriculum
    env = ArtilleryFiringEnv()
    curriculum = CurriculumScenarios(phase=4)  # Use Phase 4 for comprehensive evaluation

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load the trained model (Phase 4)
    ROOT = Path(__file__).parent.parent
    model_path = ROOT / "rl_training/models/checkpoints/dqn_curriculum_phase4.pth"
    try:
        q_network = load_dqn_model(model_path, state_dim, action_dim, device)
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Available checkpoints:")
        checkpoints_dir = ROOT / "rl_training/models/checkpoints"
        if checkpoints_dir.exists():
            for f in checkpoints_dir.glob("*.pth"):
                print(f"  {f.name}")
        return

    scenarios_to_test = get_evaluation_scenarios(curriculum)

    all_results = []  # collects results from every run
    for run_idx in range(num_runs):
        seed = run_idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        print(f"\n{'='*60}\nRUN {run_idx+1}/{num_runs}  (seed={seed})\n{'='*60}")
        results = []
        for scenario in scenarios_to_test:
            perturbed = perturb_scenario(scenario, rng)
            print(f"  Evaluating: {perturbed.name}  (range={perturbed.range_m:.0f}m)")
            result = evaluate_trajectory(env, q_network, perturbed, device,
                                         seed=seed)
            results.append(result)
        all_results.append(results)

    plot_results = all_results[args.plot_run]

    # Print a detailed summary for the same run used by the plots.
    print(f"\nDetailed traces and plots use seed={args.plot_run}.")
    print_trajectory_summary(plot_results)

    import json
    import pandas as pd
    from scipy import stats as sp_stats

    # Flatten all results into a single list with run/seed columns
    flat_rows = []
    for run_idx, run_results in enumerate(all_results):
        for r in run_results:
            flat_rows.append({
                'run': run_idx,
                'seed': run_idx,
                'scenario_name': r['scenario_name'],
                'final_hp': float(r['final_hp']),
                'fired_at_step': r['fired_at_step'],
                'firing_range': float(r['firing_range']) if r.get('firing_range') is not None else np.nan,
                'steps': r['steps'],
                'initial_range': float(r['initial_range']),
            })
    df_all = pd.DataFrame(flat_rows)

    def calc_ci95(series):
        """Calculate 95% confidence interval half-width using Student's t distribution."""
        valid = series.dropna()
        n = len(valid)
        if n < 2:
            return 0.0
        sem = valid.std() / np.sqrt(n)
        t_crit = sp_stats.t.ppf(0.975, df=n - 1)
        return float(t_crit * sem)

    # Multi-run statistics table
    if num_runs > 1:
        stats = (
            df_all.groupby('scenario_name')
            .agg(
                mean_final_hp   =('final_hp', 'mean'),
                std_final_hp    =('final_hp', 'std'),
                min_final_hp    =('final_hp', 'min'),
                max_final_hp    =('final_hp', 'max'),
                ci95_final_hp   =('final_hp', calc_ci95),
                mean_fired_step =('fired_at_step', 'mean'),
                std_fired_step  =('fired_at_step', 'std'),
                ci95_fired_step =('fired_at_step', calc_ci95),
                mean_firing_range=('firing_range', 'mean'),
                std_firing_range =('firing_range', 'std'),
                min_firing_range =('firing_range', 'min'),
                max_firing_range =('firing_range', 'max'),
                ci95_firing_range=('firing_range', calc_ci95),
                firing_rate     =('fired_at_step', lambda x: x.notna().mean() * 100),
            )
            .reset_index()
        )
        # Coefficient of variation (%) — std / mean * 100, meaningful robustness metric
        stats['cv_pct'] = (stats['std_final_hp'] / stats['mean_final_hp'] * 100).round(1)
        stats['cv_range_pct'] = (stats['std_firing_range'] / stats['mean_firing_range'] * 100).round(1)

        # ---- pretty console table ----
        col_w = [28, 8, 8, 8, 10, 10, 8, 12, 12, 7, 7]
        headers = ['Scenario', 'Mean HP', 'Std HP', '95% CI',
                   'Mean Step', 'Std Step', 'Fire%', 'Mean Range', 'Std Range', 'CV%', 'R_CV%']
        sep = '+' + '+'.join('-' * w for w in col_w) + '+'
        hdr = '|' + '|'.join(h.center(w) for h, w in zip(headers, col_w)) + '|'

        print(f"\n{'='*95}")
        print(f"ROBUSTNESS STATISTICS  ({num_runs} runs, seeds 0–{num_runs-1})")
        print(f"Parameter perturbations: range ±5%, velocity ±10%, noise ±20%")
        print(f"{'='*95}")
        print(sep); print(hdr); print(sep)
        for _, row in stats.iterrows():
            fired_step_mean = f"{row['mean_fired_step']:.1f}" if pd.notna(row['mean_fired_step']) else '—'
            fired_step_std  = f"{row['std_fired_step']:.1f}"  if pd.notna(row['std_fired_step'])  else '—'
            range_mean = f"{row['mean_firing_range']:.1f}" if pd.notna(row['mean_firing_range']) else '—'
            range_std  = f"{row['std_firing_range']:.1f}"  if pd.notna(row['std_firing_range'])  else '—'
            vals = [
                row['scenario_name'][:col_w[0]-2],
                f"{row['mean_final_hp']:.3f}",
                f"{row['std_final_hp']:.3f}",
                f"±{row['ci95_final_hp']:.3f}",
                fired_step_mean,
                fired_step_std,
                f"{row['firing_rate']:.0f}",
                range_mean,
                range_std,
                f"{row['cv_pct']:.1f}",
                f"{row['cv_range_pct']:.1f}",
            ]
            print('|' + '|'.join(v.center(w) for v, w in zip(vals, col_w)) + '|')
        print(sep)
        print("\nNote: This is a robustness (sensitivity) analysis, not a generalization")
        print("test. Perturbations simulate sensor/model uncertainty (cf. Tobin et al.,")
        print("2017, Domain Randomization; Rajeswaran et al., 2017, EPOpt). CV% is the")
        print("coefficient of variation (std/mean × 100) — lower values indicate a more")
        print("robust policy. For out-of-distribution generalization, held-out scenario")
        print("families would be required (Cobbe et al., 2019; Packer et al., 2018).")

        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        stats.to_csv(results_dir / "evaluation_summary_multiple_runs.csv", index=False)
        print(f"\nPer-run statistics saved to {results_dir / 'evaluation_summary_multiple_runs.csv'}")

    # Keep single-run diagnostics from overwriting the canonical multi-run data.
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary_name = "evaluation_summary.csv" if num_runs > 1 else "evaluation_summary_single_run.csv"
    summary_path = results_dir / summary_name
    df_all.to_csv(summary_path, index=False)
    print(f"All results saved to {summary_path}")

    # Save detailed traces for the explicitly selected plot run.
    details_data = []
    for r in plot_results:
        details_data.append({
            'seed': args.plot_run,
            'scenario_name': r['scenario_name'],
            'initial_range': float(r['initial_range']),
            'steps': int(r['steps']),
            'fired_at_step': int(r['fired_at_step']) if r['fired_at_step'] is not None else None,
            'final_hp': float(r['final_hp']),
            'firing_range': float(r['firing_range']) if r.get('firing_range') is not None else None,
            'ranges': [float(x) for x in r.get('ranges', [])],
            'target_positions': [[float(c) for c in pos] for pos in r.get('target_positions', [])],
            'hp_trace': [float(x) for x in r.get('hp_trace', [])],
            'cov_traces': [float(x) if x is not None else None for x in r.get('cov_traces', [])],
        })
    with open(results_dir / "evaluation_details.json", "w") as f:
        json.dump(details_data, f, indent=2)
    print(f"Evaluation details saved to {results_dir / 'evaluation_details.json'}")

    plot_trajectory_results(
        plot_results,
        save_path=str(results_dir / "trajectory_evaluation.png"),
        seed=args.plot_run,
    )
    plot_individual_scenarios(
        plot_results,
        plots_dir=str(results_dir / "scenario_plots"),
        seed=args.plot_run,
    )

    print("\n[OK] Evaluation complete!")
    print("Trajectory plots saved to: trajectory_evaluation.png")
    print("Individual scenario plots saved to: scenario_plots/ directory")
if __name__ == "__main__":
    main()