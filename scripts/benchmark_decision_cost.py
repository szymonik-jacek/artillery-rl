#!/usr/bin/env python3
"""
Time the computations behind one firing decision (TODO stage 1).

A threshold rule needs, at every decision step, the full hit-probability
chain: Kalman filter update, intercept point (ODE integration of the
projectile, or a firing table), error propagation along the projectile
trajectory (integration of the trajectory with its variational equations --
needed even when the intercept comes from a firing table) and HP
integration over the target. The DQN agent needs the Kalman filter update
and one forward pass. Each component is timed separately on realistic
inputs: for each evaluation scenario the filter is run on 5 s of noisy
measurements at 2 Hz, as in the environment.

Writes results/decision_cost_benchmark.json.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MEASUREMENT_DT = 0.5        # s, as in ArtilleryFiringEnv
TRACK_TIME_S = 5.0          # filter state after 5 s of tracking
PROJECTILE_SPEED = 800.0    # m/s, as in ArtilleryFiringEnv
SHOOTER = np.zeros(3)


def summarize_us(times_us):
    """Summary statistics (microseconds) of a list of timings."""
    t = np.asarray(times_us, dtype=float)
    return {"n": int(t.size), "median_us": float(np.median(t)), "mean_us": float(t.mean()),
            "p95_us": float(np.percentile(t, 95)), "p99_us": float(np.percentile(t, 99)),
            "min_us": float(t.min()), "max_us": float(t.max())}


def time_call(fn, repeats, warmup=0):
    """Wall-clock time of fn() in microseconds for each of `repeats` calls (after warmup)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) / 1000.0)
    return times


def projectile_covariance(scenario):
    """Projectile parameter covariance exactly as in ArtilleryFiringEnv._calculate_hit_probability."""
    angle_std = 2e-4 * (scenario.range_m / 1000.0) ** 0.2
    return np.diag([2.5 ** 2, angle_std ** 2, angle_std ** 2])


def decision_inputs(scenario, rng, track_time_s=TRACK_TIME_S):
    """Kalman filter (and its estimate) after track_time_s of noisy measurements."""
    from src.kalman_filter import TargetKalmanFilter

    start = np.array([scenario.range_m, 0.0, 50.0])
    velocity = np.array([scenario.target_vx, scenario.target_vy, scenario.target_vz])
    noise_std = np.asarray(scenario.measurement_noise_std, dtype=float) * np.ones(3)
    times = np.arange(0.0, track_time_s + 1e-9, MEASUREMENT_DT)
    measurements = [start + velocity * t + rng.normal(0.0, noise_std) for t in times]

    kf = TargetKalmanFilter(
        initial_state=np.concatenate([measurements[0], np.zeros(3)]),
        initial_covariance=np.diag([25.0, 25.0, 25.0, 2.0, 2.0, 2.0]),
        process_noise_std=scenario.process_noise_std,
        measurement_noise_std=scenario.measurement_noise_std,
        dt=MEASUREMENT_DT,
    )
    for z in measurements[1:]:
        kf.predict()
        kf.update(z)
    next_measurement = start + velocity * (times[-1] + MEASUREMENT_DT) + rng.normal(0.0, noise_std)
    return {"kf": kf, "next_measurement": next_measurement,
            "position": kf.state[:3].copy(), "velocity": kf.state[3:].copy(),
            "covariance": kf.covariance.copy()}


def benchmark_scenario(scenario, q_network, observation, rng, repeats):
    """Time each decision component for one scenario. Returns {component: summary}."""
    import copy

    import torch

    from src.find_optimal_firing_angles import find_optimal_firing_angles
    from src.hit_probability import HitProbabilityCalculator

    inp = decision_inputs(scenario, rng)

    def kalman_step():
        kf = copy.deepcopy(inp["kf"])
        kf.predict()
        kf.update(inp["next_measurement"])

    def deepcopy_only():
        copy.deepcopy(inp["kf"])

    def intercept():
        return find_optimal_firing_angles(SHOOTER, inp["position"], inp["velocity"],
                                          PROJECTILE_SPEED, "tpt", 30.0)

    elev, azim, impact_time, intersection_point, _ = intercept()
    calculator = HitProbabilityCalculator(projectile_velocity=PROJECTILE_SPEED)
    calculator.target_dimensions = (scenario.target_length, scenario.target_width, scenario.target_height)
    proj_cov = projectile_covariance(scenario)

    def hit_probability(use_diagonal_approx=True, mc_refine_samples=0):
        return calculator.calculate_hit_probability_analytical(
            target_position=inp["position"], target_velocity=inp["velocity"],
            projectile_param_cov=proj_cov, target_param_cov=inp["covariance"],
            elevation_angle=elev, azimuth_angle=azim,
            use_diagonal_approx=use_diagonal_approx, mc_refine_samples=mc_refine_samples,
            impact_time=impact_time, intersection_point=intersection_point,
        )

    # Projectile part of the error propagation on its own: integration of the
    # trajectory with its variational (8x8 Jacobian) equations up to impact.
    proj_state_cov = np.zeros((8, 8))
    proj_state_cov[3:6, 3:6] = proj_cov

    def projectile_propagation():
        return calculator.error_propagation.propagate_projectile_covariance(
            proj_state_cov, PROJECTILE_SPEED, elev, azim, impact_time)

    x = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)

    def dqn_forward():
        with torch.no_grad():
            return q_network(x).argmax().item()

    # The filter step is timed including a deepcopy (to keep inputs fixed); subtract it.
    kalman = np.asarray(time_call(kalman_step, repeats["fast"], warmup=50))
    copies = np.asarray(time_call(deepcopy_only, repeats["fast"], warmup=50))
    results = {
        "kalman_update": summarize_us(np.maximum(kalman - np.median(copies), 0.0)),
        "intercept_ode": summarize_us(time_call(intercept, repeats["slow"], warmup=2)),
        "projectile_error_propagation_ode": summarize_us(
            time_call(projectile_propagation, repeats["slow"], warmup=2)),
        "hp_propagation_diagonal": summarize_us(time_call(hit_probability, repeats["slow"], warmup=2)),
        "hp_propagation_full_covariance": summarize_us(
            time_call(lambda: hit_probability(use_diagonal_approx=False), repeats["slow"], warmup=2)),
        "hp_propagation_mc1000": summarize_us(
            time_call(lambda: hit_probability(mc_refine_samples=1000), repeats["mc"], warmup=1)),
        "dqn_forward": summarize_us(time_call(dqn_forward, repeats["fast"], warmup=200)),
    }
    results["_inputs"] = {"impact_time_s": float(impact_time), "elevation_rad": float(elev),
                          "hp_diagonal": float(hit_probability())}
    return results


def per_decision_totals(components):
    """Median cost per decision for the threshold rule (with/without intercept) and the DQN."""
    med = {k: v["median_us"] for k, v in components.items() if not k.startswith("_")}
    return {
        "rule_full_chain_us": med["kalman_update"] + med["intercept_ode"] + med["hp_propagation_diagonal"],
        "rule_hp_only_us": med["hp_propagation_diagonal"],
        "dqn_decision_us": med["kalman_update"] + med["dqn_forward"],
        "dqn_forward_us": med["dqn_forward"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats_fast", type=int, default=5000, help="Kalman update and DQN forward")
    parser.add_argument("--repeats_slow", type=int, default=50, help="Intercept and analytical HP")
    parser.add_argument("--repeats_mc", type=int, default=10, help="HP with Monte Carlo refinement")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "decision_cost_benchmark.json")
    args = parser.parse_args(argv)

    import torch

    from evaluate_dqn_trajectories import load_dqn_model
    from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
    from rl_training.environment import EnvironmentState
    from rl_training.utils.evaluation_protocol import get_evaluation_scenarios

    torch.set_num_threads(1)  # single-core figures, comparable with embedded targets
    checkpoint = ROOT / "rl_training/models/checkpoints/dqn_curriculum_phase4.pth"
    q_network = load_dqn_model(checkpoint, 14, 2, torch.device("cpu"))
    rng = np.random.default_rng(args.seed)
    repeats = {"fast": args.repeats_fast, "slow": args.repeats_slow, "mc": args.repeats_mc}

    per_scenario = {}
    for scenario in get_evaluation_scenarios(CurriculumScenarios(phase=4)):
        observation = EnvironmentState(scenario=scenario, time_remaining=scenario.tracking_duration / 2,
                                       current_hit_probability=0.0, episode_step=10).to_array()
        components = benchmark_scenario(scenario, q_network, observation, rng, repeats)
        per_scenario[scenario.name] = {"components": components, "per_decision": per_decision_totals(components)}
        pd_ = per_scenario[scenario.name]["per_decision"]
        print(f"{scenario.name:28s} rule chain {pd_['rule_full_chain_us'] / 1000:8.2f} ms | "
              f"HP only {pd_['rule_hp_only_us'] / 1000:8.2f} ms | DQN decision {pd_['dqn_decision_us']:7.1f} us",
              flush=True)

    payload = {
        "description": "Per-decision computation cost; medians in microseconds",
        "hardware": {"processor": platform.processor(), "machine": platform.machine(),
                     "platform": platform.platform(), "python": platform.python_version(),
                     "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
        "config": {"track_time_s": TRACK_TIME_S, "measurement_dt_s": MEASUREMENT_DT,
                   "projectile_speed_m_s": PROJECTILE_SPEED, "seed": args.seed, "repeats": repeats,
                   "hp_configuration_used_in_environment": "diagonal approximation, no Monte Carlo"},
        "per_scenario": per_scenario,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
