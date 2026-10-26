"""Exploratory diagnosis of the calibration gap (step 4b): estimate timing and filter consistency.

With --extrapolate the shots are rebuilt with the estimate predicted to the
decision time (as in the step-4e runs), and the estimate error and NEES refer to
the true state at the decision time; otherwise to the time of the last measurement.
"""
import argparse
import json
import sys
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")


def diag(row, extrapolate=False):
    from validate_hp_mc import _perturbed
    from rl_training.environment import ArtilleryFiringEnv
    from rl_training.utils.hp_validation import firing_situation, in_box

    env = ArtilleryFiringEnv(extrapolate_estimate=extrapolate)
    sc = _perturbed(row.get("rebuild_set", row["scenario_set"]), row["scenario"], int(row["seed"]))
    s = firing_situation(env, sc, int(row["seed"]), int(row["fired_at_step"]))
    elapsed = s["elapsed_s"]
    t_meas = np.floor(elapsed / 0.5 + 1e-9) * 0.5  # time of the last measurement
    p0 = np.array([sc.range_m, 0.0, 50.0]); v = np.array([sc.target_vx, sc.target_vy, sc.target_vz])
    t_est = elapsed if extrapolate else t_meas  # time the estimate refers to
    true_state_at_meas = np.concatenate([p0 + v * t_est, v])
    est = np.concatenate([s["target_position"], s["target_velocity"]])
    err = est - true_state_at_meas
    nees = float(err @ np.linalg.solve(s["target_covariance"], err))
    aim = s["intersection_point"]
    miss_truth_elapsed = aim - (p0 + v * (elapsed + s["impact_time"]))
    miss_truth_meas = aim - (p0 + v * (t_meas + s["impact_time"]))
    return {**row, "elapsed": elapsed, "lag_s": elapsed - t_meas, "pos_err": float(np.linalg.norm(err[:3])),
            "vel_err": float(np.linalg.norm(err[3:])), "nees": nees,
            "miss_now": float(np.linalg.norm(miss_truth_elapsed)), "miss_meas": float(np.linalg.norm(miss_truth_meas)),
            "hit_now": bool(in_box(miss_truth_elapsed, s["half_dims"])[0]),
            "hit_meas": bool(in_box(miss_truth_meas, s["half_dims"])[0])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots_dir", default="results/hp_validation")
    parser.add_argument("--extrapolate", action="store_true")
    args = parser.parse_args()
    shots = pd.read_csv(f"{args.shots_dir}/hp_validation_shots.csv")
    cols = ["scenario_set", "group", "scenario", "seed", "fired_at_step", "base_hp", "true_hit_rate"]
    cols += ["rebuild_set"] if "rebuild_set" in shots else []
    with ProcessPoolExecutor() as pool:
        d = pd.DataFrame(list(pool.map(partial(diag, extrapolate=args.extrapolate), shots[cols].to_dict("records"))))
    print("lag (s): mean %.2f, share > 0: %.0f%%" % (d.lag_s.mean(), 100 * (d.lag_s > 1e-9).mean()))
    print("estimate error: position %.2f m, velocity %.2f m/s (median)" % (d.pos_err.median(), d.vel_err.median()))
    print("NEES mean %.1f (consistent filter: 6.0), median %.1f" % (d.nees.mean(), d.nees.median()))
    print("nominal shot hits true target: at decision time %.0f%%, at last-measurement time %.0f%%; mean base HP %.2f, mean true MC %.2f"
          % (100 * d.hit_now.mean(), 100 * d.hit_meas.mean(), d.base_hp.mean(), d.true_hit_rate.mean()))
    for lagged, g in d.groupby(d.lag_s > 1e-9):
        print(f"  lag>0={lagged}: n={len(g)}, base HP {g.base_hp.mean():.2f}, true MC {g.true_hit_rate.mean():.2f}, "
              f"nominal hit now {g.hit_now.mean():.2f}, at meas {g.hit_meas.mean():.2f}, NEES {g.nees.median():.1f}")
    d.to_csv(f"{args.shots_dir}/calibration_diagnosis.csv", index=False)
    summary = {"extrapolate": args.extrapolate, "n": len(d), "nees_mean": float(d.nees.mean()),
               "nees_median": float(d.nees.median()), "nees_expected_if_consistent": 6.0,
               "pos_err_median_m": float(d.pos_err.median()), "vel_err_median_m_s": float(d.vel_err.median())}
    (__import__("pathlib").Path(args.shots_dir) / "calibration_diagnosis_summary.json").write_text(json.dumps(summary, indent=2))
