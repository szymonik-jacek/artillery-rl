#!/bin/bash
# Evaluation runs on the tuning set, the test set and random set 2, with the filter estimate
# predicted to the decision time (--extrapolate_estimate). The tuned rules are then selected on
# tuning_run by scripts/analyse_step4e.py.
# Re-running this script resumes each run from its last completed seed.
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/Scripts/python}"  # override, e.g. PYTHON=python bash scripts/run_step4e_eval.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
C=rl_training/models/checkpoints
AG=""; for v in bp b; do for s in 0 1 2 3 4; do AG="$AG --extra_dqn S2_${v}_s$s=$C/dqn_s2_${v}_seed${s}_phase4.pth"; done; done
for s in 0 1 2; do AG="$AG --extra_dqn V0r_s$s=$C/dqn_v0r_seed${s}_phase4.pth"; done
ENS="Ens_bp=$C/dqn_s2_bp_seed0_phase4.pth,$C/dqn_s2_bp_seed1_phase4.pth,$C/dqn_s2_bp_seed2_phase4.pth,$C/dqn_s2_bp_seed3_phase4.pth,$C/dqn_s2_bp_seed4_phase4.pth"
OUT=results/stage4e_eval; mkdir -p $OUT/logs
run() { # name, extra args
  local name=$1; shift
  "$PYTHON" -u scripts/compare_traditional_vs_rl.py "$@" $AG --ensemble "$ENS" --extrapolate_estimate \
    --output_dir $OUT/$name --table $OUT/$name/table_comparison.tex --resume >> $OUT/logs/$name.log 2>&1
  echo "$name exit $? $(date +%H:%M)"
}
run tuning_run --scenario_set tuning --num_seeds 30 &
run heldout2 --scenario_set heldout2 --num_seeds 10 &
run evaluation_100 --scenario_set evaluation --num_seeds 100 &
wait
echo "all done $(date +%H:%M)"
