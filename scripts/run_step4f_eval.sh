#!/bin/bash
# Evaluation runs on random set 1 and the four shifts of the test set, with the filter estimate
# predicted to the decision time; the tuned rules are those selected on results/stage4e_eval/tuning_run.
# Re-running this script resumes each run from its last completed seed.
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/Scripts/python}"  # override, e.g. PYTHON=python bash scripts/run_step4f_eval.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
C=rl_training/models/checkpoints
AG=""; for v in bp b; do for s in 0 1 2 3 4; do AG="$AG --extra_dqn S2_${v}_s$s=$C/dqn_s2_${v}_seed${s}_phase4.pth"; done; done
for s in 0 1 2; do AG="$AG --extra_dqn V0r_s$s=$C/dqn_v0r_seed${s}_phase4.pth"; done
ENS="Ens_bp=$C/dqn_s2_bp_seed0_phase4.pth,$C/dqn_s2_bp_seed1_phase4.pth,$C/dqn_s2_bp_seed2_phase4.pth,$C/dqn_s2_bp_seed3_phase4.pth,$C/dqn_s2_bp_seed4_phase4.pth"
SEL=results/stage4e_eval/tuning_run/benchmark_traditional_vs_rl_raw.csv
DESC="10 Phase-4 tuning scenarios x 30 seeds (results/stage4e_eval/tuning_run)"
OUT=results/stage4f_eval; mkdir -p $OUT/logs
run() { # name, extra args
  local name=$1; shift
  "$PYTHON" -u scripts/compare_traditional_vs_rl.py "$@" $AG --ensemble "$ENS" --extrapolate_estimate --select_on $SEL --selected_on "$DESC" \
    --output_dir $OUT/$name --table $OUT/$name/table_comparison.tex --resume >> $OUT/logs/$name.log 2>&1
  echo "$name exit $? $(date +%H:%M)"
}
run heldout --scenario_set heldout --num_seeds 10 &
for sh in noise1.5 noise2 size0.7 range1.2; do run shift_$sh --scenario_set evaluation --num_seeds 30 --shift $sh & done
wait
echo "all done $(date +%H:%M)"
