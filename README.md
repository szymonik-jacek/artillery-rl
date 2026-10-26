# Reinforcement Learning for Firing Timing in Anti-Aircraft Fire Control

Code for training Deep Q-Network (DQN) agents that decide when to fire in a simulated
anti-aircraft engagement, and for comparing them with 82 threshold rules on the hit
probability (HP). The simulation combines a modified point-mass projectile model, a
constant-velocity target, a Kalman filter and an analytical HP model.

## Installation

Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # requirements-lock.txt: exact versions used for the reported runs
```

All commands below are run from the repository root.

## Repository layout

```
src/                  ballistics, Kalman filter, intercept solution, analytical HP
rl_training/          environment, scenarios, DQN components, threshold rules, training, evaluation
scripts/              training, evaluation and analysis entry points (below)
config/               paper2_experiment.json: parameters of the reported experiments
rl_training/models/   checkpoints/: phase-4 checkpoints of the four agents
                      trajectory_pool/: training trajectories (not stored; built in step 1)
results/              per-episode results, analyses and checksums
tests/                pytest suite
```

## Agents

| Agent (paper) | Checkpoints (`rl_training/models/checkpoints/`) | Label in result files |
|---|---|---|
| Base-0 | `dqn_curriculum_phase4.pth` | `DQN` |
| Base | `dqn_v0r_seed{0,1,2}_phase4.pth` | `V0r_s0` … `V0r_s2` |
| Ext-steps | `dqn_s2_b_seed{0..4}_phase4.pth` | `S2_b_s0` … `S2_b_s4` |
| Ext-sec | `dqn_s2_bp_seed{0..4}_phase4.pth` | `S2_bp_s0` … `S2_bp_s4` |

`Ens_bp` in the result files is the ensemble of the five Ext-sec networks.

## Pipeline

### 1. Training

Training never overwrites existing checkpoints; use `--checkpoint_prefix` (or delete the
files) to train new ones.

```bash
# Base-0: four curriculum phases, 10,000 steps each, default parameters
python -m rl_training.train.train_dqn_curriculum --start_phase 1 --end_phase 4 --timesteps 10000 \
    --checkpoint_prefix my_base0

# Base: three seeds, parameters from config/paper2_experiment.json
python scripts/train_variants.py --set v0r

# Pool of 750 perturbed training trajectories (needed by the extended agents);
# results/trajectory_pool_sha256.txt holds the checksums of the pool used for the reported agents
python scripts/build_trajectory_pool.py

# Ext-steps: imitation warm start (scripts/pretrain_imitation.py), then curriculum training;
# this set also trains two variants (dqn_s2_a0, dqn_s2_a) that are not reported
python scripts/train_variants.py --set stage2

# Ext-sec: as Ext-steps, with the time input in seconds
python scripts/train_variants.py --set stage2e
```

### 2. Evaluation

Each launcher runs `scripts/compare_traditional_vs_rl.py` for several scenario sets in
parallel and can be restarted (finished seeds are skipped). The launchers call
`.venv/Scripts/python`; set `PYTHON` to use another interpreter, e.g.
`PYTHON=python bash scripts/run_step4e_eval.sh`.

```bash
bash scripts/run_step4e_eval.sh     # tuning set, test set, random set 2
python scripts/analyse_step4e.py    # rule selection on the tuning set; tests on these sets
bash scripts/run_step4f_eval.sh     # random set 1 and the four shifts

# Hit-probability validation (Monte Carlo and true-target check, then NEES) -> results/hp_validation_4e/
python scripts/validate_hp_mc.py --extrapolate_estimate
python scripts/diagnose_hp_calibration.py --extrapolate

# Cost of a decision (rule vs agent) and forward-pass latency
python scripts/benchmark_decision_cost.py
python scripts/benchmark_inference_latency.py
```

| Scenario set (paper) | Results folder |
|---|---|
| Tuning set | `results/stage4e_eval/tuning_run/` |
| Test set | `results/stage4e_eval/evaluation_100/` |
| Random set 1 | `results/stage4f_eval/heldout/` |
| Random set 2 | `results/stage4e_eval/heldout2/` |
| Shifts | `results/stage4f_eval/shift_{noise1.5,noise2,size0.7,range1.2}/` |

The per-episode results are in `benchmark_traditional_vs_rl_raw.csv` in each folder.

### 3. Numbers, figure and manifest

```bash
python scripts/paper_numbers.py         # results/paper_numbers.json: all values of the Results section
python scripts/analyse_discussion.py    # results/discussion_analysis.json: analyses of the Discussion
python scripts/plot_hp_vs_time.py       # figure: HP against firing time
python scripts/generate_run_manifest.py # results/paper2_run_manifest.json: checksums and environment
```

## Tests

```bash
pytest tests/
```
