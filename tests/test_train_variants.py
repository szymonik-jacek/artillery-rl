"""
Tests for scripts/train_variants.py: the run lists (V0r and
V0r on the trajectory pool), the training command line and resume status.
None of the runs puts hit probability into the agent's state.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import train_variants
from train_variants import CONFIG, build_command, pool_runs, run_status, v0r_runs
from rl_training.train.train_dqn_curriculum import checkpoint_path, parse_args


class TestRunLists:
    def test_v0r_runs(self):
        runs = dict(v0r_runs())
        assert set(runs) == {f"dqn_v0r_seed{s}" for s in range(3)}
        args = parse_args(runs["dqn_v0r_seed1"])
        assert (args.seed, args.reward_mode, args.trajectory_pool) == (1, "legacy", None)

    def test_pool_runs(self, tmp_path):
        runs = dict(pool_runs(tmp_path))
        assert set(runs) == {f"dqn_pool_v0r_seed{s}" for s in range(3)}
        args = parse_args(runs["dqn_pool_v0r_seed2"])
        assert (args.seed, args.reward_mode, args.trajectory_pool) == (2, "legacy", str(tmp_path))

    def test_prefixes_never_match_base0_checkpoints(self, tmp_path):
        prefixes = [p for p, _ in v0r_runs() + pool_runs(tmp_path)]
        assert all(not p.startswith("dqn_curriculum") for p in prefixes)
        assert len(set(prefixes)) == len(prefixes)


class TestBuildCommand:
    def test_uses_experiment_config_and_all_phases(self):
        cmd = build_command("dqn_v0r_seed0", ["--seed", "0"], python="py")
        args = parse_args(cmd[3:])
        assert cmd[:2] == ["py", "-u"]
        assert (args.start_phase, args.end_phase, args.timesteps) == (1, 4, 10000)
        assert args.config == str(CONFIG) and os.path.exists(CONFIG)
        assert args.checkpoint_prefix == "dqn_v0r_seed0" and not args.overwrite


class TestRunStatus:
    def test_pending_partial_done(self, tmp_path):
        assert run_status("x", tmp_path) == "pending"
        checkpoint_path("x", 2, tmp_path).write_bytes(b"")
        assert run_status("x", tmp_path) == "partial"
        checkpoint_path("x", 4, tmp_path).write_bytes(b"")
        assert run_status("x", tmp_path) == "done"


def test_refuses_incomplete_pool(tmp_path):
    (tmp_path / "00_Static_Close.pkl").write_bytes(b"")
    with pytest.raises(SystemExit):
        train_variants.main(["--set", "pool", "--pool_dir", str(tmp_path), "--log_dir", str(tmp_path / "logs")])


# ---------------------------------------------------------------------------
# Stage-2 runs
# ---------------------------------------------------------------------------

from train_variants import stage2_runs


class TestStage2Runs:
    def test_fifteen_runs_three_variants_five_seeds(self, tmp_path):
        runs = stage2_runs(tmp_path / "pool", checkpoint_dir=tmp_path)
        prefixes = [p for p, _, _ in runs]
        assert len(runs) == 15 and len(set(prefixes)) == 15
        assert {p.rsplit("_seed", 1)[0] for p in prefixes} == {"dqn_s2_a0", "dqn_s2_a", "dqn_s2_b"}

    def test_common_stage2_options(self, tmp_path):
        for prefix, args, _ in stage2_runs(tmp_path / "pool", checkpoint_dir=tmp_path):
            a = parse_args(args)
            assert (a.architecture, a.double_dqn, a.n_step) == ("dueling", True, 3)
            assert (a.reward_mode, a.time_cost, a.cov_feature) == ("aligned", 0.15, "log")
            assert a.trajectory_pool == str(tmp_path / "pool")

    def test_variant_differences(self, tmp_path):
        runs = {p: (args, pre) for p, args, pre in stage2_runs(tmp_path / "pool", checkpoint_dir=tmp_path)}
        a0, a, b = (parse_args(runs[f"dqn_s2_{v}_seed2"][0]) for v in ("a0", "a", "b"))
        assert a0.aux_hp_weight == 0.0 and a0.init_checkpoint is None and runs["dqn_s2_a0_seed2"][1] is None
        assert a.aux_hp_weight == 1.0 and a.init_checkpoint is None and runs["dqn_s2_a_seed2"][1] is None
        assert b.aux_hp_weight == 1.0 and b.seed == 2
        assert b.init_checkpoint == str(tmp_path / "dqn_s2_b_seed2_pretrain.pth")
        pre = runs["dqn_s2_b_seed2"][1]
        assert pre[pre.index("--prefix") + 1] == "dqn_s2_b_seed2" and pre[pre.index("--seed") + 1] == "2"


from train_variants import stage2e_runs


class TestStage2eRuns:
    def test_bprime_is_b_plus_seconds(self, tmp_path):
        b = {p: (a, pre) for p, a, pre in stage2_runs(tmp_path / "pool", checkpoint_dir=tmp_path)
             if p.startswith("dqn_s2_b_")}
        bp = stage2e_runs(tmp_path / "pool", checkpoint_dir=tmp_path)
        assert [p for p, _, _ in bp] == [f"dqn_s2_bp_seed{s}" for s in range(5)]
        for prefix, args, pretrain in bp:
            b_args, b_pre = b[prefix.replace("_bp_", "_b_")]
            assert args[-2:] == ["--time_feature", "seconds"] and pretrain[-2:] == ["--time_feature", "seconds"]
            # Otherwise identical to B, with B' file names.
            assert args[:-2] == [a.replace("dqn_s2_b_", "dqn_s2_bp_") for a in b_args]
            assert pretrain[:-2] == [a.replace("dqn_s2_b_", "dqn_s2_bp_") for a in b_pre]
            init = args[args.index("--init_checkpoint") + 1]
            assert init.endswith(f"{prefix}_pretrain.pth")
            assert pretrain[pretrain.index("--prefix") + 1] == prefix

    def test_set_option_accepts_stage2e(self, tmp_path, capsys, monkeypatch):
        import train_variants
        # Independent of which checkpoints exist on disk (trained runs are skipped).
        monkeypatch.setattr(train_variants, "run_status", lambda prefix, *args, **kwargs: "pending")
        train_variants.main(["--set", "stage2e", "--dry_run", "--pool_dir", str(tmp_path),
                             "--log_dir", str(tmp_path / "logs")])
        out = capsys.readouterr().out
        assert "dqn_s2_bp_seed0" in out and "dqn_s2_bp_seed4" in out and "--time_feature seconds" in out
