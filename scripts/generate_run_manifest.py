#!/usr/bin/env python3
"""
Generate an immutable Run Manifest for Paper 2 experiments and checkpoints.

Captures:
- Checkpoint SHA-256 checksums, sizes, parameter counts
- Git commit hash, branch, and release tag metadata
- Python, PyTorch, NumPy, system runtime details
- Experiment configuration checksum and dump
"""

import sys
import os
import platform
import json
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_git_info():
    info = {"commit": "unknown", "branch": "unknown", "tag": None, "remote": "unknown", "is_dirty": False}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        info["commit"] = commit
    except Exception:
        pass
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
        info["branch"] = branch
    except Exception:
        pass
    try:
        tags = subprocess.check_output(["git", "tag", "--points-at", "HEAD"], cwd=ROOT, text=True).splitlines()
        info["tag"] = tags[0] if tags else None
    except Exception:
        pass
    try:
        remote = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=ROOT, text=True).strip()
        info["remote"] = remote
    except Exception:
        pass
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        info["is_dirty"] = bool(status)
    except Exception:
        pass
    return info

def inspect_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        return None
    import torch
    sha256 = compute_sha256(checkpoint_path)
    size_bytes = checkpoint_path.stat().st_size
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    total_params = sum(p.numel() for p in state_dict.values())
    layers = {k: list(v.shape) for k, v in state_dict.items()}
    return {
        "file": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "total_parameters": total_params,
        "layers": layers
    }

def main():
    import torch
    import numpy as np

    config_path = ROOT / "config" / "paper2_experiment.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config_sha256 = compute_sha256(config_path)
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # Checkpoint auditing
    checkpoints_dir = ROOT / "rl_training" / "models" / "checkpoints"
    checkpoint_phase4 = checkpoints_dir / "dqn_curriculum_phase4.pth"
    phase4_info = inspect_model(checkpoint_phase4)

    all_phases = {}
    for p in sorted(checkpoints_dir.glob("*.pth")):
        all_phases[p.name] = inspect_model(p)

    # Agents of the manuscript (Section 3.7) -> phase-4 checkpoints.
    agents = {
        "Base-0": ["dqn_curriculum_phase4.pth"],
        "Base": [f"dqn_v0r_seed{s}_phase4.pth" for s in range(3)],
        "Ext-steps": [f"dqn_s2_b_seed{s}_phase4.pth" for s in range(5)],
        "Ext-sec": [f"dqn_s2_bp_seed{s}_phase4.pth" for s in range(5)],
    }
    missing = [f for files in agents.values() for f in files if not (checkpoints_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Agent checkpoints not found: {missing}")

    # Files the numbers, tables and figures of the manuscript are computed from.
    result_patterns = ("results/stage4e_eval/*/benchmark_traditional_vs_rl_raw.csv",
                       "results/stage4f_eval/*/benchmark_traditional_vs_rl_raw.csv",
                       "results/hp_validation_4e/*", "results/decision_cost_benchmark.json",
                       "results/paper_numbers.json", "results/discussion_analysis.json")
    result_files = {str(p.relative_to(ROOT)).replace("\\", "/"): compute_sha256(p)
                    for pattern in result_patterns for p in sorted(ROOT.glob(pattern)) if p.is_file()}

    git_info = get_git_info()

    from datetime import datetime, timezone

    manifest = {
        "manifest_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paper_title": "Reinforcement Learning for Firing Timing Decisions in Anti-Aircraft Fire Control Systems",
        "release_tag": git_info["tag"],
        "git": git_info,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "numpy_version": np.__version__
        },
        "experiment_configuration": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": config_sha256,
            "parameters": config_data
        },
        "published_model_phase4": phase4_info,
        "all_curriculum_checkpoints": all_phases,
        "agents": agents,
        "result_files": result_files
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    manifest_path = results_dir / "paper2_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Run manifest successfully generated at: {manifest_path}")
    print(f"Phase 4 Model Checksum: {phase4_info['sha256']}")
    print(f"Configuration Checksum: {config_sha256}")

if __name__ == "__main__":
    main()
