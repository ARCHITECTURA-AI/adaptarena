"""Pre-registration: the analysis plan is hashed and committed before the locked suite exists.

`analysis_plan.yaml` lists hypotheses, arms, seed counts, metrics, thresholds, the parity-gate
criteria and the memory-track predictions. Its SHA-256 lives in `analysis_plan.sha256`, which is
committed to git *before* the locked test suite is generated or opened. The evaluation harness
refuses to touch the suite unless the plan on disk still matches the committed hash, and it logs
every access. Deviations are dated entries in `plan_amendments.jsonl`; the original is never edited.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from typing import Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PLAN = os.path.join(REPO, "analysis_plan.yaml")
HASH = os.path.join(REPO, "analysis_plan.sha256")
AMEND = os.path.join(REPO, "plan_amendments.jsonl")


class PlanError(RuntimeError):
    pass


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


PLAN_V2_1 = os.path.join(REPO, "analysis_plan_v2_1.yaml")
HASH_V2_1 = os.path.join(REPO, "analysis_plan_v2_1.sha256")


def hash_path_for(plan: str) -> str:
    """analysis_plan.yaml -> analysis_plan.sha256, analysis_plan_v2_1.yaml -> analysis_plan_v2_1.sha256."""
    return os.path.splitext(plan)[0] + ".sha256"


def lock_plan(plan: str = PLAN, hash_path: Optional[str] = None) -> str:
    hash_path = hash_path or hash_path_for(plan)
    h = file_hash(plan)
    with open(hash_path, "w") as f:
        f.write(h + "\n")
    return h


def _git(*args) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout
    except Exception:
        return None


def git_commit_of_hash(hash_path: str = HASH) -> Optional[str]:
    out = _git("log", "-1", "--format=%H", "--", os.path.relpath(hash_path, REPO))
    return out.strip() or None if out is not None else None


def verify_plan(plan: str = PLAN, hash_path: str = HASH, require_git: bool = True) -> dict:
    """Raise PlanError unless the plan on disk matches the committed hash."""
    if not os.path.exists(hash_path):
        raise PlanError("analysis_plan.sha256 is missing: lock the plan before touching the test suite")
    locked = open(hash_path).read().strip()
    actual = file_hash(plan)
    if locked != actual:
        raise PlanError(f"analysis plan changed after locking ({actual[:12]} != {locked[:12]}); "
                        "record a dated amendment instead of editing the plan")
    info = {"plan_hash": actual, "git_commit": None, "git_verified": False}
    if _git("rev-parse", "--is-inside-work-tree") is not None:
        dirty = _git("status", "--porcelain", "--", os.path.relpath(plan, REPO), os.path.relpath(hash_path, REPO))
        commit = git_commit_of_hash(hash_path)
        if commit and not dirty:
            info.update(git_commit=commit, git_verified=True)
        elif require_git:
            raise PlanError("the plan hash is not committed to git yet (commit analysis_plan.yaml and "
                            "analysis_plan.sha256 before generating or opening the test suite)")
    return info


def log_access(log_path: str, purpose: str, info: dict) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "purpose": purpose, **info}
    head = _git("rev-parse", "HEAD")
    rec["head_commit"] = head.strip() if head else None
    amend = []
    if os.path.exists(AMEND):
        amend = [json.loads(l) for l in open(AMEND) if l.strip()]
    rec["amendments"] = len(amend)
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def add_amendment(text: str) -> dict:
    rec = {"date": time.strftime("%Y-%m-%d"), "original_plan_hash": file_hash(PLAN), "amendment": text}
    with open(AMEND, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def check_frozen(plan: str = PLAN) -> None:
    """The simulator, difficulty constants, planner and scripts must be byte-identical to what was locked."""
    import yaml
    spec = yaml.safe_load(open(plan))
    for rel, want in spec["frozen_components"].items():
        path = os.path.join(REPO, rel)
        if file_hash(path) != want:
            raise PlanError(f"{rel} changed after the plan was locked; record a dated amendment before evaluating")
