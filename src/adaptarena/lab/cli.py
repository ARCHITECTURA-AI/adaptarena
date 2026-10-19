"""Command line: `adaptarena <command> --tier smoke|dev|full`. Every command is resumable."""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="adaptarena", description="AdaptArena v2 research stack")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_, tier=True, workers=False):
        sp = sub.add_parser(name, help=help_)
        if tier:
            sp.add_argument("--tier", default="smoke", choices=["smoke", "dev", "full"])
        if workers:
            sp.add_argument("--workers", type=int, default=4)
        return sp

    add("check-env", "run the Gymnasium environment checker", tier=False)
    add("throughput", "benchmark simulator throughput", tier=False)
    add("calibrate", "difficulty calibration on validation seeds (scripts + planner only)")
    add("lock-plan", "hash analysis_plan.yaml (commit the result before touching the test suite)", tier=False)
    add("parity", "own PPO vs SB3 parity gate", workers=True)
    add("headline", "train the four arms and evaluate on the locked suite", workers=True)
    add("ablations", "single-change ablation ladder", workers=True)
    add("memory", "memory track: LSTM (and SSM) vs feedforward", workers=True)
    add("stress", "stress curves and Kaplan-Meier survival")
    add("redteam", "red team search over scenario space")
    add("probes", "what-it-learned probes")
    add("site", "build the static showcase site from results")
    add("report", "build the technical report from results")
    add("reproduce", "run every stage for a tier", workers=True)
    a = p.parse_args(argv)

    if a.cmd == "check-env":
        import gymnasium as gym
        from gymnasium.utils.env_checker import check_env
        import adaptarena  # noqa: F401
        check_env(gym.make("AdaptArena-Survival-v1").unwrapped, skip_render_check=True)
        print("Gymnasium environment checker: passed")
    elif a.cmd == "throughput":
        from .throughput import main as tp
        tp()
    elif a.cmd == "lock-plan":
        from . import plan
        print(plan.lock_plan())
    elif a.cmd == "calibrate":
        from .calibration import run
        run(a.tier)
    elif a.cmd == "parity":
        from .experiments import run_parity
        print(json.dumps({k: v for k, v in run_parity(a.tier, a.workers).items() if k != "curves"}, indent=1))
    elif a.cmd == "headline":
        from .experiments import run_headline_training, evaluate_headline
        run_headline_training(a.tier, a.workers)
        r = evaluate_headline(a.tier)
        print(json.dumps(r["decision"], indent=1))
    elif a.cmd == "ablations":
        from .ablations import run
        run(a.tier, a.workers)
    elif a.cmd == "memory":
        from .memory import run
        run(a.tier, a.workers)
    elif a.cmd == "stress":
        from .stress import run
        run(a.tier)
    elif a.cmd == "redteam":
        from .redteam import run
        run(a.tier)
    elif a.cmd == "probes":
        from .probes import run
        run(a.tier)
    elif a.cmd == "site":
        from .site import build
        build(a.tier)
    elif a.cmd == "report":
        from .report import build
        build(a.tier)
    elif a.cmd == "reproduce":
        from .reproduce import run
        run(a.tier, a.workers)


if __name__ == "__main__":
    main()
