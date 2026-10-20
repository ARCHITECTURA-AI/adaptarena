"""`make reproduce`: run every stage for a tier, in dependency order, each behind its gate."""
from __future__ import annotations

import json
import os
import time

from .experiments import (evaluate_headline, get_val_suite, load_tier, manifest, run_headline_training, run_parity,
                          tier_dir)


def run(tier_name: str, workers: int = 4) -> None:
    t0 = time.time()
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    manifest(tdir, tier)
    from . import plan as planmod
    info = planmod.verify_plan(require_git=False)
    print(f"plan hash {info['plan_hash'][:16]} (git verified: {info['git_verified']})", flush=True)
    get_val_suite(tdir, tier, 0)
    from .calibration import run as calibrate
    calibrate(tier_name)
    from .tune import run as tune
    tune(tier_name, workers=workers)
    p = run_parity(tier_name, workers)
    print("parity gate passed:", p["passed"], flush=True)
    run_headline_training(tier_name, workers)
    evaluate_headline(tier_name)
    from .ablations import run as ablate
    ablate(tier_name, workers)
    from .memory import run as memory
    memory(tier_name, workers)
    # v2.1 (analysis_plan_v2_1.yaml): audit and fresh suite first, then the queue in priority order. The one-off training
    # profile (`adaptarena v21 profile`) is not part of a reproduction. Smoke runs only the stages that finish in minutes.
    from .v2_1 import QUEUE, run_stage
    for st in ("audit", "suite", "rescore"):
        run_stage(st, tier_name, workers)
    for st in (("imitation", "mutation", "sweep") if tier_name == "smoke" else QUEUE):
        run_stage(st, tier_name, workers)
    from .stress import run as stress
    stress(tier_name)
    from .redteam import run as redteam
    redteam(tier_name)
    from .probes import run as probes
    probes(tier_name)
    from .site import build as site
    site(tier_name)
    from .report import build as report
    report(tier_name)
    print(f"reproduce[{tier_name}] finished in {(time.time() - t0) / 60:.1f} min", flush=True)
