"""Simulator throughput on the machine at hand. Every experiment budget is sized against this number."""
from __future__ import annotations

import json
import os
import platform
import time

import numpy as np

from ..core import BatchEnv, SurvivalEnv, build_scenario
from ..core.distributions import sample_train
from .plan import REPO


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure(seconds: float = 3.0) -> dict:
    rng = np.random.default_rng(0)
    out = {"machine": cpu_name(), "cores": os.cpu_count(), "platform": platform.platform(),
           "python": platform.python_version(), "numpy": np.__version__, "steps_per_second": {}}
    scs = [build_scenario(sample_train(rng)) for _ in range(64)]
    env = SurvivalEnv()
    env.reset(scs[0])
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        if env.done:
            env.reset(scs[n % 64])
        env.step(int(rng.integers(0, 6)))
        n += 1
    out["steps_per_second"]["reference_scalar"] = n / (time.time() - t0)
    for N in (8, 16, 64, 256):
        b = BatchEnv(N)
        sc = scs * (N // 64 + 1)
        b.reset_all(sc[:N])
        n, t0 = 0, time.time()
        while time.time() - t0 < seconds:
            b.step(rng.integers(0, 6, N))
            b.flat_obs()
            n += N
            for i in np.nonzero(b.done)[0]:
                b.reset_env(int(i), sc[int(rng.integers(0, 64))])
        out["steps_per_second"][f"batch_{N}_with_observations"] = n / (time.time() - t0)
    return out


def main() -> None:
    r = measure()
    path = os.path.join(REPO, "results", "throughput.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(r, open(path, "w"), indent=1)
    print(json.dumps(r, indent=1))
