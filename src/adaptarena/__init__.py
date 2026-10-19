"""AdaptArena v2. The core (environment, planner, baselines) needs only NumPy;
the Gymnasium wrapper is registered lazily and the training lab is a separate extra."""
from .core import (BatchEnv, EnvConfig, Scenario, ScenarioSpec, SurvivalEnv, build_scenario,  # noqa: F401
                   plan, verify_plan, with_intervention)

__version__ = "2.0.0"


def register_envs() -> None:
    from .gym_env import register_envs as _r
    _r()


try:  # register AdaptArena-Survival-v1 when Gymnasium is available (it is optional in the browser)
    register_envs()
except ImportError:  # pragma: no cover
    pass
