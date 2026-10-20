import pytest

import adaptarena.core.params as P


@pytest.fixture
def harsh_physics():
    """Drains and restores scaled so tiny horizons actually produce deaths (brute-force tests)."""
    keep = {k: getattr(P, k) for k in ("BASE_ENERGY_DRAIN", "BASE_HYDRATION_DRAIN", "FOOD_RESTORE",
                                       "WATER_RESTORE", "MOVE_EXTRA_ENERGY", "COLD_EXTRA_ENERGY")}
    P.BASE_ENERGY_DRAIN, P.BASE_HYDRATION_DRAIN = 520, 620
    P.FOOD_RESTORE, P.WATER_RESTORE = 2500, 2500
    P.MOVE_EXTRA_ENERGY, P.COLD_EXTRA_ENERGY = 150, 120
    yield
    for k, v in keep.items():
        setattr(P, k, v)
