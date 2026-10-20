"""Fixed constants of AdaptArena-Survival-v1.

Everything that touches simulator state is an integer. Energy and hydration are
stored in hundredths (0..10000); drains are integer hundredths; cost multipliers
are integer percents (130 means 1.3x).
"""

GRID = 15
HORIZON = 500
N_ACTIONS = 6
VIEW = 9
VIEW_R = VIEW // 2
N_CHANNELS = 8
STATS_DIM = 11
OBS_DIM = N_CHANNELS * VIEW * VIEW + STATS_DIM

STAT_MAX = 10000
MAX_SITES = 6
NEVER = 10**6  # "site never disappears"

# Actions
UP, DOWN, LEFT, RIGHT, INTERACT, WAIT = range(6)
ACTION_NAMES = ("up", "down", "left", "right", "interact", "wait")
DR = (-1, 1, 0, 0, 0, 0)
DC = (0, 0, -1, 1, 0, 0)

# Observation channels
CH_WALL, CH_FOOD, CH_WATER, CH_FOOD_DEPLETED, CH_WATER_DEPLETED, CH_HAZARD, CH_FLOOR, CH_VISIBLE = range(8)

# Physics defaults (hundredths of a stat point)
BASE_ENERGY_DRAIN = 55
BASE_HYDRATION_DRAIN = 65
MOVE_EXTRA_ENERGY = 30  # scaled by the movement-cost percent
COLD_EXTRA_ENERGY = 30
FOOD_RESTORE = 3000
WATER_RESTORE = 3000

# Death causes, in priority order for the single "primary" label.
CAUSE_HAZARD, CAUSE_DEHYDRATION, CAUSE_STARVATION = 1, 2, 4
CAUSE_ORDER = (
    (CAUSE_HAZARD, "hazard"),
    (CAUSE_DEHYDRATION, "dehydration"),
    (CAUSE_STARVATION, "starvation"),
)


def primary_cause(flags: int) -> str:
    for bit, name in CAUSE_ORDER:
        if flags & bit:
            return name
    return "none"


# Stats-vector layout
S_ENERGY, S_HYDRATION, S_COLD = 0, 1, 2
S_PREV_ACTION = slice(3, 9)
S_PREV_VALID = 9
S_LAST_DRAIN = 10
DRAIN_NORM = 400  # hundredths; (drain/100)/4

ENV_ID = "AdaptArena-Survival-v1"
