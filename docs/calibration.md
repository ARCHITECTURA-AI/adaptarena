# Difficulty calibration

Run `adaptarena calibrate --tier full`. Procedure: random policy, the two scripted baselines and the reference planner on **validation** seeds only,
no learned policy. The first calibration (4000/4000 restores, 40/50 drains) left the memory script at 500 steps on every condition, so nothing could
separate learned policies from a strong baseline; drains and restores were raised (`BASE_ENERGY_DRAIN` 55, `BASE_HYDRATION_DRAIN` 65, restores 3000),
training layouts got more walls and longer respawn delays, and the shift ranges in `core/distributions.py` were set. Those constants are hashed into
`analysis_plan.yaml` (`frozen_components`) before the locked suite was generated. The resulting per-condition numbers are in
`results/<tier>/calibration.json` and in section 2 of the report.

The difficulty rules fixed before any policy ran: a scenario enters any suite only if the planner certifies it; a condition on which every arm
reaches the horizon in more than 90% of episodes is flagged saturated and not counted as a win; stress levels without certificates are drawn as unsolvable.
