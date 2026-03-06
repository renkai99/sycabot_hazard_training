# SycaBot Hazard Rescue

Multi-robot PPO environment for hazard-aware maze rescue.

## What changed

- Joint control for `N` robots (`action = [v1,w1,...,vN,wN]`).
- Full multi-robot observation (robot states + task states + fire grid + indicators).
- Stochastic cell-based fire propagation on a discretized map.
- Robot destruction risk near burning cells (`fire_kill_prob`).
- Task workflow: `pending -> carried -> delivered` (or `contaminated`).
- Safety/task indicators included in observation and info.
- Helper rewards: move-to-task, move-to-exit after pickup, and team spread reward.

## Tunable environment args

In `SycaBotEnv(...)`:

- `num_robots`
- `num_tasks`
- `max_steps`
- `fire_spread_prob`
- `fire_kill_prob`
- `fire_cell_size`
- `robot_radius`

## Run

Train:

```bash
python3 PPO_training.py
```

Test/render:

```bash
python3 sycabot_test.py
```
