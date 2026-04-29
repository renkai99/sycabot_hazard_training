# SycaBot Hazard Rescue

Multi-robot PPO environment for hazard-aware rescue in a lab maze. Robots navigate a fire-spreading environment, pick up task items, and deliver them to exits without being destroyed.

## Environment overview

The environment is a bounded 2D workspace (`~3.1 m × 6.2 m`) with static wall obstacles and multiple exits. A stochastic cellular fire spreads from a random seed cell each episode. Robots start near exits and must pick up task items from the interior and deliver them before the fire destroys the items or the robots.

- `N` unicycle robots, jointly controlled via a flat action vector `[v₁, ω₁, ..., vₙ, ωₙ]`
- Tasks transition: `pending → carried → delivered` (or `contaminated` if fire reaches them)
- Robots are destroyed by: boundary exit, obstacle contact, inter-robot collision, or fire proximity
- Episode ends when all robots are destroyed, all tasks are contaminated, or all robots return to exits

---

## Observation space

The observation is a flat float32 vector of length `N_robots × 8 + N_tasks × 4 + grid_cells + 2`.

### Robot block — 8 features per robot (ordered by robot index)

| Feature | Description |
|---------|-------------|
| `x` | World x-position (meters) |
| `y` | World y-position (meters) |
| `theta` | Heading angle (radians, −π to π) |
| `alive` | 1.0 if robot is alive, 0.0 if destroyed |
| `carrying` | 1.0 if the robot is carrying a task item, 0.0 otherwise |
| `task_d` | Euclidean distance to the nearest pending task (meters) |
| `exit_d` | Euclidean distance to the nearest exit point (meters) |
| `fire_d` | Euclidean distance to the nearest burning grid cell center (meters) |

### Task block — 4 features per task (ordered by task index)

| Feature | Description |
|---------|-------------|
| `x` | World x-position of the task item (meters) |
| `y` | World y-position of the task item (meters) |
| `status` | 0=pending, 1=carried, 2=delivered, 3=contaminated |
| `carried` | 1.0 if currently being carried by a robot, 0.0 otherwise |

### Fire grid

Flattened binary array of shape `(grid_x, grid_y)` where each cell is `1` if burning, `0` otherwise. Cell size is controlled by `fire_cell_size` (default 0.08 m), giving approximately `39 × 78 = 3042` cells for the default lab bounds.

### Global indicators — 2 scalars

| Feature | Description |
|---------|-------------|
| `global_safety_indicator` | 1.0 at episode start; flips permanently to 0.0 on the first safety event (any robot death) |
| `global_task_indicator` | 0.0=no task touched, 1.0=at least one task picked up, 2.0=at least one task delivered |

---

## Reward components

All components are summed each step. Key weights are configurable via `SycaBotEnv(...)` constructor arguments.

| Component | Formula | Default weight | Notes |
|-----------|---------|---------------|-------|
| **Pickup reward** | `+pickup_reward × pickups_this_step` | 1000 | Sparse; triggers when a robot enters ≤0.18 m of a pending task |
| **Delivery reward** | `+delivery_reward × deliveries_this_step` | 1000 | Sparse; triggers when a carrying robot enters ≤0.20 m of an exit |
| **Task progress** | `+weight × Σ max(prev_visible_task_dist − curr_visible_task_dist, 0)` | 20 | Potential-based shaping; only counts when the task is visible (line-of-sight); resets when visibility is lost |
| **Exit progress** | `+weight × Σ max(prev_visible_exit_dist − curr_visible_exit_dist, 0)` | 20 | Identical logic, but for carrying robots approaching exits |
| **Safety event penalty** | `−3.0 × safety_events` | fixed | Counts every robot death in the step (boundary, obstacle, collision, fire) |
| **Action smoothness penalty** | `−smooth_action_weight × mean(Δv²)` | 0.30 | Penalizes abrupt linear velocity changes |
| **Turn smoothness penalty** | `−turn_smooth_weight × mean(Δω²)` | 0.20 | Penalizes abrupt angular velocity changes |
| **Jerk penalty** | `−jerk_weight × mean(jerk²)` | 0.08 | Second-order finite difference of actions |
| **Direction flip penalty** | `−direction_flip_weight × num_flips` | 0.10 | Counts robots that reverse linear direction within one step |
| **Time step penalty** | `−0.01` | fixed | Constant per-step cost to encourage task efficiency |
| **Catastrophic failure override** | `−200` (replaces everything) | fixed | Triggered if all robots are destroyed in a single step |

> **Note:** `boundary_death_penalty` (20 × boundary_deaths) and `−1.0 × obstacle_hits` are computed and exposed in `info` but are currently commented out of the total reward.

---

## Suggestions: improving the observation space

**1. Egocentric (robot-relative) coordinates**
The current observation uses absolute world coordinates for robot and task positions. Switching to positions expressed relative to each robot's own pose (translated and rotated into the robot's local frame) improves generalization across spawn locations and reduces what the policy needs to learn implicitly.

**2. sin/cos heading encoding**
The raw heading angle `theta` has a discontinuity at ±π. Replacing it with `(sin(theta), cos(theta))` eliminates this discontinuity and gives a smooth, bounded representation that neural networks handle much better.

**3. Explicit inter-robot relative positions**
Each robot's observation currently lacks direct information about where the other robots are relative to itself. Adding the relative position and heading of each teammate makes collision avoidance and coordination significantly easier to learn, especially as `num_robots` grows.

**4. LIDAR-like proximity sensors**
`fire_d` and `exit_d` are single scalars, which lose directional information. Replacing or augmenting them with N evenly-spaced range readings (obstacle distance, fire distance, wall distance) in the robot's local frame provides richer local awareness without the full fire grid.

**5. Per-task fire risk**
Include for each task the distance from the task to the nearest burning cell. This lets the policy prioritize rescuing tasks that are about to be contaminated.

**6. Normalized positions**
All distances and coordinates should be normalized by the workspace dimensions or a fixed scale factor to keep inputs in a consistent range (e.g., [−1, 1]). This avoids gradient scaling issues and makes the policy more robust to arena size changes.

**7. Action history**
Including the previous one or two joint actions as part of the observation gives the policy explicit memory of recent motion, which can reduce the need to re-learn jerk avoidance from rewards alone and helps with partially observable dynamics.

---

## Suggestions: improving the reward structure

**1. Use the team spread reward**
`_spread_reward()` computes mean pairwise distance between alive robots and returns a value in [0, 1], but it is never added to the total reward. Including it (e.g., `+0.5 × spread_reward`) encourages robots to cover different areas of the arena and reduces task competition.

**2. Continuous fire proximity penalty**
The current fire penalty is purely stochastic (kill probability near fire cells). Adding a continuous shaping term like `−k / (fire_d + ε)` gives the agent a smooth gradient away from fire long before a death event occurs, making fire avoidance much easier to learn.

**3. Contamination prevention bonus**
Give a small positive reward when a robot picks up a task that is within a threshold distance of an active fire cell (`task_fire_dist < threshold`). This rewards urgency and teaches the agent to prioritize at-risk items.

**4. Survival-per-step bonus**
A small per-step reward (+0.01 to +0.05) for each alive robot counters the tendency to sacrifice robots when it seems expedient, and balances the time-step penalty.

**5. Collision proximity shaping**
Rather than only penalizing actual deaths, add a potential-based penalty for inter-robot distance below a soft threshold (e.g., `−k × max(0, collision_distance × 2 − dist)²`). This creates a repulsive field that guides the policy away from near-collisions.

**6. Fire-weighted task progress**
Weight the task progress reward by the urgency of the targeted task: tasks closer to fire get a higher progress multiplier. This guides robots to rescue at-risk tasks first without requiring any additional manual priority logic.

---

## Tunable environment parameters

In `SycaBotEnv(...)`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_robots` | 2 | Number of jointly-controlled robots |
| `num_tasks` | 2 | Number of task items to rescue |
| `max_steps` | 1000 | Episode length cutoff |
| `fire_spread_prob` | 0.02 | Probability per step that fire spreads to each neighboring cell |
| `fire_kill_prob` | 0.2 | Probability per step that a robot near fire is destroyed |
| `fire_cell_size` | 0.08 m | Resolution of the fire grid |
| `robot_radius` | 0.08 m | Robot body radius; also sets collision and obstacle contact threshold |
| `pickup_reward` | 1000 | Reward for picking up a task item |
| `delivery_reward` | 1000 | Reward for delivering a task item to an exit |
| `task_progress_reward_weight` | 20 | Weight for move-to-task shaping reward |
| `exit_progress_reward_weight` | 20 | Weight for move-to-exit shaping reward (when carrying) |
| `smooth_action_weight` | 0.30 | Weight for linear velocity smoothness penalty |
| `turn_smooth_weight` | 0.20 | Weight for angular velocity smoothness penalty |
| `jerk_weight` | 0.08 | Weight for jerk penalty |
| `direction_flip_weight` | 0.10 | Weight for direction reversal penalty |
| `boundary_death_penalty` | 20 | Penalty magnitude for boundary deaths (currently inactive) |

---

## Run

Train:

```bash
python3 PPO_training.py
```

Test / render:

```bash
python3 sycabot_test.py
```

Monitor training in TensorBoard:

```bash
tensorboard --logdir ./ppo_sycabot_tensorboard/
```
