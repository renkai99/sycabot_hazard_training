# SycaBot Hazard Rescue

Multi-robot PPO environment for hazard-aware rescue in a lab maze. Robots navigate a fire-spreading environment, pick up task items, and deliver them to exits without being destroyed.

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone <repo-url>
cd sycabot_hazard_training

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Runtime dependencies only
pip install -r requirements.txt

# Development dependencies (adds TensorBoard and Jupyter)
pip install -r requirements-dev.txt
```

`requirements.txt` includes: `numpy`, `gymnasium`, `pygame`, `stable-baselines3[extra]`

`requirements-dev.txt` adds: `tensorboard`, `jupyter`

---

## Environment overview

The environment is a bounded 2D workspace (`~3.1 m × 6.2 m`) with static wall obstacles and multiple exits. A stochastic cellular fire spreads from a random seed cell each episode. Robots start near exits and must pick up task items from the interior and deliver them before the fire destroys the items or the robots.

- `N` unicycle robots, jointly controlled via a flat action vector `[v₁, ω₁, ..., vₙ, ωₙ]`
- Tasks transition: `pending → carried → delivered` (or `contaminated` if fire reaches them)
- Robots are destroyed by: boundary exit, obstacle contact, inter-robot collision, or fire proximity
- Episode ends when all robots are destroyed or all tasks reach a terminal state (delivered or contaminated)

---

## Observation space

The observation is a flat float32 vector of length `N_robots × 22 + N_tasks × 4 + 2 + 2 × N_robots`.

For the default 2-robot / 2-task configuration this is **58 dimensions** (vs. 3068 in the previous flat-fire-grid design).

### Robot block — 22 features per robot (ordered by robot index)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `x` | World x-position (meters) |
| 2 | `y` | World y-position (meters) |
| 3 | `sin(θ)` | Sine of heading — smooth, bounded heading encoding |
| 4 | `cos(θ)` | Cosine of heading |
| 5 | `alive` | 1.0 if alive, 0.0 if destroyed |
| 6 | `carrying` | 1.0 if carrying a task item, 0.0 otherwise |
| 7 | `task_d` | Euclidean distance to nearest pending task (meters) |
| 8 | `task_orient` | Robot-relative bearing to nearest pending task (radians, −π to π) |
| 9 | `exit_d` | Euclidean distance to nearest exit (meters) |
| 10 | `exit_orient` | Robot-relative bearing to nearest exit (radians, −π to π) |
| 11–16 | `fire{1,2,3}_d`, `fire{1,2,3}_o` | Edge distance and robot-relative bearing to the 3 nearest burning cells; distance is `max(0, centre_dist − cell_size/2)`; slots with no fire are filled with `(10.0, 0.0)` |
| 17–22 | `obs{1,2,3}_d`, `obs{1,2,3}_o` | Point-to-segment distance and robot-relative bearing to the 3 nearest obstacle segments |

### Task block — 4 features per task (ordered by task index)

| Feature | Description |
|---------|-------------|
| `x` | World x-position of the task item (meters) |
| `y` | World y-position of the task item (meters) |
| `status` | 0=pending, 1=carried, 2=delivered, 3=contaminated |
| `carried` | 1.0 if currently being carried, 0.0 otherwise |

### Global indicators — 2 scalars

| Feature | Description |
|---------|-------------|
| `safety_indicator` | 1.0 at episode start; latches permanently to 0.0 on the first robot death |
| `task_indicator` | Recomputed each step from task states: 0.0=none touched, 1.0=at least one carried, 2.0=at least one delivered |

### Action history — 2 × N_robots scalars

The previous joint action `[v₁, ω₁, ..., vₙ, ωₙ]` is appended to the observation. This gives the policy direct access to the information it needs to minimise the smoothness, jerk, and direction-flip penalties.

---

## Reward components

All components are summed each step. Weights are configurable via `SycaBotEnv(...)` constructor arguments.

| Component | Formula | Default weight | Notes |
|-----------|---------|---------------|-------|
| **Pickup reward** | `+pickup_reward × pickups_this_step` | 1000 | Sparse; triggers when a robot enters ≤0.18 m of a pending task |
| **Delivery reward** | `+delivery_reward × deliveries_this_step` | 1000 | Sparse; triggers when a carrying robot enters ≤0.20 m of an exit |
| **Task progress** | `+weight × Σ max(prev_visible_task_dist − curr_visible_task_dist, 0)` | 8 | Potential-based shaping; only counts when the task has line-of-sight; resets when visibility is lost |
| **Exit progress** | `+weight × Σ max(prev_visible_exit_dist − curr_visible_exit_dist, 0)` | 10 | Same logic for carrying robots approaching exits |
| **Death penalty** | `−death_penalty × safety_events` | 3000 | Applied per robot destroyed this step (boundary, obstacle, collision, or fire) |
| **Survival reward** | `+survival_reward_weight × Σ alive_robots` | 0.05 | Small per-step bonus for each robot still alive; counteracts sacrificing robots for task speed |
| **Action smoothness penalty** | `−smooth_action_weight × mean(Δv²)` | 1.0 | Penalises abrupt linear velocity changes |
| **Turn smoothness penalty** | `−turn_smooth_weight × mean(Δω²)` | 0.20 | Penalises abrupt angular velocity changes |
| **Jerk penalty** | `−jerk_weight × mean(jerk²)` | 0.08 | Second-order finite difference of actions |
| **Direction flip penalty** | `−direction_flip_weight × num_flips` | 0.10 | Counts robots that reverse linear direction within one step |
| **Time step penalty** | `−0.01` | fixed | Constant per-step cost to encourage efficiency |
| **Catastrophic failure override** | `−200` (replaces everything) | fixed | Triggered if all robots are destroyed in a single step |

---

## Suggestions: improving the observation space

**1. Egocentric (robot-relative) coordinates**
Task and robot positions are still in absolute world coordinates. Switching to positions expressed relative to each robot's own pose (translated and rotated into the local frame) improves generalisation across spawn locations and reduces what the policy needs to learn implicitly.

**2. Explicit inter-robot relative positions**
Each robot's block does not contain information about the other robots' locations. Adding the relative position and heading of each teammate makes collision avoidance and coordination much easier to learn, especially as `num_robots` grows.

**3. Per-task fire risk**
Include for each task the distance from the task to the nearest burning cell. This lets the policy prioritise rescuing items that are about to be contaminated.

**4. Normalised positions**
Normalising all coordinates by the workspace dimensions (e.g., to [−1, 1]) keeps network inputs on a consistent scale and makes the policy more robust to arena size changes.

---

## Suggestions: improving the reward structure

**1. Continuous fire proximity penalty**
The fire danger is currently stochastic (probabilistic kill near cells). Adding a continuous shaping term like `−k / (fire_d + ε)` gives a smooth gradient away from fire long before a death event, making avoidance much easier to learn.

**2. Contamination prevention bonus**
Give a small positive reward when a robot picks up a task that is within a threshold distance of an active fire cell. This rewards urgency and teaches the agent to prioritise at-risk items.

**3. Collision proximity shaping**
Add a potential-based repulsion for inter-robot distances below a soft threshold (e.g., `−k × max(0, 2 × collision_distance − dist)²`) so the policy learns to avoid near-collisions, not just fatal ones.

**4. Fire-weighted task progress**
Weight the task progress reward by the urgency of the targeted task: tasks closer to fire get a higher multiplier. This guides robots to rescue at-risk tasks first without additional manual priority logic.

---

## Tunable environment parameters

In `SycaBotEnv(...)`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_robots` | 2 | Number of jointly-controlled robots |
| `num_tasks` | 2 | Number of task items to rescue |
| `max_steps` | 1000 | Episode length cutoff |
| `fire_spread_prob` | 0.02 | Probability per step that fire spreads to each neighbouring cell |
| `fire_kill_prob` | 0.2 | Probability per step that a robot near fire is destroyed |
| `fire_cell_size` | 0.08 m | Resolution of the fire grid |
| `robot_radius` | 0.08 m | Robot body radius; also sets collision and obstacle contact threshold |
| `pickup_reward` | 1000 | Reward for picking up a task item |
| `delivery_reward` | 1000 | Reward for delivering a task item to an exit |
| `death_penalty` | 3000 | Penalty per robot destroyed in a step |
| `survival_reward_weight` | 0.05 | Per-step reward per alive robot |
| `task_progress_reward_weight` | 8 | Weight for move-to-task shaping reward |
| `exit_progress_reward_weight` | 10 | Weight for move-to-exit shaping reward (when carrying) |
| `smooth_action_weight` | 1.0 | Weight for linear velocity smoothness penalty |
| `turn_smooth_weight` | 0.20 | Weight for angular velocity smoothness penalty |
| `jerk_weight` | 0.08 | Weight for jerk penalty |
| `direction_flip_weight` | 0.10 | Weight for direction reversal penalty |

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
