# Lagrangian PPO for Constrained Rescue RL

This document describes how to reformulate the SycaBot hazard rescue task as a Constrained Markov Decision Process (CMDP) and how to modify the existing codebase to train with Lagrangian PPO.

---

## 1. Why constrained RL?

The current PPO setup blends safety penalties (robot deaths, fire exposure) directly into the scalar reward. This creates several problems:

- **Reward hacking**: the agent can trade off lives for faster task completion if the delivery reward dominates.
- **Tuning fragility**: the balance between safety and task performance is controlled entirely by hand-tuned penalty weights.
- **No formal guarantee**: the policy may converge to a solution that violates safety at some tolerable average rate without the designer intending this.

Constrained RL separates the objective into a **primary reward** to maximize and one or more **cost budgets** to satisfy. The agent is required to keep expected cumulative costs below specified thresholds while maximizing the reward. Lagrangian relaxation is the standard and simplest approach to enforce these constraints in policy gradient methods.

---

## 2. CMDP formulation

A Constrained MDP is defined by the standard MDP tuple plus a set of cost functions `cᵢ(s, a)` and constraint thresholds `dᵢ`:

```
maximize   E[Σ γᵗ r(s, a)]
subject to E[Σ γᵗ cᵢ(s, a)] ≤ dᵢ   for i = 1, …, K
```

The Lagrangian relaxation converts this to an unconstrained minimax problem:

```
L(π, λ) = E[Σ γᵗ r(s, a)] − Σᵢ λᵢ · (E[Σ γᵗ cᵢ(s, a)] − dᵢ)
```

where `λᵢ ≥ 0` are Lagrange multipliers. Training alternates between:

1. **Policy update** (gradient ascent on `L` w.r.t. `π`): maximize the penalized reward.
2. **Multiplier update** (gradient ascent on `−L` w.r.t. `λ`): increase `λᵢ` when constraint `i` is violated, decrease when satisfied.

The multiplier update is:

```
λᵢ ← max(0, λᵢ + α_λ · (J_Cᵢ(π) − dᵢ))
```

where `J_Cᵢ(π)` is the estimated expected discounted cost under the current policy and `α_λ` is the dual learning rate.

---

## 3. Which reward components become constraints?

### Unconstrained objective (keep in reward)

| Component | Rationale |
|-----------|-----------|
| Pickup reward | Primary task performance metric |
| Delivery reward | Primary task performance metric |
| Task / exit progress shaping | Dense guidance toward task completion |
| Time step penalty | Encourages efficiency |

### Constraint candidates (move to cost signals)

| Cost signal | Constraint meaning | Suggested budget `d` |
|-------------|-------------------|----------------------|
| **`cost_safety`** — total safety events per step (boundary + obstacle + inter-robot + fire deaths) | Robots should not be destroyed. Budget of 0 means zero tolerance; a small positive budget (e.g., 0.05 per episode) allows occasional learning failures during exploration. | 0.05 (expected deaths per episode) |
| **`cost_fire_death`** — fire-related deaths only | Fire is the primary environmental hazard and warrants its own constraint, tighter than general collision. | 0.02 per episode |
| **`cost_contamination`** — tasks contaminated by fire | Prevents the agent from letting items burn even if it improves robot survival. | 0.10 per episode |
| **`cost_smoothness`** (optional) | For physical deployment: constrain the jerk and direction-flip penalty to a hard ceiling to protect hardware. Only meaningful if robots will run on real hardware. | Deployment-specific |

The smoothness costs are optional constraints that only matter for hardware deployment; leave them as reward penalties for sim-only training.

---

## 4. Code modifications

### Step 1 — Add cost signals to the environment

In [sycabot_env.py](sycabot_env.py), extend the `info` dict returned by `step()` with the cost values. The relevant quantities are already computed:

```python
# In SycaBotEnv.step(), replace/extend the info dict:
info = {
    # ... existing keys ...
    "cost_safety":       float(safety_events),          # total robot deaths this step
    "cost_fire_death":   float(fire_deaths),             # fire-caused deaths this step
    "cost_contamination": float(                         # tasks contaminated this step
        np.sum((self.task_status == 3)) - prev_contaminated_count
    ),
}
```

Track `prev_contaminated_count` at the top of `step()`:

```python
prev_contaminated_count = int(np.sum(self.task_status == 3))
```

Remove (or reduce to zero) the `-3.0 × safety_events` penalty from the scalar reward — that signal now lives in the cost.

### Step 2 — Create a cost-tracking callback

```python
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LagrangianCallback(BaseCallback):
    """Tracks per-rollout cost returns and updates Lagrange multipliers."""

    def __init__(self, lambdas, cost_keys, budgets, dual_lr=1e-3, verbose=0):
        super().__init__(verbose)
        self.lambdas = lambdas          # dict: cost_key -> current lambda value
        self.cost_keys = cost_keys      # list of cost signal names
        self.budgets = budgets          # dict: cost_key -> budget d_i
        self.dual_lr = dual_lr
        self._episode_costs = {k: [] for k in cost_keys}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for k in self.cost_keys:
                if k in info:
                    self._episode_costs[k].append(float(info[k]))
        return True

    def _on_rollout_end(self) -> None:
        for k in self.cost_keys:
            if not self._episode_costs[k]:
                continue
            mean_cost = float(np.mean(self._episode_costs[k]))
            violation = mean_cost - self.budgets[k]
            self.lambdas[k] = max(0.0, self.lambdas[k] + self.dual_lr * violation)
            self.logger.record(f"lagrangian/lambda_{k}", self.lambdas[k])
            self.logger.record(f"lagrangian/mean_cost_{k}", mean_cost)
            self._episode_costs[k].clear()
```

### Step 3 — Build a Lagrangian-augmented reward wrapper

The cleanest way to inject the Lagrange penalty without modifying the SB3 PPO internals is a Gymnasium `RewardWrapper` that reads the current `lambdas` dict and adjusts the scalar reward each step:

```python
import gymnasium as gym

class LagrangianRewardWrapper(gym.Wrapper):
    """Subtracts λᵢ · cᵢ from the task reward each step."""

    def __init__(self, env, lambdas, cost_keys):
        super().__init__(env)
        self.lambdas = lambdas      # shared dict, mutated by the callback
        self.cost_keys = cost_keys

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        penalty = sum(self.lambdas[k] * info.get(k, 0.0) for k in self.cost_keys)
        return obs, reward - penalty, terminated, truncated, info
```

Because `lambdas` is a mutable dict shared between the wrapper and the callback, the policy always trains against the current multiplier values without any extra plumbing.

### Step 4 — Wire it together in the training script

Replace [PPO_training.py](PPO_training.py) with:

```python
from stable_baselines3 import PPO
from sycabot_env import SycaBotEnv
from lagrangian_utils import LagrangianRewardWrapper, LagrangianCallback

COST_KEYS = ["cost_safety", "cost_fire_death", "cost_contamination"]
BUDGETS   = {"cost_safety": 0.05, "cost_fire_death": 0.02, "cost_contamination": 0.10}
DUAL_LR   = 1e-3

lambdas = {k: 0.0 for k in COST_KEYS}   # start with no penalty; let constraint violations drive them up

base_env = SycaBotEnv(
    render_mode=None,
    num_robots=2,
    num_tasks=2,
    fire_spread_prob=0.02,
    fire_kill_prob=0.2,
    fire_cell_size=0.08,
)
env = LagrangianRewardWrapper(base_env, lambdas, COST_KEYS)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_sycabot_tensorboard/")

callback = LagrangianCallback(lambdas, COST_KEYS, BUDGETS, dual_lr=DUAL_LR)
model.learn(total_timesteps=int(1e7), callback=callback, progress_bar=True)
```

---

## 5. Training dynamics and what to monitor

In TensorBoard, add these panels alongside the existing reward components:

| Metric | Expected behaviour |
|--------|--------------------|
| `lagrangian/lambda_cost_safety` | Rises when robots die frequently, falls when the policy becomes safe |
| `lagrangian/lambda_cost_fire_death` | Should converge to a stable positive value if fire avoidance is learned |
| `lagrangian/mean_cost_safety` | Should decrease below `d=0.05` at convergence |
| `reward_components/progress` | Should not collapse — if lambdas grow too fast the policy may sacrifice task performance |

Common failure modes:
- **Lambda diverges upward**: dual learning rate `α_λ` is too high; reduce by 10×.
- **Policy ignores constraints**: reward signal too sparse or too large relative to costs; scale pickup/delivery rewards down or increase initial lambda values.
- **Oscillation**: alternate between feasible and infeasible policies; add a small momentum term to the lambda update or clip the maximum lambda.

---

## 6. Advanced options

### Use an existing constrained RL library

If you prefer not to implement the Lagrangian update manually, [OmniSafe](https://github.com/PKU-Alignment/omnisafe) provides a drop-in `LagrangianPPO` and `LagrangianTRPO` that accept a `Safety-Gymnasium`-compatible environment. You would need to wrap `SycaBotEnv` to expose a `cost()` method matching the Safety-Gymnasium API.

### Primal-dual with trust regions (TRPO-Lagrangian)

For more stable constraint satisfaction, replace the PPO inner loop with TRPO. The multiplier update remains identical. OmniSafe and `safe-rl` both provide this variant.

### Multiple constraint levels

For hardware deployment, add a second, tighter constraint for fire deaths with a higher initial lambda to give fire avoidance strict priority over the softer boundary-collision constraint.

---

## 7. Summary of changes per file

| File | Change |
|------|--------|
| [sycabot_env.py](sycabot_env.py) | Add `cost_safety`, `cost_fire_death`, `cost_contamination` to `info`; remove (or zero) the `-3 × safety_events` penalty from the scalar reward |
| [PPO_training.py](PPO_training.py) | Wrap env with `LagrangianRewardWrapper`; attach `LagrangianCallback` |
| `lagrangian_utils.py` (new) | `LagrangianRewardWrapper` and `LagrangianCallback` classes |
