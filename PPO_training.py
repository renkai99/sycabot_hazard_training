from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sycabot_env import SycaBotEnv


class RewardComponentTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.progress_vals = []
        self.pickup_vals = []
        self.delivery_vals = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "reward_progress" in info:
                self.progress_vals.append(float(info["reward_progress"]))
            if "reward_pickup" in info:
                self.pickup_vals.append(float(info["reward_pickup"]))
            if "reward_delivery" in info:
                self.delivery_vals.append(float(info["reward_delivery"]))
        return True

    def _on_rollout_end(self) -> None:
        if self.progress_vals:
            self.logger.record("reward_components/progress", sum(self.progress_vals) / len(self.progress_vals))
            self.progress_vals.clear()
        if self.pickup_vals:
            self.logger.record("reward_components/pickup", sum(self.pickup_vals) / len(self.pickup_vals))
            self.pickup_vals.clear()
        if self.delivery_vals:
            self.logger.record("reward_components/delivery", sum(self.delivery_vals) / len(self.delivery_vals))
            self.delivery_vals.clear()


CONTINUE_FROM_PREVIOUS = True


env = SycaBotEnv(
    render_mode=None,
    num_robots=2,
    num_tasks=2,
    fire_spread_prob=0.02,
    fire_kill_prob=0.2,
    fire_cell_size=0.08,
)

tensorboard_log = "./ppo_sycabot_tensorboard/"
device = "cuda" if hasattr(env, "device") and env.device == "cuda" else "cpu"
base_model_path = Path("ppo_sycabot.zip")

if CONTINUE_FROM_PREVIOUS and base_model_path.exists():
    model = PPO.load(
        str(base_model_path),
        env=env,
        device=device,
        tensorboard_log=tensorboard_log,
    )
else:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=device,
    )

component_callback = RewardComponentTensorboardCallback()
model.learn(
    total_timesteps=int(1e7),
    progress_bar=True,
    callback=component_callback,
    reset_num_timesteps=not CONTINUE_FROM_PREVIOUS,
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"ppo_sycabot_{timestamp}")

# tensorboard --logdir ./ppo_sycabot_tensorboard/
