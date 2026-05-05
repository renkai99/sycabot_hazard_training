from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sycabot_env import SycaBotEnv


class RewardComponentTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._keys = ["reward_progress", "reward_pickup", "reward_delivery", "reward_survival"]
        self._bufs = {k: [] for k in self._keys}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for k in self._keys:
                if k in info:
                    self._bufs[k].append(float(info[k]))
        return True

    def _on_rollout_end(self) -> None:
        for k, vals in self._bufs.items():
            if vals:
                tag = "reward_components/" + k.removeprefix("reward_")
                self.logger.record(tag, sum(vals) / len(vals))
                vals.clear()


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
