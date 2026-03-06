from stable_baselines3 import PPO
from sycabot_env import SycaBotEnv


env = SycaBotEnv(
    render_mode=None,
    num_robots=2,
    num_tasks=2,
    fire_spread_prob=0.02,
    fire_kill_prob=0.5,
    fire_cell_size=0.08,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_sycabot_tensorboard/",
    device="cuda" if hasattr(env, "device") and env.device == "cuda" else "cpu",
)

model.learn(total_timesteps=int(1e5), progress_bar=True)
model.save("ppo_sycabot")

# tensorboard --logdir ./ppo_sycabot_tensorboard/
