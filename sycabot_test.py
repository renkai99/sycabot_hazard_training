from stable_baselines3 import PPO
from sycabot_env import SycaBotEnv
import time


env = SycaBotEnv(
    render_mode="human",
    num_robots=2,
    num_tasks=2,
    fire_spread_prob=0.02,
    fire_kill_prob=0.5,
    fire_cell_size=0.08,
)

model = PPO.load("ppo_sycabot")

obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.07)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
