# stan gry i zestaw akcji sa ciagle

import gymnasium as gym

env = gym.make(
    "LunarLander-v2",
    continuous = True,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
    render_mode = "human",
)

observation, info = env.reset(seed=42)

for i in range(300):
   action = env.action_space.sample()
   print("Akcja nr ", i, ": ", action)
   observation, reward, terminated, truncated, info = env.step(action)
 
   if terminated or truncated:
      observation, info = env.reset()
env.close()