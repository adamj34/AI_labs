import gym

env = gym.make('InvertedPendulum-v4', render_mode="human")
env.reset(seed=42)

for _ in range(300):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
      observation, info = env.reset()
env.close()
