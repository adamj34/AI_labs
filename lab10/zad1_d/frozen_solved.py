import gymnasium as gym
    
env = gym.make('FrozenLake-v1', render_mode="human")


observation, info = env.reset(seed=42)

actions = [1, 0, 1, 2, 1, 1]

for i in range(len(actions)):
   action = actions[i]
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()


