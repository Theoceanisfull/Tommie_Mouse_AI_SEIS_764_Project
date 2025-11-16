import numpy as np
from maze_env import MazeEnv

# Simple test maze (0 = free, 1 = wall)
maze = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
])

# Initialize environment with human render mode
env = MazeEnv(maze, render_mode="human")

# Reset environment
obs, info = env.reset()
print(f"Initial observation: {obs}, info: {info}")

done = False
while not done:
    action = env.action_space.sample()  # random move
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Action={action}, Obs={obs}, Reward={reward}, Done={done}")
    if done:
        print("Episode complete.")
        break

env.close()
