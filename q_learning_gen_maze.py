import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from maze_generator import generate_maze  

# =========================================================
# 1️⃣ Hyperparameters
# =========================================================
EPISODES = 2000
MAX_STEPS = 300
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05

MAZE_SIZE = 5
EPISODES_PER_MAZE = 20
RENDER_INTERVAL = 400


# =========================================================
# 2️⃣ Initialize environment + Q-table
# =========================================================
initial_maze = generate_maze(size=MAZE_SIZE, difficulty=0.0)
env = MazeEnv(initial_maze)

n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

episode_rewards = []
success_history = []
wall_hits_history = []


# =========================================================
# 3️⃣ Training loop
# =========================================================
for episode in range(EPISODES):

    # Curriculum difficulty scaling (0.0 → 1.0)
    difficulty = min(1.0, episode / EPISODES)

    # Generate a new solvable maze periodically
    if episode % EPISODES_PER_MAZE == 0:
        maze = generate_maze(size=MAZE_SIZE, difficulty=difficulty)
        env = MazeEnv(maze)

    state, info = env.reset()
    total_reward = 0
    done = False
    wall_hits = 0

    for step in range(MAX_STEPS):

        # ε-greedy action
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # -----------------------------------
        # Reward shaping (correct use of hit_wall!)
        # -----------------------------------
        if info["hit_wall"]:
            reward = -5.0
            wall_hits += 1
        elif terminated:
            reward = +20.0
        else:
            reward = -1.0

        # -----------------------------------
        # Q-learning update
        # -----------------------------------
        best_next = np.argmax(Q[next_state, :])
        td_target = reward + GAMMA * Q[next_state, best_next]
        td_error = td_target - Q[state, action]
        Q[state, action] += ALPHA * np.clip(td_error, -1.0, 1.0)

        total_reward += reward
        state = next_state

        if done:
            break

    # Update epsilon
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    # Logging
    episode_rewards.append(total_reward)
    success_history.append(1 if terminated else 0)
    wall_hits_history.append(wall_hits)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        success_rate = np.mean(success_history[-100:])
        print(
            f"Episode {episode+1}/{EPISODES} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Success: {success_rate:.2f} | "
            f"ε={EPSILON:.3f}"
        )

    # Optional visualization
    if (episode + 1) % RENDER_INTERVAL == 0:
        env.render_mode = "human"
        env.reset()
        s = env._pos_to_state((0, 0))
        for _ in range(40):
            action = np.argmax(Q[s, :])
            s, _, t, tr, _ = env.step(action)
            if t or tr:
                break
        env.render_mode = None

env.close()


# =========================================================
# 4️⃣ Plot performance curves
# =========================================================
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# Reward curve
axs[0].plot(episode_rewards)
axs[0].set_title("Total Reward per Episode")
axs[0].grid(True)

# Success rate (moving average)
window = 50
success_ma = np.convolve(success_history, np.ones(window)/window, mode="same")
axs[1].plot(success_ma, color="green")
axs[1].set_title("Success Rate (Moving Avg, window=50)")
axs[1].grid(True)

# Wall hits per episode
axs[2].plot(wall_hits_history, color="red")
axs[2].set_title("Wall Hits per Episode")
axs[2].grid(True)

plt.tight_layout()
plt.show()


# =========================================================
# 5️⃣ Evaluation on unseen mazes
# =========================================================
print("\n--- Evaluation on Unseen Solvable Mazes ---\n")

for i in range(1, 6):
    difficulty = 0.75  # mid-high challenge
    maze = generate_maze(size=MAZE_SIZE, difficulty=difficulty)
    env = MazeEnv(maze, render_mode="human")

    state, _ = env.reset()
    total_reward = 0
    done = False

    for _ in range(MAX_STEPS):
        action = np.argmax(Q[state, :])
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
        done = terminated or truncated
        if done:
            break

    print(f"Test Maze {i}: Total Reward = {total_reward}")
    env.close()
