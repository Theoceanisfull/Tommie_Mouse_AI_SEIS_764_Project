import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv


# ---------------------------
# 1. Define hyperparameters
# ---------------------------
EPISODES = 1000          # total training episodes
MAX_STEPS = 500          # max steps per episode
ALPHA = 0.1              # learning rate
GAMMA = 0.9              # discount factor
EPSILON = 1.0            # exploration probability
EPSILON_DECAY = 0.995    # decay per episode
EPSILON_MIN = 0.05       # floor for epsilon

# ---------------------------
# 2. Create environment
# ---------------------------
maze = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
])

env = MazeEnv(maze, render_mode=None)  # set to "human" to visualize occasionally

n_states = env.observation_space.n
n_actions = env.action_space.n

# ---------------------------
# 3. Initialize Q-table
# ---------------------------
Q = np.zeros((n_states, n_actions))

# For logging
episode_rewards = []

# ---------------------------
# 4. Training loop
# ---------------------------
for episode in range(EPISODES):
    state, info = env.reset()
    total_reward = 0
    done = False

    for step in range(MAX_STEPS):
        # ε-greedy action selection
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(Q[state, :])     # exploit best known action

        # Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-learning update
        best_next_action = np.argmax(Q[next_state, :])
        td_target = reward + GAMMA * Q[next_state, best_next_action]
        td_error = td_target - Q[state, action]
        Q[state, action] += ALPHA * td_error

        total_reward += reward
        state = next_state

        if done:
            break

    # ε-decay (explore less over time)
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    # Optional progress printing
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1}/{EPISODES} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {EPSILON:.3f}")

    # Optional visualization
    if (episode + 1) % 200 == 0:
        env.render_mode = "human"
        _ = env.reset()
        done = False
        for _ in range(50):
            action = np.argmax(Q[state, :])
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.render_mode = None

env.close()

# ---------------------------
# 5. Plot training performance
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(episode_rewards, label="Episode reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Progress")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
