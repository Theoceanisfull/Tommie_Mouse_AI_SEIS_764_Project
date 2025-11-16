# 🐭 Tommie Mouse Reinforcement Learning Maze Project
University of St. Thomas – SEIS 764 Final Project
This project explores how reinforcement learning (RL), classical pathfinding, and deep learning solve grid-based mazes. It includes:
A fully custom Gymnasium-compatible maze environment
Tabular Q-learning on fixed and randomly generated mazes
Evaluation on a large dataset (perfect + imperfect mazes)
A fast A* shortest-path benchmark
Optional emoji visualization
Planned upgrade to Deep Q-Networks (DQN)

# 🔧 1. Activate Python Environment
source rl_env/bin/activate
Make sure required packages (Gymnasium, NumPy, Matplotlib, Pygame, Pillow) are installed.

# 🧱 2. Maze Environment (maze_env.py)
The environment defines:
Grid maze with 0 = free and 1 = wall
Agent start: (0, 0)
Goal: bottom-right cell
Actions: up, right, down, left
Optional emoji visualization (🐭 + 🧀)
Compatible with any N × N maze size
Test the environment
python test_maze_env.py
This will run random actions to verify:
Maze loads correctly
Walls block movement
Rendering works correctly
Gymnasium documentation: https://gymnasium.farama.org

# 🧠 3. Tabular Q-Learning (Fixed Maze)
File: q_learning_fixed_maze.py
This script:
Loads one maze
Uses tabular Q-learning
Learns state-action values
Uses reward shaping:
+20 when reaching the goal
-5 for hitting a wall
-1 per move
Great for learning the fundamentals and validating RL logic.

# 🔁 4. Tabular Q-Learning on Random Generated Mazes
File: q_learning_gen_maze.py
This version:
Uses maze_generator.py
Generates solvable mazes at increasing difficulty
Lets the agent learn general patterns instead of memorizing a layout
Limitations:
Tabular Q-learning cannot scale to larger mazes
Works well only on ~5×5 or similar sizes

# 📁 5. Maze Dataset (Perfect + Imperfect Mazes)
Located in:
mazes/
    perfect_maze/
    imperfect_maze/
The dataset contains 3,000 total mazes at random sizes (10×10 → 150×150+).
Scripts:
maze_dataset.py — loads maze files, handles indexing, lazy loading
maze_decoder.py — converts irregular raw text formats into numpy grids
This is required to train RL models on real, pre-generated maze distributions.

# ⭐ 6. A* Shortest-Path Benchmark
File: train_shortest_path.py
Runs optimal A* pathfinding on the dataset.
Features:
Works on all maze sizes
Finds optimal shortest path
Detects unsolvable mazes
Optional visualization
Helps benchmark maze difficulty before using RL
You can control rendering:
RENDER = True
RENDER_FIRST_K = 3
This shows only the first K solved mazes — avoids dozens of Pygame windows.

# 🎨 7. Visualization Improvements
Rendering includes:
Purple walls
White free-space
🐭 mouse for the agent
🧀 cheese for the goal
Smooth animation when stepping through a solution path
Pygame rendering can be turned off during training and on during evaluation.

# 🚀 8. Future Work: Deep Q-Network (DQN)
Because tabular Q-learning cannot generalize to large mazes (100×100+), the next stage is:
Build a Deep Q-Network (DQN):
CNN-based state encoder
Replay buffer
Target network
ε-greedy policy
Mini-batch training
The DQN pipeline will:
Convert maze grids to tensor inputs
Learn generalized navigation strategies
Scale to large mazes
Train across the entire dataset
This is the natural continuation of the project into modern RL.

# ✔ Repository Structure Summary
Component	Description
maze_env.py	Custom Gymnasium Maze Environment
test_maze_env.py	Test harness for visualization
maze_generator.py	Creates solvable random mazes
q_learning_fixed_maze.py	Tabular RL on one maze
q_learning_gen_maze.py	Tabular RL with curriculum
maze_dataset.py	Loads dataset mazes
maze_decoder.py	Converts text mazes → arrays
train_shortest_path.py	A* solver & benchmark
mazes/	Perfect + imperfect dataset
