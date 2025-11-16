"""
train_shortest_path.py

Evaluates your maze dataset using A* shortest-path search.

This version:
    ✔ No maze size filtering — uses ALL maze sizes automatically
    ✔ Works with 10×10 through 150×150 mazes
    ✔ Solves using A* (fast and optimal)
    ✔ Optional visualization through MazeEnv
"""

import heapq
import numpy as np
from tqdm import tqdm

from maze_dataset import MazeDataset, load_maze_file
from maze_env import MazeEnv


# ===============================================================
# USER CONTROLS
# ===============================================================

MAX_EVAL = 20        # Evaluate this many mazes total
USE_PERFECT = True   # Include mazes/perfect_mazes
USE_IMPERFECT = True # Include mazes/imperfect_mazes
RENDER = True        # Visualize solution in Pygame window


# ===============================================================
# A* PATHFINDING
# ===============================================================

def astar_solve(maze):
    n = maze.shape[0]
    start = (0, 0)
    goal = (n - 1, n - 1)

    if maze[start] == 1 or maze[goal] == 1:
        return None  # unsolvable

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    pq = [(0, 0, start, None)]
    visited = set()
    parents = {}

    while pq:
        _, cost, (r, c), parent = heapq.heappop(pq)

        if (r, c) in visited:
            continue
        visited.add((r, c))
        parents[(r, c)] = parent

        if (r, c) == goal:
            # reconstruct path
            path = []
            cur = (r, c)
            while cur:
                path.append(cur)
                cur = parents[cur]
            return list(reversed(path))

        # neighbors
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and maze[nr, nc] == 0:
                if (nr, nc) not in visited:
                    priority = cost + 1 + heuristic((nr, nc), goal)
                    heapq.heappush(pq, (priority, cost + 1, (nr, nc), (r, c)))

    return None  # unsolvable


# ===============================================================
# LOAD DATASET
# ===============================================================

dataset = MazeDataset("mazes", load_into_memory=False)
print(f"[INFO] Dataset loaded: {dataset.total_perfect()} perfect, {dataset.total_imperfect()} imperfect mazes")


# ===============================================================
# Build evaluation list (no size filtering)
# ===============================================================

maze_files = []

if USE_PERFECT:
    maze_files += dataset.perfect_files

if USE_IMPERFECT:
    maze_files += dataset.imperfect_files

maze_files = maze_files[:MAX_EVAL]

print(f"[INFO] Evaluating {len(maze_files)} total mazes\n")


# ===============================================================
# EVALUATION LOOP
# ===============================================================

for i, path in enumerate(maze_files, start=1):

    maze = load_maze_file(path)
    print(f"[{i}] Maze: {path} | Size: {maze.shape}")

    path_cells = astar_solve(maze)

    if path_cells is None:
        print("    ❌ Unsolvable maze")
        continue

    print(f"    ✔ Solved! Path length = {len(path_cells)}")

    if RENDER:
        env = MazeEnv(maze, render_mode="human")
        env.reset()

        # animate solution
        for (r, c) in path_cells:
            env.agent_pos = (r, c)
            env.render()

        env.close()
