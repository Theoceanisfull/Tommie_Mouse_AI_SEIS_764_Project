"""
train_shortest_path.py

Evaluates your maze dataset using A* shortest-path search.

This version:
    ✔ Uses ALL maze sizes automatically
    ✔ Works with 10×10 through 150×150 mazes
    ✔ Solves using A* (fast and optimal)
    ✔ Tracks solved %, avg path length
    ✔ Renders ONLY the final maze if enabled
"""

import heapq
import numpy as np
from tqdm import tqdm

from maze_dataset import MazeDataset, load_maze_file
from maze_env import MazeEnv


# ===============================================================
# USER CONTROLS
# ===============================================================

MAX_EVAL = 20
USE_PERFECT = True
USE_IMPERFECT = True

RENDER_DURING_EVAL = False    # <-- No renders during evaluation loops
RENDER_FINAL = True           # <-- Only render the very LAST solved maze


# ===============================================================
# A* PATHFINDING
# ===============================================================

def astar_solve(maze):
    n = maze.shape[0]
    start = (0, 0)
    goal = (n - 1, n - 1)

    if maze[start] == 1 or maze[goal] == 1:
        return None

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

    return None



# ===============================================================
# LOAD DATASET
# ===============================================================

dataset = MazeDataset("mazes", load_into_memory=False)
print(f"[INFO] Dataset loaded: {dataset.total_perfect()} perfect, {dataset.total_imperfect()} imperfect mazes")


# ===============================================================
# Build evaluation list
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

solved = 0
total_path_length = 0
last_solved_maze = None
last_solved_path = None

for i, path in enumerate(maze_files, start=1):

    maze = load_maze_file(path)
    print(f"[{i}] Maze: {path} | Size: {maze.shape}")

    path_cells = astar_solve(maze)

    if path_cells is None:
        print("    ❌ Unsolvable maze")
        continue

    print(f"    ✔ Solved! Path length = {len(path_cells)}")

    solved += 1
    total_path_length += len(path_cells)
    last_solved_maze = maze
    last_solved_path = path_cells



# ===============================================================
# METRICS SUMMARY
# ===============================================================

print("\n==================== SUMMARY ====================")
print(f"Solved: {solved}/{len(maze_files)}  ({solved/len(maze_files)*100:.1f}%)")

if solved > 0:
    print(f"Average Path Length: {total_path_length/solved:.1f}")
else:
    print("Average Path Length: N/A (no solved mazes)")

print("=================================================\n")



# ===============================================================
# RENDER FINAL MAZE ONLY
# ===============================================================

if RENDER_FINAL and last_solved_maze is not None:
    print("[INFO] Rendering FINAL solved maze...\n")

    env = MazeEnv(last_solved_maze, render_mode="human")
    env.reset()

    for (r, c) in last_solved_path:
        env.agent_pos = (r, c)
        env.render()

    env.close()

else:
    print("[INFO] No final maze rendered (either disabled or no solved maze).")

