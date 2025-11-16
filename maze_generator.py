"""
maze_generator.py
------------------

Provides solvable maze generation for RL training.

Features:
    ✔ Always solvable mazes (DFS backtracking)
    ✔ Difficulty scaling
    ✔ Optional complexity enhancements (loops, pruning)

Usage:

    from maze_generator import generate_maze

    maze = generate_maze(size=7, difficulty=0.3)
"""

import numpy as np


# ============================================================
# Utility: Add loops (optional difficulty enhancer)
# ============================================================
def add_loops(maze, probability=0.05):
    """
    Randomly removes some walls to create cycles in the maze.
    Increasing probability increases difficulty.
    """
    m = maze.copy()
    rows, cols = m.shape

    for r in range(rows):
        for c in range(cols):
            if m[r, c] == 1 and np.random.rand() < probability:
                m[r, c] = 0

    return m


# ============================================================
# Utility: Remove dead-ends (optional simplifier or complexifier)
# ============================================================
def remove_dead_ends(maze, iterations=1):
    """
    Removes dead ends by carving extra passages.
    - Low iterations -> easier maze
    - High iterations -> more complex branching
    """
    m = maze.copy()
    rows, cols = m.shape

    for _ in range(iterations):
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):

                if m[r, c] == 0:
                    # Count adjacent walls
                    walls = 0
                    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                        if m[r+dr, c+dc] == 1:
                            walls += 1

                    # Dead-end detected
                    if walls == 3:
                        # Punch a hole in one wall
                        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                            if m[r+dr, c+dc] == 1:
                                m[r+dr, c+dc] = 0
                                break

    return m


# ============================================================
# Base maze generator: DFS Backtracking (always solvable)
# ============================================================
def generate_solvable_maze(n: int) -> np.ndarray:
    """
    Always generates a fully solvable maze using DFS backtracking.
    Maze is returned as n x n grid of 0 (free) and 1 (wall).
    """

    # Work internally with odd dimensions
    size = n if n % 2 == 1 else n + 1
    maze = np.ones((size, size), dtype=int)

    # Directions: up, down, left, right (jump 2 steps)
    directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]

    # Start position
    stack = [(0, 0)]
    maze[0, 0] = 0

    while stack:
        r, c = stack[-1]
        np.random.shuffle(directions)

        carved = False
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < size and 0 <= nc < size and maze[nr, nc] == 1:
                # Carve path
                maze[r + dr // 2, c + dc // 2] = 0
                maze[nr, nc] = 0
                stack.append((nr, nc))
                carved = True
                break

        if not carved:
            stack.pop()

    # Trim to requested n×n size
    trimmed = maze[:n, :n]
    trimmed[0, 0] = 0
    trimmed[n-1, n-1] = 0

    return trimmed


# ============================================================
# Public API: Generate Maze with Difficulty
# ============================================================
def generate_maze(size: int,
                  difficulty: float = 0.0,
                  add_cycles: bool = True,
                  remove_deadends: bool = False) -> np.ndarray:
    """
    Main function used by the trainer.

    Parameters:
        size (int): size of maze (NxN)
        difficulty (float): 0.0 to 1.0
            - 0.0 = simple DFS maze
            - 1.0 = many loops + few dead-ends
        add_cycles (bool): Adds loops proportional to difficulty
        remove_deadends (bool): Smooths maze or increases complexity

    Returns:
        N x N numpy grid of 0 (path) and 1 (wall)
    """

    maze = generate_solvable_maze(size)

    # Add loops depending on difficulty
    if add_cycles and difficulty > 0:
        cycle_prob = min(0.20, difficulty * 0.20)   # Max loop prob = 20%
        maze = add_loops(maze, probability=cycle_prob)

    # Remove dead ends (optional)
    if remove_deadends:
        iterations = int(difficulty * 3)  # up to 3 pruning passes
        if iterations > 0:
            maze = remove_dead_ends(maze, iterations=iterations)

    # Ensure safe start/goal
    maze[0, 0] = 0
    maze[size - 1, size - 1] = 0

    return maze
