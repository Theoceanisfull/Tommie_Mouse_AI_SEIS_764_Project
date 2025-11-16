"""
maze_dataset.py
----------------

Loads a large dataset of pre-generated mazes from disk.

Supports:
    - Loading perfect or imperfect mazes
    - Random sampling
    - Shuffling
    - Train/test splits
    - Loading all mazes into memory OR lazily from disk
"""

import os
import numpy as np
from typing import List, Tuple
from maze_decoder import decode_raw_maze


# ============================================================
# FAST Utility: Load a single maze file into a NumPy array
# ============================================================
def load_maze_file(path: str) -> np.ndarray:
    """
    Loads a maze from text format containing 0s and 1s.
    Supports both formats:
        - "0 1 0 1"
        - "01010101"

    Automatically detects *raw expanded* maze format and decodes it
    into logical maze format.
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    matrix = []
    for line in lines:
        # Case 1: space-separated "0 1 1 0"
        if " " in line:
            row = np.fromstring(line, dtype=int, sep=" ")

        # Case 2: compact "011010"
        else:
            # frombuffer turns "0101" → array(["0","1","0","1"])
            row = np.frombuffer(line.encode(), dtype="S1").astype(int)

        matrix.append(row)

    raw = np.array(matrix, dtype=int)

    # ========================================================
    # Auto-detect raw expanded maze format
    # ========================================================
    H, W = raw.shape
    if H % 2 == 1 and W % 2 == 1 and H > 10 and W > 10:
        # Looks like a raw expanded recursive-backtracking maze
        # Example: 85×85 → logical 42×42
        logical = decode_raw_maze(raw)
        return logical

    # Otherwise return raw grid
    return raw


# ============================================================
# Class: MazeDataset
# ============================================================
class MazeDataset:
    def __init__(self,
                 root_dir: str = "mazes",
                 load_into_memory: bool = False):
        """
        root_dir must contain:
            perfect_maze/
            imperfect_maze/

        If load_into_memory=False, files are loaded only when needed.
        """

        self.root_dir = root_dir

        # Your folder names were singular
        self.perfect_dir = os.path.join(root_dir, "perfect_maze")
        self.imperfect_dir = os.path.join(root_dir, "imperfect_maze")

        if not os.path.exists(self.perfect_dir):
            raise FileNotFoundError(f"[ERROR] Folder not found: {self.perfect_dir}")

        if not os.path.exists(self.imperfect_dir):
            raise FileNotFoundError(f"[ERROR] Folder not found: {self.imperfect_dir}")

        # Collect all *.txt maze files
        self.perfect_files = sorted(
            [os.path.join(self.perfect_dir, f)
             for f in os.listdir(self.perfect_dir)
             if f.endswith(".txt")]
        )

        self.imperfect_files = sorted(
            [os.path.join(self.imperfect_dir, f)
             for f in os.listdir(self.imperfect_dir)
             if f.endswith(".txt")]
        )

        self.load_into_memory = load_into_memory

        # Optional: Load everything into memory
        if load_into_memory:
            print("[INFO] Loading all perfect mazes into RAM...")
            self.perfect_mazes = [load_maze_file(f) for f in self.perfect_files]

            print("[INFO] Loading all imperfect mazes into RAM...")
            self.imperfect_mazes = [load_maze_file(f) for f in self.imperfect_files]
        else:
            self.perfect_mazes = None
            self.imperfect_mazes = None

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------
    def total_perfect(self) -> int:
        return len(self.perfect_files)

    def total_imperfect(self) -> int:
        return len(self.imperfect_files)

    # --------------------------------------------------------
    # Get maze by index
    # --------------------------------------------------------
    def get_perfect(self, idx: int) -> np.ndarray:
        if self.load_into_memory:
            return self.perfect_mazes[idx]
        return load_maze_file(self.perfect_files[idx])

    def get_imperfect(self, idx: int) -> np.ndarray:
        if self.load_into_memory:
            return self.imperfect_mazes[idx]
        return load_maze_file(self.imperfect_files[idx])

    # --------------------------------------------------------
    # Random sampling
    # --------------------------------------------------------
    def sample_maze(self, perfect: bool = True) -> np.ndarray:
        """Randomly sample a perfect or imperfect maze."""
        if perfect:
            idx = np.random.randint(0, self.total_perfect())
            return self.get_perfect(idx)
        else:
            idx = np.random.randint(0, self.total_imperfect())
            return self.get_imperfect(idx)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------
    def train_test_split(self, split_ratio: float = 0.8):
        """
        Returns:
            perfect_train, perfect_test, imperfect_train, imperfect_test
        """
        p_total = self.total_perfect()
        i_total = self.total_imperfect()

        p_split = int(p_total * split_ratio)
        i_split = int(i_total * split_ratio)

        perfect_train = self.perfect_files[:p_split]
        perfect_test = self.perfect_files[p_split:]

        imperfect_train = self.imperfect_files[:i_split]
        imperfect_test = self.imperfect_files[i_split:]

        return perfect_train, perfect_test, imperfect_train, imperfect_test
