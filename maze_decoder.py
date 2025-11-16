import numpy as np

def decode_raw_maze(raw: np.ndarray) -> np.ndarray:
    """
    Decodes an expanded maze format like your dataset:
    - Even rows/cols are walls
    - Odd rows/cols are real logical maze cells
    - raw size = 2*N+1
    - logical size = N

    Output:
        logical_maze: (N x N) array with 0=free, 1=wall
    """
    h, w = raw.shape

    # Raw format must be odd dimension
    assert h % 2 == 1 and w % 2 == 1, \
        f"Raw maze must be odd-sized, got {raw.shape}"

    # Logical resolution
    N = (h - 1) // 2

    logical = np.ones((N, N), dtype=int)

    # Iterate logical cells
    for r in range(N):
        for c in range(N):
            raw_r = 1 + 2 * r
            raw_c = 1 + 2 * c

            # If the cell location is a 0 → open
            if raw[raw_r, raw_c] == 0:
                logical[r, c] = 0

            # Check possible walls around cell
            # Up
            if r > 0 and raw[raw_r - 1, raw_c] == 0:
                logical[r - 1, c] = 0
            # Down
            if r < N - 1 and raw[raw_r + 1, raw_c] == 0:
                logical[r + 1, c] = 0
            # Left
            if c > 0 and raw[raw_r, raw_c - 1] == 0:
                logical[r, c - 1] = 0
            # Right
            if c < N - 1 and raw[raw_r, raw_c + 1] == 0:
                logical[r, c + 1] = 0

    return logical
