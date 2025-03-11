class Node:
    """
    A node for the A* search.
    """
    def __init__(self, pos, g, h, parent=None):
        self.pos = pos  # (x, y)
        self.g = g      # Cost from start to current node.
        self.h = h      # Heuristic estimate to goal from current node.
        self.f = g + h  # Total cost.
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

def get_neighbors(pos, world):
    """
    Returns valid neighboring positions (4-connected grid) for a given position.
    Checks boundaries and obstacles using the world object.
    """
    x, y = pos
    # Moves: up, down, left, right.
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = []
    for dx, dy in moves:
        new_x = x + dx
        new_y = y + dy
        if 0 <= new_x < world.width and 0 <= new_y < world.height:
            # Note: if world.grid is indexed as (row, col), then world.is_traversable(new_y, new_x)
            if world.is_traversable(new_y, new_x):
                neighbors.append((new_x, new_y))
    return neighbors 