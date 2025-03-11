import heapq
import numpy as np
from src.planning.base_planner import BasePlanner

class DStarPlanner(BasePlanner):
    """
    Dynamic A* (D*) algorithm for path planning.
    
    D* is an incremental search algorithm that efficiently replans paths
    in dynamic environments. It propagates changes in the graph backward
    from the goal to the current position.
    
    Key features:
    - Efficient replanning in dynamic environments
    - Propagates changes backward from goal to current position
    - Handles both increasing and decreasing edge costs
    """
    
    def __init__(self, world):
        super().__init__(world)
        # State tags
        self.NEW = 0
        self.OPEN = 1
        self.CLOSED = 2
        
        # State information
        self.t = {}  # State tags (NEW, OPEN, CLOSED)
        self.h = {}  # State costs
        self.k = {}  # State backpointers
        self.open_list = []  # Priority queue
        self.open_hash = set()  # For faster lookups
        
        # Current robot position
        self.current_pos = None
        self.goal = None
        
        # Map changes
        self.changed_cells = []
    
    def get_neighbors(self, point):
        """Get valid neighboring points (4-connected grid)."""
        x, y = point
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                neighbors.append((nx, ny))
        return neighbors
    
    def cost(self, a, b):
        """
        Calculate the cost between two adjacent cells.
        Returns 1 for traversable cells, infinity for obstacles.
        """
        if not self.is_valid_point(b):
            return float('inf')
        return 1
    
    def initialize(self, start, goal):
        """Initialize the D* algorithm."""
        self.current_pos = start
        self.goal = goal
        
        # Initialize state information
        self.t = {}
        self.h = {}
        self.k = {}
        self.open_list = []
        self.open_hash = set()
        
        # Set goal state
        self.t[goal] = self.NEW
        self.h[goal] = 0
        self.insert(goal, 0)
    
    def insert(self, state, h_new):
        """Insert or update a state in the open list."""
        if state not in self.t:
            self.t[state] = self.NEW
        
        if self.t[state] == self.OPEN:
            # State is already in OPEN list, update its priority
            self.k[state] = min(self.k[state], h_new)
        else:
            # State is either NEW or CLOSED
            self.k[state] = h_new
            self.t[state] = self.OPEN
            heapq.heappush(self.open_list, (self.k[state], id(state), state))
            self.open_hash.add(state)
    
    def remove_min(self):
        """Remove and return the state with minimum k-value from the open list."""
        if not self.open_list:
            return None
        
        k_val, _, state = heapq.heappop(self.open_list)
        self.open_hash.remove(state)
        self.t[state] = self.CLOSED
        return state, k_val
    
    def get_kmin(self):
        """Get the minimum k-value in the open list."""
        if not self.open_list:
            return -1
        return self.open_list[0][0]
    
    def modify_cost(self, x, y):
        """Modify the cost of a cell and its neighbors."""
        if (x, y) not in self.t:
            return
        
        self.insert((x, y), self.h.get((x, y), float('inf')))
        
        # Process neighbors
        for neighbor in self.get_neighbors((x, y)):
            if neighbor not in self.t:
                continue
            
            old_cost = self.cost(neighbor, (x, y))
            self.insert(neighbor, self.h.get(neighbor, float('inf')))
    
    def process_state(self):
        """Process the state with minimum k-value."""
        if not self.open_list:
            return -1
        
        k_old = self.get_kmin()
        state, k_val = self.remove_min()
        
        if state is None:
            return -1
        
        # If k_old < h[state], state needs to be reinserted
        if k_old < self.h.get(state, float('inf')):
            for neighbor in self.get_neighbors(state):
                if (self.h.get(neighbor, float('inf')) <= k_old and 
                    self.h.get(state, float('inf')) > self.h.get(neighbor, float('inf')) + self.cost(neighbor, state)):
                    self.h[state] = self.h[neighbor] + self.cost(neighbor, state)
        
        # If k_old == h[state], state is correctly evaluated
        if k_old == self.h.get(state, float('inf')):
            for neighbor in self.get_neighbors(state):
                if (self.t.get(neighbor, self.NEW) == self.NEW or 
                    (self.t.get(neighbor, self.NEW) == self.OPEN and 
                     self.h.get(neighbor, float('inf')) > self.h.get(state, float('inf')) + self.cost(state, neighbor)) or 
                    (self.t.get(neighbor, self.NEW) == self.CLOSED and 
                     self.h.get(neighbor, float('inf')) > self.h.get(state, float('inf')) + self.cost(state, neighbor))):
                    self.h[neighbor] = self.h[state] + self.cost(state, neighbor)
                    self.insert(neighbor, self.h[neighbor])
        
        # If k_old > h[state], state is overconsistent
        else:
            for neighbor in self.get_neighbors(state):
                if (self.t.get(neighbor, self.NEW) == self.NEW or 
                    (self.t.get(neighbor, self.NEW) == self.OPEN and 
                     self.h.get(neighbor, float('inf')) > self.h.get(state, float('inf')) + self.cost(state, neighbor)) or 
                    (self.t.get(neighbor, self.NEW) == self.CLOSED and 
                     self.h.get(neighbor, float('inf')) > self.h.get(state, float('inf')) + self.cost(state, neighbor))):
                    self.h[neighbor] = self.h[state] + self.cost(state, neighbor)
                    self.insert(neighbor, self.h[neighbor])
                else:
                    if (self.t.get(neighbor, self.NEW) == self.OPEN and 
                        self.h.get(neighbor, float('inf')) <= self.h.get(state, float('inf')) + self.cost(state, neighbor) and 
                        state != self.goal and 
                        self.h.get(state, float('inf')) > self.h.get(neighbor, float('inf')) + self.cost(neighbor, state)):
                        self.h[state] = self.h[neighbor] + self.cost(neighbor, state)
                        self.insert(state, self.h[state])
        
        return self.get_kmin()
    
    def replan(self):
        """Replan the path from current position to goal."""
        # Process states until current position is CLOSED or open list is empty
        while self.t.get(self.current_pos, self.NEW) != self.CLOSED:
            k_min = self.process_state()
            if k_min == -1:
                # Open list is empty, no path exists
                return None
        
        # Extract path from current position to goal
        path = [self.current_pos]
        current = self.current_pos
        
        while current != self.goal:
            # Find the neighbor with minimum h-value + cost
            min_h = float('inf')
            next_node = None
            
            for neighbor in self.get_neighbors(current):
                if neighbor in self.h:
                    h_val = self.h[neighbor] + self.cost(current, neighbor)
                    if h_val < min_h:
                        min_h = h_val
                        next_node = neighbor
            
            if next_node is None or min_h == float('inf'):
                # No valid neighbor found
                return None
            
            path.append(next_node)
            current = next_node
        
        return path
    
    def update_map(self, changed_cells):
        """Update the map with changed cells and replan."""
        self.changed_cells = changed_cells
        
        # Process all changed cells
        for cell in self.changed_cells:
            x, y = cell
            self.modify_cost(x, y)
    
    def plan(self, start, goal, **kwargs):
        """
        Plan a path from start to goal using Dynamic A* (D*).
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            changed_cells: Optional list of cells that have changed.
            
        Returns:
            List of tuples representing the path from start to goal, or None if no path is found.
        """
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return None
        
        # Get changed cells if provided
        changed_cells = kwargs.get('changed_cells', [])
        
        # Initialize if this is a new search or if goal changed
        if self.goal != goal or not self.h:
            self.initialize(start, goal)
            
            # Initial planning
            while self.t.get(start, self.NEW) != self.CLOSED:
                k_min = self.process_state()
                if k_min == -1:
                    # Open list is empty, no path exists
                    return None
        else:
            # Update current position
            self.current_pos = start
            
            # Update map with changed cells
            if changed_cells:
                self.update_map(changed_cells)
        
        # Replan path
        return self.replan()
    
    def interactive_plan(self, start, goal, callback=None, **kwargs):
        """
        Interactive version of D* that can provide step-by-step visualization.
        
        Parameters:
            start: Tuple (x, y) of the start position.
            goal: Tuple (x, y) of the goal position.
            callback: Function to call after each step with current state and path.
            changed_cells: Optional list of cells that have changed.
        """
        # Implementation of interactive_plan method
        # TODO: Yet to implement this method!
        pass 