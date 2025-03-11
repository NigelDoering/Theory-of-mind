import numpy as np
from src.planning.node import Node, get_neighbors

class PathPlanner:
    """
    A class that encapsulates the path planning algorithm.
    """
    def __init__(self, world, temperature=0.5, top_n=5):
        self.world = world  # The world instance, for obstacle checking.
        self.temperature = temperature
        self.top_n = top_n

    @staticmethod
    def cost(current, neighbor):
        """
        Cost between adjacent grid cells.
        Here, we assume a uniform cost of 1.
        """
        return 1

    @staticmethod
    def heuristic(pos, goal):
        """
        Compute the Manhattan distance between pos and goal.
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    @staticmethod
    def boltzmann_sample(candidates, temperature):
        """
        Given a list of candidate nodes, sample one candidate according to a Boltzmann (softmax) distribution.
        """
        # Extract f-costs from candidates.
        f_costs = np.array([node.f for node in candidates])
        # Compute probabilities: lower f-cost should yield higher probability.
        probs = np.exp(-f_costs / temperature)
        probs /= np.sum(probs)
        index = np.random.choice(len(candidates), p=probs)
        return candidates[index]
    
    @staticmethod
    def stochastic_a_star(world, start, goal, temperature=0.5, top_n=5):
        """
        Compute a path from start to goal using a stochastic version of A*.
        """
        # Create the start node.
        start_node = Node(start, g=0, h=PathPlanner.heuristic(start, goal))
        open_list = [start_node]
        closed_set = set()
        
        while open_list:
            # Sort open_list by f-cost.
            open_list.sort(key=lambda node: node.f)
            # Get top_n candidates.
            candidates = open_list[:min(top_n, len(open_list))]
            # Sample one candidate using Boltzmann sampling.
            current = PathPlanner.boltzmann_sample(candidates, temperature)
            open_list.remove(current)
            closed_set.add(current.pos)
            
            if current.pos == goal:
                # Reconstruct path.
                path = []
                node = current
                while node is not None:
                    path.append(node.pos)
                    node = node.parent
                return path[::-1]
            
            for neighbor_pos in get_neighbors(current.pos, world):
                if neighbor_pos in closed_set:
                    continue
                tentative_g = current.g + PathPlanner.cost(current.pos, neighbor_pos)
                neighbor_node = Node(neighbor_pos, g=tentative_g, 
                                     h=PathPlanner.heuristic(neighbor_pos, goal),
                                     parent=current)
                # Check if a better path exists in open_list.
                if any(n.pos == neighbor_pos and n.f <= neighbor_node.f for n in open_list):
                    continue
                open_list.append(neighbor_node)
        
        return None  # No path found.
    
    def plan(self, start, goal):
        """
        Computes a path from start to goal using the stochastic A* algorithm.
        """
        path = PathPlanner.stochastic_a_star(self.world, start, goal, temperature=self.temperature, top_n=self.top_n)
        return path 