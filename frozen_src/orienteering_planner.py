import networkx as nx
import numpy as np
from queue import PriorityQueue

class OrienteeringPlanner:
    """Plans paths to maximize reward within a budget."""
    
    def __init__(self, agent_knowledge, remaining_budget):
        """Initialize with agent's knowledge and remaining budget."""
        self.agent_knowledge = agent_knowledge
        self.remaining_budget = remaining_budget
        self.path = []
        self.estimated_reward = 0
        self.alpha = 0.5  # Weight for distance to final goal
        self.debug = True  # Print detailed planning info
        self.goal_bonus = 2.0  # Bonus multiplier for intermediate goals
        
    def set_alpha(self, alpha):
        """Set the weight for distance to final goal."""
        self.alpha = alpha
        
    def set_goal_bonus(self, bonus):
        """Set the bonus multiplier for intermediate goals."""
        self.goal_bonus = bonus
    
    def plan(self, current_node, budget):
        """
        Plan a path from current node to maximize reward within budget.
        Uses a modified orienteering algorithm to visit multiple goals.
        """
        self.remaining_budget = budget
        
        # Reset path
        self.path = [current_node]
        
        # If we don't have a graph yet or just the current node, focus on exploration
        if len(self.agent_knowledge.graph) <= 1:
            print("No known nodes except current position, using random exploration")
            return [current_node]
        
        # Get the final goal if known
        final_goal = self.agent_knowledge.final_goal
        
        # First check: if we know the final goal and have limited budget, ensure we can reach it
        if final_goal and final_goal not in self.agent_knowledge.collected_rewards:
            try:
                path_to_final = nx.shortest_path(self.agent_knowledge.graph, current_node, final_goal)
                min_budget_needed = len(path_to_final) - 1
                
                if min_budget_needed > budget * 0.8:  # If we need 80% of budget just to reach final goal
                    print(f"Critical budget situation! Need {min_budget_needed} steps to reach final goal with {budget} budget")
                    # Just go straight to the final goal
                    self.path = path_to_final
                    return self.path
            except nx.NetworkXNoPath:
                print("No path to final goal found!")
        
        # If we have enough budget, consider exploring and collecting other goals first
        # First, build a candidate list of nodes to consider (goals and frontier)
        candidates = self._get_candidate_nodes(current_node)
        
        if len(candidates) == 0:
            print("No valid candidate nodes found")
            return [current_node]
        
        # Sort candidates by their evaluation score
        evaluated_candidates = []
        for node in candidates:
            if node == current_node:
                continue
                
            score, details = self._evaluate_node_complete(current_node, node, final_goal)
            evaluated_candidates.append((node, score, details))
        
        # Sort by score (higher is better)
        evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if self.debug:
            print("\n--- Top 5 Evaluated Nodes ---")
            for i, (node, score, details) in enumerate(evaluated_candidates[:5]):
                pos = self.agent_knowledge.graph.nodes[node]['pos']
                print(f"{i+1}. Node {node} at {pos}: Score {score:.2f}")
                print(f"   Cost: {details['cost']:.2f}, Reward: {details['reward']:.2f}, " +
                     f"Final Goal Impact: {details['final_goal_impact']:.2f}")
                if 'notes' in details:
                    print(f"   Notes: {details['notes']}")
            print("----------------------------\n")
        
        # If we have good candidates, plan a path to the best one
        if evaluated_candidates:
            best_node, best_score, _ = evaluated_candidates[0]
            try:
                path_to_best = nx.shortest_path(self.agent_knowledge.graph, current_node, best_node)
                if len(path_to_best) - 1 <= budget:
                    self.path = path_to_best
                    print(f"Planning path to node {best_node} with score {best_score:.2f}")
                    return self.path
            except nx.NetworkXNoPath:
                pass
        
        # If we couldn't use the evaluation approach, fall back to orienteering
        return self._plan_orienteering_path(current_node, budget, final_goal)
    
    def _plan_orienteering_path(self, current_node, budget, final_goal):
        """
        Plan a path that maximizes reward collection while ensuring final goal is reached.
        Uses a specialized algorithm for the orienteering problem.
        """
        graph = self.agent_knowledge.graph
        
        # Get all uncollected goals
        uncollected_goals = {}
        for node, reward in self.agent_knowledge.discovered_goals.items():
            if node not in self.agent_knowledge.collected_rewards:
                # Apply goal bonus to make intermediate goals more attractive
                if node == final_goal:
                    # Final goal keeps original reward to maintain its priority
                    uncollected_goals[node] = reward
                else:
                    # Boost intermediate goals
                    uncollected_goals[node] = reward * self.goal_bonus
        
        # If no goals or only final goal, handle specially
        if not uncollected_goals or (len(uncollected_goals) == 1 and final_goal in uncollected_goals):
            if final_goal and final_goal not in self.agent_knowledge.collected_rewards:
                try:
                    path = nx.shortest_path(graph, current_node, final_goal)
                    print(f"Planning direct path to final goal, length: {len(path)-1}")
                    return path
                except nx.NetworkXNoPath:
                    print("No path to final goal found, exploring")
            
            # No goals or can't reach final goal, explore frontier
            return self._plan_for_exploration(current_node)
        
        # Calculate shortest paths between all goals and current position
        # This creates a complete graph where nodes are goals and edges are shortest paths
        goal_graph = nx.DiGraph()
        all_nodes = list(uncollected_goals.keys()) + [current_node]
        
        # If final goal is known but not in uncollected_goals, add it
        if final_goal and final_goal not in all_nodes:
            all_nodes.append(final_goal)
        
        # Add nodes (all goals + current position)
        for node in all_nodes:
            # Use original positions for visualization later
            pos = graph.nodes[node]['pos']
            
            # Add reward information or 0 for current position
            reward = uncollected_goals.get(node, 0)
            
            # Mark if this is the final goal
            is_final = (node == final_goal)
            
            goal_graph.add_node(node, pos=pos, reward=reward, is_final=is_final)
        
        # Add edges with distances
        for i, node1 in enumerate(all_nodes):
            for node2 in all_nodes:
                if node1 != node2:
                    try:
                        # Find shortest path and distance
                        path = nx.shortest_path(graph, node1, node2)
                        distance = len(path) - 1
                        
                        # Store both distance and actual path
                        goal_graph.add_edge(node1, node2, weight=distance, path=path)
                    except nx.NetworkXNoPath:
                        # No path between these goals
                        pass
        
        # Now solve the orienteering problem
        path_nodes = self._solve_orienteering(goal_graph, current_node, final_goal, budget)
        
        if not path_nodes or len(path_nodes) <= 1:
            print("Orienteering solver couldn't find a valid path")
            if final_goal:
                try:
                    # Try direct path to final goal
                    path = nx.shortest_path(graph, current_node, final_goal)
                    return path
                except nx.NetworkXNoPath:
                    pass
            return self._plan_for_exploration(current_node)
        
        # Convert the sequence of goals to actual path segments
        complete_path = [current_node]
        
        for i in range(1, len(path_nodes)):
            start = path_nodes[i-1]
            end = path_nodes[i]
            
            # Get the path segment between these goals
            if goal_graph.has_edge(start, end):
                path_segment = goal_graph[start][end]['path'][1:]  # Skip first node to avoid duplication
                complete_path.extend(path_segment)
            else:
                print(f"Warning: No edge between {start} and {end} in goal graph")
                try:
                    # Try to find direct path
                    path_segment = nx.shortest_path(graph, start, end)[1:]
                    complete_path.extend(path_segment)
                except nx.NetworkXNoPath:
                    print(f"No path from {start} to {end}, path planning failed")
                    break
        
        print(f"Planned path visiting {len(path_nodes)-1} goals, total length: {len(complete_path)-1}")
        return complete_path
    
    def _solve_orienteering(self, goal_graph, start_node, final_goal, budget):
        """
        Solve the orienteering problem more intelligently with reward maximization.
        Uses a more sophisticated algorithm that considers complete paths.
        """
        # Start with direct path to final goal if possible
        tour = []
        tour_length = 0
        
        if final_goal and final_goal in goal_graph:
            try:
                # Start with just start and finish
                tour = [start_node, final_goal]
                tour_length = goal_graph[start_node][final_goal]['weight'] if goal_graph.has_edge(start_node, final_goal) else budget + 1
                
                if tour_length > budget:
                    print(f"Can't reach final goal directly ({tour_length} > {budget})")
                    tour = [start_node]
                    tour_length = 0
            except Exception as e:
                print(f"Error initializing tour: {e}")
                tour = [start_node]
                tour_length = 0
        else:
            # If no final goal, start with just the start node
            tour = [start_node]
            tour_length = 0
        
        # Calculate the reward of the tour
        def tour_reward(tour):
            return sum(goal_graph.nodes[n]['reward'] for n in tour)
        
        # Use a more sophisticated algorithm that optimizes the complete path
        # First, calculate the reward-to-distance ratio for each node
        value_ratios = {}
        for node in goal_graph.nodes():
            if node == start_node:
                continue
                
            # Get reward
            reward = goal_graph.nodes[node]['reward']
            is_final = goal_graph.nodes[node].get('is_final', False)
            
            if reward <= 0 and not is_final:
                continue  # Skip nodes with no reward unless it's the final goal
                
            # Calculate distance from start
            if goal_graph.has_edge(start_node, node):
                distance = goal_graph[start_node][node]['weight']
                
                # For the final goal, we want to ensure it's included, so give it a special value
                if is_final:
                    # High value but avoid division by zero
                    value_ratios[node] = 1000 if distance < 1 else (100 / distance)
                else:
                    # Normal value calculation: reward per unit distance
                    value_ratios[node] = reward / max(distance, 1)
        
        # Sort nodes by their value ratio (descending)
        sorted_nodes = sorted(value_ratios.keys(), key=lambda n: value_ratios[n], reverse=True)
        
        # Use dynamic programming to find the optimal path
        best_reward = 0
        best_path = [start_node]
        
        # Try different high-value nodes as the "anchor" nodes for our path
        for anchor_node in sorted_nodes[:min(5, len(sorted_nodes))]:
            # Try to build a path through this anchor node
            current_path = [start_node]
            current_length = 0
            current_reward = 0
            remaining_nodes = set(n for n in goal_graph.nodes() if n != start_node and goal_graph.nodes[n]['reward'] > 0)
            
            # First, try to add the anchor node
            if goal_graph.has_edge(start_node, anchor_node):
                distance = goal_graph[start_node][anchor_node]['weight']
                if distance <= budget:
                    current_path.append(anchor_node)
                    current_length += distance
                    current_reward += goal_graph.nodes[anchor_node]['reward']
                    remaining_nodes.remove(anchor_node)
                    
                    # Now greedily add more nodes if possible
                    while remaining_nodes and current_length < budget:
                        best_next = None
                        best_ratio = -1
                        best_dist = 0
                        
                        for node in remaining_nodes:
                            if goal_graph.has_edge(current_path[-1], node):
                                dist = goal_graph[current_path[-1]][node]['weight']
                                if current_length + dist <= budget:
                                    ratio = goal_graph.nodes[node]['reward'] / max(dist, 0.1)
                                    if ratio > best_ratio:
                                        best_ratio = ratio
                                        best_next = node
                                        best_dist = dist
                        
                        if best_next:
                            current_path.append(best_next)
                            current_length += best_dist
                            current_reward += goal_graph.nodes[best_next]['reward']
                            remaining_nodes.remove(best_next)
                        else:
                            break
                    
                    # If the final goal isn't in our path and we have budget left, try to add it
                    if final_goal and final_goal not in current_path:
                        if goal_graph.has_edge(current_path[-1], final_goal):
                            dist_to_final = goal_graph[current_path[-1]][final_goal]['weight'] 
                            if current_length + dist_to_final <= budget:
                                current_path.append(final_goal)
                                current_length += dist_to_final
                                current_reward += goal_graph.nodes[final_goal]['reward']
                    
                    # Check if this path is better than our best so far
                    if current_reward > best_reward:
                        best_reward = current_reward
                        best_path = current_path
        
        # Ensure the final goal is in the path if we have enough budget
        if final_goal and final_goal not in best_path:
            if best_path and goal_graph.has_edge(best_path[-1], final_goal):
                dist_to_final = goal_graph[best_path[-1]][final_goal]['weight']
                if dist_to_final + tour_length <= budget:
                    best_path.append(final_goal)
        
        if best_reward > 0:
            print(f"Found optimized path with reward {best_reward:.2f}")
            return best_path
        
        # Fall back to the original tour if our optimization didn't work
        if tour and len(tour) > 1:
            return tour
        
        # Last resort: try direct path to final goal
        if final_goal:
            try:
                path = [start_node, final_goal]
                if goal_graph.has_edge(start_node, final_goal) and goal_graph[start_node][final_goal]['weight'] <= budget:
                    return path
            except Exception:
                pass
        
        # If all else fails
        return [start_node]
    
    def _plan_for_exploration(self, current_node):
        """Plan a path that prioritizes exploration."""
        graph = self.agent_knowledge.graph
        frontier = list(self.agent_knowledge.frontier)
        
        # If there's nothing in the frontier, return current node
        if not frontier:
            self.path = [current_node]
            return self.path
        
        # Find closest frontier node
        best_frontier_node = None
        best_distance = float('inf')
        best_path = None
        
        for node in frontier:
            try:
                path = nx.shortest_path(graph, current_node, node)
                distance = len(path) - 1
                
                if distance < best_distance and distance <= self.remaining_budget:
                    best_distance = distance
                    best_frontier_node = node
                    best_path = path
            except nx.NetworkXNoPath:
                continue
        
        if best_path:
            self.path = best_path
            return self.path
        
        # If we can't reach any frontier node, stay in place
        self.path = [current_node]
        return self.path

    def _prioritize_goals(self, goals, current_node, final_goal, budget):
        """
        Adjust goal rewards based on distance, current budget, and strategic value.
        
        Args:
            goals: Dictionary of node_id -> reward
            current_node: Current position
            final_goal: Final goal node
            budget: Remaining budget
            
        Returns:
            Dictionary of node_id -> adjusted_reward
        """
        graph = self.agent_knowledge.graph
        adjusted_goals = {}
        
        # Calculate distance to final goal if known
        distance_to_final = float('inf')
        if final_goal:
            try:
                path_to_final = nx.shortest_path(graph, current_node, final_goal)
                distance_to_final = len(path_to_final) - 1
            except nx.NetworkXNoPath:
                pass
        
        # Determine if we're in "rush mode" (need to prioritize final goal)
        rush_mode = (distance_to_final > budget * 0.7)
        
        for node, reward in goals.items():
            # Skip final goal - keep its original reward
            if node == final_goal:
                adjusted_goals[node] = reward
                continue
                
            try:
                # Calculate distances
                dist_to_goal = len(nx.shortest_path(graph, current_node, node)) - 1
                
                # Check if reaching this goal would endanger reaching the final goal
                if final_goal:
                    try:
                        dist_from_goal_to_final = len(nx.shortest_path(graph, node, final_goal)) - 1
                        total_trip = dist_to_goal + dist_from_goal_to_final
                        
                        # If we're in rush mode, only consider goals on the way to final goal
                        if rush_mode:
                            # Only prioritize goals that don't add much detour
                            detour = total_trip - distance_to_final
                            
                            if detour <= 2:  # Minimal detour
                                # On the way to final goal - boost it
                                adjusted_goals[node] = reward * self.goal_bonus * 1.5
                            else:
                                # Too far out of the way in rush mode
                                adjusted_goals[node] = reward * 0.5
                        else:
                            # Normal mode - if total trip is feasible
                            if total_trip <= budget * 0.9:
                                # Apply standard goal bonus
                                adjusted_goals[node] = reward * self.goal_bonus
                                
                                # Add extra bonus for "on the way" goals
                                if total_trip <= distance_to_final + 3:  # Small detour
                                    adjusted_goals[node] *= 1.3
                            else:
                                # Reaching this might endanger final goal
                                adjusted_goals[node] = reward * 0.8
                    except nx.NetworkXNoPath:
                        # Can't reach final from this goal
                        adjusted_goals[node] = reward * 0.5
                else:
                    # No final goal known yet, use standard bonus
                    adjusted_goals[node] = reward * self.goal_bonus
            except nx.NetworkXNoPath:
                # Can't reach this goal
                adjusted_goals[node] = 0
        
        return adjusted_goals

    def _get_candidate_nodes(self, current_node):
        """
        Get list of candidate nodes to consider for planning.
        Includes:
        1. Uncollected goals
        2. Frontier nodes
        3. Final goal (if known)
        """
        candidates = set()
        
        # Add all uncollected goals
        for node, reward in self.agent_knowledge.discovered_goals.items():
            if node not in self.agent_knowledge.collected_rewards:
                candidates.add(node)
        
        # Add frontier nodes
        candidates.update(self.agent_knowledge.frontier)
        
        # Ensure the final goal is included if known
        if self.agent_knowledge.final_goal:
            candidates.add(self.agent_knowledge.final_goal)
        
        # Remove current node and ensure all candidates are actually in the graph
        candidates = {n for n in candidates if n in self.agent_knowledge.graph and n != current_node}
        
        return list(candidates)

    def _evaluate_node_complete(self, current_node, target_node, final_goal=None):
        """
        Evaluate a node using the formula:
        Cost to node (-ve) + Reward (+ve or 0) + Alpha * Distance to final node (-ve)
        
        Always considers final goal impact if the final goal is known.
        
        Returns:
            tuple: (overall_score, details_dict)
        """
        graph = self.agent_knowledge.graph
        details = {
            'cost': 0,
            'reward': 0,
            'final_goal_impact': 0,
            'notes': ""
        }
        
        # Get cost to node (negative of distance)
        try:
            path = nx.shortest_path(graph, current_node, target_node)
            cost_to_node = -(len(path) - 1)  # Negative cost
            details['cost'] = cost_to_node
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            details['notes'] = "No path to target"
            return float('-inf'), details
        
        # Get reward (0 if not a goal)
        reward = graph.nodes[target_node].get('reward', 0)
        details['reward'] = reward
        
        # Add bonus if it's a frontier node (unexplored area)
        if target_node in self.agent_knowledge.frontier:
            exploration_bonus = 0.1
            reward += exploration_bonus
            details['notes'] += "Frontier node (+0.1 bonus). "
        
        # Get distance to final goal impact - ALWAYS calculate if final goal is known
        # This is the key fix
        if final_goal is None:
            final_goal = self.agent_knowledge.final_goal
            
        final_goal_impact = 0
        if final_goal is not None:
            try:
                # Calculate distance from target node to final goal
                path_to_final = nx.shortest_path(graph, target_node, final_goal)
                distance_to_final = len(path_to_final) - 1
                
                # Calculate distance from current node to final goal as baseline
                direct_path_to_final = nx.shortest_path(graph, current_node, final_goal)
                direct_distance = len(direct_path_to_final) - 1
                
                # Calculate how much this node improves/worsens the path to final goal
                # Negative if it takes us farther, positive if it takes us closer
                improvement = direct_distance - distance_to_final
                
                # Scale by alpha - higher value means we care more about getting to final goal
                final_goal_impact = self.alpha * improvement
                details['final_goal_impact'] = final_goal_impact
                
                if target_node == final_goal:
                    details['notes'] += "This is the final goal. "
                elif improvement > 0:
                    details['notes'] += f"Gets us {improvement} steps closer to final goal. "
                elif improvement < 0:
                    details['notes'] += f"Takes us {-improvement} steps farther from final goal. "
                else:
                    details['notes'] += f"Neutral distance to final goal. "
                    
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                details['notes'] += "Cannot reach final goal from here. "
                # Penalize nodes that can't reach the final goal
                final_goal_impact = -5 * self.alpha
                details['final_goal_impact'] = final_goal_impact
        
        # Calculate final score
        score = cost_to_node + reward + final_goal_impact
        
        return score, details