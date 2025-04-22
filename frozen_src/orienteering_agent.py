import os
import sys
import time
from frozen_src.world_graph import WorldGraph
from frozen_src.agent_knowledge import AgentKnowledge
from frozen_src.orienteering_planner import OrienteeringPlanner
from frozen_src.visualization_module import InteractiveVisualizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OrienteeringAgent:
    """
    Agent that navigates a partially observable environment to maximize rewards.
    Combines perception, knowledge representation, planning, and action.
    """
    
    def __init__(self, world_graph, budget, perception_radius=2, alpha=0.5, goal_bonus=2.0):
        """
        Initialize the agent.
        
        Args:
            world_graph: The world graph (ground truth)
            budget: Movement budget
            perception_radius: How far the agent can see
            alpha: Weight for final goal distance in node evaluation
            goal_bonus: Bonus multiplier for intermediate goals
        """
        self.world_graph = world_graph
        self.max_budget = budget
        self.remaining_budget = budget
        self.perception_radius = perception_radius
        self.alpha = alpha
        self.goal_bonus = goal_bonus
        
        # Initialize knowledge
        self.knowledge = AgentKnowledge(world_graph, perception_radius)
        
        # Initialize planner
        self.planner = OrienteeringPlanner(self.knowledge, budget)
        self.planner.set_alpha(alpha)
        self.planner.set_goal_bonus(goal_bonus)
        
        # Initialize visualization
        self.visualizer = InteractiveVisualizer(world_graph)
        
        # Initialize position to the start node
        self.current_node = world_graph.get_start_node()
        
        # Initialize counters
        self.total_reward = 0
        self.step = 0
        
        # Planning
        self.planned_path = []
        self.replan_frequency = 3  # Replan every N steps
        
        # Debug flags
        self.debug = True  # Print detailed information during execution
    
    def observe(self):
        """Observe surroundings and update knowledge."""
        self.knowledge.update(self.current_node)
        
        # Check if current node has reward
        node_info = self.world_graph.get_node_info(self.current_node)
        
        # Get reward if this is our first visit to a goal node
        if node_info['reward'] > 0 and self.current_node not in self.knowledge.collected_rewards:
            reward = node_info['reward']
            self.knowledge.collected_rewards[self.current_node] = reward
            self.total_reward += reward  # THIS LINE NEEDS TO BE CHECKED
            return reward
        
        return 0
    
    def plan(self):
        """Plan a path to maximize reward."""
        if self.debug:
            print(f"\n--- Planning from node {self.current_node} with budget {self.remaining_budget} ---")
            
        self.planned_path = self.planner.plan(self.current_node, self.remaining_budget)
        
        if self.debug:
            if len(self.planned_path) > 1:
                print(f"Planned path of length {len(self.planned_path)}: {self.planned_path}")
            else:
                print("No path planned, staying at current node")
                
        return self.planned_path
    
    def move(self):
        """Move to the next node in the planned path."""
        # Check if we need to replan
        if len(self.planned_path) <= 1 or self.step % self.replan_frequency == 0:
            # No path or time to replan
            self.plan()
        
        if len(self.planned_path) > 1:
            # Get the next step in the path
            next_node = self.planned_path[1]
            
            # Make sure next_node is actually connected to current_node in the knowledge graph
            if next_node in self.knowledge.graph and self.knowledge.graph.has_edge(self.current_node, next_node):
                # Valid move
                prev_node = self.current_node
                self.current_node = next_node
                self.planned_path.pop(0)  # Remove the current node from path
                self.remaining_budget -= 1
                self.step += 1
                
                if self.debug:
                    print(f"Step {self.step}: Moving from node {prev_node} to {next_node}")
                    
                return True
            else:
                # Edge doesn't exist in our knowledge or we need to replan
                print(f"Warning: Planned move to node {next_node} invalid, replanning...")
                self.plan()
                if len(self.planned_path) <= 1:
                    print("No valid moves found after replanning")
                    return False
                return self.move()  # Try again with new plan
        else:
            print("No planned path to follow")
            return False
    
    def visualize(self, save=True):
        """Visualize the current state."""
        self.visualizer.update(
            agent_knowledge=self.knowledge,
            current_node=self.current_node,
            planned_path=self.planned_path,
            total_reward=self.total_reward,
            budget=self.remaining_budget,
            step=self.step,
            save=save
        )
    
    def run_episode(self, max_steps=None, visualize_every=1, delay=0.5):
        """Run a complete episode with improved reward-seeking behavior."""
        if max_steps is None:
            max_steps = self.max_budget
        
        print(f"Starting episode with budget: {self.remaining_budget}")
        
        # Initial observation and planning
        reward = self.observe()
        if reward > 0:
            print(f"Initial position has reward: {reward:.2f}")
            
        self.plan()
        
        # Initial visualization
        self.visualize()
        
        # Track highest-value goals
        known_goals = {}
        step_count = 0
        
        while step_count < max_steps and self.remaining_budget > 0:
            # Update known goals and their values
            for node, reward in self.knowledge.discovered_goals.items():
                if node not in self.knowledge.collected_rewards:
                    known_goals[node] = reward
            
            # If we know goals but haven't planned a path to any yet, force replan
            if known_goals and not any(node in known_goals for node in self.planned_path[1:] if len(self.planned_path) > 1):
                print("Discovered goals but not heading to any - replanning...")
                self.plan()
            
            # Move to next node
            moved = self.move()
            
            if not moved:
                # Try to replan completely fresh
                print("Agent cannot move with current plan, forcing replan")
                self.plan()
                moved = self.move()
                
                if not moved:
                    # If still can't move, we're stuck
                    print("Agent is stuck, cannot find valid moves")
                    break
            
            # Observe and get reward
            reward = self.observe()
            if reward > 0:
                print(f"Step {step_count}: Collected reward {reward:.2f} at node {self.current_node}")
                # Remove from known goals
                if self.current_node in known_goals:
                    del known_goals[self.current_node]
            
            # Visualize periodically
            if step_count % visualize_every == 0:
                self.visualize()
                time.sleep(delay)  # Add delay to see progress
            
            # Check if we've reached the final goal
            if self.current_node == self.world_graph.get_final_node():
                print(f"*** REACHED FINAL GOAL at node {self.current_node}! ***")
                # Check if we've collected all rewards
                if len(self.knowledge.collected_rewards) == len(self.knowledge.discovered_goals):
                    print("All known rewards collected!")
                    self.visualize()  # Final visualization
                    break
                elif self.remaining_budget < 10:
                    print("Reached final goal with low budget - ending episode")
                    self.visualize()
                    break
                else:
                    print(f"Reached final goal but {len(self.knowledge.discovered_goals) - len(self.knowledge.collected_rewards)} rewards remain")
                    # Continue if we have budget and uncollected rewards
            
            step_count += 1
            
            # Replan periodically or when we've reached a planned goal
            if (step_count % 5 == 0) or (len(self.planned_path) <= 1):
                self.plan()
        
        # Final visualization and summary
        self.visualize()
        
        # Create animation - with error handling
        try:
            self.visualizer.create_animation()
        except Exception as e:
            print(f"Error creating animation: {e}")
        
        # Final summary
        print("\n===== EPISODE SUMMARY =====")
        print(f"Total reward collected: {self.total_reward:.2f}")
        print(f"Steps taken: {self.step}")
        print(f"Remaining budget: {self.remaining_budget}")
        print(f"Goals found: {len(self.knowledge.discovered_goals)}/{len(self.world_graph.goal_generator.goal_positions)}")
        print(f"Goals reached: {len(self.knowledge.collected_rewards)}/{len(self.knowledge.discovered_goals)}")
        
        # Check if final goal was reached
        final_goal = self.world_graph.get_final_node()
        final_reached = final_goal in self.knowledge.collected_rewards if final_goal else False
        print(f"Final goal reached: {final_reached}")
        
        known_percent = len(self.knowledge.graph)/len(self.world_graph.graph)*100
        print(f"Map explored: {len(self.knowledge.graph)}/{len(self.world_graph.graph)} nodes ({known_percent:.1f}%)")
        print("==========================")
        
        # Close the visualization when done
        self.visualizer.close()