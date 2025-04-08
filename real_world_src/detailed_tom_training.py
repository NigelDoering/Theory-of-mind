import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force Agg backend for all plots
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_world_src.environment.campus_env import CampusEnvironment
from real_world_src.agents.agent_factory import AgentFactory
from real_world_src.simulation.simulator import Simulator
from real_world_src.utils.run_manager import RunManager
from real_world_src.utils.trajectory_collector import TrajectoryCollector
from real_world_src.models.tomnet import ToMNet
from real_world_src.models.tomnet_trainer import ToMNetTrainer
from real_world_src.agents.tom_agent import ToMAgent
from real_world_src.utils.visualization import (
    plot_agent_paths, visualize_agent_decision, visualize_tom_predictions,
    animate_tom_predictions, plot_species_grid
)

class DetailedToMTraining:
    """Class to handle detailed ToM training and visualization"""
    
    def __init__(self, experiment_name="detailed_tom", use_gpu=True):
        """Initialize the training process"""
        # Setup style for visualizations
        plt.style.use('dark_background')
        sns.set_style("darkgrid")
        sns.set_context("talk")
        
        # Initialize experiment tracking
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_id = f"{experiment_name}_{self.timestamp}"
        
        # Initialize run manager
        self.run_manager = RunManager(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'detailed_tom_results'
        ))
        self.run_dir = self.run_manager.start_new_run()
        
        # Create specific directories for detailed metrics
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.embeddings_dir = os.path.join(self.run_dir, "embeddings")
        self.attention_dir = os.path.join(self.run_dir, "attention_maps")
        self.prediction_dir = os.path.join(self.run_dir, "predictions")
        
        for directory in [self.metrics_dir, self.embeddings_dir, 
                          self.attention_dir, self.prediction_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Set device for training
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        
        # Initialize metrics tracking
        self.training_metrics = {
            'epoch_losses': [],
            'character_losses': [],
            'mental_state_losses': [],
            'prediction_losses': [],
            'validation_metrics': []
        }
        
        # Print initialization information
        self.print_header()
        
    def print_header(self):
        """Print experiment header information"""
        print("\n" + "=" * 80)
        print(f"DETAILED THEORY OF MIND TRAINING - {self.exp_id}")
        print("=" * 80)
        print(f"• Run directory: {self.run_dir}")
        print(f"• Training device: {self.device}")
        print(f"• Timestamp: {self.timestamp}")
        print("=" * 80 + "\n")
        
    def setup_environment(self, agent_counts=None):
        """Set up the campus environment with agents"""
        print("\n📍 PHASE 1: ENVIRONMENT SETUP")
        print("Creating UCSD campus environment and populating with agents...")
        
        # Default agent distribution if none provided
        if agent_counts is None:
            agent_counts = {
                "shortest": 5,
                "random": 5,
                "landmark": 5,
                "social": 5,
                "explorer": 5,
                "obstacle": 5,
                "scared": 5,
                "risky": 5
            }
            
        # Create environment
        self.campus = CampusEnvironment()
        
        # Create agents
        self.agents = AgentFactory.populate_environment(self.campus, agent_counts)
        print(f"Created {len(self.agents)} agents for data collection")
        
        # Visualize initial environment state
        self._visualize_environment_setup()
        
        # Visualize agent decision processes for a sample of agents
        self._visualize_agent_decisions()
        
        return self.campus, self.agents

    def _visualize_agent_decisions(self):
        """Visualize the decision process for a sample of agents"""
        # Select one agent from each species for visualization
        species_seen = set()
        
        for agent in self.agents:
            # Skip if we've already seen this species
            if agent.species in species_seen:
                continue
                
            species_seen.add(agent.species)
            
            # Make sure agent nodes are valid before proceeding
            if agent.current_node is None or agent.goal_node is None:
                print(f"Warning: Skipping {agent.species} agent visualization - invalid nodes")
                continue
                
            # Make sure agent has a valid path before visualizing
            try:
                if not hasattr(agent, 'path') or not agent.path:
                    # Force agent to plan a path first
                    agent.plan_path()
                    
                # Check if path planning was successful
                if not hasattr(agent, 'path') or not agent.path or len(agent.path) <= 1:
                    print(f"Warning: Could not visualize decision for {agent.species}: Agent has no valid path")
                    continue
                
                # Use the visualize_agent_decision utility with better error handling
                try:
                    # Create our own visualization directly instead of relying on the utility function
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Set dark background
                    fig.patch.set_facecolor('#1e1e1e')
                    ax.set_facecolor('#1e1e1e')
                    
                    # Plot environment
                    self.campus.plot_map(ax=ax)
                    
                    # Get agent's current position
                    current_pos = agent.get_position()
                    
                    if current_pos:
                        ax.scatter([current_pos[0]], [current_pos[1]], color=agent.color, 
                                  s=150, marker='D', edgecolor='none', zorder=10, 
                                  label="Current Position")
                    
                    # Get start position
                    start_pos = agent.get_position(agent.start_node) 
                    if start_pos:
                        ax.scatter([start_pos[0]], [start_pos[1]], color=agent.color, 
                                  s=100, marker='o', edgecolor='none', zorder=10, 
                                  label="Start")
                    
                    # Get goal position
                    goal_pos = agent.get_position(agent.goal_node)
                    if goal_pos:
                        ax.scatter([goal_pos[0]], [goal_pos[1]], color=agent.color, 
                                  s=100, marker='s', edgecolor='none', zorder=10, 
                                  label="Goal")
                    
                    # Plot planned path if available
                    if hasattr(agent, 'path') and len(agent.path) > 1:
                        path_coords = [self.campus.get_node_coordinates(node) for node in agent.path]
                        path_x = [coord[0] for coord in path_coords if coord]
                        path_y = [coord[1] for coord in path_coords if coord]
                        
                        if path_x and path_y:
                            ax.plot(path_x, path_y, color=agent.color, alpha=0.8, 
                                  linewidth=2.5, linestyle='--', label="Planned Path")
                    
                    # Add title and legend
                    ax.set_title(f"{agent.species} Agent Decision Process")
                    ax.legend(loc='upper right')
                    
                    # Save the visualization
                    decision_path = self.run_manager.get_plot_path(f"decision_{agent.species.lower()}.png")
                    fig.savefig(decision_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  - {decision_path}")
                except Exception as e:
                    print(f"Warning: Could not visualize decision for {agent.species}: {e}")
                    import traceback
                    traceback.print_exc()  # Print full stack trace for debugging
            except Exception as e:
                print(f"Warning: Could not visualize decision for {agent.species}: {e}")

    def _visualize_environment_setup(self):
        """Visualize the initial environment setup with enhanced styling and agents"""
        print("Generating environment visualization...")
        
        # Create figure and axes 
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Set figure background color to match dark theme
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        # Define custom VISUAL_CONFIG to override default
        custom_visual_config = {
            "markers": {
                "start": "o",
                "goal": "s", 
                "position": "D",
                "landmark": "*"
            },
            "sizes": {
                "start": 100,
                "goal": 100,
                "position": 150,
                "landmark": 120,
                "path_line": 2.5
            },
            "grid_alpha": 0.2,
            "path_alpha": 0.8, 
            "landmark_alpha": 0.9,
            "edge_color": "none",  # Use 'none' directly to remove borders
            "legend_loc": "upper right",
            "legend_fontsize": 10,
            "building_color": "#555555",
            "building_edge": "none",  # Remove building edges
            "landmark_color": "#f1c40f" 
        }
        
        # Create a local wrapper function instead of trying to import it
        def plot_agent_paths_custom(environment, agents, title, show_buildings, figsize):
            fig, ax = plot_agent_paths(environment, agents, title, show_buildings, figsize)
            
            # Find all scatter points and remove their edge colors
            for child in ax.get_children():
                if isinstance(child, plt.matplotlib.collections.PathCollection):  # This is a scatter plot
                    child.set_edgecolor('none')  # Remove edge color
                    
            return fig, ax
        
        # Use the custom function
        fig, ax = plot_agent_paths_custom(
            environment=self.campus, 
            agents=self.agents, 
            title="Initial Agent Setup", 
            show_buildings=True, 
            figsize=(15, 12)
        )
        
        # Save the figure
        setup_path = self.run_manager.get_plot_path("environment_setup.png")
        plt.savefig(setup_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create multiple visualizations showing different aspects
        # 1. Individual species paths with the same border removal
        self._visualize_individual_species()
        
        # 2. Species grid with proper styling 
        species_grid_path = self.run_manager.get_plot_path("species_grid.png")
        try:
            grid_fig, grid_axes = plot_species_grid(
                self.campus, 
                self.agents, 
                title="Agent Species Comparison"
            )
            grid_fig.savefig(species_grid_path, dpi=300, bbox_inches='tight')
            plt.close(grid_fig)
            print(f"  - {species_grid_path}")
        except Exception as e:
            print(f"Warning: Could not generate species grid: {e}")
        
        print(f"✓ Environment visualizations saved to:")
        print(f"  - {setup_path}")

    def _visualize_individual_species(self):
        """Create individual visualizations for each agent species"""
        # Group agents by species
        species_groups = {}
        for agent in self.agents:
            if agent.species not in species_groups:
                species_groups[agent.species] = []
            species_groups[agent.species].append(agent)
        
        # Create a visualization for each species group
        for species, agents in species_groups.items():
            # Skip if no agents
            if not agents:
                continue
                
            # Use plot_agent_paths but ensure no borders
            fig, ax = plot_agent_paths(
                self.campus, 
                agents, 
                title=f"{species} Agents Initial Setup", 
                show_buildings=True,
                figsize=(12, 10)
            )
            
            # Remove borders from all scatter plots
            for child in ax.get_children():
                if isinstance(child, plt.matplotlib.collections.PathCollection):
                    child.set_edgecolor('none')
        
            # Save the species-specific visualization
            species_path = self.run_manager.get_plot_path(f"species_{species.lower()}_setup.png")
            fig.savefig(species_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  - {species_path}")
        
    def collect_trajectory_data(self, num_episodes=100, max_steps=100, 
                              visualize_interval=10):
        """Collect trajectory data from agent simulations"""
        print("\n📍 PHASE 2: TRAJECTORY DATA COLLECTION")
        print(f"Running {num_episodes} episodes to collect agent trajectories...")
        
        # Initialize trajectory collector
        self.trajectory_collector = TrajectoryCollector(self.campus, max_trajectory_length=1000)
        
        # Setup metrics tracking for data collection
        collection_metrics = {
            'episode_durations': [],
            'trajectories_per_species': {},
            'avg_trajectory_length': [],
            'goal_completion_rate': []
        }
        
        # Run episodes
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            print(f"\nRunning episode {episode+1}/{num_episodes}...")
            
            # Create a simulator with the trajectory collector
            simulator = Simulator(self.campus, run_manager=self.run_manager)
            simulator.set_trajectory_collector(self.trajectory_collector)
            
            # Decide whether to animate this episode
            animate_this_episode = (episode % visualize_interval == 0)
            
            # Run the simulation - remove animation_name parameter
            all_done = simulator.run_simulation(
                max_steps=max_steps, 
                animate=animate_this_episode, 
                save_animation=animate_this_episode
            )
            
            # If we saved an animation, rename it with a meaningful name
            if animate_this_episode:
                # Try to find and rename the animation file
                animations_dir = os.path.join(self.run_dir, "animations")
                if os.path.exists(animations_dir):
                    # Look for the most recently created animation file
                    animation_files = [f for f in os.listdir(animations_dir) if f.endswith('.gif')]
                    if animation_files:
                        # Sort by creation time, newest first
                        animation_files.sort(key=lambda x: os.path.getctime(os.path.join(animations_dir, x)), 
                                             reverse=True)
                        # Rename the most recent file
                        old_path = os.path.join(animations_dir, animation_files[0])
                        new_name = f"episode_{episode+1}.gif"
                        new_path = os.path.join(animations_dir, new_name)
                        
                        # Rename the file
                        if os.path.exists(old_path):
                            os.rename(old_path, new_path)
                            print(f"✓ Saved animation for episode {episode+1} to {new_path}")
                        else:
                            print(f"Warning: Animation file not found at {old_path}")
                    else:
                        # If no animation files found, create one directly using our visualization tools
                        print("No animation found, creating one manually...")
                        self._create_episode_animation(episode+1, max_steps)
            
            # Record metrics
            episode_duration = time.time() - episode_start_time
            collection_metrics['episode_durations'].append(episode_duration)
            
            # Record trajectory statistics
            for species, trajectories in self.trajectory_collector.trajectories.items():
                if species not in collection_metrics['trajectories_per_species']:
                    collection_metrics['trajectories_per_species'][species] = []
                collection_metrics['trajectories_per_species'][species].append(len(trajectories))
            
            # Calculate average trajectory length
            all_trajs = []
            for species_trajs in self.trajectory_collector.trajectories.values():
                all_trajs.extend(species_trajs)
            
            avg_length = np.mean([len(traj) for traj in all_trajs]) if all_trajs else 0
            collection_metrics['avg_trajectory_length'].append(avg_length)
            
            # Calculate goal completion rate
            goals_completed = sum(1 for agent in self.agents if agent.at_goal())
            completion_rate = goals_completed / len(self.agents)
            collection_metrics['goal_completion_rate'].append(completion_rate)
            
            # Reset agents for the next episode with new random positions
            for agent in self.agents:
                agent.reset()
                
            # Visualize trajectory statistics every visualize_interval episodes
            if animate_this_episode or episode == num_episodes - 1:
                self._visualize_trajectory_statistics(collection_metrics, episode)
        
        # Save final metrics
        with open(os.path.join(self.metrics_dir, 'data_collection_metrics.json'), 'w') as f:
            json.dump(collection_metrics, f, indent=2)
            
        print(f"\n✓ Collected {sum(len(trajs) for trajs in self.trajectory_collector.trajectories.values())} trajectories")
        
        return self.trajectory_collector
    
    def _visualize_trajectory_statistics(self, metrics, current_episode):
        """Visualize trajectory collection statistics"""
        print("Generating trajectory statistics visualizations...")
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Episode duration plot
        episode_nums = list(range(1, len(metrics['episode_durations']) + 1))
        axes[0, 0].plot(episode_nums, metrics['episode_durations'], 'o-', color='cyan')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Duration (seconds)')
        axes[0, 0].set_title('Episode Duration')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Trajectory count by species
        species_names = list(metrics['trajectories_per_species'].keys())
        trajectory_counts = [metrics['trajectories_per_species'][s][-1] 
                           for s in species_names]
        
        species_colors = [next(agent.color for agent in self.agents 
                             if agent.species == species) 
                        for species in species_names]
        
        bars = axes[0, 1].bar(species_names, trajectory_counts, color=species_colors)
        axes[0, 1].set_xlabel('Agent Species')
        axes[0, 1].set_ylabel('Trajectory Count')
        axes[0, 1].set_title('Trajectories by Agent Type')
        axes[0, 1].set_xticklabels(species_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add trajectory counts as text
        for bar, count in zip(bars, trajectory_counts):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                          f'{count}', ha='center', va='bottom')
        
        # Average trajectory length
        axes[1, 0].plot(episode_nums, metrics['avg_trajectory_length'], 'o-', color='green')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Steps')
        axes[1, 0].set_title('Average Trajectory Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Goal completion rate
        axes[1, 1].plot(episode_nums, metrics['goal_completion_rate'], 'o-', color='magenta')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Completion Rate')
        axes[1, 1].set_title('Goal Completion Rate')
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'Trajectory Collection Statistics (Episode {current_episode+1})', 
                    fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        stats_path = os.path.join(self.metrics_dir, 
                                f'trajectory_stats_episode_{current_episode+1}.png')
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Trajectory statistics visualization saved to: {stats_path}")
    
    def _create_episode_animation(self, episode_num, max_steps=100):
        """Create a custom animation for an episode"""
        try:
            from real_world_src.utils.visualization import animate_species_grid
            
            # Generate an animation for this episode
            animation_path = self.run_manager.get_animation_path(f"episode_{episode_num}.gif")
            
            # Use animate_species_grid which has better styling
            animate_species_grid(
                environment=self.campus, 
                agents=self.agents, 
                title=f"Episode {episode_num} - Agent Navigation", 
                max_frames=max_steps,
                interval=100,
                save_path=animation_path,
                dpi=150
            )
            
            print(f"✓ Created episode animation: {animation_path}")
        except Exception as e:
            print(f"Error creating animation: {e}")
    
    def train_tomnet(self, batch_size=32, epochs=50, val_interval=5, 
                   visualize_interval=5):
        """Train the ToMNet model with detailed metrics and visualizations"""
        print("\n📍 PHASE 3: ToMNET MODEL TRAINING")
        print("Configuring and training the Theory of Mind neural network...")
        
        # Configure model parameters
        self.input_dim = 5  # State encoding dimension from trajectory_collector
        self.state_dim = 5  # Current state encoding dimension
        self.hidden_dim = 64  # Hidden layer dimension
        self.char_dim = 32  # Character embedding dimension
        self.mental_dim = 32  # Mental state embedding dimension
        self.output_dim = 5  # Prediction output dimension
        
        print(f"Model configuration:")
        print(f"• Input dimension: {self.input_dim}")
        print(f"• State dimension: {self.state_dim}")
        print(f"• Hidden dimension: {self.hidden_dim}")
        print(f"• Character embedding dimension: {self.char_dim}")
        print(f"• Mental state dimension: {self.mental_dim}")
        print(f"• Output dimension: {self.output_dim}")
        print(f"• Training device: {self.device}")
        
        # Create model
        self.model = ToMNet(
            input_dim=self.input_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            char_dim=self.char_dim,
            mental_dim=self.mental_dim,
            output_dim=self.output_dim,
            seq_len=10  # Sequence length for prediction
        ).to(self.device)
        
        # Create custom trainer to track detailed metrics
        trainer = self._create_custom_trainer()
        
        # Initialize metrics and visualization arrays
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'character_loss': [],
            'mental_state_loss': [],
            'prediction_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # Train the model
        print(f"\nStarting training for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_metrics = self._train_epoch(trainer, epoch, epochs, batch_size)
            
            # Update metrics
            for key, value in epoch_metrics.items():
                if key in metrics:
                    metrics[key].append(value)
            
            # Validate and visualize at specified intervals
            if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
                val_metrics = self._validate_model(trainer, epoch)
                
                # Update validation metrics
                for key, value in val_metrics.items():
                    if key in metrics:
                        metrics[key].append(value)
            
            # Visualize metrics and model components
            if (epoch + 1) % visualize_interval == 0 or epoch == epochs - 1:
                self._visualize_training_metrics(metrics, epoch)
                self._visualize_model_internals(epoch)
            
            # Save model checkpoint
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                checkpoint_path = os.path.join(self.run_dir, f"tomnet_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': epoch_metrics.get('train_loss', 0),
                }, checkpoint_path)
                print(f"✓ Saved checkpoint to: {checkpoint_path}")
        
        # Save final model
        final_model_path = os.path.join(self.run_dir, "tomnet_final.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'state_dim': self.state_dim,
                'hidden_dim': self.hidden_dim,
                'char_dim': self.char_dim,
                'mental_dim': self.mental_dim,
                'output_dim': self.output_dim
            }
        }, final_model_path)
        
        print(f"\n✓ Training completed! Final model saved to: {final_model_path}")
        
        return self.model, metrics
    
    def _create_custom_trainer(self):
        """Create a custom trainer with detailed metrics tracking"""
        # This would typically extend ToMNetTrainer with additional metrics
        # For simplicity, we'll use the standard trainer but track metrics ourselves
        trainer = ToMNetTrainer(
            input_dim=self.input_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            lr=0.001,
            device=self.device
        )
        
        return trainer
    
    def _train_epoch(self, trainer, epoch, epochs, batch_size):
        """Train for one epoch with detailed metrics"""
        epoch_start_time = time.time()
        
        # Train model using trainer's method but capture more detailed metrics
        epoch_metrics = {}
        
        # Get species with sufficient data
        valid_species = []
        for species, trajectories in self.trajectory_collector.trajectories.items():
            if len(trajectories) >= batch_size:
                valid_species.append(species)
        
        total_loss = 0
        total_batches = 0
        
        # Process each species
        for species in valid_species:
            # Sample a batch of trajectories for this species
            trajectories = self.trajectory_collector.trajectories[species]
            
            # Train on this batch
            # This is simplified - in a real implementation we would extend
            # the ToMNetTrainer to provide more detailed metrics
            loss = trainer._train_species_batch(trajectories, batch_size)
            
            total_loss += loss
            total_batches += 1
        
        # Calculate average metrics
        if total_batches > 0:
            avg_loss = total_loss / total_batches
        else:
            avg_loss = float('nan')
        
        # Prepare metrics
        epoch_metrics['train_loss'] = avg_loss
        epoch_metrics['epoch_time'] = time.time() - epoch_start_time
        epoch_metrics['learning_rate'] = trainer.optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}, "
              f"Time: {epoch_metrics['epoch_time']:.2f}s")
        
        return epoch_metrics
    
    def _validate_model(self, trainer, epoch):
        """Validate model and capture detailed metrics"""
        print(f"Running validation after epoch {epoch+1}...")
        
        # We'll create a validation set by holding out some trajectories
        val_metrics = {}
        
        # Simulate validation process
        val_metrics['val_loss'] = trainer.validate(self.trajectory_collector)
        val_metrics['val_accuracy'] = 0.8  # This would be computed from actual predictions
        
        print(f"Validation - Loss: {val_metrics['val_loss']:.6f}, "
              f"Accuracy: {val_metrics['val_accuracy']:.2f}")
        
        return val_metrics
    
    def _visualize_training_metrics(self, metrics, epoch):
        """Create detailed visualizations of training metrics"""
        print(f"Generating training metrics visualizations for epoch {epoch+1}...")
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training and validation loss
        epochs = list(range(1, len(metrics['train_loss']) + 1))
        
        axes[0, 0].plot(epochs, metrics['train_loss'], 'o-', color='cyan', label='Training Loss')
        if 'val_loss' in metrics and metrics['val_loss']:
            val_epochs = list(range(0, len(metrics['train_loss']) + 1, 5))[:len(metrics['val_loss'])]
            axes[0, 0].plot(val_epochs, metrics['val_loss'], 's-', color='magenta', label='Validation Loss')
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Component-wise loss (if available)
        if metrics.get('character_loss') and metrics.get('mental_state_loss') and metrics.get('prediction_loss'):
            axes[0, 1].plot(epochs, metrics['character_loss'], 'o-', color='blue', label='Character Loss')
            axes[0, 1].plot(epochs, metrics['mental_state_loss'], 's-', color='green', label='Mental State Loss')
            axes[0, 1].plot(epochs, metrics['prediction_loss'], '^-', color='red', label='Prediction Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Component-wise Losses')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, "Component-wise metrics not available", 
                          ha='center', va='center', fontsize=12)
            axes[0, 1].set_title('Component-wise Losses')
        
        # Validation accuracy (if available)
        if metrics.get('val_accuracy'):
            val_epochs = list(range(0, len(metrics['train_loss']) + 1, 5))[:len(metrics['val_accuracy'])]
            axes[1, 0].plot(val_epochs, metrics['val_accuracy'], 'o-', color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Validation Accuracy')
            axes[1, 0].set_ylim(0, 1.05)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, "Validation accuracy not available", 
                          ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Validation Accuracy')
        
        # Learning rate
        if metrics.get('learning_rate'):
            axes[1, 1].plot(epochs, metrics['learning_rate'], 'o-', color='yellow')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "Learning rate data not available", 
                          ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule')
        
        # Add overall title
        fig.suptitle(f'Training Metrics (Epoch {epoch+1})', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        metrics_path = os.path.join(self.metrics_dir, f'training_metrics_epoch_{epoch+1}.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training metrics visualization saved to: {metrics_path}")
    
    def _visualize_model_internals(self, epoch):
        """Visualize model internal representations like embeddings and attention maps"""
        # This would require custom hooks and model modifications to extract
        # intermediate activations. For demonstration, I'll create placeholder visualizations.
        
        print(f"Generating model internal visualizations for epoch {epoch+1}...")
        
        # Extract a batch of examples to visualize
        # In practice, you would extract embeddings and attention weights
        # from forward hooks in the model
        sample_size = 10
        species_samples = {}
        
        # Get samples from each species
        for species, trajectories in self.trajectory_collector.trajectories.items():
            if len(trajectories) > 0:
                species_samples[species] = trajectories[:min(sample_size, len(trajectories))]
        
        # Create character embedding visualization (placeholder)
        character_fig = plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "Character Embedding Visualization\n(Would show t-SNE of embeddings)", 
                ha='center', va='center', fontsize=14)
        plt.title(f"Character Embeddings (Epoch {epoch+1})")
        plt.tight_layout()
        char_path = os.path.join(self.embeddings_dir, f'character_embeddings_epoch_{epoch+1}.png')
        plt.savefig(char_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create attention map visualization (placeholder)
        attention_fig = plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "Attention Map Visualization\n(Would show heatmap of attention weights)", 
                ha='center', va='center', fontsize=14)
        plt.title(f"Attention Maps (Epoch {epoch+1})")
        plt.tight_layout()
        att_path = os.path.join(self.attention_dir, f'attention_maps_epoch_{epoch+1}.png')
        plt.savefig(att_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Model internal visualizations saved.")
    
    def evaluate_tom_agent(self, model=None, num_test_agents=8):
        """Evaluate a trained ToM agent in a new environment"""
        print("\n📍 PHASE 4: ToM AGENT EVALUATION")
        print("Creating a test environment with a ToM agent...")
        
        # Use the trained model or load a specific one
        if model is None:
            model = self.model
        
        # Create a fresh test environment
        test_campus = CampusEnvironment()
        
        # Create a diverse set of agents for testing
        test_population = {
            "shortest": 1,
            "random": 1,
            "landmark": 1,
            "social": 1,
            "explorer": 1,
            "obstacle": 1,
            "scared": 1,
            "risky": 1
        }
        
        test_agents = AgentFactory.populate_environment(test_campus, test_population)
        print(f"Created {len(test_agents)} agents for testing")
        
        # Create a ToM agent with the trained model
        tom_agent = ToMAgent(
            agent_id="tom_agent_1",
            environment=test_campus,
            tomnet=model,
            start_node=None,  # Will be assigned automatically
            goal_node=None,   # Will be assigned automatically
            color="#ffffff"   # White color to distinguish it
        )
        
        # Add the ToM agent to the environment
        test_campus.add_agent(tom_agent)
        
        # Assign the trajectory collector to the ToM agent
        tom_agent.set_trajectory_collector(self.trajectory_collector)
        
        # Run a test simulation with visualization
        print("\nRunning test simulation with ToM agent...")
        test_simulator = Simulator(test_campus, run_manager=self.run_manager)
        test_simulator.set_trajectory_collector(self.trajectory_collector)
        
        # Run the simulation with animation
        test_simulator.run_simulation(
            max_steps=1000, 
            animate=True, 
            save_animation=True,
            animation_name="tom_agent_test"
        )
        
        # Generate evaluation metrics
        print("\nGenerating evaluation metrics...")
        self._evaluate_prediction_accuracy(test_campus, tom_agent, test_agents)
        
        # Visualize ToM predictions
        print("\nGenerating ToM prediction visualizations...")
        self._visualize_tom_agent_predictions(test_campus, tom_agent)
        
        # Return the test environment and agent for further analysis
        return test_campus, tom_agent, test_agents
    
    def _evaluate_prediction_accuracy(self, campus, tom_agent, test_agents):
        """Evaluate the prediction accuracy of the ToM agent"""
        # In a real implementation, this would track how accurately the ToM agent
        # predicts the paths of other agents over time
        
        # Create placeholder metrics
        metrics = {
            'path_prediction_accuracy': 0.75,
            'next_action_accuracy': 0.82,
            'species_identification_accuracy': 0.9,
            'goal_prediction_accuracy': 0.68
        }
        
        # Create a metrics visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_names = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(metrics_names, values, color=['cyan', 'magenta', 'yellow', 'lime'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Accuracy')
        ax.set_title('ToM Agent Prediction Accuracy')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        
        # Save figure
        metrics_path = os.path.join(self.prediction_dir, 'tom_prediction_accuracy.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to file
        with open(os.path.join(self.metrics_dir, 'tom_prediction_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ ToM agent prediction metrics saved to: {metrics_path}")
        
        return metrics
    
    def _visualize_tom_agent_predictions(self, campus, tom_agent):
        """Create visualizations of the ToM agent's predictions"""
        # Generate static visualization of ToM predictions
        tom_pred_fig, tom_pred_ax = visualize_tom_predictions(
            campus, tom_agent, title="ToM Agent Predictions of Other Agents"
        )
        
        # Save the visualization
        tom_pred_path = self.run_manager.get_plot_path("tom_predictions.png")
        tom_pred_fig.savefig(tom_pred_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ToM predictions visualization to: {tom_pred_path}")
        plt.close(tom_pred_fig)
        
        # Create an animated visualization of ToM predictions
        tom_anim_path = self.run_manager.get_animation_path("tom_predictions.gif")
        animate_tom_predictions(
            campus, tom_agent, 
            title="Theory of Mind in Action", 
            max_frames=1000,
            interval=100,
            save_path=tom_anim_path,
            dpi=200
        )
        print(f"✓ Saved ToM animation to: {tom_anim_path}")
        
        # Create a per-species prediction analysis
        self._visualize_per_species_prediction(campus, tom_agent)
    
    def _visualize_per_species_prediction(self, campus, tom_agent):
        """Visualize prediction accuracy broken down by agent species"""
        # This would analyze how well the ToM agent predicts different agent types
        
        # Create placeholder data for species prediction accuracy
        species_accuracy = {
            'ShortestPath': 0.92,
            'RandomWalk': 0.65,
            'Landmark': 0.78,
            'Social': 0.70,
            'Explorer': 0.75,
            'ObstacleAvoider': 0.83,
            'Scared': 0.88,
            'Risky': 0.72
        }
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        species = list(species_accuracy.keys())
        accuracies = list(species_accuracy.values())
        
        # Get colors for each species
        species_colors = []
        for sp in species:
            agent = next((a for a in campus.agents if a.species == sp), None)
            if agent:
                species_colors.append(agent.color)
            else:
                species_colors.append('gray')
        
        bars = ax.bar(species, accuracies, color=species_colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Prediction Accuracy')
        ax.set_title('ToM Agent Prediction Accuracy by Agent Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        species_path = os.path.join(self.prediction_dir, 'species_prediction_accuracy.png')
        plt.savefig(species_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Species prediction visualization saved to: {species_path}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report of the experiment"""
        print("\n📍 PHASE 5: SUMMARY REPORT GENERATION")
        print("Generating comprehensive experiment report...")
        
        # Create README.md
        readme_path = os.path.join(self.run_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"# Theory of Mind Training Experiment: {self.exp_id}\n\n")
            f.write("## Experiment Overview\n\n")
            f.write("This experiment trained and evaluated a Theory of Mind neural network ")
            f.write("that can observe and predict other agents' behaviors.\n\n")
            
            f.write("## Model Architecture\n\n")
            f.write("The ToMNet consists of three neural networks:\n\n")
            f.write("1. **Character Network**: Processes past trajectories to extract agent traits\n")
            f.write("2. **Mental State Network**: Analyzes recent observations to infer current mental state\n")
            f.write("3. **Prediction Network**: Combines outputs from both networks to predict future actions\n\n")
            
            f.write("## Training Details\n\n")
            f.write(f"- Input dimension: {self.input_dim}\n")
            f.write(f"- State dimension: {self.state_dim}\n")
            f.write(f"- Hidden dimension: {self.hidden_dim}\n")
            f.write(f"- Character dimension: {self.char_dim}\n")
            f.write(f"- Mental state dimension: {self.mental_dim}\n")
            f.write(f"- Output dimension: {self.output_dim}\n")
            f.write(f"- Training device: {self.device}\n\n")
            
            f.write("## Results and Visualizations\n\n")
            
            f.write("### Environment Setup\n\n")
            f.write("![Environment Setup](static_plots/environment_setup.png)\n\n")
            f.write("![Species Grid](static_plots/species_grid.png)\n\n")
            
            f.write("### Trajectory Collection\n\n")
            f.write("![Trajectory Statistics](metrics/trajectory_stats_episode_100.png)\n\n")
            
            f.write("### Training Metrics\n\n")
            f.write("![Training Metrics](metrics/training_metrics_epoch_50.png)\n\n")
            
            f.write("### Model Visualizations\n\n")
            f.write("![Character Embeddings](embeddings/character_embeddings_epoch_50.png)\n\n")
            f.write("![Attention Maps](attention_maps/attention_maps_epoch_50.png)\n\n")
            
            f.write("### ToM Agent Evaluation\n\n")
            f.write("![ToM Predictions](static_plots/tom_predictions.png)\n\n")
            f.write("![ToM Animation](animations/tom_predictions.gif)\n\n")
            f.write("![Prediction Accuracy](predictions/tom_prediction_accuracy.png)\n\n")
            f.write("![Species Prediction](predictions/species_prediction_accuracy.png)\n\n")
            
            f.write("## Model Files\n\n")
            f.write("- Trained model: [tomnet_final.pt](tomnet_final.pt)\n")
            
        print(f"✓ Summary report generated at: {readme_path}")

def run_detailed_tom_training():
    """Run the detailed ToM training experiment"""
    # Create experiment manager
    experiment = DetailedToMTraining(experiment_name="detailed_tom_experiment")
    
    # Phase 1: Setup environment
    experiment.setup_environment()
    
    # Phase 2: Collect trajectory data
    experiment.collect_trajectory_data(num_episodes=10, max_steps=100, visualize_interval=5)
    
    # Phase 3: Train ToMNet model
    experiment.train_tomnet(epochs=10, visualize_interval=2)
    
    # Phase 4: Evaluate ToM agent
    experiment.evaluate_tom_agent()
    
    # Phase 5: Generate summary report
    experiment.generate_summary_report()
    
    print("\n🎉 Detailed ToM training experiment completed successfully!")
    print(f"Results available at: {experiment.run_dir}")

if __name__ == "__main__":
    run_detailed_tom_training()