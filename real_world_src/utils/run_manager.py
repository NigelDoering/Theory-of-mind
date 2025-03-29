import os
import json
import datetime
import shutil

class RunManager:
    """Manages run numbering and folder creation for simulations."""
    
    def __init__(self, base_dir='visuals'):
        """
        Initialize the run manager.
        
        Args:
            base_dir: Base directory for all visualization outputs
        """
        self.base_dir = os.path.abspath(base_dir)
        self.run_info_file = os.path.join(self.base_dir, 'run_info.json')
        self.current_run = None
        self.run_dir = None
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
    def start_new_run(self):
        """Start a new run, creating appropriate folders and updating run number."""
        # Load existing run info or create new
        if os.path.exists(self.run_info_file):
            with open(self.run_info_file, 'r') as f:
                run_info = json.load(f)
                last_run = run_info.get('last_run', 0)
        else:
            run_info = {'last_run': 0, 'runs': {}}
            last_run = 0
            
        # Increment run number
        self.current_run = last_run + 1
        
        # Create run-specific directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_dir, f"run_{self.current_run}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create subdirectories for different types of visualizations
        os.makedirs(os.path.join(self.run_dir, 'animations'), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'static_plots'), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'agent_decisions'), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'species_plots'), exist_ok=True)
        
        # Update run info
        run_info['last_run'] = self.current_run
        run_info['runs'][str(self.current_run)] = {
            'timestamp': timestamp,
            'directory': self.run_dir
        }
        
        with open(self.run_info_file, 'w') as f:
            json.dump(run_info, f, indent=2)
            
        print(f"Started new simulation run #{self.current_run}")
        print(f"All visualizations will be saved to: {self.run_dir}")
        
        return self.run_dir
        
    def get_animation_path(self, filename):
        """Get path for an animation file in the current run directory."""
        if not self.run_dir:
            self.start_new_run()
        return os.path.join(self.run_dir, 'animations', filename)
    
    def get_plot_path(self, filename):
        """Get path for a static plot file in the current run directory."""
        if not self.run_dir:
            self.start_new_run()
        return os.path.join(self.run_dir, 'static_plots', filename)
    
    def get_agent_decision_path(self, filename):
        """Get path for an agent decision visualization in the current run directory."""
        if not self.run_dir:
            self.start_new_run()
        return os.path.join(self.run_dir, 'agent_decisions', filename)
    
    def get_species_plot_path(self, filename):
        """Get path for a species-specific plot in the current run directory."""
        if not self.run_dir:
            self.start_new_run()
        return os.path.join(self.run_dir, 'species_plots', filename)