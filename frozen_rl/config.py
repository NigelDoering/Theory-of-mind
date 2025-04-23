class Config:
    def __init__(self):
        self.default_params = {
            "grid_size": 8,
            "num_goals": 5,
            "is_slippery": False,
            "final_goal_reward": 10.0,
            "hole_reward": -1.0,
            "step_reward": -0.1,
            "alpha_high": 0.2,
            "alpha_low": 0.5,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "min_epsilon": 0.01,
            "seed": 42
        }

    def get_params(self):
        return self.default_params

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.default_params:
                self.default_params[key] = value

    def save_config(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.default_params, f)

    def load_config(self, filepath):
        import json
        with open(filepath, 'r') as f:
            self.default_params = json.load(f)