import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FarmingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        super().__init__()
        self.n_agents = 2
        self.action_space = [spaces.Discrete(4) for _ in range(self.n_agents)]
        self.observation_space = [spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32) for _ in range(self.n_agents)]
        self.state = [np.zeros(4, dtype=np.float32) for _ in range(self.n_agents)]
        self.episode_count = 0  # Initialize episode counter
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1   # Increment episode count at start of each episode
        self.state = [np.random.rand(4).astype(np.float32) for _ in range(self.n_agents)]
        return self.state, {}
    
    def step(self, actions):
        rewards = [0.0 for _ in range(self.n_agents)]
        terminated = False
        truncated = False
        info = {}
        
        for idx, action in enumerate(actions):
            if action == 0:
                rewards[idx] = 0.0
            elif action == 1:
                rewards[idx] = 1.0
            elif action == 2:
                rewards[idx] = 0.5
            elif action == 3:
                rewards[idx] = 0.2
            
            self.state[idx] = np.random.rand(4).astype(np.float32)
        
        return self.state, rewards, terminated, truncated, info
    
    def render(self, mode='human'):
        if mode == 'human':
            # Print agent states only every 1000 episodes
            if self.episode_count % 1000 == 0:
                print(f"Episode {self.episode_count}: Agent states: {self.state}")