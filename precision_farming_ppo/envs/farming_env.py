import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FarmingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super().__init__()
        self.n_agents = 2
        self.n_actions = 4
        self.state_dim = 4
        self.action_space = spaces.Tuple([spaces.Discrete(self.n_actions) for _ in range(self.n_agents)])
        self.observation_space = spaces.Tuple([spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32) for _ in range(self.n_agents)])
        self.max_steps = 50
        self.reset()
        
    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.state = [np.random.rand(self.state_dim).astype(np.float32) for _ in range(self.n_agents)]
        return tuple(self.state), {}

    def step(self, actions):
        # Example transition; replace with domain-specific logic
        rewards = [float(action == self.current_step % self.n_actions) for action in actions]
        self.state = [np.random.rand(self.state_dim).astype(np.float32) for _ in range(self.n_agents)]
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return tuple(self.state), tuple(rewards), terminated, False, {}

    def render(self, mode='human'):
        print(f"Current state: {self.state}")
