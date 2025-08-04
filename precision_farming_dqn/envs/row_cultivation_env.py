import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RowCultivationEnv(gym.Env):
    """Custom Gymnasium environment for two agents in row cultivation."""

    def __init__(self):
        super(RowCultivationEnv, self).__init__()
        self.n_agents = 2
        self.state_dim = 8      # Example state: [x1, y1, x2, y2, ...]
        self.action_dim = 4     # [forward, back, left, right], per agent

        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.action_dim] * self.n_agents)
        self.reset()

    def reset(self, seed=None, options=None):
        # Initialize positions randomly within bounds
        self.state = np.random.rand(self.state_dim).astype(np.float32)
        self.steps = 0
        info = {}
        return self.state, info

    def step(self, actions):
        """Process both agent actions and update state."""
        # Dummy logic: random walk updates
        assert len(actions) == self.n_agents
        for idx, action in enumerate(actions):
            # For real world, map actions (0...3) to state changes
            if action == 0:  # forward
                self.state[idx*4] += 0.05
            elif action == 1:  # back
                self.state[idx*4] -= 0.05
            elif action == 2:  # left
                self.state[idx*4+1] -= 0.05
            elif action == 3:  # right
                self.state[idx*4+1] += 0.05
        self.state = np.clip(self.state, 0, 1)

        # Dummy reward logic: negative for leaving bounds, positive for "progress"
        reward = np.sum(self.state) / self.state_dim
        done = False
        self.steps += 1
        if self.steps >= 200:
            done = True
        info = {}
        return self.state, reward, done, False, info  # (obs, reward, terminated, truncated, info)
