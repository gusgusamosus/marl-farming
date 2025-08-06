import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RowCultivationEnv(gym.Env):
    def __init__(self, n_agents=3, field_length=10):
        super(RowCultivationEnv, self).__init__()
        self.n_agents = n_agents
        self.field_length = field_length
        self.action_space = spaces.Discrete(4)  # e.g., move forward, stay, turn
        self.observation_space = spaces.Box(low=0, high=field_length, shape=(n_agents, 2), dtype=np.float32)
        self.reset()

    def reset(self):
        # Initialize agents at random positions (row major)
        self.agent_positions = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.done = False
        return self.agent_positions

    def step(self, actions):
        rewards = np.zeros(self.n_agents)
        for i, action in enumerate(actions):
            if action == 0:  # Move forward
                self.agent_positions[i, 0] += 1
            elif action == 1:  # Stay
                pass
            elif action == 2:  # Turn (simulate side movement)
                self.agent_positions[i, 1] += 1
            # Reward logic: reward for not colliding and for moving forward
            if self.agent_positions[i, 0] < self.field_length:
                rewards[i] = 1
            else:
                rewards[i] = -1  # penalty for moving past edge
        self.done = np.any(self.agent_positions[:, 0] >= self.field_length)
        return self.agent_positions.copy(), rewards, self.done, {}

    def render(self, mode='human'):
        print("Agent positions:", self.agent_positions)