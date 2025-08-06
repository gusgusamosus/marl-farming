import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RowCultivationEnv(gym.Env):
    def __init__(self, n_agents=2, row_length=20, max_steps=20):
        super(RowCultivationEnv, self).__init__()
        self.n_agents = n_agents
        self.row_length = row_length
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)

        # Observations: agent's normalized position on the row [0, 1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_agents, 1), dtype=np.float32)

        self.reset()

    def reset(self):
        self.agent_positions = np.zeros(self.n_agents, dtype=np.int32)  # All agents start at position 0
        self.step_count = 0
        self.diseased = np.random.rand(self.row_length) < 0.15  # 15% plants diseased
        self.inspected = [set() for _ in range(self.n_agents)]  # track inspected plants per agent
        self.done = False
        # Reset trackers for stats for hits, false_positives, repeats per episode
        self.hits = 0
        self.false_positives = 0
        self.repeats = 0
        return self._get_obs()

    def _get_obs(self):
        # Normalized positions [0,1]
        return (self.agent_positions / (self.row_length - 1)).reshape(self.n_agents, 1).astype(np.float32)

    def step(self, actions):
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        self.step_count += 1

        for i, action in enumerate(actions):
            pos = self.agent_positions[i]
            if action == 0:
                # No-op
                reward = 0.0
            elif action == 1:
                # Move forward with +0.1 penalty → reward = -0.1
                if pos < self.row_length - 1:
                    self.agent_positions[i] += 1
                reward = -0.1
            elif action == 2:
                # Move backward with -0.01 penalty → reward = -0.01
                if pos > 0:
                    self.agent_positions[i] -= 1
                reward = -0.01
            elif action == 3:
                # Inspect plant
                plant_idx = self.agent_positions[i]
                if plant_idx in self.inspected[i]:
                    reward = -0.2
                    self.repeats += 1
                else:
                    self.inspected[i].add(plant_idx)
                    if self.diseased[plant_idx]:
                        reward = 5.0
                        self.hits += 1
                    else:
                        reward = -0.1
                        self.false_positives += 1
            else:
                reward = 0.0

            rewards[i] = reward

        self.done = self.step_count >= self.max_steps
        return self._get_obs(), rewards, self.done, {}

    def render(self, mode='human'):
        row = ["_"] * self.row_length
        for idx in np.where(self.diseased)[0]:
            row[idx] = "D"
        agent_symbols = ["A", "B"]
        for i, pos in enumerate(self.agent_positions):
            row[pos] = agent_symbols[i]
        print("".join(row))
        print(f"Step: {self.step_count} Positions: {self.agent_positions}")
