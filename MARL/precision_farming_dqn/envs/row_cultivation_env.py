import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RowCultivationEnv(gym.Env):
    def __init__(self):
        super(RowCultivationEnv, self).__init__()
        self.n_agents = 2
        self.n_plants = 20

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(42,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([4, 4])

        self.max_steps = 20
        self.reset()

    def reset(self, seed=None, options=None):
        self.plants_diseased = (np.random.rand(self.n_plants) < 0.2).astype(np.float32)
        self.plants_inspected = np.zeros(self.n_plants, dtype=bool)
        self.agent_positions = np.array([0, self.n_plants - 1], dtype=np.int32)
        self.steps = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, actions):
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        for i, action in enumerate(actions):
            pos = self.agent_positions[i]

            if action == 0:
                rewards[i] = 0.0

            elif action == 1:
                if pos < self.n_plants - 1:
                    self.agent_positions[i] += 1
                rewards[i] = 0.1

            elif action == 2:
                if pos > 0:
                    self.agent_positions[i] -= 1
                rewards[i] = -0.01

            elif action == 3:
                plant_idx = self.agent_positions[i]
                if self.plants_inspected[plant_idx]:
                    rewards[i] = -0.2
                else:
                    self.plants_inspected[plant_idx] = True
                    if self.plants_diseased[plant_idx] == 1:
                        rewards[i] = 5.0
                    else:
                        rewards[i] = -0.1
            else:
                raise ValueError(f"Invalid action {action} for agent {i}")

        reward = float(np.sum(rewards))

        self.steps += 1
        done = self.steps >= self.max_steps

        obs = self._get_obs()
        truncated = False
        info = {
            'agent_positions': self.agent_positions.copy(),
            'plants_diseased': self.plants_diseased.copy(),
            'plants_inspected': self.plants_inspected.copy(),
            'individual_rewards': rewards.copy()
        }
        return obs, reward, done, truncated, info

    def _get_obs(self):
        norm_pos = self.agent_positions / (self.n_plants - 1)
        obs = []
        obs.extend(norm_pos.tolist())
        obs.extend(self.plants_diseased.tolist())
        obs.extend(self.plants_inspected.astype(float).tolist())
        return np.array(obs, dtype=np.float32)
