import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FarmingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, n_agents=2, n_plants=20, max_steps=50):
        super().__init__()
        self.n_agents = n_agents
        self.n_plants = n_plants
        self.max_steps = max_steps
        self.action_space = [spaces.Discrete(4) for _ in range(self.n_agents)]
        self.observation_space = [
            spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.episode_count = 0
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.episode_count += 1
        self.field = np.zeros(self.n_plants, dtype=int)
        # Ensure at least 1 diseased plant present
        n_diseased = max(1, np.random.randint(1, self.n_plants // 4 + 1))
        diseased_indices = np.random.choice(self.n_plants, size=n_diseased, replace=False)
        self.field[diseased_indices] = 1
        self.agent_positions = np.zeros(self.n_agents, dtype=int)
        self.already_sampled = np.zeros(self.n_plants, dtype=bool)
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for agent in range(self.n_agents):
            pos = self.agent_positions[agent]
            obs.append([
                pos / (self.n_plants - 1),
                self.field[pos],
                float(self.already_sampled[pos]),
                1.0 - self.steps / self.max_steps
            ])
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        rewards = [0.0 for _ in range(self.n_agents)]
        self.steps += 1

        for idx, action in enumerate(actions):
            pos = self.agent_positions[idx]
            # Actions: 0=noop, 1=forward, 2=backward, 3=inspect
            if action == 1 and pos < self.n_plants - 1:
                self.agent_positions[idx] += 1
            elif action == 2 and pos > 0:
                self.agent_positions[idx] -= 1
            elif action == 3:
                if not self.already_sampled[pos]:
                    if self.field[pos] == 1:
                        rewards[idx] += 5.0   # Increased positive reward
                    else:
                        rewards[idx] -= 0.1   # Reduced false positive penalty
                    self.already_sampled[pos] = True
                else:
                    rewards[idx] -= 0.2       # Reduced repeat inspection penalty

        for idx, action in enumerate(actions):
            if action in [1, 2]:
                rewards[idx] -= 0.01          # Reduced movement penalty

        obs = self._get_obs()
        terminated = self.steps >= self.max_steps or np.all(self.already_sampled)
        truncated = False
        info = {}

        return obs, rewards, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human' and self.episode_count % 1000 == 0:
            print(f"Episode {self.episode_count}:")
            print(f"Field (0:healthy,1:diseased): {self.field}")
            print(f"Agent positions: {self.agent_positions}")
            print(f"Already sampled: {self.already_sampled}")
