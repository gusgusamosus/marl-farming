import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FarmingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super().__init__()
        self.n_agents = 2
        self.n_actions = 4
        self.n_plants = 20
        self.max_steps = 20
        self.state_dim = 6  # We'll use 6-dimensional obs: normalized pos(1), last_action(4 onehot), plant_status(1)

        # Define action and observation spaces
        self.action_space = spaces.Tuple([
            spaces.Discrete(self.n_actions) for _ in range(self.n_agents)
        ])

        # Observation per agent: vector with
        # [normalized_position] + [last_action_onehot, 4 dims] + [plant_disease (0/1) under agent's position]
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ])

        # Environment attributes
        self.plant_states = None  # np array size 20, 0 = healthy, 1 = diseased
        self.agent_positions = None  # agents' indices in 0..19
        self.last_actions = None  # last actions for each agent (integer 0-3)
        self.inspected_plants = None  # list of sets, tracks indices previously inspected by each agent

        self.current_step = 0
        self.reset()

    def reset(self, *, seed=None, options=None):
        # Randomly assign diseased plants (~20% diseased)
        self.plant_states = np.zeros(self.n_plants, dtype=np.int32)
        diseased_indices = np.random.choice(self.n_plants, size=int(self.n_plants * 0.2), replace=False)
        self.plant_states[diseased_indices] = 1
        
        # Agents start positions: start at left (pos 0) and right (pos 19)
        self.agent_positions = [0, self.n_plants - 1]  
        self.last_actions = [0 for _ in range(self.n_agents)]  # start with no-op
        self.inspected_plants = [set() for _ in range(self.n_agents)]
        self.current_step = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for i in range(self.n_agents):
            pos_norm = self.agent_positions[i] / (self.n_plants - 1)  # normalize position 0 to 1
            last_action_oh = np.zeros(self.n_actions, dtype=np.float32)
            last_action_oh[self.last_actions[i]] = 1.0
            plant_status = float(self.plant_states[self.agent_positions[i]])
            observation = np.concatenate([[pos_norm], last_action_oh, [plant_status]])
            obs.append(observation.astype(np.float32))
        return tuple(obs)

    def step(self, actions):
        rewards = []
        info = {}
        # For each agent, apply action and compute rewards
        for i, action in enumerate(actions):
            pos = self.agent_positions[i]
            reward = 0.0
            
            # Apply action penalty/reward
            if action == 0:
                # No-op
                reward = 0.0
            elif action == 1:
                # Move forward (increase position by 1 but max at n_plants-1)
                if pos < self.n_plants - 1:
                    self.agent_positions[i] += 1
                reward = +0.1
            elif action == 2:
                # Move backward (decrease position by 1 but min at 0)
                if pos > 0:
                    self.agent_positions[i] -= 1
                reward = -0.01
            elif action == 3:
                # Inspect current plant at agent position
                plant_idx = self.agent_positions[i]
                diseased = self.plant_states[plant_idx] == 1
                repeated_inspection = plant_idx in self.inspected_plants[i]
                if repeated_inspection:
                    reward = -0.2
                else:
                    if diseased:
                        reward = 5.0
                    else:
                        reward = -0.1
                    # Mark plant as inspected by this agent
                    self.inspected_plants[i].add(plant_idx)

            else:
                raise ValueError(f"Invalid action {action}")

            rewards.append(reward)
            self.last_actions[i] = action  # update last action

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        obs = self._get_obs()
        return obs, tuple(rewards), terminated, False, info

    def render(self, mode='human'):
        # Display agent positions, plant disease status, or just print a simple summary
        row_str = ""
        for i in range(self.n_plants):
            plant_char = "D" if self.plant_states[i] == 1 else "."
            if i == self.agent_positions[0] and i == self.agent_positions[1]:
                row_str += "[B]"  # Both agents
            elif i == self.agent_positions[0]:
                row_str += "[A]"
            elif i == self.agent_positions[1]:
                row_str += "[C]"
            else:
                row_str += f" {plant_char} "
        print(f"Step: {self.current_step} | Row: {row_str}")
