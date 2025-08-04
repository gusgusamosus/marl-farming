import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay_buffer import ReplayBuffer

class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, state):
        return self.fc(state)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, buffer_size=50000, batch_size=128, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQNNet(state_dim, action_dim).to(self.device)
        self.target_net = DQNNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = nn.MSELoss()
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.learn_steps = 0
        self.target_update_freq = 1000

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.model.fc[-1].out_features)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(state)
        return q.argmax(dim=1).item()

    def store_transition(self, s, a, r, s_, done):
        self.buffer.push((s, a, r, s_, done))

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_curr = self.model(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + (1 - dones) * self.gamma * q_next
        loss = self.loss_fn(q_curr, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        # Update target network
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.model.state_dict())

        # Decay epsilon
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
