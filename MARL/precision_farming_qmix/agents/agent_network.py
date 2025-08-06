import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(AgentNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        return self.fc2(x)
