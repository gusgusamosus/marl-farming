import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBase(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super(MLPBase, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, obs):
        x = self.activation(self.fc1(obs))
        x = self.activation(self.fc2(x))
        return x

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.base = MLPBase(obs_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        base_out = self.base(obs)
        action_logits = self.actor(base_out)
        state_value = self.critic(base_out)
        return action_logits, state_value

    def act(self, obs):
        action_logits, _ = self.forward(obs)
        action_prob = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_prob)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_actions(self, obs, actions):
        action_logits, state_value = self.forward(obs)
        action_prob = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_prob)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_log_probs, torch.squeeze(state_value), dist_entropy
