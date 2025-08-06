import torch
import torch.nn as nn
import torch.nn.functional as F

class QMIXMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super(QMIXMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w1(states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)
        y = torch.bmm(hidden, w2) + b2
        q_tot = y.view(-1, 1)
        return q_tot
