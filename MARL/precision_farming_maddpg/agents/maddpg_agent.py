import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # To convert list of arrays efficiently

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(total_state_dim + total_action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, total_state_dim, total_action_dim, lr=1e-3):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(total_state_dim, total_action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(total_state_dim, total_action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.update_targets(1.0)  # hard update at start
    
    def update_targets(self, tau=0.01):
        # Soft update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        probs = self.actor(state_tensor)
        action = torch.argmax(probs).item()
        return action
    
    def update(self, samples, agent_idx, agents, gamma=0.95):
        # samples is a list of tuples: (states, actions, rewards, next_states, dones)
        states, actions, rewards, next_states, dones = zip(*samples)
        batch_size = len(samples)
        
        # Efficient conversion from list of numpy arrays to torch tensors
        states = torch.from_numpy(np.array(states)).float()            # Shape (batch, n_agents, state_dim)
        actions = torch.LongTensor(actions)                             # Shape (batch, n_agents)
        rewards = torch.FloatTensor(rewards)                            # Shape (batch, n_agents)
        next_states = torch.from_numpy(np.array(next_states)).float()  # Shape (batch, n_agents, state_dim)
        dones = torch.BoolTensor(dones)                                 # Shape (batch,)
        
        # Flatten states and one-hot encode actions for critic
        states_flat = states.view(batch_size, -1)
        next_states_flat = next_states.view(batch_size, -1)
        actions_onehot = F.one_hot(actions, num_classes=4).float().view(batch_size, -1)
        
        # Compute next actions from target actors for all agents
        next_actions = []
        for i, agent in enumerate(agents):
            next_prob = agent.target_actor(next_states[:, i, :])
            next_action = torch.argmax(next_prob, dim=1)
            next_actions.append(F.one_hot(next_action, num_classes=4).float())
        next_actions_flat = torch.cat(next_actions, dim=1)
        
        # Critic update
        target_q = self.target_critic(next_states_flat, next_actions_flat).squeeze()
        expected_q = rewards[:, agent_idx] + gamma * target_q * (~dones)
        current_q = self.critic(states_flat, actions_onehot).squeeze()
        critic_loss = F.mse_loss(current_q, expected_q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        curr_policy_actions = []
        for i, agent in enumerate(agents):
            if i == agent_idx:
                prob = agent.actor(states[:, i, :])
                action = torch.argmax(prob, dim=1)
                curr_policy_actions.append(F.one_hot(action, num_classes=4).float())
            else:
                with torch.no_grad():
                    prob = agent.actor(states[:, i, :])
                    action = torch.argmax(prob, dim=1)
                    curr_policy_actions.append(F.one_hot(action, num_classes=4).float())
        curr_policy_actions_flat = torch.cat(curr_policy_actions, dim=1)
        
        actor_loss = -self.critic(states_flat, curr_policy_actions_flat).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_targets()