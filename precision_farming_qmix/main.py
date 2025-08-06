import os
import csv
import yaml
import torch
import torch.optim as optim
import numpy as np

from envs.row_cultivation_env import RowCultivationEnv
from agents.agent_network import AgentNet
from qmix.qmix_network import QMIXMixer
from qmix.replay_buffer import ReplayBuffer

# ---- Config ----
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_agents = config['n_agents']
field_length = config['field_length']
buffer_size = config['buffer_size']
batch_size = config['batch_size']
episodes = config['episodes']
gamma = config['gamma']
epsilon_start = config['epsilon_start']
epsilon_final = config['epsilon_final']
epsilon_decay = config['epsilon_decay']
save_interval = config.get('save_interval', 100)
lr = config.get('lr', 1e-3)
hidden_dim = config.get('hidden_dim', 64)
target_update_interval = config.get('target_update_interval', 100)

# ---- Environment and Networks ----
env = RowCultivationEnv(n_agents=n_agents, field_length=field_length)
obs_dim = env.observation_space.shape[1]
action_dim = env.action_space.n

# Fix: state_dim = n_agents * obs_dim for concatenated agent observations as global state input
state_dim = n_agents * obs_dim

agents = [AgentNet(obs_dim, action_dim, hidden_dim).to(device) for _ in range(n_agents)]
target_agents = [AgentNet(obs_dim, action_dim, hidden_dim).to(device) for _ in range(n_agents)]
mixer = QMIXMixer(n_agents, state_dim).to(device)
target_mixer = QMIXMixer(n_agents, state_dim).to(device)

for i in range(n_agents):
    target_agents[i].load_state_dict(agents[i].state_dict())
target_mixer.load_state_dict(mixer.state_dict())

agent_opts = [optim.Adam(agent.parameters(), lr=lr) for agent in agents]
mixer_opt = optim.Adam(mixer.parameters(), lr=lr)

replay_buffer = ReplayBuffer(buffer_size)

# ---- CSV Logging Setup ----
log_file = "episode_scores.csv"
# Overwrite file (empty) at start
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "TotalReward"])  # header

# ---- Helper Functions ----

def select_actions(obs_batch, epsilon):
    actions = []
    for i, agent in enumerate(agents):
        obs = torch.tensor(obs_batch[i], dtype=torch.float32).unsqueeze(0).to(device)
        if np.random.rand() < epsilon:
            action = np.random.randint(action_dim)
        else:
            with torch.no_grad():
                q_values = agent(obs)
                action = q_values.argmax().item()
        actions.append(action)
    return np.array(actions)

def update_targets():
    for i in range(n_agents):
        target_agents[i].load_state_dict(agents[i].state_dict())
    target_mixer.load_state_dict(mixer.state_dict())

def pad_episode(episode, max_steps):
    states, actions, rewards, next_states, dones = episode
    n_agents = states.shape[1]
    obs_dim = states.shape[2]

    pad_len = max_steps - states.shape[0]

    # Pad states and next_states: (episode_len, n_agents, obs_dim)
    states_pad = np.pad(states, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
    next_states_pad = np.pad(next_states, ((0, pad_len), (0, 0), (0, 0)), mode='constant')

    # Pad actions and rewards: (episode_len, n_agents)
    actions_pad = np.pad(actions, ((0, pad_len), (0, 0)), mode='constant')
    rewards_pad = np.pad(rewards, ((0, pad_len), (0, 0)), mode='constant')

    # Pad dones: (episode_len,), pad with True to mask out
    dones_pad = np.pad(dones, (0, pad_len), mode='constant', constant_values=True)

    return (states_pad, actions_pad, rewards_pad, next_states_pad, dones_pad)

# ---- Training Loop ----
epsilon = epsilon_start
epsilon_decay_rate = (epsilon_start - epsilon_final) / epsilon_decay

for ep in range(episodes):
    obs = env.reset()
    states, actions, rewards, next_states, dones = [], [], [], [], []
    done = False
    ep_reward = 0.0

    while not done:
        action = select_actions(obs, epsilon)
        next_obs, reward, done, _ = env.step(action)

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_obs)
        dones.append(done)

        ep_reward += reward.sum()
        obs = next_obs

    # Convert lists to numpy arrays
    states_np = np.array(states)
    actions_np = np.array(actions)
    rewards_np = np.array(rewards)
    next_states_np = np.array(next_states)
    dones_np = np.array(dones)

    replay_buffer.push((states_np, actions_np, rewards_np, next_states_np, dones_np))

    # Log episode reward into CSV
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ep + 1, ep_reward])

    # Anneal epsilon
    epsilon = max(epsilon_final, epsilon - epsilon_decay_rate)

    # Training step
    if len(replay_buffer.buffer) >= batch_size:
        batch = replay_buffer.sample(batch_size)

        max_ep_len = max(ep_data[0].shape[0] for ep_data in batch)
        padded_batch = [pad_episode(ep_data, max_ep_len) for ep_data in batch]

        s_batch = np.array([ep[0] for ep in padded_batch])   # (batch, max_ep_len, n_agents, obs_dim)
        a_batch = np.array([ep[1] for ep in padded_batch])   # (batch, max_ep_len, n_agents)
        r_batch = np.array([ep[2] for ep in padded_batch])   # (batch, max_ep_len, n_agents)
        ns_batch = np.array([ep[3] for ep in padded_batch])  # (batch, max_ep_len, n_agents, obs_dim)
        d_batch = np.array([ep[4] for ep in padded_batch])   # (batch, max_ep_len)

        for t in range(max_ep_len):
            obs_t = torch.tensor(s_batch[:, t], dtype=torch.float32).to(device)          # [batch, n_agents, obs_dim]
            acts_t = torch.tensor(a_batch[:, t], dtype=torch.long).to(device)             # [batch, n_agents]
            rews_t = torch.tensor(r_batch[:, t], dtype=torch.float32).to(device)          # [batch, n_agents]
            next_obs_t = torch.tensor(ns_batch[:, t], dtype=torch.float32).to(device)     # [batch, n_agents, obs_dim]
            done_t = torch.tensor(d_batch[:, t], dtype=torch.bool).to(device)             # [batch]

            agent_qs = []
            target_agent_qs = []
            for i_agent in range(n_agents):
                q = agents[i_agent](obs_t[:, i_agent, :])
                tq = target_agents[i_agent](next_obs_t[:, i_agent, :])
                agent_qs.append(q.gather(1, acts_t[:, i_agent].unsqueeze(-1)).squeeze())
                target_agent_qs.append(tq.max(dim=1)[0])
            agent_qs = torch.stack(agent_qs, dim=1)          # [batch, n_agents]
            target_agent_qs = torch.stack(target_agent_qs, dim=1)  # [batch, n_agents]

            # Flatten global state (concatenated agent obs)
            state_flat = obs_t.view(agent_qs.size(0), -1)
            next_state_flat = next_obs_t.view(agent_qs.size(0), -1)

            q_tot = mixer(agent_qs, state_flat)
            with torch.no_grad():
                target_q_tot = target_mixer(target_agent_qs, next_state_flat)
                targets = rews_t.sum(dim=1, keepdim=True) + gamma * target_q_tot * (~done_t).unsqueeze(1).float()

            loss = torch.nn.functional.mse_loss(q_tot, targets)

            for opt in agent_opts:
                opt.zero_grad()
            mixer_opt.zero_grad()
            loss.backward()
            for opt in agent_opts:
                opt.step()
            mixer_opt.step()

    # Update target networks periodically
    if (ep + 1) % target_update_interval == 0:
        update_targets()

    # Save checkpoint
    if (ep + 1) % save_interval == 0:
        os.makedirs("checkpoints", exist_ok=True)
        for i, agent in enumerate(agents):
            torch.save(agent.state_dict(), f"checkpoints/agent_{i}_ep{ep + 1}.pth")
        torch.save(mixer.state_dict(), f"checkpoints/mixer_ep{ep + 1}.pth")
        print(f"[Episode {ep + 1}] Model saved, epsilon={epsilon:.3f}")

    if (ep + 1) % 10 == 0:
        print(f"[Episode {ep + 1}] Epsilon: {epsilon:.3f} | Episode Reward: {ep_reward:.2f}")

print("Training finished.")