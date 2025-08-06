import torch
import torch.nn.functional as F
import numpy as np
import csv
from envs.farming_env import FarmingEnv
from models.networks import ActorNet, CriticNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CSV Logging Setup ---
log_file = 'training_log.csv'
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Header with combined FP, hits, repeats
    writer.writerow([
        'Episode',
        'AvgReward',
        'FalsePositives_Combined',
        'Hits_Combined',
        'Repeats_Combined'
    ])

def log_training_data(episode, avg_reward,
                      false_positives, hits, repeats):
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            episode,
            avg_reward,
            false_positives,
            hits,
            repeats
        ])

# --- Hyperparameters ---
n_agents = 2
n_actions = 4
obs_dim = 6  # matches env observation shape
lr = 2e-3
gamma = 0.98
gae_lambda = 0.95
eps_clip = 0.2
K_epochs = 4
batch_size = 32
max_episodes = 5000
max_steps = 20

env = FarmingEnv()
actors = [ActorNet(obs_dim, n_actions).to(device) for _ in range(n_agents)]
critics = [CriticNet(obs_dim).to(device) for _ in range(n_agents)]
optimizers = [torch.optim.Adam(list(actors[i].parameters()) + list(critics[i].parameters()), lr=lr) for i in range(n_agents)]

def select_action(agent, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    logits = actors[agent](obs_t)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    return action.cpu().numpy(), dist.log_prob(action).detach().cpu().numpy(), dist.entropy().mean().item()

def compute_gae(rewards, values, gamma=0.98, lam=0.95):
    rewards = np.array(rewards)
    values = np.array(values + [0])
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[step])
    return returns

for episode in range(max_episodes):
    memory = [{} for _ in range(n_agents)]
    for ag in range(n_agents):
        memory[ag]['obs'] = []
        memory[ag]['actions'] = []
        memory[ag]['rewards'] = []
        memory[ag]['log_probs'] = []
        memory[ag]['values'] = []
        memory[ag]['entropies'] = []

    # Track combined false positives, hits, repeats per agent
    false_positives_per_agent = [0, 0]
    hits_per_agent = [0, 0]
    repeats_per_agent = [0, 0]

    obs, _ = env.reset()
    t = 0
    terminated = False

    while t < max_steps and not terminated:
        acts, lps, ents, vals = [], [], [], []
        for i in range(n_agents):
            a, lp, ent = select_action(i, obs[i])
            acts.append(a)
            lps.append(lp)
            ents.append(ent)
            with torch.no_grad():
                val = critics[i](torch.tensor(obs[i], dtype=torch.float32, device=device)).cpu().item()
            vals.append(val)
            memory[i]['obs'].append(obs[i])
            memory[i]['actions'].append(a)
            memory[i]['log_probs'].append(lp)
            memory[i]['values'].append(val)
            memory[i]['entropies'].append(ent)

        next_obs, rewards, terminated, _, _ = env.step(tuple(acts))
        obs = next_obs
        t += 1

        for i in range(n_agents):
            memory[i]['rewards'].append(rewards[i])
            # Count hits, false positives, repeats based on reward heuristics:
            # - hits: reward == 5.0 (inspect diseased plant)
            # - false positives: reward == -0.1 (inspect healthy plant)
            # - repeats: reward == -0.2 (repeat inspection)
            if rewards[i] == 5.0:
                hits_per_agent[i] += 1
            elif rewards[i] == -0.1:
                false_positives_per_agent[i] += 1
            elif rewards[i] == -0.2:
                repeats_per_agent[i] += 1

    # Compute returns and advantages & PPO update
    for i in range(n_agents):
        with torch.no_grad():
            last_val = critics[i](torch.tensor(obs[i], dtype=torch.float32, device=device)).cpu().item()
        memory[i]['values'].append(last_val)
        returns = compute_gae(memory[i]['rewards'], memory[i]['values'], gamma, gae_lambda)
        advantages = np.array(returns) - np.array(memory[i]['values'][:-1])

        obs_batch = torch.tensor(np.array(memory[i]['obs']), dtype=torch.float32, device=device)
        actions_batch = torch.tensor(np.array(memory[i]['actions']), dtype=torch.long, device=device)
        logprobs_batch = torch.tensor(np.array(memory[i]['log_probs']), dtype=torch.float32, device=device)
        returns_batch = torch.tensor(np.array(returns), dtype=torch.float32, device=device)
        adv_batch = torch.tensor(advantages, dtype=torch.float32, device=device)

        for _ in range(K_epochs):
            idx = np.arange(len(obs_batch))
            np.random.shuffle(idx)
            for start in range(0, len(obs_batch), batch_size):
                end = start + batch_size
                mb_idx = idx[start:end]
                obs_b = obs_batch[mb_idx]
                act_b = actions_batch[mb_idx]
                old_logprobs_b = logprobs_batch[mb_idx]
                ret_b = returns_batch[mb_idx]
                adv_b = adv_batch[mb_idx]

                logits = actors[i](obs_b)
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(act_b)
                ratio = (new_logprobs - old_logprobs_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                value = critics[i](obs_b)
                critic_loss = F.mse_loss(value, ret_b)

                ent_loss = -dist.entropy().mean() * 0.01

                total_loss = actor_loss + 0.5 * critic_loss + ent_loss

                optimizers[i].zero_grad()
                total_loss.backward()
                optimizers[i].step()

    # Compute average episode rewards for logging
    avg_rewards = [np.mean(memory[i]['rewards']) for i in range(n_agents)]
    total_reward = sum(avg_rewards)
    avg_reward = total_reward / n_agents
    false_positives_combined = sum(false_positives_per_agent)
    hits_combined = sum(hits_per_agent)
    repeats_combined = sum(repeats_per_agent)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1} | Avg Reward: {avg_reward:.3f}, "
              f"FP combined: {false_positives_combined}, Hits combined: {hits_combined}, Repeats combined: {repeats_combined}")

    log_training_data(episode + 1, 
                      avg_reward,
                      false_positives_combined, hits_combined, repeats_combined)

print("Training completed and logged to", log_file)
