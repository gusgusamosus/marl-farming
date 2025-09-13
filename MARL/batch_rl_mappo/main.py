import torch
import torch.nn.functional as F
import numpy as np
import csv
import os
from envs.farming_env import FarmingEnv
from models.networks import ActorNet, CriticNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CSV Logging Setup ---
log_file = 'training_log.csv'
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
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


def get_last_logged_episode(log_file_path):
    """Return the largest episode number found in the CSV log (0 if none).

    This ensures new runs continue numbering from the last logged episode
    even if no checkpoint is present.
    """
    if not os.path.exists(log_file_path):
        return 0
    last = 0
    try:
        with open(log_file_path, newline='') as f:
            reader = csv.reader(f)
            # skip header
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                try:
                    val = int(row[0])
                    if val > last:
                        last = val
                except (ValueError, TypeError):
                    # skip malformed rows
                    continue
    except Exception as e:
        print(f"Warning: could not read {log_file_path}: {e}")
    return last

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

# --- Environment ---
env = FarmingEnv()

# --- Agents ---
actors = [ActorNet(obs_dim, n_actions).to(device) for _ in range(n_agents)]
critics = [CriticNet(obs_dim).to(device) for _ in range(n_agents)]
optimizers = [torch.optim.Adam(list(actors[i].parameters()) + list(critics[i].parameters()), lr=lr) for i in range(n_agents)]

# --- Checkpoint helpers ---
def save_checkpoint(actors, critics, optimizers, episode, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save({
        'actors': [a.state_dict() for a in actors],
        'critics': [c.state_dict() for c in critics],
        'optimizers': [o.state_dict() for o in optimizers],
        'episode': episode
    }, filename)

def load_checkpoint(filename):
    if os.path.exists(filename):
        # First check the stored episode number without committing to loading weights
        try:
            meta = torch.load(filename, map_location="cpu")
        except TypeError:
            # fallback for older torch versions
            meta = torch.load(filename, map_location="cpu")
        except Exception as e:
            print(f"Warning: failed to read checkpoint metadata {filename}: {e}")
            return 0

        # Ensure checkpoint is a dict and contains an episode key
        if not isinstance(meta, dict) or 'episode' not in meta:
            print(f"Found checkpoint at {filename} but it lacks an 'episode' field; ignoring it")
            return 0

        saved_episode = int(meta.get('episode', 0))
        # Only accept checkpoints that represent a completed run of at least max_episodes
        if saved_episode < max_episodes:
            print(f"Found checkpoint at {filename} but episode={saved_episode} < required {max_episodes}; ignoring checkpoint")
            return 0

        # The checkpoint looks like a completed run; load full weights onto the configured device
        try:
            checkpoint = torch.load(filename, map_location=device)
            for i in range(n_agents):
                actors[i].load_state_dict(checkpoint['actors'][i])
                critics[i].load_state_dict(checkpoint['critics'][i])
                optimizers[i].load_state_dict(checkpoint['optimizers'][i])
            print(f"Resuming training from episode {checkpoint['episode']}")
            return checkpoint['episode']
        except Exception as e:
            print(f"Error loading checkpoint weights from {filename}: {e}")
            return 0
    return 0

# --- PPO Helpers ---
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

# --- Training ---
checkpoint_path = "ppo_checkpoints/ppo_final.pth"
# episode from checkpoint (0 if none)
ckpt_episode = load_checkpoint(checkpoint_path)
# episode from the CSV log (0 if none)
last_logged = get_last_logged_episode(log_file)
# start from whichever is larger so logging continues sequentially
start_episode = max(ckpt_episode, last_logged)
if start_episode > ckpt_episode:
    print(f"No checkpoint or checkpoint older than log; starting numbering at {start_episode + 1} based on {log_file}")
total_episodes = start_episode + max_episodes  # extend by max_episodes every run

for episode in range(start_episode, total_episodes):
    memory = [{} for _ in range(n_agents)]
    for ag in range(n_agents):
        memory[ag]['obs'] = []
        memory[ag]['actions'] = []
        memory[ag]['rewards'] = []
        memory[ag]['log_probs'] = []
        memory[ag]['values'] = []
        memory[ag]['entropies'] = []

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
            if rewards[i] == 5.0:
                hits_per_agent[i] += 1
            elif rewards[i] == -0.1:
                false_positives_per_agent[i] += 1
            elif rewards[i] == -0.2:
                repeats_per_agent[i] += 1

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

    avg_rewards = [np.mean(memory[i]['rewards']) for i in range(n_agents)]
    avg_reward = sum(avg_rewards) / n_agents
    false_positives_combined = sum(false_positives_per_agent)
    hits_combined = sum(hits_per_agent)
    repeats_combined = sum(repeats_per_agent)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1} | Avg Reward: {avg_reward:.3f}, "
              f"FP combined: {false_positives_combined}, Hits combined: {hits_combined}, Repeats combined: {repeats_combined}")

    log_training_data(episode + 1, avg_reward,
                      false_positives_combined, hits_combined, repeats_combined)

# Save final checkpoint
save_checkpoint(actors, critics, optimizers, total_episodes, checkpoint_path)
print(f"Training completed and final checkpoint saved at {checkpoint_path}")



