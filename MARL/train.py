import os
import glob
import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from cnn_encoder import CNNEncoder
from image_classification_env import ImageClassificationEnv
from policy import ActorCritic
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, episode, checkpoint_dir='checkpoints', max_to_keep=10):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{episode}.pt')
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f'[Checkpoint] Saved at {checkpoint_path}')
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt')), key=os.path.getctime)
    while len(checkpoints) > max_to_keep:
        os.remove(checkpoints[0])
        checkpoints = checkpoints[1:]


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'[Checkpoint] Loaded from {checkpoint_path}, episode={checkpoint["episode"]}')
    return checkpoint['episode']


def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt'))
    if not checkpoints:
        return None

    # Extract episode numbers and filenames
    checkpoint_episodes = []
    for ckpt in checkpoints:
        try:
            # Filename format: checkpoint_XXXX.pt
            base = os.path.basename(ckpt)
            ep_str = base.split('_')[1].split('.')[0]
            ep_num = int(ep_str)
            checkpoint_episodes.append((ep_num, ckpt))
        except Exception as e:
            print(f"Warning: Failed to parse checkpoint {ckpt}: {e}")
    
    if not checkpoint_episodes:
        return None

    # Pick checkpoint with highest episode number
    latest_ckpt = max(checkpoint_episodes, key=lambda x: x[0])[1]
    return latest_ckpt



def log_detailed_episode_result(filename, episode_num, total_reward, hits, false_positives, repeats, avg_loss):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Episode', 'Total Reward', 'Hits', 'False Positives', 'Repeats', 'Avg Loss'])
        writer.writerow([episode_num, total_reward, hits, false_positives, repeats, avg_loss])


def compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95):
    """Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    return advantages


def visualize_environment_batch(images, true_labels, pred_labels, class_names, num_cols=5):
    num_samples = len(images)
    num_rows = (num_samples + num_cols - 1) // num_cols
    plt.figure(figsize=(3 * num_cols, 3.4 * num_rows))
    for i in range(num_samples):
        plt.subplot(num_rows, num_cols, i + 1)
        img = images[i]
        if torch.is_tensor(img):
            img = img.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.axis('off')
        correct = pred_labels[i] == true_labels[i]
        pred = class_names[pred_labels[i]].replace("Tomato_", "")
        true = class_names[true_labels[i]].replace("Tomato_", "")
        color = "green" if correct else "red"
        plt.title(f"GT: {true}\nPred: {pred}", color=color, fontsize=8, pad=4)
    plt.subplots_adjust(hspace=0.35, wspace=0.15)
    plt.tight_layout()
    plt.show()


# ----- Device & Data Setup -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enhanced data augmentation
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='TomatoDataset', transform=transform_train)
print(f"Dataset size: {len(train_dataset)} images")
print(f"Number of classes: {len(train_dataset.classes)}")

# ----- Agent and Environment Setup -----
encoder = CNNEncoder().to(device)
encoder.eval()  # Keep frozen for stability

num_classes = len(train_dataset.classes)
num_agents = 2
episode_len = 10
batch_size = 32  # Increased from 16

policy = ActorCritic(obs_dim=512, action_dim=num_classes).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-4, weight_decay=1e-5)  # Lower LR + regularization
scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)  # Smooth LR decay


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []


buffer = RolloutBuffer()

# Improved hyperparameters
clip_param = 0.2  # Balanced clipping
ppo_epochs = 12  # More optimization steps
entropy_coeff = 0.05  # Moderate exploration
value_loss_coeff = 0.5
max_grad_norm = 0.5  # Gradient clipping
num_episodes = 20500  # Longer training
save_every = 500

latest_ckpt = find_latest_checkpoint()
start_episode = 0
if latest_ckpt:
    start_episode = load_checkpoint(policy, optimizer, latest_ckpt)
    # Restore scheduler state
    for _ in range(start_episode):
        scheduler.step()


def get_episode_sample_indices(num_samples):
    dataset_size = len(train_dataset)
    return torch.randperm(dataset_size)[:num_samples]


class_names = [
    "Tomato_Bacterial_Spot", "Tomato_Early_Blight", "Tomato_Healthy", "Tomato_Late_Blight",
    "Tomato_Leaf_Mold", "Tomato_Mosaic_Virus", "Tomato_Septoria_Leaf_Spot",
    "Tomato_Spider_mites Two-spotted_spider_mite", "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus"
]

# Training loop
print(f"\nStarting training from episode {start_episode + 1}...")
best_reward = float('-inf')
running_rewards = []

for episode in range(start_episode, num_episodes):
    obs_ep = []
    actions_ep = []
    log_probs_ep = []
    rewards_ep = []
    dones_ep = []
    values_ep = []

    hits = 0
    false_positives = 0
    repeats = 0
    last_action = [None for _ in range(num_agents)]
    total_reward = 0

    idxs = get_episode_sample_indices(episode_len * num_agents)
    idxs = idxs.view(episode_len, num_agents)

    images_to_visualize = []
    true_labels_vis = []
    pred_labels_vis = []

    # Epsilon decay for exploration
    epsilon = max(0.01, 0.3 - (episode / num_episodes) * 0.29)

    for t in range(episode_len):
        obs_t = []
        actions_t = []
        log_probs_t = []
        rewards_t = []
        dones_t = []
        values_t = []
        
        for agent_id in range(num_agents):
            idx = idxs[t, agent_id].item()
            img, label = train_dataset[idx]
            with torch.no_grad():
                img_tensor = img
                img = img.unsqueeze(0).to(device)
                embedding = encoder(img).squeeze(0)
            
            obs_tensor = embedding
            
            # Epsilon-greedy exploration
            if np.random.rand() < epsilon:
                action = np.random.randint(0, num_classes)
                with torch.no_grad():
                    action_logits, _ = policy.forward(obs_tensor.unsqueeze(0))
                    dist = torch.distributions.Categorical(logits=action_logits)
                    log_prob = dist.log_prob(torch.tensor(action).to(device))
                # Make sure log_prob is a tensor with shape [1]
                log_prob = log_prob.unsqueeze(0) if log_prob.dim() == 0 else log_prob
            else:
                action, log_prob = policy.act(obs_tensor)
                # Make sure log_prob is tensor with shape [1]
                log_prob = log_prob.unsqueeze(0) if log_prob.dim() == 0 else log_prob
                with torch.no_grad():
                    _, value = policy.forward(obs_tensor.unsqueeze(0))

            # Simplified & balanced reward
            if action == label:
                reward = 5.0  # Correct prediction
            else:
                reward = -1.0  # Wrong prediction

            # Strong repeat penalty
            if t > 0 and last_action[agent_id] is not None and last_action[agent_id] == action:
                repeats += 1
                reward += -2.0  # Stronger penalty
            last_action[agent_id] = action

            if action == label:
                hits += 1
            else:
                false_positives += 1

            obs_t.append(obs_tensor)
            actions_t.append(torch.tensor(action).to(device))
            log_probs_t.append(log_prob)
            rewards_t.append(torch.tensor(reward, dtype=torch.float32).to(device))
            dones_t.append(0)
            values_t.append(value.squeeze())
            total_reward += reward

            images_to_visualize.append(img_tensor)
            true_labels_vis.append(label)
            pred_labels_vis.append(action)

        obs_ep.append(torch.stack(obs_t))
        actions_ep.append(torch.stack(actions_t))
        log_probs_ep.append(torch.stack(log_probs_t))
        rewards_ep.append(torch.stack(rewards_t))
        dones_ep.append(torch.tensor(dones_t, dtype=torch.bool))
        values_ep.append(torch.stack(values_t))

    # Visualize every 200 episodes
    if episode % 200 == 0 and episode > 0:
        visualize_environment_batch(images_to_visualize, true_labels_vis, pred_labels_vis, class_names)

    buffer.obs.append(torch.stack(obs_ep))
    buffer.actions.append(torch.stack(actions_ep))
    buffer.log_probs.append(torch.stack(log_probs_ep))
    buffer.rewards.append(torch.stack(rewards_ep))
    buffer.dones.append(torch.stack(dones_ep))
    buffer.values.append(torch.stack(values_ep))

    running_rewards.append(total_reward)
    
    # PPO Update
    if (episode + 1) % batch_size == 0:
        obs_batch = torch.stack(buffer.obs).view(-1, 512)
        actions_batch = torch.stack(buffer.actions).view(-1)
        old_log_probs_batch = torch.stack(buffer.log_probs).view(-1)
        rewards_batch = torch.stack(buffer.rewards).view(-1)
        values_batch = torch.stack(buffer.values).view(-1)

        # GAE computation
        values_list = values_batch.cpu().detach().numpy().tolist()
        rewards_list = rewards_batch.cpu().numpy().tolist()
        advantages = compute_gae(rewards_list, values_list, gamma=0.99, gae_lambda=0.95)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = advantages + values_batch[:len(advantages)]

        epoch_losses = []
        for _ in range(ppo_epochs):
            action_logits, state_values = policy.forward(obs_batch)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_batch.detach())
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages.detach()

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns.detach())
            loss = actor_loss + value_loss_coeff * critic_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        scheduler.step()
        buffer.clear()

        # Save best model
        avg_recent_reward = np.mean(running_rewards[-batch_size:])
        if avg_recent_reward > best_reward:
            best_reward = avg_recent_reward
            save_checkpoint(policy, optimizer, episode + 1, checkpoint_dir='checkpoints_best', max_to_keep=3)
    else:
        avg_loss = 0.0

    if (episode + 1) % save_every == 0:
        save_checkpoint(policy, optimizer, episode + 1)

    if (episode + 1) % 20 == 0:
        avg_reward_20 = np.mean(running_rewards[-20:]) if len(running_rewards) >= 20 else total_reward
        print(f"Ep {episode + 1}/{num_episodes} | Reward: {total_reward:.1f} (Avg20: {avg_reward_20:.1f}) | "
              f"Hits: {hits}/{episode_len*num_agents} | FP: {false_positives} | Rep: {repeats} | "
              f"ε: {epsilon:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    log_detailed_episode_result('detailed_training_log.csv', episode + 1, total_reward, hits, false_positives, repeats, avg_loss)

print("\n✓ Training complete!")
