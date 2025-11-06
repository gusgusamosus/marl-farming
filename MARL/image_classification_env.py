import numpy as np
import torch
from gymnasium import spaces
import matplotlib.pyplot as plt


class ImageClassificationEnv:
    def __init__(self, dataset, encoder, num_classes, device='cpu'):
        self.dataset = dataset
        self.encoder = encoder.to(device).eval()
        self.device = device
        self.num_classes = num_classes
        self.action_space = spaces.Discrete(num_classes)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)
        
        # State variables
        self.idx = None
        self.label = None
        self.last_action = None
        self.visited_indices = set()
        
        # Episode management
        self.episode_length = 0
        self.max_episode_length = 20  # Fixed episode length
        
        # Class names
        self.class_names = [
            'Tomato_Bacterial_Spot',
            'Tomato_Early_Blight',
            'Tomato_Healthy',
            'Tomato_Late_Blight',
            'Tomato_Leaf_Mold',
            'Tomato_Mosaic_Virus',
            'Tomato_Septoria_Leaf_Spot',
            'Tomato_Spider_mites Two-spotted_spider_mite',
            'Tomato_Target_Spot',
            'Tomato_Yellow_Leaf_Curl_Virus'
        ]
        
        # Reward parameters (improved)
        self.correct_reward = 5.0
        self.incorrect_reward = -1.0
        self.repeat_penalty = -1.0  # Stronger penalty
        self.revisit_penalty = -0.5  # Penalty for revisiting same sample
        
        # Visualization
        self.visualize_each_step = False  # Disable by default for training speed
        self.fig = None
        self.ax = None

    def reset(self):
        """Reset environment to initial state"""
        self.idx = np.random.randint(len(self.dataset))
        img, label = self.dataset[self.idx]
        self.label = label
        self.last_action = None
        self.visited_indices = {self.idx}
        self.episode_length = 0
        
        with torch.no_grad():
            img = img.unsqueeze(0).to(self.device)
            embedding = self.encoder(img).squeeze(0).cpu().numpy()
        
        return embedding

    def step(self, action):
        """Execute one step in the environment"""
        true_label_name = self.class_names[self.label]
        pred_label_name = self.class_names[action]
        
        # Calculate base reward (simplified)
        if action == self.label:
            reward = self.correct_reward  # Correct prediction
        else:
            reward = self.incorrect_reward  # Wrong prediction
        
        # Repeat penalty (same action as last time)
        if self.last_action is not None and action == self.last_action:
            reward += self.repeat_penalty
        
        self.last_action = action
        self.episode_length += 1
        
        # Move to next sample (sequential or random)
        self.idx = (self.idx + 1) % len(self.dataset)
        
        # Revisit penalty (optional - discourage seeing same sample twice)
        if self.idx in self.visited_indices:
            reward += self.revisit_penalty
        self.visited_indices.add(self.idx)
        
        # Get next observation
        img, label = self.dataset[self.idx]
        self.label = label
        with torch.no_grad():
            img = img.unsqueeze(0).to(self.device)
            embedding = self.encoder(img).squeeze(0).cpu().numpy()
        
        # Check if episode is done
        done = (self.episode_length >= self.max_episode_length)
        
        # Info dict
        info = {
            'true_label': true_label_name,
            'predicted_label': pred_label_name,
            'reward': reward,
            'current_index': self.idx,
            'episode_length': self.episode_length,
            'correct': (action == self.label)
        }
        
        # Visualization (if enabled)
        if self.visualize_each_step:
            self.visualize()
        
        return embedding, reward, done, info

    def visualize(self):
        """Visualize agent position in dataset"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 2))
            plt.ion()
        
        self.ax.clear()
        
        # Show a subset of dataset (e.g., first 100 samples)
        max_display = min(100, len(self.dataset))
        x = np.arange(max_display)
        
        # Plot all positions
        self.ax.scatter(x, np.zeros_like(x), color='lightgray', s=300, alpha=0.5)
        
        # Highlight visited positions
        visited_display = [idx for idx in self.visited_indices if idx < max_display]
        if visited_display:
            self.ax.scatter(visited_display, np.zeros(len(visited_display)), 
                          color='blue', s=350, alpha=0.6, label='Visited')
        
        # Show current position
        if self.idx < max_display:
            self.ax.scatter([self.idx], [0], color='red', s=500, 
                          marker='*', label='Current', zorder=10)
        
        self.ax.set_xlim(-1, max_display)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_xticks(range(0, max_display, 10))
        self.ax.set_yticks([])
        self.ax.set_title(f'Agent Position: {self.idx} | Episode Step: {self.episode_length}/{self.max_episode_length}')
        self.ax.legend(loc='upper right')
        
        plt.pause(0.01)
        plt.draw()

    def close(self):
        """Clean up visualization"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def get_stats(self):
        """Get environment statistics"""
        return {
            'episode_length': self.episode_length,
            'max_episode_length': self.max_episode_length,
            'visited_count': len(self.visited_indices),
            'current_index': self.idx,
            'current_label': self.class_names[self.label] if self.label is not None else None
        }
