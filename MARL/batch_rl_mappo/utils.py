import pandas as pd
import matplotlib.pyplot as plt

def plot_rewards(csv_file='training_log.csv'):
    # Load CSV data
    data = pd.read_csv(csv_file)

    # Create a single figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    fig.suptitle('PPO Training Metrics Over Episodes', fontsize=16)

    # Plot 1: Total Reward vs Episode
    axs[0, 0].plot(data['Episode'], data['AvgReward'], label='Episode Reward', color="blue")
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Plot 2: Hits_Combined Progression
    axs[0, 1].plot(data['Episode'], data['Hits_Combined'], label='Hits', color="red")
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Plot 3: False Positives Progression
    axs[1, 0].plot(data['Episode'], data['FalsePositives_Combined'], label='False Positives', color="green")
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Plot 4: Repeats_Combined Progression
    axs[1, 1].plot(data['Episode'], data['Repeats_Combined'], label='Repeats', color="yellow")
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

plot_rewards()
