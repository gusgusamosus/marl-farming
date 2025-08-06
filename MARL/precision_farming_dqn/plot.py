import pandas as pd
import matplotlib.pyplot as plt

def plot_rewards(csv_file='dqn_training_log.csv'):
    # Load CSV data
    data = pd.read_csv(csv_file)

    # Plot total_reward vs episode
    plt.figure(figsize=(10,6))
    plt.plot(data['episode'], data['total_reward'], label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Reward Progression')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_rewards()
