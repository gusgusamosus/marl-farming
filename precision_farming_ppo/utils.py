import matplotlib.pyplot as plt
import csv

def plot_training_log(csv_file='training_log.csv'):
    episodes = []
    agent0_rewards = []
    agent1_rewards = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            episodes.append(int(row['Episode']))
            agent0_rewards.append(float(row['Agent0_AvgReward']))
            agent1_rewards.append(float(row['Agent1_AvgReward']))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, agent0_rewards, label='Agent 0 Avg Reward')
    plt.plot(episodes, agent1_rewards, label='Agent 1 Avg Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    plot_training_log()