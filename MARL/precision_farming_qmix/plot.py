import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file with the actual columns; header is present
df = pd.read_csv("episode_scores.csv")  # 'Episode' and 'TotalReward'

# Plot total reward per episode
plt.plot(df['Episode'], df['TotalReward'])
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
