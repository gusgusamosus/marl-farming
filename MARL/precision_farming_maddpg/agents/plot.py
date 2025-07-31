import pandas as pd
import matplotlib.pyplot as plt

# Specify column names since CSV has no header
df = pd.read_csv("training_log.csv", names=["episode", "score", "hits", "false_positives", "repeats"])

plt.plot(df['episode'], df['score'])
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()

plt.plot(df['episode'], df['hits'])
plt.title("Hits per Episode")
plt.xlabel("Episode")
plt.ylabel("Hits")
plt.show()

plt.plot(df['episode'], df['false_positives'])
plt.title("False Positives per Episode")
plt.xlabel("Episode")
plt.ylabel("False Positives")
plt.show()

plt.plot(df['episode'], df['repeats'])
plt.title("Repeats per Episode")
plt.xlabel("Episode")
plt.ylabel("Repeats")
plt.show()