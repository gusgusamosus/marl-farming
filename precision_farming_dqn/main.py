import numpy as np
import torch
import csv
import os
from envs.row_cultivation_env import RowCultivationEnv
from dqn_agent import DQNAgent

NUM_EPISODES = 50000
MAX_STEPS = 200
LOG_FILE = "dqn_training_log.csv"

def init_csv_logger(log_file):
    """Always initialize the CSV log file with header, clearing previous content."""
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward"])

def log_to_csv(log_file, episode, reward):
    """Append episode results to CSV file."""
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, reward])

def main():
    env = RowCultivationEnv()
    n_agents = env.n_agents
    state_dim = env.state_dim
    action_dim = env.action_dim

    agents = [DQNAgent(state_dim, action_dim) for _ in range(n_agents)]

    init_csv_logger(LOG_FILE)

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(MAX_STEPS):
            actions = [agent.select_action(state) for agent in agents]
            next_state, reward, done, truncated, info = env.step(actions)
            for agent in agents:
                agent.store_transition(state, agent.select_action(state), reward, next_state, done)
                agent.train()
            state = next_state
            episode_reward += reward
            if done:
                break

        log_to_csv(LOG_FILE, ep, episode_reward)

        if ep % 10 == 0:
            print(f"Episode {ep}, Total Reward: {episode_reward}")

if __name__ == "__main__":
    main()
