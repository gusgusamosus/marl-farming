import numpy as np
import torch
import csv
import os
from envs.row_cultivation_env import RowCultivationEnv
from dqn_agent import DQNAgent

NUM_EPISODES = 5000
MAX_STEPS = 20
LOG_FILE = "dqn_training_log.csv"

def init_csv_logger(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "avg_score",
            "avg_hits_per_agent",
            "avg_false_positives_per_agent",
            "avg_repeats_per_agent"
        ])

def log_to_csv(log_file, episode, avg_score,
               avg_hits, avg_false_positives, avg_repeats):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode, avg_score, avg_hits, avg_false_positives, avg_repeats
        ])

def main():
    env = RowCultivationEnv()
    n_agents = env.n_agents
    state_dim = env.observation_space.shape[0]
    action_dim = 4

    agents = [DQNAgent(state_dim, action_dim) for _ in range(n_agents)]

    init_csv_logger(LOG_FILE)

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        done = False

        agent_rewards = np.zeros(n_agents, dtype=np.float32)
        total_steps = 0

        hits = np.zeros(n_agents, dtype=np.int32)
        false_positives = np.zeros(n_agents, dtype=np.int32)
        repeats = np.zeros(n_agents, dtype=np.int32)

        for step in range(MAX_STEPS):
            actions = [agent.select_action(state) for agent in agents]
            next_state, reward, done, truncated, info = env.step(actions)
            rewards_indiv = info['individual_rewards']

            for i in range(n_agents):
                r = rewards_indiv[i]
                if r == 5.0:
                    hits[i] += 1
                elif r == -0.1:
                    false_positives[i] += 1
                elif r == -0.2:
                    repeats[i] += 1

            for i, agent in enumerate(agents):
                agent.store_transition(state, actions[i], rewards_indiv[i], next_state, done)
                agent.train()

            state = next_state
            agent_rewards += rewards_indiv
            total_steps += 1

            if done:
                break

        total_reward = agent_rewards.sum()
        avg_score = total_reward / n_agents if total_steps > 0 else 0.0
        avg_hits = hits.mean()
        avg_false_positives = false_positives.mean()
        avg_repeats = repeats.mean()
        avg_score = total_reward / total_steps if total_steps > 0 else 0.0

        log_to_csv(LOG_FILE, ep, avg_score, avg_hits, avg_false_positives, avg_repeats)

        if ep % 10 == 0:
            print(f"Episode {ep}: Agent1 reward = {agent_rewards[0]:.3f}, Agent2 reward = {agent_rewards[1]:.3f}, "
                  f"Total = {total_reward:.3f}, Avg score/step = {avg_score:.3f}, Hits = {avg_hits:.1f}, "
                  f"False Positives = {avg_false_positives:.1f}, Repeats = {avg_repeats:.1f}")

if __name__ == "__main__":
    main()
