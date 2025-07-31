import random
import csv
import os
from envs.farming_env import FarmingEnv
from agents.maddpg_agent import MADDPGAgent
from utils import ReplayBuffer

# --- Experiment parameters ---
n_agents = 2 #change this if needed but for this project 2/4/4 will suffice
state_dim = 4 #ensure that these two...
action_dim = 4 #...are the same
n_plants = 20
n_episodes = 5000 #change this as needed
max_steps = 25 #and this
batch_size = 32

# --- Logging setup ---
logfile = "training_log.csv"
with open(logfile, 'w') as f:
    pass  # This will clear the file

# --- Env and agent setup ---
env = FarmingEnv(n_agents=n_agents, n_plants=n_plants, max_steps=max_steps)
agents = [
    MADDPGAgent(state_dim, action_dim, total_state_dim=state_dim * n_agents, total_action_dim=action_dim * n_agents)
    for _ in range(n_agents)
]
replay_buffer = ReplayBuffer()

episode_scores = []
detection_stats = []

for episode in range(n_episodes):
    states, _ = env.reset()
    terminated = False
    truncated = False
    step = 0
    total_episode_reward = 0
    hits = 0
    false_positives = 0
    repeats = 0

    while not (terminated or truncated) and step < max_steps:
        epsilon = max(0.05, 1 - episode / 10000)  # epsilon-greedy exploration
        actions = []
        for agent, state in zip(agents, states):
            if random.random() < epsilon:
                actions.append(random.randint(0, 3))
            else:
                actions.append(agent.select_action(state))

        pre_sampled = env.already_sampled.copy()  # snapshot before env.step

        next_states, rewards, terminated, truncated, _ = env.step(actions)

        # Correct event logging
        for idx, action in enumerate(actions):
            pos = env.agent_positions[idx]
            if action == 3:
                if pre_sampled[pos]:
                    repeats += 1
                elif env.field[pos] == 1:
                    hits += 1
                else:
                    false_positives += 1

        replay_buffer.add((states, actions, rewards, next_states, terminated or truncated))

        total_episode_reward += sum(rewards)
        states = next_states
        step += 1

        if len(replay_buffer) >= batch_size:
            samples = replay_buffer.sample(batch_size)
            for agent_idx, agent in enumerate(agents):
                agent.update(samples, agent_idx, agents)

    episode_scores.append(total_episode_reward)
    detection_stats.append((hits, false_positives, repeats))

    # --- Write to CSV log ---
    with open(logfile, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode + 1, total_episode_reward, hits, false_positives, repeats])

    # Print summary every 100 episodes only
    if (episode + 1) % 100 == 0:
        recent_scores = episode_scores[-100:]
        recent_detections = detection_stats[-100:]
        avg_score = sum(recent_scores) / len(recent_scores)
        avg_hits = sum(h for h, _, _ in recent_detections) / 100
        avg_false = sum(f for _, f, _ in recent_detections) / 100
        avg_repeats = sum(r for _, _, r in recent_detections) / 100
        print(
            f"Episode {episode+1}: Average Score={avg_score:.2f}, "
            f"Hits={avg_hits:.1f}, False Positives={avg_false:.1f}, Repeats={avg_repeats:.1f}"
        )

print(f"\nTraining completed. Final average score: {sum(episode_scores) / len(episode_scores):.2f}")
print(f"Training log saved to: {logfile}")
