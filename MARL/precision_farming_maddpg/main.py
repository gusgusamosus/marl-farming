from envs.farming_env import FarmingEnv
from agents.maddpg_agent import MADDPGAgent
from utils import ReplayBuffer

n_agents = 2
state_dim = 4
action_dim = 4
n_episodes = 50000      
max_steps = 50         
batch_size = 32

env = FarmingEnv()
agents = [
    MADDPGAgent(state_dim, action_dim, total_state_dim=state_dim*n_agents, total_action_dim=action_dim*n_agents)
    for _ in range(n_agents)
]
replay_buffer = ReplayBuffer()

episode_scores = []

for episode in range(n_episodes):
    states, _ = env.reset()
    terminated = False
    truncated = False
    step = 0
    total_episode_reward = 0

    while not (terminated or truncated) and step < max_steps:
        actions = [agent.select_action(state) for agent, state in zip(agents, states)]
        next_states, rewards, terminated, truncated, _ = env.step(actions)

        replay_buffer.add((states, actions, rewards, next_states, terminated or truncated))

        total_episode_reward += sum(rewards)

        states = next_states
        step += 1

        # Uncomment to render environment (slow for large runs)
        # env.render()

        if len(replay_buffer) >= batch_size:
            samples = replay_buffer.sample(batch_size)
            for agent_idx, agent in enumerate(agents):
                agent.update(samples, agent_idx, agents)

    episode_scores.append(total_episode_reward)

    # Print average only every 100 episodes
    if (episode + 1) % 1000 == 0:
        recent_scores = episode_scores[-1000:]
        running_avg = sum(recent_scores) / len(recent_scores)
        print(
            f"Episode {episode+1}: average score (last 1000 episodes): {running_avg:.2f}"
        )

print(f"\nTraining completed. Final average score over all {n_episodes} episodes: {sum(episode_scores) / len(episode_scores):.2f}")